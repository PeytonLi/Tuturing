"""CPU-only smoke workflow (PRD 11.2, gate G4): ten scripted sessions through the public API.

The workflow drives Person A's public API in-process with the CPU response stub against a
learning service (Person B's real service over HTTP, or A's transaction stand-in), exercises a
retry after an injected dropped connection on every session, and writes a JSON report. It needs
no GPU, credential, dataset or internet access.

    export TUTOR_LEARNING_SERVICE_TOKEN=dev
    uv run python -m tools.learning.serve --config configs/learning/rules.yaml --port 8100
    uv run python -m tools.response.smoke_sessions --learning-url http://127.0.0.1:8100 \
        --report work/public_api/smoke_report.json

The scripted learner messages are synthetic fixtures written by Person A; the report records
message lengths and metadata, never message text. A passing run shows the transaction and
recovery path works end to end; it is not a measurement of teaching quality.
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.public_api.clients import HttpLearningClient, InProcessResponseClient
from apps.public_api.config import LearningConfig, PublicApiConfig
from apps.public_api.contract import ROOT, find_internal_fields, load_contract
from apps.public_api.errors import ApiError
from apps.public_api.service import PublicApi
from apps.public_api.storage import RequestStore

log = logging.getLogger("tutor.smoke")

ENABLED_GOALS = ("C01", "C02", "C03", "C06", "C08")
PACES = ("normal", "concise", "detailed")

# Ten scripted learners. The first turn is always the empty first message (source null).
# Messages avoid concepts outside B's enabled set because B rejects such requests.
SCRIPTS: list[dict[str, Any]] = [
    {"name": "novice", "messages": ["idk", "1/2", "is it 0.25?", "hint please", "3/4", "stop"]},
    {"name": "confident", "messages": ["1/2", "0.5", "50%", "1/4", "2/3"]},
    {"name": "skipper", "messages": ["skip", "can we go faster", "1/2", "0.3"]},
    {"name": "slow", "messages": ["can you slow down", "I think 1/2", "maybe 2/4", "1/8"]},
    {"name": "answer_seeker", "messages": ["just tell me the answer", "idk", "0.5", "hint"]},
    {"name": "goal_change", "messages": ["switch the goal to sample space", "1/2", "0.5"]},
    {"name": "typos", "messages": ["i dont know", "1 / 2", "30 percent", "0.3?"]},
    {"name": "rambler", "messages": ["so I was thinking " * 150 + "1/2", "1/2", "0.5"]},
    {"name": "immediate_stop", "messages": ["stop"]},
    {"name": "budget", "messages": ["1/2"] * 16},
]


class DroppedConnection(RuntimeError):
    """Injected: the committed reply never reached the client."""


def _turn(request_id: str, version: int, source: str | None, message: str) -> dict:
    return {
        "request_id": request_id,
        "expected_state_version": version,
        "source_response_id": source,
        "message": message,
    }


def run_smoke(
    api: PublicApi,
    *,
    caller: str = "smoke",
    scripts: list[dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
    learning_label: str = "unknown",
) -> dict:
    scripts = SCRIPTS if scripts is None else scripts
    contract = load_contract()
    run_id = f"smoke-{secrets.token_hex(4)}"
    started = datetime.now(UTC)
    original_hook = api.hook
    sessions: list[dict] = []
    latencies: list[float] = []
    modes: Counter[str] = Counter()
    moves: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    errors: list[dict] = []
    retries = {"exercised": 0, "identical": 0, "hash_checked": 0, "hash_matches_learning": 0}

    def public(schema: str, body: Any) -> None:
        problems = contract.public_issues(schema, body) + find_internal_fields(body)
        if problems:
            raise AssertionError(f"non-conforming {schema}: {problems}")

    for index, script in enumerate(scripts):
        prefix = f"{run_id}-s{index:02d}"
        goal = ENABLED_GOALS[index % len(ENABLED_GOALS)]
        pace = PACES[index % len(PACES)]
        record: dict[str, Any] = {
            "script": script["name"],
            "goal_concept_id": goal,
            "pace_preference": pace,
            "turns": [],
            "end_reason": None,
            "ended_by": None,
            "errors": [],
        }
        sessions.append(record)
        try:
            created = api.create_session(
                caller,
                {
                    "request_id": f"{prefix}-create",
                    "goal_concept_id": goal,
                    "pace_preference": pace,
                },
            )
            public("CreateSessionResult", created.body)
            session_id = created.body["session_id"]
            record["session_id"] = session_id
            version, source = 0, None
            retry_turn = min(1, len(script["messages"]))
            for turn_index, message in enumerate(["", *script["messages"]]):
                request_id = f"{prefix}-t{turn_index:02d}"
                body = _turn(request_id, version, source, message)
                dropped = False
                if turn_index == retry_turn:

                    def drop_once(point: str, context: dict, _rid=request_id) -> None:
                        original_hook(point, context)
                        if point == "before_reply" and context["record"].request_id == _rid:
                            api.hook = original_hook
                            raise DroppedConnection(_rid)

                    api.hook = drop_once
                    try:
                        api.turn(caller, session_id, body)
                    except DroppedConnection:
                        dropped = True
                    finally:
                        api.hook = original_hook
                started_turn = time.perf_counter()
                reply = api.turn(caller, session_id, body)
                elapsed_ms = (time.perf_counter() - started_turn) * 1000
                latencies.append(elapsed_ms)
                public("TurnResult", reply.body)
                if dropped:
                    retries["exercised"] += 1
                    again = api.turn(caller, session_id, body)
                    stored = api.store.get_request(caller, request_id)
                    if again.body == reply.body and (
                        "session_end" in reply.body
                        or reply.body["response"]["response_id"]
                        == stored.staged_response["response_id"]
                    ):
                        retries["identical"] += 1
                    if "response" in reply.body:
                        retries["hash_checked"] += 1
                        status = api.learning.plan_status(
                            reply.body["response"]["plan_id"], trace_id=stored.trace_id
                        )
                        if status.get("content_hash") == stored.content_hash:
                            retries["hash_matches_learning"] += 1
                if "session_end" in reply.body:
                    record["end_reason"] = reply.body["session_end"]["reason"]
                    record["ended_by"] = "turn"
                    record["turns"].append(
                        {
                            "request_id": request_id,
                            "message_chars": len(message),
                            "result": "session_end",
                            "latency_ms": round(elapsed_ms, 1),
                            "retried": dropped,
                        }
                    )
                    break
                response = reply.body["response"]
                stored = api.store.get_request(caller, request_id)
                move = stored.staged_response["effective_move"]["move"]
                modes[response["delivery_mode"]] += 1
                moves[move] += 1
                record["turns"].append(
                    {
                        "request_id": request_id,
                        "message_chars": len(message),
                        "result": "response",
                        "plan_id": response["plan_id"],
                        "response_id": response["response_id"],
                        "move": move,
                        "delivery_mode": response["delivery_mode"],
                        "response_words": len(response["text"].split()),
                        "state_version": reply.body["public_state"]["state_version"],
                        "latency_ms": round(elapsed_ms, 1),
                        "retried": dropped,
                    }
                )
                source = response["response_id"]
                version = reply.body["public_state"]["state_version"]
            if record["end_reason"] is None:
                stop = api.stop(
                    caller,
                    session_id,
                    {"request_id": f"{prefix}-stop", "expected_state_version": version},
                )
                public("StopResult", stop.body)
                record["end_reason"] = stop.body["session_end"]["reason"]
                record["ended_by"] = "stop"
            view = api.get_session(caller, session_id)
            public("SessionView", view.body)
            record["transcript_entries"] = len(view.body["transcript"])
            record["final_state_version"] = view.body["public_state"]["state_version"]
            record["teaching_turns"] = view.body["public_state"]["teaching_turns"]
            ends[record["end_reason"]] += 1
        except ApiError as error:
            failure = {"script": script["name"], "code": error.code, "status": error.status}
            record["errors"].append(failure)
            errors.append(failure)
            log.error("smoke session %s failed: %s %s", script["name"], error.code, error.message)
    finished = datetime.now(UTC)
    report = {
        "schema_version": "smoke_report_v1",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "learning_service": learning_label,
        "response_service": "in_process_stub",
        "summary": {
            "sessions": len(sessions),
            "sessions_completed": sum(1 for s in sessions if s["end_reason"] is not None),
            "turns": sum(len(s["turns"]) for s in sessions),
            "responses": sum(modes.values()),
            "delivery_modes": dict(modes),
            "moves": dict(moves),
            "end_reasons": dict(ends),
            "retries": retries,
            "errors": len(errors),
            "latency_ms": {
                "max": round(max(latencies), 1) if latencies else None,
                "median": round(statistics.median(latencies), 1) if latencies else None,
            },
        },
        "sessions_detail": sessions,
        "errors": errors,
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ten-session CPU smoke workflow")
    parser.add_argument("--config", default="configs/response/public_api_local.yaml")
    parser.add_argument(
        "--learning-url", help="B's learning service base URL (sets learning.mode http)"
    )
    parser.add_argument("--report", default="work/public_api/smoke_report.json")
    parser.add_argument("--store", help="SQLite request store for this run (default: fresh file)")
    parser.add_argument("--sessions", type=int, default=len(SCRIPTS))
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level)
    config = PublicApiConfig.load(args.config)
    if args.learning_url:
        config = PublicApiConfig(
            host=config.host,
            port=config.port,
            tokens_env=config.tokens_env,
            storage_path=config.storage_path,
            learning=LearningConfig(
                mode="http",
                base_url=args.learning_url,
                service_token_env=config.learning.service_token_env,
                timeout_s=config.learning.timeout_s,
            ),
            response=config.response,
            limits=config.limits,
        )
    store_path = (
        Path(args.store)
        if args.store
        else ROOT
        / "work/public_api/smoke"
        / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-requests.sqlite")
    )
    if config.learning.mode == "http":
        learning: Any = HttpLearningClient(
            config.learning.base_url,
            service_token_env=config.learning.service_token_env,
            timeout_s=config.learning.timeout_s,
        )
        label = f"http {config.learning.base_url}"
    else:
        from tests.integration.response.learning_standin import LearningStandIn

        learning = LearningStandIn(plan_ttl_s=config.learning.standin_plan_ttl_s)
        label = "person_a_standin"
    api = PublicApi(RequestStore(store_path), learning, InProcessResponseClient(), config=config)
    report = run_smoke(
        api, scripts=SCRIPTS[: args.sessions], report_path=args.report, learning_label=label
    )
    summary = report["summary"]
    print(json.dumps(summary, indent=2))
    print(f"report written to {args.report}")
    ok = (
        summary["errors"] == 0
        and summary["retries"]["identical"] == summary["retries"]["exercised"]
    )
    return 0 if ok and summary["sessions_completed"] == summary["sessions"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
