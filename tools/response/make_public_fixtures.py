"""Generate Person A's public contract fixtures under contracts/fixtures/response/public.

The fixtures exercise contracts/public_v1.openapi.yaml: each turn result variant, a retry that
must return the identical body, a stale-state conflict, request_id reuse, every error code at
its PRD 9.5 status, an unknown field, an oversize message and a leaked internal field. Tutor
text comes from the CPU response stub applied to B's PROBE fixture plan, so the fixtures stay
consistent with the generation contract. Run after changing the contract or the stub:

    uv run python -m tools.response.make_public_fixtures
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from apps.public_api.contract import PRD_ERROR_STATUS, PUBLIC_FIXTURES, ROOT
from packages.response.stub import generate

SESSION = "ses-0123456789abcdef01234567"
HEADERS = {"Authorization": "Bearer <token from TUTOR_PUBLIC_API_TOKENS>"}


def public_state(
    version: int, turns: int, *, ended: bool = False, reason: str | None = None
) -> dict:
    return {
        "session_id": SESSION,
        "state_version": version,
        "current_concept_id": "C01",
        "goal_concept_id": "C01",
        "teaching_turns": turns,
        "status": "ended" if ended else "active",
        "end_reason": reason,
    }


def stub_response(request_id: str) -> dict:
    plan = json.loads(
        (ROOT / "contracts/fixtures/learning/valid/teaching_plan_probe.json").read_text("utf-8")
    )
    plan["session_id"] = SESSION
    generated = generate(
        {
            "schema_version": "1.0",
            "request_id": request_id,
            "trace_id": "trace-fixture",
            "plan": plan,
            "response_artifact_id": "stub_v1",
            "generation_seed": 0,
            "deadline_ms": 1000,
        }
    )
    return {
        "response_id": generated["response_id"],
        "plan_id": generated["plan_id"],
        "text": generated["text"],
        "delivery_mode": generated["delivery_mode"],
    }


def error(request_id: str, code: str, retryable: bool, version: int | None, message: str) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "trace_id": "trc-fixture",
        "code": code,
        "message": message,
        "retryable": retryable,
        "current_state_version": version,
    }


def exchange(operation: str, request, status: int, body, *, session: bool = True, **extra) -> dict:
    fixture = {
        "kind": "exchange",
        "operation": operation,
        "path_params": {"session_id": SESSION} if session else {},
        "headers": dict(HEADERS, **extra.pop("headers", {})),
        "request": request,
        "response": {"status": status, "body": body},
    }
    fixture.update(extra)
    return fixture


def dump(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def valid_fixtures() -> dict[str, dict]:
    first = stub_response("req-turn-1")
    second = stub_response("req-turn-2")
    turn_1 = {
        "request_id": "req-turn-1",
        "expected_state_version": 0,
        "source_response_id": None,
        "message": "",
    }
    turn_2 = {
        "request_id": "req-turn-2",
        "expected_state_version": 1,
        "source_response_id": first["response_id"],
        "message": "is it 1/2?",
        "session_id": SESSION,
    }
    turn_1_result = {
        "request_id": "req-turn-1",
        "response": first,
        "public_state": public_state(1, 1),
    }
    session_end = {
        "schema_version": "1.0",
        "session_id": SESSION,
        "state_version": 3,
        "reason": "learner_stop",
        "summary_content_id": "summary-ses-0123456789abcdef01234567",
        "public_state": public_state(3, 2, ended=True, reason="learner_stop"),
    }
    fixtures = {
        "create_session": exchange(
            "createSession",
            {"request_id": "req-create-1", "goal_concept_id": "C01", "pace_preference": "normal"},
            201,
            {"session_id": SESSION, "public_state": public_state(0, 0)},
            session=False,
            headers={"Idempotency-Key": "req-create-1"},
        ),
        "turn_first_response_variant": exchange("postTurn", turn_1, 200, turn_1_result),
        "turn_learner_response_variant": exchange(
            "postTurn",
            turn_2,
            200,
            {"request_id": "req-turn-2", "response": second, "public_state": public_state(2, 2)},
        ),
        "turn_session_end_variant": exchange(
            "postTurn",
            {
                "request_id": "req-turn-3",
                "expected_state_version": 2,
                "source_response_id": second["response_id"],
                "message": "stop",
            },
            200,
            {"request_id": "req-turn-3", "session_end": session_end},
        ),
        "turn_retry_identical_response": {
            "kind": "retry",
            "operation": "postTurn",
            "path_params": {"session_id": SESSION},
            "headers": HEADERS,
            "request": turn_1,
            "attempts": [
                {"status": 200, "body": turn_1_result},
                {"status": 200, "body": copy.deepcopy(turn_1_result)},
            ],
        },
        "stop_session_end": exchange(
            "stopSession",
            {"request_id": "req-stop-1", "expected_state_version": 2},
            200,
            {"request_id": "req-stop-1", "session_end": session_end},
        ),
        "get_session_transcript": {
            "kind": "exchange",
            "operation": "getSession",
            "path_params": {"session_id": SESSION},
            "headers": HEADERS,
            "request": None,
            "response": {
                "status": 200,
                "body": {
                    "session_id": SESSION,
                    "public_state": public_state(2, 2),
                    "transcript": [
                        {"role": "tutor", "request_id": "req-turn-1", "response": first},
                        {"role": "learner", "request_id": "req-turn-2", "text": "is it 1/2?"},
                        {"role": "tutor", "request_id": "req-turn-2", "response": second},
                    ],
                },
            },
        },
        "turn_stale_state_conflict": exchange(
            "postTurn",
            dict(turn_2, request_id="req-turn-stale", expected_state_version=0),
            409,
            error(
                "req-turn-stale",
                "STATE_CONFLICT",
                False,
                2,
                "expected_state_version is stale; refresh the session and send a new request",
            ),
        ),
        "turn_request_id_reuse_conflict": exchange(
            "postTurn",
            dict(turn_2, message="a different message"),
            409,
            error(
                "req-turn-2",
                "STATE_CONFLICT",
                False,
                2,
                "request_id was already used with different content; use a new request_id",
            ),
        ),
        "turn_competing_request_conflict": exchange(
            "postTurn",
            dict(turn_2, request_id="req-turn-competing"),
            409,
            error(
                "req-turn-competing",
                "STATE_CONFLICT",
                True,
                1,
                "Another request for this session is in progress; retry after it completes.",
            ),
        ),
        "turn_generation_unavailable": exchange(
            "postTurn",
            turn_2,
            503,
            error(
                "req-turn-2",
                "GENERATION_UNAVAILABLE",
                True,
                1,
                "The tutor could not produce a validated reply; retry the same request.",
            ),
        ),
        "turn_message_too_long": exchange(
            "postTurn",
            dict(turn_2, request_id="req-turn-long", message="x" * 8001),
            422,
            error(
                "req-turn-long", "INVALID_SCHEMA", False, None, "message exceeds 8000 characters"
            ),
        ),
    }
    status_errors = {
        "INVALID_SCHEMA": "request does not match TurnRequest: $: additionalProperties",
        "UNSUPPORTED_VERSION": "Only contract major version 1 is supported.",
        "UNAUTHORIZED": "A valid bearer token is required.",
        "FORBIDDEN": "No such session for this caller.",
        "STATE_CONFLICT": "source_response_id is not the last committed response of this session",
        "PLAN_EXPIRED": "The reserved teaching plan expired before delivery; refresh the session"
        " and send a new turn.",
        "SESSION_ENDED": "The session has ended.",
        "RATE_LIMITED": "Too many requests; retry after the Retry-After delay.",
        "GENERATION_UNAVAILABLE": "The tutor could not produce a validated reply; retry the same"
        " request.",
        "LEARNING_UNAVAILABLE": "The learning service is unavailable; retry the same request.",
        "DEADLINE_EXCEEDED": "The request deadline elapsed before a validated reply; retry the"
        " request.",
    }
    for code, message in status_errors.items():
        retryable = code in {
            "RATE_LIMITED",
            "GENERATION_UNAVAILABLE",
            "LEARNING_UNAVAILABLE",
            "DEADLINE_EXCEEDED",
        }
        fixtures[f"error_{code.lower()}"] = {
            "kind": "schema",
            "schema": "Error",
            "http_status": PRD_ERROR_STATUS[code],
            "instance": error("req-error", code, retryable, None, message),
        }
    fixtures["public_state_projection"] = {
        "kind": "schema",
        "schema": "PublicState",
        "instance": public_state(2, 2),
    }
    return fixtures


def invalid_fixtures() -> dict[str, dict]:
    first = stub_response("req-turn-1")
    turn_1 = {
        "request_id": "req-turn-1",
        "expected_state_version": 0,
        "source_response_id": None,
        "message": "",
    }
    ok = {"request_id": "req-turn-1", "response": first, "public_state": public_state(1, 1)}
    session_end = {
        "schema_version": "1.0",
        "session_id": SESSION,
        "state_version": 1,
        "reason": "learner_stop",
        "summary_content_id": "summary-1",
        "public_state": public_state(1, 0, ended=True, reason="learner_stop"),
    }
    leaked = dict(ok, response=dict(first, content_hash="a" * 64, checks={"schema_valid": True}))
    return {
        "turn_result_both_variants": {
            "kind": "schema",
            "schema": "TurnResult",
            "instance": dict(ok, session_end=session_end),
        },
        "turn_result_no_variant": {
            "kind": "schema",
            "schema": "TurnResult",
            "instance": {"request_id": "req-turn-1"},
        },
        "turn_result_leaks_checks_and_hash": {
            "kind": "schema",
            "schema": "TurnResult",
            "instance": leaked,
        },
        "turn_result_leaks_plan": {
            "kind": "schema",
            "schema": "TurnResult",
            "instance": dict(ok, plan={"constraints": {"forbidden_answer_fingerprints": ["1/2"]}}),
        },
        "turn_request_unknown_field": {
            "kind": "schema",
            "schema": "TurnRequest",
            "instance": dict(turn_1, unexpected="not allowed"),
        },
        "turn_request_message_too_long": {
            "kind": "schema",
            "schema": "TurnRequest",
            "instance": dict(turn_1, message="x" * 8001),
        },
        "turn_request_missing_source": {
            "kind": "schema",
            "schema": "TurnRequest",
            "instance": {"request_id": "req", "expected_state_version": 0, "message": "hi"},
        },
        "create_request_bad_pace": {
            "kind": "schema",
            "schema": "CreateSessionRequest",
            "instance": {"request_id": "req", "goal_concept_id": "C01", "pace_preference": "fast"},
        },
        "error_unknown_code": {
            "kind": "schema",
            "schema": "Error",
            "instance": error("req", "NOT_A_CODE", False, None, "unknown"),
        },
        "error_missing_state_version": {
            "kind": "schema",
            "schema": "Error",
            "instance": {
                key: value
                for key, value in error("req", "SESSION_ENDED", False, 1, "ended").items()
                if key != "current_state_version"
            },
        },
        "public_state_leaks_mastery": {
            "kind": "schema",
            "schema": "PublicState",
            "instance": dict(public_state(1, 1), mastery=0.5),
        },
        "session_end_with_active_state": {
            "kind": "schema",
            "schema": "TurnResult",
            "instance": {
                "request_id": "req-turn-1",
                "session_end": dict(session_end, public_state=public_state(1, 0)),
            },
        },
        "exchange_state_conflict_wrong_status": exchange(
            "postTurn",
            turn_1,
            400,
            error("req-turn-1", "STATE_CONFLICT", False, 1, "stale"),
        ),
        "exchange_session_id_mismatch": exchange(
            "postTurn",
            dict(turn_1, session_id="ses-other"),
            200,
            ok,
        ),
        "exchange_idempotency_key_mismatch": exchange(
            "postTurn",
            turn_1,
            200,
            ok,
            headers={"Idempotency-Key": "req-other"},
        ),
        "exchange_response_request_id_mismatch": exchange(
            "postTurn",
            turn_1,
            200,
            dict(ok, request_id="req-other"),
        ),
        "retry_returns_different_response": {
            "kind": "retry",
            "operation": "postTurn",
            "path_params": {"session_id": SESSION},
            "headers": HEADERS,
            "request": turn_1,
            "attempts": [
                {"status": 200, "body": ok},
                {"status": 200, "body": dict(ok, response=stub_response("req-other"))},
            ],
        },
    }


def main() -> None:
    for name, fixture in valid_fixtures().items():
        dump(PUBLIC_FIXTURES / "valid" / f"{name}.json", fixture)
    for name, fixture in invalid_fixtures().items():
        dump(PUBLIC_FIXTURES / "invalid" / f"{name}.json", fixture)
    nan_path = PUBLIC_FIXTURES / "invalid" / "nan_value.json"
    nan_path.write_text(
        '{"kind": "schema", "schema": "PublicState", "instance": {"state_version": NaN}}\n',
        encoding="utf-8",
    )
    print(f"wrote public fixtures under {PUBLIC_FIXTURES}")


if __name__ == "__main__":
    main()
