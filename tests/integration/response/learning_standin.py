"""Person A's stand-in for Person B's learning service transaction contract.

B's real service (`apps/learning_service`, started with `tools.learning.serve`) is the reference;
`tests/integration/response/test_smoke_workflow.py` runs against it. This stand-in exists for
fast, deterministic unit tests of the public API's recovery logic and mirrors B's wire shapes
(docs/learning_service.md): create allocates the session ID, observe returns
`{"kind": "plan" | "session_end", ...}`, plan status carries no public state, and every POST
needs an Idempotency-Key. Delete it once B's mock covers these tests.

It replays B's valid TeachingPlan fixtures from `contracts/fixtures/learning/valid` in a fixed
order and implements only transaction behavior: state versions, one pending plan per session,
idempotent reservation per request_id, commit with hash and move checks, plan status, cancel,
conflicts and expiry. It never estimates student state, grades evidence, or chooses actions; the
fixture order has nothing to do with the learner.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from apps.public_api.contract import PRD_ERROR_STATUS, ROOT, parse_json_strict
from apps.public_api.errors import ApiError

FIXTURES = ROOT / "contracts/fixtures/learning/valid"
PLAN_ORDER = (
    "teaching_plan_probe",
    "teaching_plan_hint",
    "teaching_plan_explain_concept",
    "teaching_plan_worked_example",
    "teaching_plan_practice",
    "teaching_plan_challenge",
    "teaching_plan_check_mastery",
    "teaching_plan_reflect",
)
TEACHING_BUDGET = 16
CREATE_FIELDS = {"schema_version", "request_id", "goal_concept_id", "pace_preference"}
OBSERVE_FIELDS = {
    "schema_version",
    "request_id",
    "session_id",
    "expected_state_version",
    "evidence",
}
CANCEL_FIELDS = {"schema_version", "request_id", "session_id", "plan_id"}


def load_fixture_plans(root: Path = FIXTURES) -> list[dict]:
    return [json.loads((root / f"{name}.json").read_text(encoding="utf-8")) for name in PLAN_ORDER]


@dataclass
class PendingPlan:
    plan: dict
    request_id: str
    expires_at: float


@dataclass
class CommittedPlan:
    plan: dict
    content_hash: str
    state_version: int
    public_state: dict
    response_id: str
    response: dict


@dataclass
class StandInSession:
    session_id: str
    goal_concept_id: str
    pace_preference: str
    state_version: int = 0
    teaching_turns: int = 0
    status: str = "active"
    end_reason: str | None = None
    last_response_id: str | None = None
    last_plan: dict | None = None
    pending: PendingPlan | None = None
    applied_events: set[str] = field(default_factory=set)
    plan_counter: int = 0

    def public_state(self) -> dict:
        return {
            "session_id": self.session_id,
            "state_version": self.state_version,
            "current_concept_id": self.goal_concept_id,
            "goal_concept_id": self.goal_concept_id,
            "teaching_turns": self.teaching_turns,
            "status": self.status,
            "end_reason": self.end_reason,
        }


def _error(code: str, message: str, version: int | None = None) -> ApiError:
    retryable = code in {"LEARNING_UNAVAILABLE", "RATE_LIMITED", "DEADLINE_EXCEEDED"}
    return ApiError(code, message, retryable=retryable, current_state_version=version)


class LearningStandIn:
    """In-process implementation of the LearningClient protocol. B's real service replaces it."""

    def __init__(self, *, plan_ttl_s: float = 600.0, clock=time.monotonic) -> None:
        self.plan_ttl_s = plan_ttl_s
        self.clock = clock
        self.plans = load_fixture_plans()
        self.sessions: dict[str, StandInSession] = {}
        self.created: dict[str, dict] = {}  # request_id -> create response (idempotency)
        self.pending_by_plan: dict[str, str] = {}
        self.committed: dict[str, CommittedPlan] = {}
        self.canceled: dict[str, str] = {}
        self.fail_next: list[Exception] = []
        self.calls: list[str] = []
        self._lock = threading.Lock()

    # Fault injection ----------------------------------------------------------------------------

    def _maybe_fail(self, operation: str) -> None:
        self.calls.append(operation)
        if self.fail_next:
            raise self.fail_next.pop(0)

    # Endpoints ----------------------------------------------------------------------------------

    def create_session(self, spec: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("create_session")
            if set(spec) != CREATE_FIELDS:
                raise _error("INVALID_SCHEMA", "request does not match the schema")
            saved = self.created.get(spec["request_id"])
            if saved is not None:
                return copy.deepcopy(saved)
            if spec.get("pace_preference") not in {"concise", "normal", "detailed"}:
                raise _error("INVALID_SCHEMA", "pace_preference is invalid")
            session = StandInSession(
                f"s-{uuid.uuid4().hex}", spec["goal_concept_id"], spec["pace_preference"]
            )
            self.sessions[session.session_id] = session
            response = {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "public_state": session.public_state(),
            }
            self.created[spec["request_id"]] = response
            return copy.deepcopy(response)

    def observe_and_plan(self, request: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("observe_and_plan")
            if set(request) != OBSERVE_FIELDS:
                raise _error("INVALID_SCHEMA", "request does not match the schema")
            session = self._session(request["session_id"])
            self._expire(session)
            if session.pending is not None:
                if session.pending.request_id == request["request_id"]:
                    return {"kind": "plan", "teaching_plan": copy.deepcopy(session.pending.plan)}
                raise _error("STATE_CONFLICT", "A plan is already pending", session.state_version)
            if request["expected_state_version"] != session.state_version:
                raise _error("STATE_CONFLICT", "State version is stale", session.state_version)
            evidence = request["evidence"]
            if session.last_response_id is None:
                if evidence is not None:
                    raise _error(
                        "INVALID_SCHEMA",
                        "First observation requires null evidence",
                        session.state_version,
                    )
            else:
                if (
                    evidence is None
                    or evidence["session_id"] != session.session_id
                    or evidence["source_response_id"] != session.last_response_id
                ):
                    raise _error(
                        "INVALID_SCHEMA",
                        "Evidence must reference the last committed response",
                        session.state_version,
                    )
                assert session.last_plan is not None
                if evidence["problem_instance_id"] != session.last_plan["problem_instance_id"]:
                    raise _error(
                        "INVALID_SCHEMA",
                        "Evidence problem reference does not match",
                        session.state_version,
                    )
                if evidence["event_id"] not in session.applied_events:
                    session.applied_events.add(evidence["event_id"])
                    if evidence["explicit_request"] == "stop":
                        return self._end(session, "learner_stop")
            if session.teaching_turns >= TEACHING_BUDGET:
                return self._end(session, "teaching_budget")
            plan = copy.deepcopy(self.plans[session.plan_counter % len(self.plans)])
            session.plan_counter += 1
            plan["plan_id"] = f"{session.session_id}-plan-{session.plan_counter}"
            plan["session_id"] = session.session_id
            plan["trace_id"] = trace_id
            plan["based_on_state_version"] = session.state_version
            plan["goal_concept_id"] = session.goal_concept_id
            expires = datetime.now(UTC) + timedelta(seconds=self.plan_ttl_s)
            plan["expires_at"] = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
            session.pending = PendingPlan(
                plan, request["request_id"], self.clock() + self.plan_ttl_s
            )
            self.pending_by_plan[plan["plan_id"]] = session.session_id
            return {"kind": "plan", "teaching_plan": copy.deepcopy(plan)}

    def commit_plan(self, request: dict, *, trace_id: str, idempotency_key: str) -> dict:
        with self._lock:
            self._maybe_fail("commit_plan")
            plan_id = request["plan_id"]
            committed = self.committed.get(plan_id)
            if committed is not None:
                if committed.content_hash != request["content_hash"]:
                    raise _error("STATE_CONFLICT", "Plan already committed with a different body")
                return copy.deepcopy(committed.response)
            if plan_id in self.canceled:
                raise _error("PLAN_EXPIRED", f"Plan has {self.canceled[plan_id]}")
            session = self._session(request["session_id"])
            self._expire(session)
            if plan_id in self.canceled:
                raise _error("PLAN_EXPIRED", "Plan has expired", session.state_version)
            pending = session.pending
            if pending is None or pending.plan["plan_id"] != plan_id:
                raise _error("STATE_CONFLICT", "Plan is not pending", session.state_version)
            if request["based_on_state_version"] != session.state_version:
                raise _error("STATE_CONFLICT", "State version is stale", session.state_version)
            if request["effective_move"] != pending.plan["teaching_action"]:
                raise _error("INVALID_SCHEMA", "Effective move must equal the reserved action")
            session.state_version += 1
            session.teaching_turns += 1
            session.last_response_id = request["response_id"]
            session.last_plan = pending.plan
            session.pending = None
            self.pending_by_plan.pop(plan_id, None)
            response = {
                "schema_version": "1.0",
                "plan_id": plan_id,
                "status": "committed",
                "state_version": session.state_version,
                "public_state": session.public_state(),
            }
            self.committed[plan_id] = CommittedPlan(
                pending.plan,
                request["content_hash"],
                session.state_version,
                session.public_state(),
                request["response_id"],
                response,
            )
            return copy.deepcopy(response)

    def plan_status(self, plan_id: str, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("plan_status")
            committed = self.committed.get(plan_id)
            if committed is not None:
                return self._status(plan_id, committed.plan["session_id"], "committed", committed)
            session_id = self.pending_by_plan.get(plan_id)
            if session_id is not None:
                self._expire(self.sessions[session_id])
            if plan_id in self.canceled:
                return self._status(plan_id, plan_id.rsplit("-plan-", 1)[0], "canceled", None)
            if plan_id in self.pending_by_plan:
                return self._status(plan_id, self.pending_by_plan[plan_id], "pending", None)
            raise _error("INVALID_SCHEMA", "Unknown plan")

    def cancel_plan(self, request: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("cancel_plan")
            if set(request) != CANCEL_FIELDS:
                raise _error("INVALID_SCHEMA", "request does not match the schema")
            plan_id = request["plan_id"]
            session = self._session(request["session_id"])
            if plan_id in self.committed:
                raise _error(
                    "STATE_CONFLICT", "Committed plans cannot be canceled", session.state_version
                )
            if session.pending is not None and session.pending.plan["plan_id"] == plan_id:
                session.pending = None  # no teaching turn consumed, no state change
                self.pending_by_plan.pop(plan_id, None)
                self.canceled[plan_id] = "canceled"
            elif plan_id not in self.canceled:
                raise _error("INVALID_SCHEMA", "Unknown plan", session.state_version)
            return self._status(plan_id, session.session_id, "canceled", None)

    # Internals ----------------------------------------------------------------------------------

    @staticmethod
    def _status(
        plan_id: str, session_id: str, status: str, committed: CommittedPlan | None
    ) -> dict:
        return {
            "schema_version": "1.0",
            "plan_id": plan_id,
            "session_id": session_id,
            "status": status,
            "content_hash": committed.content_hash if committed else None,
            "response_id": committed.response_id if committed else None,
        }

    def _session(self, session_id: str) -> StandInSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise _error("INVALID_SCHEMA", "Unknown session")
        if session.status == "ended":
            raise _error("SESSION_ENDED", "Session has ended", session.state_version)
        return session

    def _expire(self, session: StandInSession) -> None:
        pending = session.pending
        if pending is not None and self.clock() >= pending.expires_at:
            session.pending = None
            self.pending_by_plan.pop(pending.plan["plan_id"], None)
            self.canceled[pending.plan["plan_id"]] = "expired"

    def _end(self, session: StandInSession, reason: str) -> dict:
        session.state_version += 1
        session.status = "ended"
        session.end_reason = reason
        return {
            "kind": "session_end",
            "session_end": {
                "schema_version": "1.0",
                "session_id": session.session_id,
                "state_version": session.state_version,
                "reason": reason,
                "summary_content_id": f"summary-{reason}",
                "public_state": session.public_state(),
            },
        }


# HTTP wrapper with B's transport rules so HttpLearningClient can be exercised without B ------


class _StandInHandler(BaseHTTPRequestHandler):
    server: StandInServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _authorized(self) -> bool:
        trace = self.headers.get("X-Trace-Id") or ""
        return (
            self.headers.get("Authorization") == f"Bearer {self.server.token}"
            and 1 <= len(trace) <= 128
            and trace.isascii()
        )

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._error("UNAUTHORIZED", "Valid bearer token and X-Trace-Id are required")
            return
        if not self.path.startswith("/internal/v1/plans/"):
            self._error("INVALID_SCHEMA", "Unknown route")
            return
        plan_id = self.path.rsplit("/", 1)[1]
        self._call(lambda: self.server.standin.plan_status(plan_id, trace_id=self._trace()))

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._error("UNAUTHORIZED", "Valid bearer token and X-Trace-Id are required")
            return
        key = self.headers.get("Idempotency-Key") or ""
        if not (1 <= len(key) <= 128 and key.isascii()):
            self._error("INVALID_SCHEMA", "A valid Idempotency-Key is required")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = parse_json_strict(self.rfile.read(length).decode("utf-8"))
        if "request_id" in body and body["request_id"] != key:
            self._error("INVALID_SCHEMA", "Idempotency-Key must equal request_id")
            return
        standin = self.server.standin
        trace_id = self._trace()
        routes = {
            "/internal/v1/sessions": lambda: standin.create_session(body, trace_id=trace_id),
            "/internal/v1/observe-and-plan": lambda: standin.observe_and_plan(
                body, trace_id=trace_id
            ),
            "/internal/v1/commit-plan": lambda: standin.commit_plan(
                body, trace_id=trace_id, idempotency_key=key
            ),
            "/internal/v1/cancel-plan": lambda: standin.cancel_plan(body, trace_id=trace_id),
        }
        handler = routes.get(self.path)
        if handler is None:
            self._error("INVALID_SCHEMA", "Unknown route")
            return
        self._call(handler)

    def _trace(self) -> str:
        return self.headers.get("X-Trace-Id", "trace-standin")

    def _error(self, code: str, message: str) -> None:
        error = ApiError(code, message, retryable=code == "LEARNING_UNAVAILABLE")
        self._send(
            error.status,
            error.body(self.headers.get("Idempotency-Key") or "unknown", self._trace()),
        )

    def _call(self, handler) -> None:
        try:
            self._send(200, handler())
        except ApiError as error:
            self._send(
                error.status,
                error.body(self.headers.get("Idempotency-Key") or "unknown", self._trace()),
            )
        except Exception as error:  # fault injection surfaces as 503
            self._send(
                PRD_ERROR_STATUS["LEARNING_UNAVAILABLE"],
                ApiError("LEARNING_UNAVAILABLE", type(error).__name__, retryable=True).body(
                    "unknown", self._trace()
                ),
            )

    def _send(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class StandInServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, standin: LearningStandIn, token: str) -> None:
        super().__init__(("127.0.0.1", 0), _StandInHandler)
        self.standin = standin
        self.token = token

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"
