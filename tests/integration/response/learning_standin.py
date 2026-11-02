"""Person A's stand-in for Person B's learning service transaction contract.

B's real mock observe/commit/status/conflict service replaces this module as soon as it ships;
the public API tests then run unchanged against it through `HttpLearningClient`.

The stand-in replays B's valid TeachingPlan fixtures from `contracts/fixtures/learning/valid` in a
fixed order and implements only transaction behavior: state versions, one pending plan per
session, idempotent reservation per request_id, commit with hash and move checks, plan status,
cancel, conflicts and expiry. It never estimates student state, grades evidence, or chooses
actions; the fixture order has nothing to do with the learner.
"""

from __future__ import annotations

import copy
import json
import threading
import time
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
    """In-process implementation of the LearningClient protocol. B's real mock replaces it."""

    def __init__(self, *, plan_ttl_s: float = 600.0, clock=time.monotonic) -> None:
        self.plan_ttl_s = plan_ttl_s
        self.clock = clock
        self.plans = load_fixture_plans()
        self.sessions: dict[str, StandInSession] = {}
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
            session = self.sessions.get(spec["session_id"])
            if session is None:
                if spec.get("pace_preference") not in {"concise", "normal", "detailed"}:
                    raise _error("INVALID_SCHEMA", "pace_preference is invalid")
                session = StandInSession(
                    spec["session_id"], spec["goal_concept_id"], spec["pace_preference"]
                )
                self.sessions[session.session_id] = session
            return {"schema_version": "1.0", "public_state": session.public_state()}

    def observe_and_plan(self, request: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("observe_and_plan")
            session = self._session(request["session_id"])
            self._expire(session)
            if session.status == "ended":
                raise _error("SESSION_ENDED", "session has ended", session.state_version)
            if session.pending is not None:
                if session.pending.request_id == request["request_id"]:
                    return copy.deepcopy(session.pending.plan)  # idempotent reservation
                raise _error("STATE_CONFLICT", "another plan is pending", session.state_version)
            if request["expected_state_version"] != session.state_version:
                raise _error("STATE_CONFLICT", "stale state version", session.state_version)
            evidence = request.get("evidence")
            if evidence is not None:
                expected_source = session.last_response_id or "session-start"
                if evidence["source_response_id"] != expected_source:
                    raise _error("STATE_CONFLICT", "unknown source response", session.state_version)
                if evidence["event_id"] in session.applied_events:
                    raise _error(
                        "STATE_CONFLICT", "evidence already applied", session.state_version
                    )
                session.applied_events.add(evidence["event_id"])
                if evidence["explicit_request"] == "stop":
                    return self._end(session, "learner_stop")
            elif session.last_response_id is not None:
                raise _error("STATE_CONFLICT", "evidence required after the first turn")
            if session.teaching_turns >= TEACHING_BUDGET:
                return self._end(session, "teaching_budget")
            plan = copy.deepcopy(self.plans[session.plan_counter % len(self.plans)])
            session.plan_counter += 1
            plan["plan_id"] = f"plan-{session.session_id}-{session.plan_counter}"
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
            return copy.deepcopy(plan)

    def commit_plan(self, request: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("commit_plan")
            plan_id = request["plan_id"]
            committed = self.committed.get(plan_id)
            if committed is not None:
                if committed.content_hash != request["content_hash"]:
                    raise _error("STATE_CONFLICT", "plan committed with a different hash")
                return self._commit_response(committed)
            if plan_id in self.canceled:
                raise _error("PLAN_EXPIRED", f"plan {self.canceled[plan_id]}")
            session = self._session(request["session_id"])
            self._expire(session)
            if plan_id in self.canceled:
                raise _error("PLAN_EXPIRED", "plan expired", session.state_version)
            pending = session.pending
            if pending is None or pending.plan["plan_id"] != plan_id:
                raise _error("STATE_CONFLICT", "no such pending plan", session.state_version)
            if request["based_on_state_version"] != session.state_version:
                raise _error("STATE_CONFLICT", "stale state version", session.state_version)
            if request["effective_move"] != pending.plan["teaching_action"]:
                raise _error("STATE_CONFLICT", "effective move differs from the plan")
            session.state_version += 1
            session.teaching_turns += 1
            session.last_response_id = request["response_id"]
            session.pending = None
            self.pending_by_plan.pop(plan_id, None)
            committed = CommittedPlan(
                pending.plan,
                request["content_hash"],
                session.state_version,
                session.public_state(),
                request["response_id"],
            )
            self.committed[plan_id] = committed
            return self._commit_response(committed)

    def plan_status(self, plan_id: str, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("plan_status")
            committed = self.committed.get(plan_id)
            if committed is not None:
                return {
                    "schema_version": "1.0",
                    "plan_id": plan_id,
                    "status": "committed",
                    "content_hash": committed.content_hash,
                    "state_version": committed.state_version,
                    "public_state": committed.public_state,
                }
            session_id = self.pending_by_plan.get(plan_id)
            if session_id is not None:
                self._expire(self.sessions[session_id])
            if plan_id in self.canceled:
                return {
                    "schema_version": "1.0",
                    "plan_id": plan_id,
                    "status": "canceled",
                    "content_hash": None,
                    "state_version": None,
                    "public_state": None,
                }
            if plan_id in self.pending_by_plan:
                return {
                    "schema_version": "1.0",
                    "plan_id": plan_id,
                    "status": "pending",
                    "content_hash": None,
                    "state_version": None,
                    "public_state": None,
                }
            raise _error("FORBIDDEN", "unknown plan")

    def cancel_plan(self, request: dict, *, trace_id: str) -> dict:
        with self._lock:
            self._maybe_fail("cancel_plan")
            plan_id = request["plan_id"]
            if plan_id in self.committed:
                raise _error("STATE_CONFLICT", "a committed plan cannot be canceled")
            session = self._session(request["session_id"])
            if session.pending is not None and session.pending.plan["plan_id"] == plan_id:
                session.pending = None  # no teaching turn consumed, no state change
                self.pending_by_plan.pop(plan_id, None)
                self.canceled[plan_id] = "canceled"
            elif plan_id not in self.canceled:
                raise _error("STATE_CONFLICT", "no such pending plan", session.state_version)
            return {"schema_version": "1.0", "plan_id": plan_id, "status": "canceled"}

    # Internals ----------------------------------------------------------------------------------

    def _session(self, session_id: str) -> StandInSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise _error("FORBIDDEN", "unknown session")
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
            "schema_version": "1.0",
            "session_id": session.session_id,
            "state_version": session.state_version,
            "reason": reason,
            "summary_content_id": f"summary-{session.session_id}",
            "public_state": session.public_state(),
        }

    @staticmethod
    def _commit_response(committed: CommittedPlan) -> dict:
        return {
            "schema_version": "1.0",
            "plan_id": committed.plan["plan_id"],
            "status": "committed",
            "state_version": committed.state_version,
            "public_state": copy.deepcopy(committed.public_state),
        }


# HTTP wrapper so HttpLearningClient can be exercised without B's service ---------------------


class _StandInHandler(BaseHTTPRequestHandler):
    server: StandInServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:  # noqa: N802
        if not self.path.startswith("/internal/v1/plans/"):
            self._send(404, {"error": "not found"})
            return
        plan_id = self.path.rsplit("/", 1)[1]
        self._call(lambda: self.server.standin.plan_status(plan_id, trace_id=self._trace()))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = parse_json_strict(self.rfile.read(length).decode("utf-8"))
        standin = self.server.standin
        trace_id = self._trace()
        routes = {
            "/internal/v1/sessions": lambda: standin.create_session(body, trace_id=trace_id),
            "/internal/v1/observe-and-plan": lambda: standin.observe_and_plan(
                body, trace_id=trace_id
            ),
            "/internal/v1/commit-plan": lambda: standin.commit_plan(body, trace_id=trace_id),
            "/internal/v1/cancel-plan": lambda: standin.cancel_plan(body, trace_id=trace_id),
        }
        handler = routes.get(self.path)
        if handler is None:
            self._send(404, {"error": "not found"})
            return
        self._call(handler)

    def _trace(self) -> str:
        return self.headers.get("X-Trace-Id", "trace-standin")

    def _call(self, handler) -> None:
        if self.headers.get("Authorization") != f"Bearer {self.server.token}":
            self._send(
                401,
                ApiError("UNAUTHORIZED", "bad service token", retryable=False).body(
                    "standin", self._trace()
                ),
            )
            return
        try:
            self._send(200, handler())
        except ApiError as error:
            self._send(error.status, error.body("standin", self._trace()))
        except Exception as error:  # fault injection surfaces as 503
            self._send(
                PRD_ERROR_STATUS["LEARNING_UNAVAILABLE"],
                ApiError("LEARNING_UNAVAILABLE", type(error).__name__, retryable=True).body(
                    "standin", self._trace()
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
