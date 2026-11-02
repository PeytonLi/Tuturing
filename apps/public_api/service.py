"""Public API orchestration (PRD 8.1 to 8.3). Sequences calls; never decides what to teach.

Turn flow: durable request cache, evidence extraction, B's observe-and-plan, A's generate,
durable staging of the exact response and its hash, B's commit-plan, completion. Every step is
recorded in the request store so a retry with the same request_id resumes where the previous
attempt stopped and returns the identical committed response. Regeneration after a commit never
happens: once a response is staged it is the only response that can be committed or returned.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from apps.public_api.clients import GenerationFailure, ServiceUnavailable
from apps.public_api.config import Limits, PublicApiConfig
from apps.public_api.contract import PublicContract, find_internal_fields, load_contract
from apps.public_api.errors import (
    ApiError,
    generation_unavailable,
    invalid,
    learning_unavailable,
    plan_expired,
    session_ended,
    state_conflict,
)
from apps.public_api.storage import RequestRecord, RequestStore, SessionRecord, fingerprint
from packages.evidence_extractor.extract import extract
from packages.learning.plan_rules import validate_plan_rules

log = logging.getLogger("tutor.public_api")

OPERATION_SCHEMAS = {
    "createSession": ("CreateSessionRequest", "CreateSessionResult"),
    "postTurn": ("TurnRequest", "TurnResult"),
    "stopSession": ("StopRequest", "StopResult"),
}

Hook = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Reply:
    status: int
    body: Any
    trace_id: str
    retry_after_s: int | None = None


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(12)}"


def generation_seed(request_id: str) -> int:
    """Deterministic per request so a retry asks the generator for the same realization."""
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)


class PublicApi:
    def __init__(
        self,
        store: RequestStore,
        learning: Any,
        response: Any,
        *,
        config: PublicApiConfig | None = None,
        contract: PublicContract | None = None,
        hook: Hook | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.learning = learning
        self.response = response
        self.config = config or PublicApiConfig()
        self.limits: Limits = self.config.limits
        self.contract = contract or load_contract()
        self.hook = hook or (lambda point, context: None)
        self.clock = clock
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # Public operations ------------------------------------------------------------------------

    def create_session(
        self, caller: str, body: Any, headers: dict[str, str] | None = None
    ) -> Reply:
        return self._run("createSession", caller, None, body, headers or {}, self._create_work)

    def turn(
        self, caller: str, session_id: str, body: Any, headers: dict[str, str] | None = None
    ) -> Reply:
        return self._run("postTurn", caller, session_id, body, headers or {}, self._turn_work)

    def stop(
        self, caller: str, session_id: str, body: Any, headers: dict[str, str] | None = None
    ) -> Reply:
        return self._run("stopSession", caller, session_id, body, headers or {}, self._stop_work)

    def get_session(self, caller: str, session_id: str) -> Reply:
        trace_id = new_id("trc")
        self._check_id(session_id, "session_id")
        session = self._session_or_forbidden(caller, session_id)
        body = {
            "session_id": session_id,
            "public_state": session.public_state,
            "transcript": self.store.transcript(caller, session_id),
        }
        self._assert_public("SessionView", body)
        return Reply(200, body, trace_id)

    # Request lifecycle ------------------------------------------------------------------------

    def _run(
        self,
        operation: str,
        caller: str,
        path_session_id: str | None,
        body: Any,
        headers: dict[str, str],
        work: Callable[..., tuple[int, Any]],
    ) -> Reply:
        started = self.clock()
        deadline = started + self.limits.request_deadline_ms / 1000.0
        trace_id = new_id("trc")
        request_id = body.get("request_id") if isinstance(body, dict) else None
        self._check_id(request_id, "request_id")
        key = headers.get("Idempotency-Key")
        if key is not None and key != request_id:
            raise invalid("Idempotency-Key must equal request_id")
        if path_session_id is not None:
            self._check_id(path_session_id, "session_id")
        request_schema, result_schema = OPERATION_SCHEMAS[operation]
        issues = self.contract.public_issues(request_schema, body)
        if issues:
            raise invalid(f"request does not match {request_schema}: " + "; ".join(issues))
        if "session_id" in body and body["session_id"] != path_session_id:
            raise invalid("session_id in the body must equal the path session ID")
        if "message" in body and len(body["message"]) > self.limits.max_message_chars:
            raise invalid(f"message exceeds {self.limits.max_message_chars} characters")
        if path_session_id is not None:
            self._session_or_forbidden(caller, path_session_id)
        digest = fingerprint(operation, path_session_id, body)
        record = self.store.begin_request(
            caller, request_id, operation, path_session_id, digest, trace_id
        )
        is_new = record.trace_id == trace_id
        trace_id = record.trace_id
        if record.fingerprint != digest or record.operation != operation:
            raise state_conflict(
                "request_id was already used with different content; use a new request_id",
                self._known_version(caller, record.session_id),
                terminal=False,
            )
        if record.complete:
            return self._stored(record)
        lock = self._session_lock(caller, record.session_id or request_id)
        wait = 0.0 if is_new else max(0.0, deadline - self.clock())
        if not lock.acquire(timeout=wait):
            raise ApiError(
                "STATE_CONFLICT",
                "Another request for this session is in progress; retry after it completes.",
                retryable=True,
                current_state_version=self._known_version(caller, record.session_id),
                retry_after_s=1,
            )
        try:
            record = self._record(caller, request_id)
            if record.complete:
                return self._stored(record)
            try:
                status, result = work(caller, record, body, deadline)
            except ApiError as error:
                if error.terminal:
                    self.store.complete_request(
                        caller, request_id, error.status, error.body(request_id, trace_id)
                    )
                self._log(operation, record, error.status, code=error.code)
                raise
            except ServiceUnavailable as error:
                self._log(operation, record, 503, code="LEARNING_UNAVAILABLE", detail=str(error))
                raise learning_unavailable() from error
            self._assert_public(result_schema, result)
            self.store.complete_request(caller, request_id, status, result)
            self._log(operation, self._record(caller, request_id), status)
            return Reply(status, result, trace_id)
        finally:
            lock.release()

    # Session creation (PRD 8.1) ---------------------------------------------------------------

    def _create_work(
        self, caller: str, record: RequestRecord, body: dict, deadline: float
    ) -> tuple[int, Any]:
        session_id = record.session_id
        session = self.store.get_session(caller, session_id) if session_id else None
        if session is None:
            # B allocates the session ID; the same request_id (idempotency key) returns the
            # same session on a retry, so a crash between B's answer and our record is safe.
            spec = {
                "schema_version": "1.0",
                "request_id": record.request_id,
                "goal_concept_id": body["goal_concept_id"],
                "pace_preference": body["pace_preference"],
            }
            created = self.learning.create_session(spec, trace_id=record.trace_id)
            public_state = created.get("public_state") if isinstance(created, dict) else None
            problems = self.contract.learning_issues("PublicState", public_state)
            if (
                problems
                or not isinstance(created.get("session_id"), str)
                or public_state["session_id"] != created["session_id"]
                or self.contract.public_issues("Id", created["session_id"])
            ):
                raise ServiceUnavailable("learning service returned an invalid initial state")
            session_id = created["session_id"]
            self.store.create_session(caller, session_id, public_state)
            self.store.set_request_session(caller, record.request_id, session_id)
            session = self.store.get_session(caller, session_id)
        assert session is not None
        return 201, {"session_id": session.session_id, "public_state": session.public_state}

    # Learner turn (PRD 8.2) -------------------------------------------------------------------

    def _turn_work(
        self, caller: str, record: RequestRecord, body: dict, deadline: float
    ) -> tuple[int, Any]:
        session_id = record.session_id
        assert session_id is not None
        if record.plan is None:
            session = self._session_or_forbidden(caller, session_id)
            if session.ended:
                raise session_ended(session.state_version)
            self._supersede_pending(caller, session_id, record)
            session = self._session_or_forbidden(caller, session_id)
            self._check_expectations(session, body["expected_state_version"], body)
            evidence = None
            if body["source_response_id"] is not None:
                evidence = self._evidence(body["message"], session, record)
            result = self.learning.observe_and_plan(
                self._observe_request(session, record, evidence), trace_id=record.trace_id
            )
            kind, payload = self._check_plan_or_end(result, session, body["expected_state_version"])
            self.store.stage_plan(caller, record.request_id, payload, kind)
            self.hook("after_observe", {"record": record, "result": payload})
            record = self._record(caller, record.request_id)
        if record.plan_kind == "session_end":
            return self._finish_session_end(caller, record, body.get("message"))
        plan = record.plan
        assert plan is not None
        if record.staged_response is None:
            generated = self._generate(plan, record, deadline)
            self.store.stage_response(caller, record.request_id, generated)
            self.hook("after_stage_response", {"record": record, "response": generated})
            record = self._record(caller, record.request_id)
        staged = record.staged_response
        assert staged is not None
        if record.committed is None:
            committed = None
            if record.commit_attempted:
                committed = self._recover_commit(caller, record, plan, staged)
            if committed is None:
                committed = self._commit(caller, record, plan, staged)
            self.store.stage_commit(caller, record.request_id, committed)
            self.hook("after_stage_commit", {"record": record, "committed": committed})
            record = self._record(caller, record.request_id)
        return self._finish_turn(caller, record, body.get("message"))

    def _evidence(self, message: str, session: SessionRecord, record: RequestRecord) -> dict:
        evidence = extract(
            message,
            session.last_plan,
            session.last_response_id,
            session_id=session.session_id,
            trace_id=record.trace_id,
            raw_message_ref=f"msg-{record.request_id}",
        )
        problems = self.contract.learning_issues("Evidence", evidence)
        if problems:
            raise ServiceUnavailable("extractor produced invalid evidence: " + "; ".join(problems))
        return evidence

    def _observe_request(
        self, session: SessionRecord, record: RequestRecord, evidence: dict | None
    ) -> dict:
        return {
            "schema_version": "1.0",
            "request_id": record.request_id,
            "session_id": session.session_id,
            "expected_state_version": session.state_version,
            "evidence": evidence,
        }

    def _check_expectations(self, session: SessionRecord, expected: int, body: dict) -> None:
        if expected != session.state_version:
            raise state_conflict(
                "expected_state_version is stale; refresh the session and send a new request",
                session.state_version,
            )
        if "source_response_id" in body and body["source_response_id"] != session.last_response_id:
            raise state_conflict(
                "source_response_id is not the last committed response of this session",
                session.state_version,
            )

    def _check_plan_or_end(
        self, result: Any, session: SessionRecord, expected: int
    ) -> tuple[str, dict]:
        """Unwrap B's `{"kind": "plan" | "session_end", ...}` observe result and validate it."""
        kind = result.get("kind") if isinstance(result, dict) else None
        if kind == "plan":
            plan = result.get("teaching_plan")
            problems = self.contract.learning_issues("TeachingPlan", plan)
            problems += validate_plan_rules(plan) if not problems else []
            if (
                problems
                or plan["session_id"] != session.session_id
                or plan["based_on_state_version"] != expected
            ):
                raise ServiceUnavailable("learning service returned an invalid TeachingPlan")
            return "plan", plan
        if kind == "session_end":
            session_end = result.get("session_end")
            problems = self.contract.learning_issues("SessionEnd", session_end)
            if problems or session_end["session_id"] != session.session_id:
                raise ServiceUnavailable("learning service returned an invalid SessionEnd")
            return "session_end", session_end
        raise ServiceUnavailable("learning service returned neither a plan nor a SessionEnd")

    def _generate(self, plan: dict, record: RequestRecord, deadline: float) -> dict:
        request = {
            "schema_version": "1.0",
            "request_id": record.request_id,
            "trace_id": record.trace_id,
            "plan": plan,
            "response_artifact_id": self.config.response.response_artifact_id,
            "generation_seed": generation_seed(record.request_id),
            "deadline_ms": 0,
        }
        failure: Exception | None = None
        for attempt in range(2):  # PRD 14.3: at most one retry with the same plan and key
            remaining_ms = int((deadline - self.clock()) * 1000)
            if remaining_ms <= 0:
                raise ApiError(
                    "DEADLINE_EXCEEDED",
                    "The request deadline elapsed before a validated reply; retry the request.",
                    retryable=True,
                )
            request["deadline_ms"] = min(self.limits.generation_deadline_ms, remaining_ms)
            try:
                return self.response.generate(request)
            except GenerationFailure as error:
                failure = error
                log.warning(
                    "generation attempt failed trace_id=%s plan_id=%s attempt=%d reason=%s",
                    record.trace_id,
                    plan["plan_id"],
                    attempt + 1,
                    type(error).__name__,
                )
        raise generation_unavailable() from failure

    def _recover_commit(
        self, caller: str, record: RequestRecord, plan: dict, staged: dict
    ) -> dict | None:
        """After a crash between B's commit and our record of it, ask B what happened."""
        status = self.learning.plan_status(plan["plan_id"], trace_id=record.trace_id)
        state = status.get("status") if isinstance(status, dict) else None
        if state == "committed":
            if status.get("content_hash") != staged["content_hash"] or (
                status.get("response_id") not in (None, staged["response_id"])
            ):
                raise learning_unavailable(retryable=False)
            public_state = status.get("public_state")
            if public_state is None:
                # B's plan status carries no state (docs/learning_service.md), so the public
                # state after the commit is derived from the plan and our last committed state:
                # commit-plan is the only operation that advances the version and turn count.
                assert record.session_id is not None
                session = self._session_or_forbidden(caller, record.session_id)
                public_state = {
                    "session_id": session.session_id,
                    "state_version": plan["based_on_state_version"] + 1,
                    "current_concept_id": plan["concept_id"],
                    "goal_concept_id": plan["goal_concept_id"],
                    "teaching_turns": session.public_state["teaching_turns"] + 1,
                    "status": "active",
                    "end_reason": None,
                }
                log.warning(
                    "derived public state after recovery trace_id=%s plan_id=%s",
                    record.trace_id,
                    plan["plan_id"],
                )
            committed = {
                "schema_version": "1.0",
                "plan_id": plan["plan_id"],
                "status": "committed",
                "state_version": public_state["state_version"],
                "public_state": public_state,
            }
            if self.contract.learning_issues("CommitPlanResponse", committed):
                raise learning_unavailable(retryable=False)
            return committed
        if state == "pending":
            return None
        if state in {"canceled", "expired"}:
            raise plan_expired(self._known_version(caller, record.session_id))
        raise ServiceUnavailable("learning service returned an unknown plan status")

    def _commit(self, caller: str, record: RequestRecord, plan: dict, staged: dict) -> dict:
        self.store.mark_commit_attempt(caller, record.request_id)
        request = {
            "schema_version": "1.0",
            "session_id": plan["session_id"],
            "plan_id": plan["plan_id"],
            "based_on_state_version": plan["based_on_state_version"],
            "response_id": staged["response_id"],
            "content_hash": staged["content_hash"],
            "effective_move": staged["effective_move"],
            "delivery_mode": staged["delivery_mode"],
        }
        committed = self.learning.commit_plan(
            request, trace_id=record.trace_id, idempotency_key=record.request_id
        )
        self.hook("after_commit", {"record": record, "committed": committed})
        problems = self.contract.learning_issues("CommitPlanResponse", committed)
        if problems or committed["plan_id"] != plan["plan_id"]:
            raise ServiceUnavailable("learning service returned an invalid CommitPlanResponse")
        return committed

    def _finish_turn(
        self, caller: str, record: RequestRecord, message: str | None
    ) -> tuple[int, Any]:
        plan, staged, committed = record.plan, record.staged_response, record.committed
        assert plan is not None and staged is not None and committed is not None
        public_response = {
            "response_id": staged["response_id"],
            "plan_id": staged["plan_id"],
            "text": staged["text"],
            "delivery_mode": staged["delivery_mode"],
        }
        entries = self._learner_entry(record, message)
        entries.append(
            (
                record.request_id,
                {"role": "tutor", "request_id": record.request_id, "response": public_response},
            )
        )
        assert record.session_id is not None
        self.store.update_session(
            caller,
            record.session_id,
            committed["public_state"],
            last_response_id=staged["response_id"],
            last_plan=plan,
            entries=entries,
        )
        self.hook("before_reply", {"record": record})
        return 200, {
            "request_id": record.request_id,
            "response": public_response,
            "public_state": committed["public_state"],
        }

    def _finish_session_end(
        self, caller: str, record: RequestRecord, message: str | None
    ) -> tuple[int, Any]:
        session_end = record.plan
        assert session_end is not None and record.session_id is not None
        self.store.update_session(
            caller,
            record.session_id,
            session_end["public_state"],
            entries=self._learner_entry(record, message),
        )
        self.hook("before_reply", {"record": record})
        return 200, {"request_id": record.request_id, "session_end": session_end}

    @staticmethod
    def _learner_entry(record: RequestRecord, message: str | None) -> list[tuple[str, dict]]:
        if not message:
            return []
        entry = {"role": "learner", "request_id": record.request_id, "text": message}
        return [(record.request_id, entry)]

    def _supersede_pending(self, caller: str, session_id: str, current: RequestRecord) -> None:
        """Resolve older uncommitted turns of the session before reserving a new plan.

        A plan whose response was never committed was never shown to the learner, so B may cancel
        it without consuming a teaching turn (PRD 8.3). A plan that B did commit is finalized so
        the session state advances; the newer request then fails its version check.
        """
        for older in self.store.pending_turns(caller, session_id):
            if older.request_id == current.request_id or older.plan is None:
                continue
            committed = older.committed
            if committed is None and older.commit_attempted and older.staged_response is not None:
                try:
                    committed = self._recover_commit(
                        caller, older, older.plan, older.staged_response
                    )
                except ApiError as error:
                    if error.code != "PLAN_EXPIRED":
                        raise
            if committed is not None:
                self.store.stage_commit(caller, older.request_id, committed)
                older = self._record(caller, older.request_id)
                status, result = self._finish_turn(caller, older, None)
                self.store.complete_request(caller, older.request_id, status, result)
                continue
            self.learning.cancel_plan(
                {
                    "schema_version": "1.0",
                    "request_id": f"cancel-{older.request_id}",
                    "session_id": session_id,
                    "plan_id": older.plan["plan_id"],
                },
                trace_id=older.trace_id,
            )
            error = plan_expired(self._known_version(caller, session_id))
            self.store.complete_request(
                caller, older.request_id, error.status, error.body(older.request_id, older.trace_id)
            )

    # Learner stop (PRD 9.5 stop endpoint) ------------------------------------------------------

    def _stop_work(
        self, caller: str, record: RequestRecord, body: dict, deadline: float
    ) -> tuple[int, Any]:
        session_id = record.session_id
        assert session_id is not None
        if record.plan is None:
            session = self._session_or_forbidden(caller, session_id)
            if session.ended:
                raise session_ended(session.state_version)
            self._supersede_pending(caller, session_id, record)
            session = self._session_or_forbidden(caller, session_id)
            self._check_expectations(session, body["expected_state_version"], body)
            if session.last_response_id is None:
                # B accepts evidence, and therefore a stop request, only after a committed
                # response (docs/learning_service.md); recorded as an open point in D11.
                raise state_conflict(
                    "The session has no committed response yet; send the first turn before"
                    " stopping, or abandon the session.",
                    session.state_version,
                )
            evidence = self._evidence("stop", session, record)
            assert evidence["explicit_request"] == "stop"
            result = self.learning.observe_and_plan(
                self._observe_request(session, record, evidence), trace_id=record.trace_id
            )
            kind, payload = self._check_plan_or_end(result, session, body["expected_state_version"])
            if kind != "session_end":
                raise ServiceUnavailable("learning service did not end the session on a stop")
            self.store.stage_plan(caller, record.request_id, payload, kind)
            record = self._record(caller, record.request_id)
        return self._finish_session_end(caller, record, None)

    # Helpers ------------------------------------------------------------------------------------

    def _check_id(self, value: Any, name: str) -> None:
        if self.contract.public_issues("Id", value):
            raise invalid(f"{name} must be an ASCII string of 1 to 128 characters")

    def _session_or_forbidden(self, caller: str, session_id: str) -> SessionRecord:
        session = self.store.get_session(caller, session_id)
        if session is None:
            raise ApiError("FORBIDDEN", "No such session for this caller.", retryable=False)
        return session

    def _known_version(self, caller: str, session_id: str | None) -> int | None:
        if session_id is None:
            return None
        session = self.store.get_session(caller, session_id)
        return None if session is None else session.state_version

    @staticmethod
    def _stored(record: RequestRecord) -> Reply:
        """Replay a completed request exactly; a stored error raises like the first attempt."""
        status = record.status or 200
        if status >= 400 and isinstance(record.body, dict):
            raise ApiError(
                record.body["code"],
                record.body["message"],
                retryable=bool(record.body["retryable"]),
                current_state_version=record.body.get("current_state_version"),
                terminal=not record.body["retryable"],
            )
        return Reply(status, record.body, record.trace_id)

    def _record(self, caller: str, request_id: str) -> RequestRecord:
        record = self.store.get_request(caller, request_id)
        if record is None:
            raise ServiceUnavailable("request record disappeared")
        return record

    def _session_lock(self, caller: str, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault((caller, session_id), threading.Lock())

    def _assert_public(self, schema: str, body: Any) -> None:
        """Last line of defense: a public body must match the contract and leak nothing."""
        problems = self.contract.public_issues(schema, body) + find_internal_fields(body)
        if problems:
            raise ServiceUnavailable(f"refusing to return a non-conforming {schema}: {problems}")

    def _log(self, operation: str, record: RequestRecord, status: int, **fields: Any) -> None:
        # Metadata only: raw learner text never enters application logs (PRD 14.3, 23.1).
        extra = " ".join(f"{key}={value}" for key, value in fields.items())
        log.info(
            "%s status=%d trace_id=%s request_id=%s session_id=%s plan_id=%s %s",
            operation,
            status,
            record.trace_id,
            record.request_id,
            record.session_id,
            (record.plan or {}).get("plan_id"),
            extra,
        )
