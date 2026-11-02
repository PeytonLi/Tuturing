"""Public API recovery and boundary tests (PRD 8.2, 8.3, 9.5, 21.1 item 5, gate G4).

The learning side is Person A's transaction stand-in (learning_standin.py), which mirrors B's
wire shapes; test_smoke_workflow.py runs the same public API against B's real service. The
response side is the in-process CPU stub. Nothing here estimates student state.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

import pytest

from apps.public_api.clients import (
    GenerationFailure,
    HttpLearningClient,
    InProcessResponseClient,
)
from apps.public_api.config import LearningConfig, PublicApiConfig
from apps.public_api.contract import (
    PRD_ERROR_STATUS,
    contract_problems,
    find_internal_fields,
    load_contract,
    load_json_file,
)
from apps.public_api.errors import ApiError
from apps.public_api.server import PublicApiServer
from apps.public_api.service import PublicApi
from apps.public_api.storage import RequestStore
from tests.integration.response.learning_standin import (
    FIXTURES,
    LearningStandIn,
    StandInServer,
)

CALLER = "alice"
CONTRACT = load_contract()


class SimulatedCrash(RuntimeError):
    """Injected process failure; the request record stays durable."""


class CrashOnce:
    def __init__(self, point: str) -> None:
        self.point = point
        self.fired = False

    def __call__(self, point: str, context: dict) -> None:
        if point == self.point and not self.fired:
            self.fired = True
            raise SimulatedCrash(point)


class FlakyResponseClient:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.inner = InProcessResponseClient()

    def generate(self, request: dict) -> dict:
        self.calls += 1
        if self.failures > 0:
            self.failures -= 1
            raise GenerationFailure("injected generation failure")
        return self.inner.generate(request)


def make_api(tmp_path, *, hook=None, response=None, standin=None, config=None):
    standin = standin or LearningStandIn()
    store = RequestStore(tmp_path / "requests.sqlite")
    api = PublicApi(
        store,
        standin,
        response or InProcessResponseClient(),
        config=config or PublicApiConfig(),
        hook=hook,
    )
    return api, standin, store


def assert_public(schema: str, body) -> None:
    assert CONTRACT.public_issues(schema, body) == []
    assert find_internal_fields(body) == []


def create(api: PublicApi, request_id: str = "create-1") -> str:
    reply = api.create_session(
        CALLER, {"request_id": request_id, "goal_concept_id": "C01", "pace_preference": "normal"}
    )
    assert reply.status == 201
    assert_public("CreateSessionResult", reply.body)
    return reply.body["session_id"]


def turn_body(request_id: str, version: int, source: str | None, message: str) -> dict:
    return {
        "request_id": request_id,
        "expected_state_version": version,
        "source_response_id": source,
        "message": message,
    }


def first_turn(api: PublicApi, session_id: str, request_id: str = "turn-1"):
    reply = api.turn(CALLER, session_id, turn_body(request_id, 0, None, ""))
    assert reply.status == 200
    assert_public("TurnResult", reply.body)
    return reply


# Contract placement (D11) ------------------------------------------------------------------------


def test_public_contract_references_b_projections_and_prd_error_table():
    assert contract_problems(CONTRACT) == []
    learning = CONTRACT.learning["$defs"]
    schemas = CONTRACT.document["components"]["schemas"]
    assert CONTRACT.deref(schemas["PublicState"]) == learning["PublicState"]
    assert CONTRACT.deref(schemas["SessionEnd"]) == learning["SessionEnd"]
    session_end = load_json_file(FIXTURES / "session_end.json")
    assert CONTRACT.public_issues("SessionEnd", session_end) == []
    assert find_internal_fields(session_end) == []
    assert CONTRACT.error_status == PRD_ERROR_STATUS
    assert sorted(learning["Error"]["properties"]["code"]["enum"]) == sorted(PRD_ERROR_STATUS)
    leaked = dict(session_end["public_state"], mastery=1)
    assert CONTRACT.public_issues("PublicState", leaked) != []


# Session creation and turns -----------------------------------------------------------------


def test_create_session_first_turn_and_learner_turn(tmp_path):
    api, standin, store = make_api(tmp_path)
    session_id = create(api)
    again = api.create_session(
        CALLER, {"request_id": "create-1", "goal_concept_id": "C01", "pace_preference": "normal"}
    )
    assert again.body["session_id"] == session_id
    assert standin.calls.count("create_session") == 1

    first = first_turn(api, session_id)
    response = first.body["response"]
    assert set(response) == {"response_id", "plan_id", "text", "delivery_mode"}
    assert first.body["public_state"]["state_version"] == 1
    stored = store.get_request(CALLER, "turn-1")
    assert standin.committed[response["plan_id"]].content_hash == stored.content_hash

    second = api.turn(
        CALLER, session_id, turn_body("turn-2", 1, response["response_id"], "is it 1/2?")
    )
    assert second.status == 200
    assert_public("TurnResult", second.body)
    assert second.body["public_state"]["state_version"] == 2
    assert len(standin.sessions[session_id].applied_events) == 1

    view = api.get_session(CALLER, session_id)
    assert_public("SessionView", view.body)
    assert [entry["role"] for entry in view.body["transcript"]] == ["tutor", "learner", "tutor"]
    assert view.body["transcript"][1]["text"] == "is it 1/2?"


def test_retry_after_dropped_connection_returns_same_response_id_and_hash(tmp_path):
    hook = CrashOnce("before_reply")  # the reply never reaches the client
    api, standin, store = make_api(tmp_path, hook=hook)
    session_id = create(api)
    with pytest.raises(SimulatedCrash):
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    interrupted = store.get_request(CALLER, "turn-1")
    assert not interrupted.complete and interrupted.committed is not None

    reply = api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert reply.status == 200
    assert reply.body["response"]["response_id"] == interrupted.staged_response["response_id"]
    plan_id = reply.body["response"]["plan_id"]
    assert standin.committed[plan_id].content_hash == interrupted.content_hash
    assert standin.calls.count("commit_plan") == 1
    assert standin.sessions[session_id].state_version == 1
    assert api.turn(CALLER, session_id, turn_body("turn-1", 0, None, "")).body == reply.body
    view = api.get_session(CALLER, session_id)
    assert [entry["role"] for entry in view.body["transcript"]] == ["tutor"]


def test_request_id_reuse_with_different_content_is_a_conflict(tmp_path):
    api, standin, _ = make_api(tmp_path)
    session_id = create(api)
    first = first_turn(api, session_id)
    with pytest.raises(ApiError) as excinfo:
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, "different"))
    assert excinfo.value.code == "STATE_CONFLICT"
    assert excinfo.value.status == 409
    assert excinfo.value.retryable is False
    assert api.turn(CALLER, session_id, turn_body("turn-1", 0, None, "")).body == first.body
    assert standin.sessions[session_id].state_version == 1


def test_competing_requests_on_the_same_version_get_one_reservation(tmp_path):
    reserved, proceed = threading.Event(), threading.Event()

    def hold_after_reservation(point: str, context: dict) -> None:
        if point == "after_observe" and context["record"].request_id == "turn-a":
            reserved.set()
            assert proceed.wait(5)

    api, standin, _ = make_api(tmp_path, hook=hold_after_reservation)
    session_id = create(api)
    results: dict[str, object] = {}

    def run_a() -> None:
        results["a"] = api.turn(CALLER, session_id, turn_body("turn-a", 0, None, ""))

    thread = threading.Thread(target=run_a)
    thread.start()
    assert reserved.wait(5)
    with pytest.raises(ApiError) as excinfo:
        api.turn(CALLER, session_id, turn_body("turn-b", 0, None, ""))
    assert excinfo.value.code == "STATE_CONFLICT"
    proceed.set()
    thread.join(5)
    assert results["a"].status == 200
    assert standin.sessions[session_id].state_version == 1
    assert len(standin.committed) == 1
    with pytest.raises(ApiError) as stale:  # after the winner commits, the version is stale
        api.turn(CALLER, session_id, turn_body("turn-c", 0, None, ""))
    assert stale.value.code == "STATE_CONFLICT"
    assert stale.value.current_state_version == 1


def test_crash_after_commit_recovers_the_identical_committed_response(tmp_path):
    api, standin, store = make_api(tmp_path, hook=CrashOnce("after_commit"))
    session_id = create(api)
    with pytest.raises(SimulatedCrash):
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    record = store.get_request(CALLER, "turn-1")
    assert record.commit_attempted and record.committed is None
    assert standin.sessions[session_id].state_version == 1

    reply = api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert reply.status == 200
    assert_public("TurnResult", reply.body)
    assert reply.body["response"]["response_id"] == record.staged_response["response_id"]
    assert reply.body["response"]["text"] == record.staged_response["text"]
    assert reply.body["public_state"] == standin.sessions[session_id].public_state()
    assert standin.calls.count("commit_plan") == 1
    assert standin.calls.count("plan_status") == 1
    assert standin.calls.count("observe_and_plan") == 1
    # the recovered state is the base for the next turn
    second = api.turn(
        CALLER, session_id, turn_body("turn-2", 1, reply.body["response"]["response_id"], "1/2")
    )
    assert second.body["public_state"]["state_version"] == 2


def test_crash_after_storing_commit_finishes_without_calling_b(tmp_path):
    api, standin, store = make_api(tmp_path, hook=CrashOnce("after_stage_commit"))
    session_id = create(api)
    with pytest.raises(SimulatedCrash):
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    calls_before = list(standin.calls)
    reply = api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert reply.status == 200
    assert standin.calls == calls_before
    stored = store.get_request(CALLER, "turn-1")
    assert reply.body["response"]["response_id"] == stored.staged_response["response_id"]


def test_generation_failure_is_retryable_and_the_plan_stays_pending(tmp_path):
    flaky = FlakyResponseClient(failures=2)
    api, standin, store = make_api(tmp_path, response=flaky)
    session_id = create(api)
    with pytest.raises(ApiError) as excinfo:
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert excinfo.value.code == "GENERATION_UNAVAILABLE"
    assert excinfo.value.status == 503 and excinfo.value.retryable is True
    assert flaky.calls == 2  # PRD 14.3: one retry with the same plan and key
    pending = standin.sessions[session_id].pending
    assert pending is not None and pending.request_id == "turn-1"
    assert standin.committed == {}
    assert store.get_request(CALLER, "turn-1").plan["plan_id"] == pending.plan["plan_id"]

    reply = api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert reply.status == 200
    assert reply.body["response"]["plan_id"] == pending.plan["plan_id"]
    assert standin.calls.count("observe_and_plan") == 1
    assert standin.sessions[session_id].state_version == 1


def test_expired_plan_after_generation_failure_returns_plan_expired(tmp_path):
    standin = LearningStandIn(plan_ttl_s=0.05)
    api, _, _ = make_api(tmp_path, response=FlakyResponseClient(failures=2), standin=standin)
    session_id = create(api)
    with pytest.raises(ApiError):
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    time.sleep(0.1)
    with pytest.raises(ApiError) as excinfo:
        api.turn(CALLER, session_id, turn_body("turn-1", 0, None, ""))
    assert excinfo.value.code == "PLAN_EXPIRED"
    assert standin.sessions[session_id].state_version == 0  # no teaching turn consumed
    reply = api.turn(CALLER, session_id, turn_body("turn-2", 0, None, ""))
    assert reply.status == 200


def test_stop_returns_session_end_and_later_turns_are_rejected(tmp_path):
    api, standin, _ = make_api(tmp_path)
    session_id = create(api)
    first = first_turn(api, session_id)
    stop_body = {"request_id": "stop-1", "expected_state_version": 1}
    stop = api.stop(CALLER, session_id, stop_body)
    assert stop.status == 200
    assert_public("StopResult", stop.body)
    assert stop.body["session_end"]["reason"] == "learner_stop"
    assert stop.body["session_end"]["public_state"]["status"] == "ended"
    assert api.stop(CALLER, session_id, stop_body).body == stop.body
    assert api.get_session(CALLER, session_id).body["public_state"]["status"] == "ended"
    with pytest.raises(ApiError) as excinfo:
        api.turn(
            CALLER, session_id, turn_body("turn-2", 2, first.body["response"]["response_id"], "hi")
        )
    assert excinfo.value.code == "SESSION_ENDED"
    assert standin.sessions[session_id].teaching_turns == 1


def test_stop_before_any_committed_response_is_a_conflict(tmp_path):
    api, standin, _ = make_api(tmp_path)
    session_id = create(api)
    with pytest.raises(ApiError) as excinfo:
        api.stop(CALLER, session_id, {"request_id": "stop-0", "expected_state_version": 0})
    assert excinfo.value.code == "STATE_CONFLICT"
    assert standin.calls.count("observe_and_plan") == 0


def test_stop_cancels_an_undelivered_pending_plan(tmp_path):
    api, standin, _ = make_api(tmp_path, response=FlakyResponseClient(failures=0))
    session_id = create(api)
    first = first_turn(api, session_id)
    api.response.failures = 2
    source = first.body["response"]["response_id"]
    with pytest.raises(ApiError) as failed:
        api.turn(CALLER, session_id, turn_body("turn-2", 1, source, "1/2"))
    assert failed.value.code == "GENERATION_UNAVAILABLE"
    stop = api.stop(CALLER, session_id, {"request_id": "stop-1", "expected_state_version": 1})
    assert stop.status == 200
    assert standin.calls.count("cancel_plan") == 1
    assert standin.sessions[session_id].teaching_turns == 1
    with pytest.raises(ApiError) as excinfo:
        api.turn(CALLER, session_id, turn_body("turn-2", 1, source, "1/2"))
    assert excinfo.value.code == "PLAN_EXPIRED"


def test_session_end_from_observe_and_plan_is_the_other_turn_variant(tmp_path):
    api, _, _ = make_api(tmp_path)
    session_id = create(api)
    first = first_turn(api, session_id)
    reply = api.turn(
        CALLER, session_id, turn_body("turn-2", 1, first.body["response"]["response_id"], "stop")
    )
    assert reply.status == 200
    assert_public("TurnResult", reply.body)
    assert set(reply.body) == {"request_id", "session_end"}
    assert reply.body["session_end"]["reason"] == "learner_stop"


def test_validation_errors_are_sanitized_and_never_log_learner_text(tmp_path, caplog):
    api, _, _ = make_api(tmp_path)
    session_id = create(api)
    secret = "SECRET-LEARNER-TEXT-4242"
    with caplog.at_level(logging.INFO):
        first = first_turn(api, session_id)
        source = first.body["response"]["response_id"]
        api.turn(CALLER, session_id, turn_body("turn-2", 1, source, secret))
        with pytest.raises(ApiError) as too_long:
            api.turn(CALLER, session_id, turn_body("turn-3", 2, "x", secret * 400))
        with pytest.raises(ApiError) as unknown_field:
            api.turn(CALLER, session_id, dict(turn_body("turn-4", 2, "x", "hi"), extra=secret))
        with pytest.raises(ApiError) as mismatch:
            body = dict(turn_body("turn-5", 2, "x", "hi"), session_id="other")
            api.turn(CALLER, session_id, body)
        with pytest.raises(ApiError) as forbidden:
            api.turn(CALLER, "ses-unknown", turn_body("turn-6", 0, None, ""))
    for error in (too_long, unknown_field, mismatch):
        assert error.value.code == "INVALID_SCHEMA" and error.value.status == 422
        assert secret not in error.value.message
    assert forbidden.value.code == "FORBIDDEN"
    assert secret not in caplog.text


# HTTP transport --------------------------------------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    api, standin, _ = make_api(tmp_path)
    server = PublicApiServer(
        ("127.0.0.1", 0), api, {"token-alice": CALLER}, max_request_bytes=65536
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, standin
    finally:
        server.shutdown()
        server.server_close()


def call(base_url: str, method: str, path: str, body=None, *, token="token-alice", headers=None):
    if body is None or isinstance(body, bytes):
        data = body
    else:
        data = json.dumps(body).encode()
    request = urllib.request.Request(base_url + path, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read()), dict(error.headers)


def test_http_flow_limits_and_malformed_requests(http_server):
    server, standin = http_server
    base = server.base_url
    status, body, headers = call(base, "POST", "/v1/sessions", {"request_id": "c"}, token=None)
    assert status == 401 and body["code"] == "UNAUTHORIZED"
    assert headers["WWW-Authenticate"].startswith("Bearer")
    assert CONTRACT.public_issues("Error", body) == []

    status, body, _ = call(base, "POST", "/v2/sessions", {"request_id": "c"})
    assert status == 400 and body["code"] == "UNSUPPORTED_VERSION"

    create_body = {"request_id": "create-1", "goal_concept_id": "C01", "pace_preference": "normal"}
    status, body, headers = call(base, "POST", "/v1/sessions", create_body)
    assert status == 201 and headers["Tutor-Trace-Id"].startswith("trc-")
    assert_public("CreateSessionResult", body)
    session_id = body["session_id"]
    turns = f"/v1/sessions/{session_id}/turns"

    status, body, _ = call(base, "POST", turns, b"{not json")
    assert status == 422 and body["code"] == "INVALID_SCHEMA"
    status, body, _ = call(base, "POST", turns, b'{"request_id": "d", "request_id": "d"}')
    assert status == 422
    status, body, _ = call(base, "POST", turns, turn_body("long", 0, None, "x" * 8001))
    assert status == 422 and body["request_id"] == "long"
    raw = json.dumps(turn_body("big", 0, None, "x" * 70000)).encode()
    assert len(raw) > 65536
    status, body, _ = call(base, "POST", turns, raw)
    assert status == 422 and "65536" in body["message"]
    status, body, _ = call(base, "POST", turns, dict(turn_body("u", 0, None, ""), extra=1))
    assert status == 422
    status, body, _ = call(base, "POST", turns, dict(turn_body("m", 0, None, ""), session_id="x"))
    assert status == 422
    status, body, _ = call(
        base, "POST", turns, turn_body("k", 0, None, ""), headers={"Idempotency-Key": "other"}
    )
    assert status == 422
    unknown = "/v1/sessions/ses-unknown/turns"
    status, body, _ = call(base, "POST", unknown, turn_body("f", 0, None, ""))
    assert status == 403 and body["code"] == "FORBIDDEN"
    status, body, _ = call(base, "GET", "/v1/sessions/ses-unknown")
    assert status == 403
    status, body, _ = call(base, "POST", "/v1/nothing", {"request_id": "n"})
    assert status == 422

    status, body, _ = call(base, "POST", turns, turn_body("turn-1", 0, None, ""))
    assert status == 200
    assert_public("TurnResult", body)
    response_id = body["response"]["response_id"]
    status, again, _ = call(base, "POST", turns, turn_body("turn-1", 0, None, ""))
    assert status == 200 and again == body
    status, body, _ = call(base, "POST", turns, turn_body("turn-2", 1, response_id, "1/2"))
    assert status == 200 and body["public_state"]["state_version"] == 2
    status, body, _ = call(base, "POST", turns, turn_body("turn-stale", 1, response_id, "1/2"))
    assert status == 409 and body["code"] == "STATE_CONFLICT"
    assert body["current_state_version"] == 2 and body["retryable"] is False
    status, body, _ = call(base, "GET", f"/v1/sessions/{session_id}")
    assert status == 200
    assert_public("SessionView", body)
    assert len(body["transcript"]) == 3
    stop_body = {"request_id": "stop", "expected_state_version": 2}
    status, body, _ = call(base, "POST", f"/v1/sessions/{session_id}/stop", stop_body)
    assert status == 200
    assert_public("StopResult", body)
    assert standin.sessions[session_id].status == "ended"


def test_http_learning_client_against_the_stand_in_server(tmp_path, monkeypatch):
    standin = LearningStandIn()
    server = StandInServer(standin, "service-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("TUTOR_LEARNING_SERVICE_TOKEN", "service-token")
        learning = HttpLearningClient(
            server.base_url, service_token_env="TUTOR_LEARNING_SERVICE_TOKEN", timeout_s=5
        )
        config = PublicApiConfig(learning=LearningConfig(mode="http", base_url=server.base_url))
        api = PublicApi(
            RequestStore(tmp_path / "requests.sqlite"),
            learning,
            InProcessResponseClient(),
            config=config,
        )
        session_id = create(api)
        first = first_turn(api, session_id)
        response_id = first.body["response"]["response_id"]
        second = api.turn(CALLER, session_id, turn_body("turn-2", 1, response_id, "idk"))
        assert second.status == 200 and second.body["public_state"]["state_version"] == 2
        assert standin.calls.count("commit_plan") == 2
        with pytest.raises(ApiError) as stale:  # B's conflict record maps onto the public error
            learning.observe_and_plan(
                {
                    "schema_version": "1.0",
                    "request_id": "stale-observe",
                    "session_id": session_id,
                    "expected_state_version": 1,
                    "evidence": None,
                },
                trace_id="trace-test",
            )
        assert stale.value.code == "STATE_CONFLICT" and stale.value.current_state_version == 2
        status = learning.plan_status(first.body["response"]["plan_id"], trace_id="trace-test")
        assert status["status"] == "committed" and status["response_id"] == response_id
        monkeypatch.setenv("TUTOR_LEARNING_SERVICE_TOKEN", "wrong")
        with pytest.raises(ApiError) as unavailable:
            api.stop(CALLER, session_id, {"request_id": "stop", "expected_state_version": 2})
        assert unavailable.value.code == "LEARNING_UNAVAILABLE"
    finally:
        server.shutdown()
        server.server_close()
