import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from apps.learning_service.engine import LearningEngine
from apps.learning_service.http import LearningHTTPServer
from apps.learning_service.schemas import validate


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("TUTOR_LEARNING_SERVICE_TOKEN", "test-token")
    engine = LearningEngine({}, tmp_path / "nested" / "runtime" / "learning.sqlite3")
    server = LearningHTTPServer(("127.0.0.1", 0), engine)
    thread = Thread(target=server.serve_forever)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", engine
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()
    engine.close()
    assert not thread.is_alive()


def call(service, path, body=None, *, key=None, omit=(), raw=None, extra_headers=None):
    headers = {
        "Authorization": "Bearer test-token",
        "X-Trace-Id": "http-test",
        "Content-Type": "application/json",
    }
    if key is not None:
        headers["Idempotency-Key"] = key
    headers.update(extra_headers or {})
    for name in omit:
        headers.pop(name, None)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = Request(service[0] + path, data=data, headers=headers)
    try:
        response = urlopen(request, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        assert response.headers.get_content_type() == "application/json"
        result = json.load(response)
        if response.status != 200:
            validate("Error", result)
        return response.status, result


def test_http_smoke(service):
    path = "/internal/v1/sessions"
    body = dict(
        schema_version="1.0", request_id="create", goal_concept_id="C01", pace_preference="normal"
    )
    for missing in ("Authorization", "X-Trace-Id"):
        status, error = call(service, path, key="create", omit=(missing,), raw=b"not-json")
        assert status == 401
        assert error["code"] == "UNAUTHORIZED"
    status, created = call(service, path, body, key="create")
    assert status == 200
    validate("CreateSessionResponse", created)
    assert call(service, path, body, key="create") == (status, created)
    observe = dict(
        schema_version="1.0",
        request_id="observe",
        session_id=created["session_id"],
        expected_state_version=0,
        evidence=None,
    )
    status, result = call(service, "/internal/v1/observe-and-plan", observe, key="observe")
    assert status == 200
    validate("ObserveResponse", result)
    plan = result["teaching_plan"]
    validate("TeachingPlan", plan)
    assert plan["trace_id"] == "http-test"
    assert call(service, "/internal/v1/observe-and-plan", observe, key="observe") == (200, result)
    status, conflict = call(
        service,
        "/internal/v1/observe-and-plan",
        {**observe, "request_id": "competing"},
        key="competing",
    )
    assert status == 409
    assert conflict["code"] == "STATE_CONFLICT"
    assert conflict["current_state_version"] == 0
    commit = dict(
        schema_version="1.0",
        session_id=created["session_id"],
        plan_id=plan["plan_id"],
        based_on_state_version=0,
        response_id="response-one",
        content_hash="a" * 64,
        effective_move=plan["teaching_action"],
        delivery_mode="model",
    )
    status, committed = call(service, "/internal/v1/commit-plan", commit, key="commit")
    assert status == 200
    validate("CommitPlanResponse", committed)
    assert committed["state_version"] == 1
    assert call(service, "/internal/v1/commit-plan", commit, key="commit") == (200, committed)
    status, record = call(service, "/internal/v1/plans/" + plan["plan_id"])
    assert status == 200
    validate("PlanStatus", record)
    assert record["status"] == "committed"
    assert record["response_id"] == "response-one"
    status, grade = call(
        service,
        "/internal/v1/grade",
        dict(
            schema_version="1.0",
            request_id="grade",
            problem_instance_id="missing",
            normalized_answer="0",
        ),
        key="grade",
    )
    assert status == 200
    validate("GradeResponse", grade)
    assert grade["correctness"] == "unknown"
    status, error = call(
        service,
        "/internal/v1/cancel-plan",
        dict(
            schema_version="1.0",
            request_id="cancel",
            session_id=created["session_id"],
            plan_id=plan["plan_id"],
        ),
        key="cancel",
    )
    assert status == 409
    assert error["code"] == "STATE_CONFLICT"


@pytest.mark.parametrize(
    "raw,key,headers",
    [
        (b"{}", None, {}),
        (b"{", "bad", {}),
        (b"[]", "bad", {}),
        (b'{"request_id":"other"}', "bad", {}),
        (b"x" * 65537, "bad", {}),
        (b'{"value":NaN}', "bad", {}),
        (b'{"value":1e999}', "bad", {}),
        (b"{}", "bad", {"Content-Length": "-1"}),
        (b"{}", "bad", {"Transfer-Encoding": "chunked"}),
    ],
)
def test_invalid_requests(service, raw, key, headers):
    status, error = call(service, "/internal/v1/sessions", key=key, raw=raw, extra_headers=headers)
    assert status == 422
    assert error["code"] == "INVALID_SCHEMA"


@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer wrong"},
        {"X-Trace-Id": ""},
        {"X-Trace-Id": "x" * 129},
        {"X-Trace-Id": "é"},
    ],
)
def test_auth_headers(service, headers):
    status, error = call(service, "/internal/v1/sessions", raw=b"{", extra_headers=headers)
    assert status == 401
    assert error["code"] == "UNAUTHORIZED"


def test_unknown_route_and_sanitized_failure(service, monkeypatch):
    status, error = call(service, "/internal/v1/unknown")
    assert status == 422
    assert error["code"] == "INVALID_SCHEMA"

    def unavailable(*args, **kwargs):
        raise RuntimeError("internal-answer-and-stack-details")

    monkeypatch.setattr(service[1], "create_session", unavailable)
    status, error = call(service, "/internal/v1/sessions", {"request_id": "fail"}, key="fail")
    assert status == 503
    assert error["code"] == "LEARNING_UNAVAILABLE"
    assert error["retryable"] is True
    assert error["message"] == "Learning service is unavailable"


def test_server_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("TUTOR_LEARNING_SERVICE_TOKEN", raising=False)
    engine = LearningEngine({}, tmp_path / "db")
    try:
        with pytest.raises(ValueError, match="must not be empty"):
            LearningHTTPServer(("127.0.0.1", 0), engine)
    finally:
        engine.close()
