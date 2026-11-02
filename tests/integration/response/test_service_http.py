"""HTTP round trips against the real response service on a loopback port."""

import copy
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from apps.response_service.config import load_config
from apps.response_service.server import (
    GENERATE_PATH,
    LOGGER,
    ResponseService,
    build_service,
    make_server,
)
from packages.response.contracts import schema_errors
from packages.response.pipeline import PipelineSettings, ResponsePipeline, stub_pipeline
from packages.response.responders import ResponderOutput
from tests.unit.response.pipeline_helpers import (
    LEARNING_INVALID,
    ROOT,
    lane_plan,
    learning_plans,
    load_json,
    make_request,
)

TOKEN = "integration-test-token"
STUB_CONFIG = ROOT / "configs" / "response" / "serve_stub.yaml"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def serve(service, config):
    server = make_server(config, service)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    return server, thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def server():
    config = load_config(STUB_CONFIG, overrides={"port": 0})
    service = build_service(config, environ={"TUTOR_RESPONSE_SERVICE_TOKEN": TOKEN})
    server, thread = serve(service, config)
    yield server
    stop(server, thread)


def url(server, path=GENERATE_PATH):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


def call(server, body=None, *, token=TOKEN, path=GENERATE_PATH, method="POST", raw=None,
         headers=None):
    data = raw if raw is not None else (
        None if body is None else json.dumps(body).encode("utf-8")
    )
    request = urllib.request.Request(url(server, path), data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with OPENER.open(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def assert_error(body, code, request_id="unknown"):
    assert body["schema_version"] == "1.0"
    assert body["code"] == code
    assert body["request_id"] == request_id
    assert isinstance(body["retryable"], bool)
    assert body["current_state_version"] is None
    assert set(body) == {
        "schema_version", "request_id", "trace_id", "code", "message", "retryable",
        "current_state_version",
    }


def test_health_reports_the_artifact_and_versions_without_auth(server):
    for path in ("/internal/v1/health", "/healthz"):
        status, body = call(server, token=None, path=path, method="GET")
        assert status == 200
        assert body["status"] == "ok"
        assert body["artifact_id"] == "response_stub_pipeline_v1"
        assert body["responder_kind"] == "template_stub"
        assert body["versions"]["prompt"] == "prompt_v1"
        assert body["auth"] == "bearer"


def test_every_learning_plan_round_trips_with_all_checks_true(server):
    plans = learning_plans()
    assert len(plans) == 9
    for name, plan in plans.items():
        request = make_request(plan, f"http-{name}")
        status, body = call(server, request)
        assert status == 200, body
        assert schema_errors(body, "GenerateResponse") == []
        assert all(body["checks"].values())
        expected = stub_pipeline().generate(copy.deepcopy(request))
        body["usage"].pop("elapsed_ms")
        expected["usage"].pop("elapsed_ms")
        assert body == expected


def test_retry_returns_the_same_response_and_conflicting_reuse_is_rejected(server):
    request = make_request(learning_plans()["teaching_plan_hint"], "http-retry")
    first_status, first = call(server, request)
    second_status, second = call(server, request)
    assert first_status == second_status == 200
    first["usage"].pop("elapsed_ms")
    second["usage"].pop("elapsed_ms")
    assert first == second
    changed = copy.deepcopy(request)
    changed["plan"]["response_context"]["last_student_message"] = "something else"
    status, body = call(server, changed)
    assert status == 409
    assert_error(body, "STATE_CONFLICT", "http-retry")
    assert body["retryable"] is False


def test_requests_without_a_valid_bearer_token_are_refused(server):
    request = make_request(learning_plans()["teaching_plan_probe"], "http-auth")
    for token in (None, "", "wrong-token", TOKEN + "x"):
        status, body = call(server, request, token=token)
        assert status == 401
        assert_error(body, "UNAUTHORIZED")
    status, body = call(server, request, token=None, headers={"Authorization": "Basic abc"})
    assert status == 401


def test_oversized_body_is_refused_before_it_is_read(server):
    host, port = server.server_address[:2]
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(
            (
                f"POST {GENERATE_PATH} HTTP/1.1\r\nHost: {host}\r\n"
                f"Authorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n"
                "Content-Length: 262145\r\n\r\n"
            ).encode("ascii")
        )
        sock.settimeout(10)
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = sock.recv(65536)
            if not chunk:
                break
            raw += chunk
        head, _, rest = raw.partition(b"\r\n\r\n")
        length_lines = [
            line for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")
        ]
        length = int(length_lines[0].split(b":")[1])
        while len(rest) < length:
            chunk = sock.recv(65536)
            if not chunk:
                break
            rest += chunk
    assert head.startswith(b"HTTP/1.1 422")
    assert b"connection: close" in head.lower()
    body = json.loads(rest.decode("utf-8"))
    assert_error(body, "INVALID_SCHEMA")
    assert "262144" in body["message"]
    status, body = call(server, raw=b"{}" * 140_000)
    assert status == 422
    assert_error(body, "INVALID_SCHEMA")


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        (b"not json", "not valid JSON"),
        (b'{"schema_version": "1.0", "deadline_ms": NaN}', "not valid JSON"),
        (b"[1, 2]", "GenerateRequest"),
        (b"{}", "required"),
    ],
)
def test_malformed_bodies_return_invalid_schema(server, raw, fragment):
    status, body = call(server, raw=raw)
    assert status == 422
    assert_error(body, "INVALID_SCHEMA")
    assert fragment in body["message"]


def test_independent_reveal_plan_is_rejected_over_http(server):
    plan = load_json(LEARNING_INVALID / "independent_reveal_solution.json")
    status, body = call(server, make_request(plan, "http-reveal"))
    assert status == 422
    assert_error(body, "INVALID_SCHEMA", "http-reveal")
    assert "independent plans must not carry a reveal_solution" in body["message"]
    assert "1/2" not in body["message"]


def test_unsupported_versions_and_unknown_paths(server):
    request = make_request(learning_plans()["teaching_plan_practice"], "http-version")
    request["schema_version"] = "2.0"
    status, body = call(server, request)
    assert status == 400
    assert_error(body, "UNSUPPORTED_VERSION", "http-version")
    request = make_request(
        learning_plans()["teaching_plan_practice"], "http-artifact", artifact="other_artifact"
    )
    status, body = call(server, request)
    assert status == 400
    assert_error(body, "UNSUPPORTED_VERSION", "http-artifact")
    for path, method in (("/internal/v1/other", "POST"), ("/", "GET")):
        status, body = call(server, {} if method == "POST" else None, path=path, method=method)
        assert status == 404
        assert body == {"message": "not found"}


def test_logs_never_contain_learner_text_plan_content_or_the_token(server, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER.name)
    plan = lane_plan("practice")
    sentinels = {
        "learner": "LEARNER-SENTINEL my phone number is 555-0100",
        "dialogue": "DIALOGUE-SENTINEL earlier message",
        "stem": plan["content"]["problem_stem"],
        "fingerprint": plan["constraints"]["forbidden_answer_fingerprints"][0],
    }
    plan["response_context"]["last_student_message"] = sentinels["learner"]
    plan["response_context"]["recent_dialogue"] = [
        {"role": "student", "text": sentinels["dialogue"]}
    ]
    ok_status, ok_body = call(server, make_request(plan, "http-log-ok"))
    bad = make_request(plan, "http-log-bad")
    bad["plan"]["content"]["reveal_solution"] = "SECRET-SENTINEL 3/10"
    bad_status, _ = call(server, bad)
    wrong_token = "WRONG-TOKEN-SENTINEL"
    unauthorized_status, _ = call(
        server, make_request(plan, "http-log-auth"), token=wrong_token
    )
    assert (ok_status, bad_status, unauthorized_status) == (200, 422, 401)
    lines = [record.getMessage() for record in caplog.records if record.name == LOGGER.name]
    assert len(lines) >= 3
    text = "\n".join(lines)
    for value in (*sentinels.values(), "SECRET-SENTINEL", TOKEN, wrong_token, ok_body["text"]):
        assert value not in text
    entries = [json.loads(line) for line in lines]
    accepted = next(entry for entry in entries if entry["request_id"] == "http-log-ok")
    assert accepted["status"] == 200
    assert accepted["move"] == "PRACTICE"
    assert accepted["delivery_mode"] == "model"
    assert accepted["checks"]["disclosure_allowed"] is True
    assert accepted["attempts"] == [{"stage": "model", "outcome": "accepted"}]
    rejected = next(entry for entry in entries if entry["request_id"] == "http-log-bad")
    assert (rejected["status"], rejected["code"]) == (422, "INVALID_SCHEMA")
    unauthorized = next(entry for entry in entries if entry.get("code") == "UNAUTHORIZED")
    assert unauthorized["request_id"] == "unknown"


def test_loopback_stub_without_a_token_serves_unauthenticated_requests():
    config = load_config(STUB_CONFIG, overrides={"port": 0})
    service = build_service(config, environ={})
    assert not service.authenticated
    server, thread = serve(service, config)
    try:
        status, body = call(
            server, make_request(learning_plans()["teaching_plan_reflect"], "http-open"),
            token=None,
        )
        assert status == 200
        assert body["delivery_mode"] == "model"
        status, health = call(server, token=None, path="/healthz", method="GET")
        assert health["auth"] == "none"
    finally:
        stop(server, thread)


class GatedResponder:
    """Blocks inside generate until released, to fill the admission limit deterministically."""

    kind = "gated"
    io_bound = True

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def describe(self):
        return {"kind": self.kind}

    def generate(self, messages, *, plan, max_tokens, timeout_s):
        self.entered.set()
        self.release.wait(timeout=5)
        return ResponderOutput(text="Here is a hint: List the two outcomes. What is next?")

    def repair(self, messages, draft, *, plan, limits, max_tokens, timeout_s):
        return ResponderOutput(text=draft)


def test_admission_limit_returns_rate_limited():
    responder = GatedResponder()
    pipeline = ResponsePipeline(responder, PipelineSettings(artifact_id="gated_v1"))
    service = ResponseService(pipeline, token=None, max_concurrent=1)
    config = load_config(STUB_CONFIG, overrides={"port": 0})
    server, thread = serve(service, config)
    plan = learning_plans()["teaching_plan_hint"]
    results = {}

    def first():
        results["first"] = call(
            server, make_request(plan, "http-slot-1", artifact="gated_v1"), token=None
        )

    worker = threading.Thread(target=first, daemon=True)
    try:
        worker.start()
        assert responder.entered.wait(timeout=5)
        status, body = call(
            server, make_request(plan, "http-slot-2", artifact="gated_v1"), token=None
        )
        assert status == 429
        assert_error(body, "RATE_LIMITED", "http-slot-2")
        assert body["retryable"] is True
    finally:
        responder.release.set()
        worker.join(timeout=10)
        stop(server, thread)
    assert results["first"][0] == 200
    assert results["first"][1]["delivery_mode"] == "model"


def test_server_starts_and_answers_quickly():
    started = time.monotonic()
    config = load_config(STUB_CONFIG, overrides={"port": 0})
    server, thread = serve(build_service(config, environ={}), config)
    try:
        status, _ = call(server, token=None, path="/healthz", method="GET")
    finally:
        stop(server, thread)
    assert status == 200
    assert time.monotonic() - started < 5
