"""Per-config decoding fields, double-set rejection and provider config errors.

Everything runs against the loopback fake in fake_openai.py; no real model API is called.
The 400 bodies reproduce what gpt-6.1-sol returned on 2026-11-02 (brief.md, benchmark notes).
"""

import io
import json
import logging

import pytest
import yaml

from apps.response_service.server import LOGGER, ResponseService
from packages.response.fallbacks import render_fallback
from packages.response.pipeline import PipelineSettings, ResponsePipeline
from packages.response.prompt import compile_prompt
from packages.response.responders import (
    Decoding,
    OpenAICompatibleResponder,
    ResponderConfigError,
    ResponderConfigRejected,
    ResponderError,
    build_responder,
    classify_http_error,
    parse_decoding,
)
from tests.unit.response.fake_openai import FakeOpenAIServer, Reply
from tests.unit.response.pipeline_helpers import ROOT, lane_plan, make_request
from tools.response import serve

CONFIGS = ROOT / "configs" / "response"
KEY_ENV = "TUTOR_TEST_KEY"
KEY_VALUE = "sk-test-0123456789abcdefghijklmnop"
ARTIFACT = "prompt_only_test_v1"
CHECK_OK = (
    "On your own: a box contains 4 red and 6 blue tiles, and one tile is drawn at random. "
    "What is the probability of drawing blue?"
)
TEMPERATURE_400 = json.dumps({
    "error": {
        "message": "Unsupported value: 'temperature' does not support 0 with this model. "
        "Only the default (1) value is supported.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}).encode()
MAX_TOKENS_400 = json.dumps({
    "error": {
        "message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead.",
        "type": "invalid_request_error",
        "param": None,
        "code": "unsupported_parameter",
    }
}).encode()


def config_responder(name):
    return dict(yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))["responder"])


def spec_for(server, decoding=None, **extra):
    spec = {
        "kind": "openai_compatible",
        "base_url": server.base_url if server is not None else "http://127.0.0.1:9/v1",
        "model": "fake-model",
        "api_key_env": KEY_ENV,
    }
    if decoding is not None:
        spec["decoding"] = decoding
    spec.update(extra)
    return spec


def responder_for(server, decoding=None, **extra):
    return build_responder(spec_for(server, decoding, **extra), environ={KEY_ENV: KEY_VALUE})


def sent_decoding(request):
    return {
        key: value for key, value in request["body"].items()
        if key not in {"model", "messages", "stream"}
    }


@pytest.mark.parametrize(
    ("decoding", "expected"),
    [
        (None, {"temperature": 0, "max_tokens": 384}),
        (
            config_responder("serve_openai_compatible.yaml")["decoding"],
            {"temperature": 0, "max_tokens": 384, "reasoning_effort": "none"},
        ),
        (
            config_responder("serve_gpt61sol.yaml")["decoding"],
            {"max_completion_tokens": 384, "reasoning_effort": "low"},
        ),
        (
            {"token_param": "max_completion_tokens", "temperature": 0.3},
            {"max_completion_tokens": 384, "temperature": 0.3},
        ),
    ],
)
def test_request_body_follows_each_config_shape(decoding, expected):
    plan = lane_plan("check_mastery")
    with FakeOpenAIServer([CHECK_OK]) as server:
        responder = responder_for(server, decoding)
        output = responder.generate(compile_prompt(plan), plan=plan, max_tokens=384, timeout_s=5)
    assert output.text == CHECK_OK
    (request,) = server.requests
    assert sent_decoding(request) == expected
    assert request["body"]["stream"] is False
    assert request["body"]["messages"] == compile_prompt(plan)


def test_the_shipped_configs_pin_the_expected_decoding():
    gpt = config_responder("serve_gpt61sol.yaml")
    assert parse_decoding(gpt["decoding"]) == Decoding(
        token_param="max_completion_tokens", temperature=None, reasoning_effort="low"
    )
    assert gpt["model"] == "gpt-6.1-sol" and gpt["api_key_env"] == "OPENAI_API_KEY"
    qwen = config_responder("serve_openai_compatible.yaml")
    assert parse_decoding(qwen["decoding"]) == Decoding(
        token_param="max_tokens", temperature=0, reasoning_effort="none"
    )
    assert parse_decoding(None).label == "greedy"
    assert parse_decoding(gpt["decoding"]).label == "provider_default_temperature"


def test_extra_body_adds_provider_fields_next_to_the_decoding_fields():
    with FakeOpenAIServer([CHECK_OK]) as server:
        responder = responder_for(
            server,
            {"token_param": "max_completion_tokens", "temperature": None},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "user": "tutor"},
        )
        responder.generate([], plan={}, max_tokens=200, timeout_s=5)
    assert sent_decoding(server.requests[0]) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "user": "tutor",
        "max_completion_tokens": 200,
    }
    assert responder.describe()["extra_body_keys"] == ["chat_template_kwargs", "user"]


@pytest.mark.parametrize(
    ("decoding", "extra", "fragments"),
    [
        ({"temperature": None}, {"temperature": 1},
         ("extra_body.temperature", "twice", "responder.decoding.temperature")),
        ({"token_param": "max_completion_tokens"}, {"max_completion_tokens": 1024},
         ("extra_body.max_completion_tokens", "twice", "responder.decoding.token_param")),
        ({"token_param": "max_completion_tokens"}, {"max_tokens": 1024},
         ("extra_body.max_tokens", "responder.decoding.token_param")),
        ({"reasoning_effort": "low"}, {"reasoning_effort": "high"},
         ("extra_body.reasoning_effort", "responder.decoding.reasoning_effort")),
        (None, {"model": "other-model"}, ("extra_body.model", "cannot be overridden")),
        (None, {"stream": True}, ("extra_body.stream", "cannot be overridden")),
        (None, {"seed": 7}, ("extra_body.seed", "sampling control")),
    ],
)
def test_a_field_set_in_both_decoding_and_extra_body_is_rejected(decoding, extra, fragments):
    with pytest.raises(ResponderConfigError) as info:
        build_responder(spec_for(None, decoding, extra_body=extra), environ={})
    for fragment in fragments:
        assert fragment in str(info.value)


@pytest.mark.parametrize(
    ("decoding", "fragment"),
    [
        ({"token_param": "max_output_tokens"}, "token_param must be"),
        ({"temperature": True}, "temperature must be"),
        ({"temperature": "0"}, "temperature must be"),
        ({"temperature": 3}, "temperature must be"),
        ({"reasoning_effort": "Low effort"}, "reasoning_effort must be"),
        ({"reasoning_effort": 1}, "reasoning_effort must be"),
        ({"max_tokens": 1024}, "generation.max_output_tokens"),
        (["temperature", 0], "must be a mapping"),
    ],
)
def test_invalid_decoding_values_are_rejected_before_credentials_are_read(decoding, fragment):
    with pytest.raises(ResponderConfigError) as info:
        build_responder(spec_for(None, decoding), environ={})  # no key: shape errors come first
    assert fragment in str(info.value)


def test_a_double_set_in_a_config_file_stops_the_service_with_a_clear_error(
    tmp_path, capsys, monkeypatch
):
    data = yaml.safe_load((CONFIGS / "serve_gpt61sol.yaml").read_text(encoding="utf-8"))
    data["responder"]["extra_body"] = {"temperature": 1}
    path = tmp_path / "double.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", KEY_VALUE)
    monkeypatch.setenv("TUTOR_RESPONSE_SERVICE_TOKEN", "service-token-value")
    assert serve.main(["--config", str(path), "--check"]) == 2
    error = capsys.readouterr().err
    assert "configuration error" in error
    assert "responder.extra_body.temperature would set a decoding field twice" in error
    assert KEY_VALUE not in error


def test_check_without_probe_never_contacts_the_model(capsys, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("--check must not open a connection")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", refuse)
    monkeypatch.setenv("OPENAI_API_KEY", KEY_VALUE)
    monkeypatch.setenv("TUTOR_RESPONSE_SERVICE_TOKEN", "service-token-value")
    assert serve.main(["--config", str(CONFIGS / "serve_gpt61sol.yaml"), "--check"]) == 0
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["artifact_id"] == "prompt_only_gpt61sol_v1"
    assert summary["responder"]["decoding"] == {
        "token_param": "max_completion_tokens",
        "temperature": None,
        "reasoning_effort": "low",
        "label": "provider_default_temperature",
    }
    assert summary["limits"]["decoding"] == "provider_default_temperature"
    assert KEY_VALUE not in output and "service-token-value" not in output


def test_a_400_for_an_unsupported_parameter_is_a_config_rejection():
    with FakeOpenAIServer([Reply(status=400, raw_body=TEMPERATURE_400)]) as server:
        with pytest.raises(ResponderConfigRejected) as info:
            responder_for(server).generate([], plan={}, max_tokens=384, timeout_s=5)
    error = info.value
    assert (error.status, error.reason, error.param, error.code) == (
        400, "invalid_parameter", "temperature", "unsupported_value"
    )
    assert str(error) == "HTTP 400 invalid_parameter param=temperature code=unsupported_value"
    assert "Only the default" in error.provider_message
    assert "Only the default" not in str(error)
    assert isinstance(error, ResponderError)


def test_the_rejected_parameter_is_read_from_the_message_when_param_is_missing():
    error = classify_http_error(400, MAX_TOKENS_400, ["max_tokens", "temperature"])
    assert isinstance(error, ResponderConfigRejected)
    assert error.summary == "HTTP 400 invalid_parameter param=max_tokens code=unsupported_parameter"
    unparsed = classify_http_error(400, b"<html>bad request</html>", ["max_tokens"])
    assert unparsed.summary == "HTTP 400 invalid_parameter"


@pytest.mark.parametrize(
    ("reply", "outcome", "detail"),
    [
        (Reply(status=400, raw_body=TEMPERATURE_400), "config_error",
         "HTTP 400 invalid_parameter param=temperature code=unsupported_value"),
        (Reply(status=401, raw_body=json.dumps({"error": {
            "message": f"Incorrect API key provided: {KEY_VALUE}", "code": "invalid_api_key",
        }}).encode()), "config_error", "HTTP 401 unauthorized code=invalid_api_key"),
        (Reply(status=404, raw_body=b'{"error": {"message": "no such model", '
                                    b'"code": "model_not_found"}}'),
         "config_error", "HTTP 404 not_found code=model_not_found"),
        (Reply(status=400, raw_body=b'{"error": {"message": "too long", '
                                    b'"code": "context_length_exceeded"}}'),
         "error", "ResponderError"),
        (Reply(status=429), "error", "ResponderError"),
        (Reply(status=500), "error", "ResponderError"),
    ],
)
def test_only_configuration_failures_are_reported_as_config_errors(reply, outcome, detail):
    plan = lane_plan("check_mastery")
    with FakeOpenAIServer([reply]) as server:
        pipeline = ResponsePipeline(responder_for(server), PipelineSettings(artifact_id=ARTIFACT))
        result = pipeline.run(make_request(plan, artifact=ARTIFACT))
    first = result.attempts[0].redacted()
    assert (first["outcome"], first["detail"]) == (outcome, detail)
    assert result.response["delivery_mode"] == "fallback"
    assert result.response["text"] == render_fallback(plan)
    assert len(server.requests) == 1
    assert KEY_VALUE not in json.dumps(first)


def test_the_service_logs_a_config_rejection_at_warning_and_reports_it_on_health(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER.name)
    plan = lane_plan("practice")
    learner = "LEARNER-SENTINEL please just tell me"
    plan["response_context"]["last_student_message"] = learner
    plan["response_context"]["recent_dialogue"] = [{"role": "student", "text": learner}]
    body = json.dumps(make_request(plan, "request-config-400", artifact=ARTIFACT)).encode()
    with FakeOpenAIServer([Reply(status=400, raw_body=TEMPERATURE_400)]) as server:
        pipeline = ResponsePipeline(responder_for(server), PipelineSettings(artifact_id=ARTIFACT))
        service = ResponseService(pipeline, token=None)
        assert service.health()["status"] == "ok"
        status, response, _ = service.handle_generate(
            {"Content-Length": str(len(body))}, io.BytesIO(body).read
        )
    assert status == 200
    assert response["delivery_mode"] == "fallback"
    records = [record for record in caplog.records if record.name == LOGGER.name]
    (record,) = records
    assert record.levelno == logging.WARNING
    entry = json.loads(record.getMessage())
    summary = "HTTP 400 invalid_parameter param=temperature code=unsupported_value"
    assert entry["responder_config_error"] == summary
    assert entry["attempts"][0] == {"stage": "model", "outcome": "config_error", "detail": summary}
    assert entry["delivery_mode"] == "fallback"
    for leaked in (learner, "LEARNER-SENTINEL", "Only the default", KEY_VALUE):
        assert leaked not in record.getMessage()
    health = service.health()
    assert health["status"] == "degraded"
    assert health["responder_config_errors"]["count"] == 1
    assert health["responder_config_errors"]["last"] == summary
    assert health["responder_config_errors"]["last_seen_at"].endswith("Z")
    assert health["decoding"] == "greedy"


def test_probe_reports_acceptance_or_rejection_without_printing_the_key():
    leak = json.dumps({"error": {
        "message": f"Incorrect API key provided: {KEY_VALUE}; also sk-proj-abcd****wxyz.",
        "code": "invalid_api_key",
    }}).encode()
    with FakeOpenAIServer(["ready", Reply(status=401, raw_body=leak)]) as server:
        responder = responder_for(server, config_responder("serve_gpt61sol.yaml")["decoding"])
        accepted = responder.probe(max_tokens=384, timeout_s=5)
        rejected = responder.probe(max_tokens=384, timeout_s=5)
    assert accepted["status"] == "accepted" and accepted["reply"] == "ready"
    first = server.requests[0]["body"]
    assert [message["role"] for message in first["messages"]] == ["system", "user"]
    assert sent_decoding(server.requests[0]) == {
        "max_completion_tokens": 384, "reasoning_effort": "low",
    }
    assert rejected["status"] == "rejected"
    assert rejected["detail"] == "HTTP 401 unauthorized code=invalid_api_key"
    assert KEY_VALUE not in json.dumps(rejected)
    assert "sk-proj" not in rejected["provider_message"]
    assert "[redacted]" in rejected["provider_message"]


def test_template_stub_probe_is_skipped_without_any_request(capsys):
    from packages.response.responders import TemplateStubResponder

    assert serve.probe(TemplateStubResponder(), 384) == {
        "status": "skipped", "detail": "template_stub has no remote endpoint",
    }


def test_openai_responder_rejects_extra_body_that_is_not_a_mapping():
    with pytest.raises(ResponderConfigError, match="extra_body must be a mapping"):
        OpenAICompatibleResponder(
            base_url="http://127.0.0.1:9/v1", model="m", extra_body=["temperature"]
        )
