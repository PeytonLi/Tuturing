import json

import pytest
import yaml

from apps.response_service.config import (
    ConfigError,
    build_pipeline,
    load_config,
    resolve_responder,
    resolve_token,
)
from apps.response_service.server import build_service, describe_service
from packages.response.responders import OpenAICompatibleResponder, TemplateStubResponder
from tests.unit.response.pipeline_helpers import ROOT
from tools.response import serve

CONFIG_DIR = ROOT / "configs" / "response"
STUB_CONFIG = CONFIG_DIR / "serve_stub.yaml"
MODEL_CONFIG = CONFIG_DIR / "serve_openai_compatible.yaml"
MODEL_ENV = {
    "TUTOR_LLM_BASE_URL": "http://127.0.0.1:1/v1",
    "TUTOR_LLM_MODEL": "test-model-name",
    "TUTOR_LLM_API_KEY": "secret-key-value",
    "TUTOR_RESPONSE_SERVICE_TOKEN": "secret-token-value",
}


def write_config(tmp_path, mutate):
    data = yaml.safe_load(STUB_CONFIG.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_stub_config_builds_the_template_pipeline_without_credentials():
    config = load_config(STUB_CONFIG)
    assert config.artifact_id == "response_stub_pipeline_v1"
    assert config.compatible_artifact_ids == ("response_stub_v1",)
    assert config.max_request_bytes == 262_144
    assert config.internal_deadline_ms == 8000
    assert config.max_output_tokens == 384
    pipeline = build_pipeline(config, environ={})
    assert isinstance(pipeline.responder, TemplateStubResponder)
    assert pipeline.settings.compatible_artifact_ids == (
        "response_stub_pipeline_v1", "response_stub_v1",
    )
    assert resolve_token(config, environ={}) is None
    assert resolve_token(config, environ={"TUTOR_RESPONSE_SERVICE_TOKEN": "t"}) == "t"
    public = load_config(STUB_CONFIG, overrides={"host": "0.0.0.0"})
    with pytest.raises(ConfigError, match="TUTOR_RESPONSE_SERVICE_TOKEN"):
        resolve_token(public, environ={})


def test_openai_config_names_environment_variables_only():
    text = MODEL_CONFIG.read_text(encoding="utf-8")
    for value in MODEL_ENV.values():
        assert value not in text
    assert "api_key:" not in text
    config = load_config(MODEL_CONFIG)
    assert config.responder["kind"] == "openai_compatible"
    assert config.responder["base_url_env"] == "TUTOR_LLM_BASE_URL"
    assert "base_url" not in config.responder
    with pytest.raises(ConfigError, match="TUTOR_LLM_BASE_URL"):
        build_pipeline(config, environ={})
    with pytest.raises(ConfigError, match="TUTOR_LLM_MODEL"):
        build_pipeline(config, environ={"TUTOR_LLM_BASE_URL": "http://127.0.0.1:1/v1"})
    with pytest.raises(ConfigError, match="TUTOR_RESPONSE_SERVICE_TOKEN"):
        build_service(config, environ={k: v for k, v in MODEL_ENV.items() if "TOKEN" not in k})
    service = build_service(config, environ=MODEL_ENV)
    responder = service.pipeline.responder
    assert isinstance(responder, OpenAICompatibleResponder)
    assert responder.base_url == "http://127.0.0.1:1/v1"
    assert responder.model == "test-model-name"
    assert service.authenticated
    summary = json.dumps(describe_service(config, service))
    assert "secret-key-value" not in summary
    assert "secret-token-value" not in summary
    assert "TUTOR_LLM_API_KEY" in summary


def test_optional_api_key_is_omitted_for_a_local_server():
    spec = {
        "kind": "openai_compatible",
        "base_url_env": "B",
        "model": "default-model",
        "model_env": "M",
        "api_key_env": "K",
        "api_key_required": False,
    }
    resolved = resolve_responder(spec, {"B": "http://127.0.0.1:1/v1"})
    assert resolved == {
        "kind": "openai_compatible",
        "base_url": "http://127.0.0.1:1/v1",
        "model": "default-model",
        "api_key_env": None,
    }
    resolved = resolve_responder(spec, {"B": "http://127.0.0.1:1/v1", "M": "m", "K": "k"})
    assert resolved["model"] == "m"
    assert resolved["api_key_env"] == "K"
    with pytest.raises(ConfigError, match="B"):
        resolve_responder(dict(spec, api_key_required=True), {"M": "m"})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d["service"].update(max_request_bytes=1_000_000), "max_request_bytes"),
        (lambda d: d["generation"].update(internal_deadline_ms=9000), "internal_deadline_ms"),
        (lambda d: d["generation"].update(max_output_tokens=512), "max_output_tokens"),
        (lambda d: d["generation"].update(max_repairs=2), "max_repairs"),
        (lambda d: d["responder"].update(kind="sampling"), "responder.kind"),
        (lambda d: d["responder"].update(api_key="inline"), "unknown keys"),
        (lambda d: d["service"]["auth"].update(token_env="not a name"), "token_env"),
        (lambda d: d["artifact"].update(id=""), "artifact.id"),
        (lambda d: d["versions"].update(prompt="prompt_v9"), "prompt_v9"),
        (lambda d: d.update(responder={"kind": "template_stub", "model": "PLACEHOLDER"}),
         "placeholder"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutate, message):
    path = write_config(tmp_path, mutate)
    with pytest.raises(ConfigError, match=message):
        build_pipeline(load_config(path), environ={})


def test_serve_check_prints_a_summary_without_binding(capsys, monkeypatch):
    monkeypatch.delenv("TUTOR_RESPONSE_SERVICE_TOKEN", raising=False)
    assert serve.main(["--config", str(STUB_CONFIG), "--check", "--port", "0"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["artifact_id"] == "response_stub_pipeline_v1"
    assert summary["auth"] == "none (loopback only)"
    assert summary["port"] == 0
    assert summary["responder"] == {"kind": "template_stub", "version": "template_stub_v1"}
    assert summary["limits"]["decoding"] == "greedy"
    assert serve.main(["--config", str(MODEL_CONFIG), "--check"]) == 2
    assert "TUTOR_LLM_BASE_URL" in capsys.readouterr().err
