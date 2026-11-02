"""Load and validate a response service config (configs/response/serve_*.yaml).

A config names environment variables for every secret and never contains one. Unknown keys,
out-of-range values, unresolved placeholders (PRD 22.4) and version pins that disagree with
the code are rejected before anything is served.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from packages.response.errors import is_id
from packages.response.fallbacks import FALLBACK_VERSION
from packages.response.pipeline import (
    INTERNAL_DEADLINE_CEILING_MS,
    PipelineSettings,
    ResponsePipeline,
)
from packages.response.prompt import PROMPT_VERSION
from packages.response.responders import MAX_OUTPUT_TOKENS, RESPONDER_KINDS, build_responder
from packages.response.validate import VALIDATOR_VERSION

CONFIG_SCHEMA_VERSION = 1
DEFAULT_MAX_REQUEST_BYTES = 262_144
CODE_VERSIONS = {
    "prompt": PROMPT_VERSION,
    "validator": VALIDATOR_VERSION,
    "fallback": FALLBACK_VERSION,
}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLACEHOLDER = re.compile(r"(?i)\bplaceholder\b|\bchangeme\b|\$\{")
_TOP_KEYS = {"schema_version", "service", "artifact", "responder", "generation", "versions"}
_SERVICE_KEYS = {
    "host", "port", "auth", "max_request_bytes", "max_concurrent", "idempotency_cache_size",
}
_AUTH_KEYS = {"token_env", "allow_unauthenticated_loopback"}
_ARTIFACT_KEYS = {"id", "compatible_ids"}
_GENERATION_KEYS = {"internal_deadline_ms", "max_output_tokens", "max_repairs"}
_RESPONDER_ENV_KEYS = {"base_url_env": "base_url", "model_env": "model"}
_RESPONDER_EXTRA_KEYS = set(_RESPONDER_ENV_KEYS) | {"api_key_required"}


class ConfigError(ValueError):
    """The config cannot be used; the message names keys and variables, never values."""


@dataclass(frozen=True)
class ServiceConfig:
    path: str
    host: str
    port: int
    token_env: str
    allow_unauthenticated_loopback: bool
    max_request_bytes: int
    max_concurrent: int
    idempotency_cache_size: int
    artifact_id: str
    compatible_artifact_ids: tuple[str, ...]
    responder: dict
    internal_deadline_ms: int
    max_output_tokens: int
    max_repairs: int
    expected_versions: dict


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _mapping(value, name: str, allowed: set[str]) -> dict:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{name} has unknown keys: " + ", ".join(unknown))
    return value


def _int(value, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}")
    return value


def _bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _env_name(value, name: str) -> str:
    if not isinstance(value, str) or not _ENV_NAME.fullmatch(value):
        raise ConfigError(f"{name} must be an environment variable name")
    return value


def _placeholders(value, path: str = "config") -> list[str]:
    """Paths of string values that look unresolved (PRD 22.4); comments are not values."""
    found = []
    if isinstance(value, str):
        if _PLACEHOLDER.search(value):
            found.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_placeholders(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_placeholders(item, f"{path}[{index}]"))
    return found


def load_config(path, *, overrides: dict | None = None) -> ServiceConfig:
    """Parse and validate a service config; ``overrides`` may replace host and port."""
    path = Path(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    unresolved = _placeholders(loaded)
    if unresolved:
        raise ConfigError(
            "config contains unresolved placeholder values at: " + ", ".join(unresolved)
        )
    data = _mapping(loaded, "config", _TOP_KEYS)
    if data.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - {"host", "port"})
    if unknown:
        raise ConfigError("unsupported overrides: " + ", ".join(unknown))

    service = _mapping(data.get("service"), "service", _SERVICE_KEYS)
    auth = _mapping(service.get("auth"), "service.auth", _AUTH_KEYS)
    host = overrides.get("host", service.get("host", "127.0.0.1"))
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("service.host must be a nonempty string")
    artifact = _mapping(data.get("artifact"), "artifact", _ARTIFACT_KEYS)
    artifact_id = artifact.get("id")
    if not is_id(artifact_id):
        raise ConfigError("artifact.id must be a nonempty ASCII ID of at most 128 characters")
    compatible = artifact.get("compatible_ids") or []
    if not isinstance(compatible, list) or not all(is_id(item) for item in compatible):
        raise ConfigError("artifact.compatible_ids must be a list of ASCII IDs")

    responder = _mapping(
        data.get("responder"), "responder", set(RESPONDER_KINDS) | _RESPONDER_EXTRA_KEYS | {
            "kind", "base_url", "model", "api_key_env", "request_timeout_s", "extra_body",
        },
    )
    if responder.get("kind") not in RESPONDER_KINDS:
        raise ConfigError("responder.kind must be one of " + ", ".join(RESPONDER_KINDS))
    for key in _RESPONDER_ENV_KEYS:
        if key in responder:
            _env_name(responder[key], f"responder.{key}")
    if "api_key_env" in responder and responder["api_key_env"] is not None:
        _env_name(responder["api_key_env"], "responder.api_key_env")
    if "api_key_required" in responder:
        _bool(responder["api_key_required"], "responder.api_key_required")
    for key in ("base_url", "model"):
        if key in responder and not isinstance(responder[key], str):
            raise ConfigError(f"responder.{key} must be a string")

    generation = _mapping(data.get("generation"), "generation", _GENERATION_KEYS)
    versions = _mapping(data.get("versions"), "versions", set(CODE_VERSIONS))
    for key, value in versions.items():
        if not isinstance(value, str):
            raise ConfigError(f"versions.{key} must be a string")

    return ServiceConfig(
        path=str(path),
        host=host,
        port=_int(overrides.get("port", service.get("port", 8081)), "service.port", 0, 65535),
        token_env=_env_name(
            auth.get("token_env", "TUTOR_RESPONSE_SERVICE_TOKEN"), "service.auth.token_env"
        ),
        allow_unauthenticated_loopback=_bool(
            auth.get("allow_unauthenticated_loopback", False),
            "service.auth.allow_unauthenticated_loopback",
        ),
        max_request_bytes=_int(
            service.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES),
            "service.max_request_bytes",
            1024,
            DEFAULT_MAX_REQUEST_BYTES,
        ),
        max_concurrent=_int(service.get("max_concurrent", 4), "service.max_concurrent", 1, 64),
        idempotency_cache_size=_int(
            service.get("idempotency_cache_size", 1024),
            "service.idempotency_cache_size",
            0,
            100_000,
        ),
        artifact_id=artifact_id,
        compatible_artifact_ids=tuple(compatible),
        responder=dict(responder),
        internal_deadline_ms=_int(
            generation.get("internal_deadline_ms", INTERNAL_DEADLINE_CEILING_MS),
            "generation.internal_deadline_ms",
            1,
            INTERNAL_DEADLINE_CEILING_MS,
        ),
        max_output_tokens=_int(
            generation.get("max_output_tokens", MAX_OUTPUT_TOKENS),
            "generation.max_output_tokens",
            1,
            MAX_OUTPUT_TOKENS,
        ),
        max_repairs=_int(generation.get("max_repairs", 1), "generation.max_repairs", 0, 1),
        expected_versions=dict(versions),
    )


def check_versions(config: ServiceConfig) -> None:
    """Refuse to serve when a pinned prompt, validator or fallback version is not the code's."""
    for key, expected in config.expected_versions.items():
        if CODE_VERSIONS[key] != expected:
            raise ConfigError(
                f"versions.{key} pins {expected} but the code provides {CODE_VERSIONS[key]}"
            )


def resolve_responder(spec: dict, environ) -> dict:
    """Replace ``*_env`` names with values from ``environ``; values never enter the config."""
    resolved = {key: value for key, value in spec.items() if key not in _RESPONDER_EXTRA_KEYS}
    for env_key, target in _RESPONDER_ENV_KEYS.items():
        name = spec.get(env_key)
        if name is None:
            continue
        value = environ.get(name)
        if value:
            resolved[target] = value
        elif not resolved.get(target):
            raise ConfigError(
                f"environment variable {name} is not set and responder.{target} has no default"
            )
    key_env = spec.get("api_key_env")
    if key_env and spec.get("api_key_required") is False and not environ.get(key_env):
        resolved["api_key_env"] = None
    return resolved


def build_pipeline(config: ServiceConfig, environ=None) -> ResponsePipeline:
    environ = os.environ if environ is None else environ
    check_versions(config)
    responder = build_responder(resolve_responder(config.responder, environ), environ=environ)
    settings = PipelineSettings(
        artifact_id=config.artifact_id,
        compatible_artifact_ids=config.compatible_artifact_ids,
        internal_deadline_ms=config.internal_deadline_ms,
        max_output_tokens=config.max_output_tokens,
        max_repairs=config.max_repairs,
    )
    return ResponsePipeline(responder, settings)


def resolve_token(config: ServiceConfig, environ=None) -> str | None:
    """The service bearer token, or None when unauthenticated loopback serving is allowed."""
    environ = os.environ if environ is None else environ
    token = environ.get(config.token_env)
    if token:
        return token
    if config.allow_unauthenticated_loopback and is_loopback_host(config.host):
        return None
    raise ConfigError(
        f"environment variable {config.token_env} is not set; it must hold the service "
        "bearer token"
    )
