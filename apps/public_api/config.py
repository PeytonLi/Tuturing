"""Configuration of the public API process.

Configs name environment variables for secrets; they never contain token values. Paths are
relative to the repository root unless absolute, and the request store must live outside Git
(`work/` is ignored).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from apps.public_api.contract import ROOT


@dataclass(frozen=True)
class Limits:
    max_request_bytes: int = 65536
    max_message_chars: int = 8000
    request_deadline_ms: int = 10000
    generation_deadline_ms: int = 8000


@dataclass(frozen=True)
class LearningConfig:
    mode: str = "http"  # http | standin
    base_url: str = "http://127.0.0.1:8081"
    service_token_env: str = "TUTOR_LEARNING_SERVICE_TOKEN"
    timeout_s: float = 3.0
    standin_plan_ttl_s: float = 600.0


@dataclass(frozen=True)
class ResponseConfig:
    mode: str = "in_process"  # in_process | http
    base_url: str = "http://127.0.0.1:8082"
    service_token_env: str = "TUTOR_RESPONSE_SERVICE_TOKEN"
    response_artifact_id: str = "stub_v1"
    timeout_s: float = 8.0


@dataclass(frozen=True)
class PublicApiConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    tokens_env: str = "TUTOR_PUBLIC_API_TOKENS"
    storage_path: Path = field(default_factory=lambda: ROOT / "work/public_api/requests.sqlite")
    learning: LearningConfig = field(default_factory=LearningConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    limits: Limits = field(default_factory=Limits)

    @classmethod
    def load(cls, path: str | Path) -> PublicApiConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        server = raw.get("server", {})
        auth = raw.get("auth", {})
        storage = raw.get("storage", {})
        storage_path = Path(storage.get("path", "work/public_api/requests.sqlite"))
        if not storage_path.is_absolute():
            storage_path = ROOT / storage_path
        return cls(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8080)),
            tokens_env=str(auth.get("tokens_env", "TUTOR_PUBLIC_API_TOKENS")),
            storage_path=storage_path,
            learning=LearningConfig(**raw.get("learning", {})),
            response=ResponseConfig(**raw.get("response", {})),
            limits=Limits(**raw.get("limits", {})),
        )


def load_callers(tokens_env: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Map bearer token to caller name from `caller=token` entries separated by commas.

    A bare token is accepted too; its caller name derives from a digest of the token so the
    token value never appears in logs or records.
    """
    environ = os.environ if environ is None else environ
    callers: dict[str, str] = {}
    for entry in (environ.get(tokens_env) or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        name, separator, token = entry.partition("=")
        if separator:
            callers[token.strip()] = name.strip()
        else:
            callers[entry] = "caller-" + hashlib.sha256(entry.encode("utf-8")).hexdigest()[:12]
    return callers
