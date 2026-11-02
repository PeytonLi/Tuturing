"""Clients of the internal services (PRD 9.5).

LearningClient talks to Person B's learning service (`apps/learning_service`, documented in
docs/learning_service.md); ResponseClient talks to Person A's `POST /internal/v1/generate`, with
an in-process option that calls `packages.response.stub` so the whole public API runs without any
server. Both HTTP clients use only the standard library.

B's transport rules, mirrored here: every call carries `Authorization: Bearer <token>` and
`X-Trace-Id`; every POST carries `Content-Type: application/json`, `Content-Length` and an
`Idempotency-Key` equal to the body's `request_id` (commit-plan has no `request_id`, so its key
is the public request's ID); success is HTTP 200; errors are PRD Error records.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from apps.public_api.contract import PRD_ERROR_STATUS, load_contract, parse_json_strict
from apps.public_api.errors import ApiError
from packages.response.hashing import content_hash
from packages.response.stub import ResponseValidationError, generate

INTERNAL_ERROR_PREFIX = "internal error from"


class ServiceUnavailable(Exception):
    """The service could not be reached or answered with something unusable."""


class GenerationFailure(ServiceUnavailable):
    """The response service failed to produce a validated response."""


class LearningClient(Protocol):
    """B's transaction contract as the public API uses it. `trace_id` travels as a header."""

    def create_session(self, spec: dict, *, trace_id: str) -> dict: ...

    def observe_and_plan(self, request: dict, *, trace_id: str) -> dict: ...

    def commit_plan(self, request: dict, *, trace_id: str, idempotency_key: str) -> dict: ...

    def plan_status(self, plan_id: str, *, trace_id: str) -> dict: ...

    def cancel_plan(self, request: dict, *, trace_id: str) -> dict: ...


class ResponseClient(Protocol):
    def generate(self, request: dict) -> dict: ...


def _service_headers(
    token_env: str, trace_id: str, idempotency_key: str | None = None
) -> dict[str, str]:
    headers = {"X-Trace-Id": trace_id, "Accept": "application/json"}
    token = os.environ.get(token_env)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key is not None:
        headers["Content-Type"] = "application/json"
        headers["Idempotency-Key"] = idempotency_key
    return headers


def http_json(
    method: str,
    url: str,
    body: dict | None,
    headers: dict[str, str],
    timeout_s: float,
    *,
    max_bytes: int = 1 << 20,
) -> tuple[int, Any]:
    """One JSON exchange. Transport failures raise ServiceUnavailable."""
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return response.status, _decode(response.read(max_bytes + 1), max_bytes)
    except urllib.error.HTTPError as error:
        try:
            payload = _decode(error.read(max_bytes + 1), max_bytes)
        except ServiceUnavailable:
            payload = None
        return error.code, payload
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        raise ServiceUnavailable(f"{method} {url}: {type(error).__name__}") from error


def _decode(raw: bytes, max_bytes: int) -> Any:
    if len(raw) > max_bytes:
        raise ServiceUnavailable("response body too large")
    try:
        return parse_json_strict(raw.decode("utf-8")) if raw else None
    except (ValueError, UnicodeDecodeError) as error:
        raise ServiceUnavailable("response body is not JSON") from error


def raise_for_error(status: int, payload: Any, service: str) -> None:
    """Translate an internal Error record into the public ApiError it implies."""
    if 200 <= status < 300:
        return
    if isinstance(payload, dict) and payload.get("code") in PRD_ERROR_STATUS:
        code = payload["code"]
        version = payload.get("current_state_version")
        version = version if isinstance(version, int) and not isinstance(version, bool) else None
        retryable = bool(payload.get("retryable", False))
        if code in {"STATE_CONFLICT", "PLAN_EXPIRED", "SESSION_ENDED"}:
            raise ApiError(
                code,
                _public_message(code),
                retryable=False,
                current_state_version=version,
                terminal=True,
            )
        if code in {"RATE_LIMITED", "DEADLINE_EXCEEDED", "LEARNING_UNAVAILABLE"}:
            raise ApiError(code, _public_message(code), retryable=True, retry_after_s=1)
        if code == "GENERATION_UNAVAILABLE":
            raise GenerationFailure(f"{service}: GENERATION_UNAVAILABLE")
        if code in {"INVALID_SCHEMA", "UNSUPPORTED_VERSION"}:
            raise ApiError(
                code,
                f"The learning service rejected the request ({code}); change the request.",
                retryable=False,
                current_state_version=version,
                terminal=True,
            )
        raise ServiceUnavailable(f"{service}: {code} ({'retryable' if retryable else 'final'})")
    raise ServiceUnavailable(f"{INTERNAL_ERROR_PREFIX} {service}: HTTP {status}")


def _public_message(code: str) -> str:
    return {
        "STATE_CONFLICT": "The expected state version or source response is stale; refresh the"
        " session and send a new request.",
        "PLAN_EXPIRED": "The reserved teaching plan expired before delivery; refresh the session"
        " and send a new turn.",
        "SESSION_ENDED": "The session has ended.",
        "RATE_LIMITED": "The learning service is rate limited; retry the same request later.",
        "DEADLINE_EXCEEDED": "The learning service deadline elapsed; retry the same request.",
        "LEARNING_UNAVAILABLE": "The learning service is unavailable; retry the same request.",
    }[code]


class HttpLearningClient:
    """Client of B's `/internal/v1` endpoints. Service auth comes from an env var."""

    def __init__(self, base_url: str, *, service_token_env: str, timeout_s: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token_env = service_token_env
        self.timeout_s = timeout_s

    def _call(
        self, method: str, path: str, body: dict | None, trace_id: str, key: str | None = None
    ) -> dict:
        status, payload = http_json(
            method,
            f"{self.base_url}{path}",
            body,
            _service_headers(self.service_token_env, trace_id, key),
            self.timeout_s,
        )
        raise_for_error(status, payload, "learning service")
        if not isinstance(payload, dict):
            raise ServiceUnavailable("learning service returned a non-object body")
        return payload

    def create_session(self, spec: dict, *, trace_id: str) -> dict:
        return self._call("POST", "/internal/v1/sessions", spec, trace_id, spec["request_id"])

    def observe_and_plan(self, request: dict, *, trace_id: str) -> dict:
        return self._call(
            "POST", "/internal/v1/observe-and-plan", request, trace_id, request["request_id"]
        )

    def commit_plan(self, request: dict, *, trace_id: str, idempotency_key: str) -> dict:
        return self._call("POST", "/internal/v1/commit-plan", request, trace_id, idempotency_key)

    def plan_status(self, plan_id: str, *, trace_id: str) -> dict:
        return self._call("GET", f"/internal/v1/plans/{plan_id}", None, trace_id)

    def cancel_plan(self, request: dict, *, trace_id: str) -> dict:
        return self._call(
            "POST", "/internal/v1/cancel-plan", request, trace_id, request["request_id"]
        )


def check_generate_response(request: dict, response: Any) -> dict:
    """Accept only a GenerateResponse that matches the contract, the request and its own hash."""
    contract = load_contract()
    issues = contract.response_issues("GenerateResponse", response)
    if issues:
        raise GenerationFailure("GenerateResponse violates response_v1: " + "; ".join(issues))
    if response["request_id"] != request["request_id"]:
        raise GenerationFailure("GenerateResponse request_id mismatch")
    if response["plan_id"] != request["plan"]["plan_id"]:
        raise GenerationFailure("GenerateResponse plan_id mismatch")
    if not all(response["checks"].values()):
        raise GenerationFailure("GenerateResponse has failed checks")
    if content_hash(response["text"], response["effective_move"]) != response["content_hash"]:
        raise GenerationFailure("GenerateResponse content_hash does not match its text")
    return response


class InProcessResponseClient:
    """Calls Person A's CPU response stub directly; no server needed."""

    def generate(self, request: dict) -> dict:
        try:
            response = generate(request)  # type: ignore[arg-type]
        except ResponseValidationError as error:
            raise GenerationFailure(f"stub rejected the plan: {error}") from error
        return check_generate_response(request, response)


class HttpResponseClient:
    """Client of `POST /internal/v1/generate` (PRD 9.5) on the response service."""

    def __init__(self, base_url: str, *, service_token_env: str, timeout_s: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_token_env = service_token_env
        self.timeout_s = timeout_s

    def generate(self, request: dict) -> dict:
        try:
            status, payload = http_json(
                "POST",
                f"{self.base_url}/internal/v1/generate",
                request,
                _service_headers(
                    self.service_token_env, request["trace_id"], request["request_id"]
                ),
                self.timeout_s,
            )
            raise_for_error(status, payload, "response service")
        except ServiceUnavailable as error:
            raise GenerationFailure(str(error)) from error
        return check_generate_response(request, payload)
