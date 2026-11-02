"""Error codes and Error objects for the response service (PRD 9.5).

The Error record itself is defined in contracts/learning_v1.schema.json; this module only
builds instances of it. Messages must never contain answer keys, plan content, prompt
internals or learner text.
"""

from __future__ import annotations

import re

ERROR_SCHEMA_VERSION = "1.0"
ERROR_STATUS = {
    "INVALID_SCHEMA": 422,
    "UNSUPPORTED_VERSION": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "STATE_CONFLICT": 409,
    "PLAN_EXPIRED": 409,
    "SESSION_ENDED": 409,
    "RATE_LIMITED": 429,
    "GENERATION_UNAVAILABLE": 503,
    "LEARNING_UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
}
UNKNOWN_ID = "unknown"
_ID = re.compile(r"[\x00-\x7f]{1,128}")


def is_id(value: object) -> bool:
    """True for a PRD 9.1 ID: a nonempty ASCII string of at most 128 characters."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def safe_id(value: object, default: str = UNKNOWN_ID) -> str:
    return value if is_id(value) else default


class GenerationError(Exception):
    """A request the pipeline will not answer with a GenerateResponse.

    ``attempts`` carries redactable diagnostics for logs; it never reaches the client.
    """

    def __init__(self, code: str, message: str, *, retryable: bool, attempts=()):
        if code not in ERROR_STATUS:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
        self.attempts = tuple(attempts)

    @property
    def http_status(self) -> int:
        return ERROR_STATUS[self.code]


def error_object(
    code: str,
    message: str,
    *,
    request_id: object,
    trace_id: object,
    retryable: bool,
    current_state_version: int | None = None,
) -> dict:
    if code not in ERROR_STATUS:
        raise ValueError(f"unknown error code: {code}")
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "request_id": safe_id(request_id),
        "trace_id": safe_id(trace_id),
        "code": code,
        "message": message,
        "retryable": bool(retryable),
        "current_state_version": current_state_version,
    }
