"""Public API errors (PRD 9.5). The HTTP status of each code comes from the contract table."""

from __future__ import annotations

from apps.public_api.contract import PRD_ERROR_STATUS


class ApiError(Exception):
    """An Error response.

    `retryable` follows the public contract: resending the identical request with the same
    request_id may succeed later. `terminal` marks an outcome that is stored as the request's
    final result, so every retry of that request_id returns the same body.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        current_state_version: int | None = None,
        terminal: bool = False,
        retry_after_s: int | None = None,
    ) -> None:
        if code not in PRD_ERROR_STATUS:
            raise ValueError(f"unknown error code {code}")
        if terminal and retryable:
            raise ValueError("a terminal error cannot be retryable")
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.current_state_version = current_state_version
        self.terminal = terminal
        self.retry_after_s = retry_after_s

    @property
    def status(self) -> int:
        return PRD_ERROR_STATUS[self.code]

    def body(self, request_id: str, trace_id: str) -> dict:
        return {
            "schema_version": "1.0",
            "request_id": request_id,
            "trace_id": trace_id,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "current_state_version": self.current_state_version,
        }


def invalid(message: str) -> ApiError:
    return ApiError("INVALID_SCHEMA", message, retryable=False)


def state_conflict(message: str, version: int | None, *, terminal: bool = True) -> ApiError:
    return ApiError(
        "STATE_CONFLICT",
        message,
        retryable=False,
        current_state_version=version,
        terminal=terminal,
    )


def session_ended(version: int | None) -> ApiError:
    return ApiError(
        "SESSION_ENDED",
        "The session has ended.",
        retryable=False,
        current_state_version=version,
        terminal=True,
    )


def plan_expired(version: int | None) -> ApiError:
    return ApiError(
        "PLAN_EXPIRED",
        "The reserved teaching plan expired before delivery; refresh the session and send a"
        " new turn.",
        retryable=False,
        current_state_version=version,
        terminal=True,
    )


def learning_unavailable(*, retryable: bool = True) -> ApiError:
    return ApiError(
        "LEARNING_UNAVAILABLE",
        "The learning service is unavailable; retry the same request."
        if retryable
        else "The learning service returned an inconsistent result; contact support with the"
        " trace_id.",
        retryable=retryable,
        retry_after_s=1 if retryable else None,
    )


def generation_unavailable() -> ApiError:
    return ApiError(
        "GENERATION_UNAVAILABLE",
        "The tutor could not produce a validated reply; retry the same request.",
        retryable=True,
        retry_after_s=1,
    )
