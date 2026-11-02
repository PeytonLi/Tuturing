"""Authenticated internal HTTP transport. LearningEngine owns all mutations."""

import hmac
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from apps.learning_service.engine import ServiceError, error_body

MAX_BODY_BYTES = 64 * 1024
POST_ROUTES = {
    "/internal/v1/sessions": "create_session",
    "/internal/v1/observe-and-plan": "observe_and_plan",
    "/internal/v1/commit-plan": "commit_plan",
    "/internal/v1/cancel-plan": "cancel_plan",
    "/internal/v1/grade": "grade",
}


def _valid_id(value):
    return isinstance(value, str) and 1 <= len(value) <= 128 and value.isascii()


def _correlation(value):
    return value if _valid_id(value) else "unknown"


def _reject_constant(value):
    raise ValueError("Non-finite JSON number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


class LearningHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, engine):
        token = os.environ.get("TUTOR_LEARNING_SERVICE_TOKEN", "")
        if not token:
            raise ValueError("TUTOR_LEARNING_SERVICE_TOKEN must not be empty")
        self.token = token
        self.engine = engine
        super().__init__(address, LearningRequestHandler)


class LearningRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, status, body):
        payload = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _dispatch(self):
        trace_id = _correlation(self.headers.get("X-Trace-Id"))
        request_id = _correlation(self.headers.get("Idempotency-Key"))
        try:
            authorization = self.headers.get("Authorization", "")
            authorized = hmac.compare_digest(
                authorization.encode("utf-8"), ("Bearer " + self.server.token).encode("utf-8")
            )
            if not authorized or not _valid_id(self.headers.get("X-Trace-Id")):
                raise ServiceError("UNAUTHORIZED", "Valid bearer token and X-Trace-Id are required")
            path = urlsplit(self.path).path
            if self.command == "GET":
                prefix = "/internal/v1/plans/"
                if (
                    not path.startswith(prefix)
                    or not path[len(prefix) :]
                    or "/" in path[len(prefix) :]
                ):
                    raise ServiceError("INVALID_SCHEMA", "Unknown route")
                result = self.server.engine.get_plan(
                    unquote(path[len(prefix) :]), trace_id=trace_id
                )
            elif self.command == "POST" and path in POST_ROUTES:
                key = self.headers.get("Idempotency-Key")
                if not _valid_id(key):
                    raise ServiceError("INVALID_SCHEMA", "A valid Idempotency-Key is required")
                length = self.headers.get("Content-Length", "")
                if (
                    self.headers.get("Transfer-Encoding") is not None
                    or not length.isascii()
                    or not length.isdecimal()
                    or len(length) > 10
                    or int(length) > MAX_BODY_BYTES
                ):
                    raise ServiceError(
                        "INVALID_SCHEMA", "Body requires Content-Length of at most 64 KB"
                    )
                raw = self.rfile.read(int(length))
                try:
                    body = json.loads(
                        raw, parse_constant=_reject_constant, parse_float=_finite_float
                    )
                except (ValueError, UnicodeError, RecursionError):
                    raise ServiceError("INVALID_SCHEMA", "Body must be valid JSON") from None
                if not isinstance(body, dict):
                    raise ServiceError("INVALID_SCHEMA", "Body must be a JSON object")
                request_id = _correlation(body.get("request_id", key))
                if "request_id" in body and body["request_id"] != key:
                    raise ServiceError("INVALID_SCHEMA", "Idempotency-Key must equal request_id")
                operation = getattr(self.server.engine, POST_ROUTES[path])
                if path == "/internal/v1/commit-plan":
                    result = operation(body, key, trace_id=trace_id)
                else:
                    result = operation(body, trace_id=trace_id)
            else:
                raise ServiceError("INVALID_SCHEMA", "Unknown route")
        except ServiceError as error:
            self._send(error.http_status, error_body(error, request_id, trace_id))
        except Exception:
            error = ServiceError(
                "LEARNING_UNAVAILABLE", "Learning service is unavailable", retryable=True
            )
            self._send(error.http_status, error_body(error, request_id, trace_id))
        else:
            self._send(200, result)

    do_GET = _dispatch
    do_POST = _dispatch
