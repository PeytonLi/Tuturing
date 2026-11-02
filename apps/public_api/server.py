"""HTTP transport of the public API: auth, size limits, routing, error bodies.

Standard library only (`http.server`). The handler never logs request bodies.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from apps.public_api.contract import parse_json_strict
from apps.public_api.errors import ApiError, invalid
from apps.public_api.service import PublicApi, Reply, new_id

log = logging.getLogger("tutor.public_api.http")

ROUTE = re.compile(r"^/v1/sessions(?:/(?P<session_id>[^/]+)(?:/(?P<action>turns|stop))?)?/?$")
VERSION_PREFIX = re.compile(r"^/v(?P<major>\d+)(?:/|$)")


class PublicApiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        api: PublicApi,
        callers: dict[str, str],
        *,
        max_request_bytes: int = 65536,
    ) -> None:
        super().__init__(address, PublicApiHandler)
        self.api = api
        self.callers = callers
        self.max_request_bytes = max_request_bytes

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class PublicApiHandler(BaseHTTPRequestHandler):
    server: PublicApiServer
    server_version = "TutorPublicApi/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("%s " + format, self.address_string(), *args)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def _handle(self, method: str) -> None:
        request_id = f"srv-{new_id('rq')[3:]}"
        trace_id = new_id("trc")
        body: Any = None
        try:
            caller = self._authenticate()
            match = self._route(method)
            if method == "POST":
                body = self._read_body()
                if isinstance(body, dict) and isinstance(body.get("request_id"), str):
                    request_id = body["request_id"] or request_id
            reply = self._dispatch(method, match, caller, body)
        except ApiError as error:
            self._write_error(error, request_id, trace_id)
        except Exception:  # unexpected failure: keep the body out of the log
            log.error("unhandled failure trace_id=%s\n%s", trace_id, traceback.format_exc())
            self._write_error(
                ApiError(
                    "LEARNING_UNAVAILABLE",
                    "Internal failure; retry the same request.",
                    retryable=True,
                    retry_after_s=1,
                ),
                request_id,
                trace_id,
            )
        else:
            self._write(reply.status, reply.body, reply.trace_id, retry_after=reply.retry_after_s)

    def _authenticate(self) -> str:
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        token = token.strip()
        if scheme.lower() == "bearer" and token:
            for candidate, caller in self.server.callers.items():
                if hmac.compare_digest(candidate.encode("utf-8"), token.encode("utf-8")):
                    return caller
        raise ApiError("UNAUTHORIZED", "A valid bearer token is required.", retryable=False)

    def _route(self, method: str) -> dict[str, str | None]:
        path = self.path.split("?", 1)[0]
        version = VERSION_PREFIX.match(path)
        if version and version.group("major") != "1":
            raise ApiError(
                "UNSUPPORTED_VERSION",
                "Only contract major version 1 is supported.",
                retryable=False,
            )
        declared = self.headers.get("Tutor-Contract-Version")
        if declared is not None and declared.split(".", 1)[0] != "1":
            raise ApiError(
                "UNSUPPORTED_VERSION",
                "Only contract major version 1 is supported.",
                retryable=False,
            )
        match = ROUTE.match(path)
        if match is None:
            raise invalid("unknown operation")
        groups = match.groupdict()
        expected_method = "GET" if groups["session_id"] and not groups["action"] else "POST"
        if method != expected_method:
            raise invalid("unknown operation")
        return groups

    def _read_body(self) -> Any:
        limit = self.server.max_request_bytes
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise invalid("Content-Length must be an integer") from error
        if length > limit:
            self.close_connection = True
            raise invalid(f"request body exceeds {limit} bytes")
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            return parse_json_strict(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise invalid("request body is not valid JSON") from error

    def _dispatch(self, method: str, match: dict[str, str | None], caller: str, body: Any) -> Reply:
        api = self.server.api
        session_id, action = match["session_id"], match["action"]
        headers = {}
        key = self.headers.get("Idempotency-Key")
        if key is not None:
            headers["Idempotency-Key"] = key
        if session_id is None:
            return api.create_session(caller, body, headers)
        if action is None:
            return api.get_session(caller, session_id)
        if action == "turns":
            return api.turn(caller, session_id, body, headers)
        return api.stop(caller, session_id, body, headers)

    def _write_error(self, error: ApiError, request_id: str, trace_id: str) -> None:
        extra = {}
        if error.code == "UNAUTHORIZED":
            extra["WWW-Authenticate"] = 'Bearer realm="tutor"'
        self._write(
            error.status,
            error.body(request_id, trace_id),
            trace_id,
            retry_after=error.retry_after_s,
            extra=extra,
        )

    def _write(
        self,
        status: int,
        body: Any,
        trace_id: str,
        *,
        retry_after: int | None = None,
        extra: dict[str, str] | None = None,
    ) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Tutor-Trace-Id", trace_id)
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)
