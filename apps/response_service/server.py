"""HTTP front for the response pipeline (PRD 9.5 internal endpoint, 14.3 and 23).

``POST /internal/v1/generate`` takes a GenerateRequest and returns a GenerateResponse or an
Error object with the PRD 9.5 code and status. ``GET /internal/v1/health`` (alias
``/healthz``) reports the artifact and versions. Requests carry a bearer token from the
environment variable named in the config; bodies over 256 KB are refused before they are
read; a repeated ``request_id`` with the same content returns the same response, and with
different content a ``STATE_CONFLICT``. Each request produces one JSON log line holding IDs,
the action, versions, check results and timings: never learner text, plan content, prompts,
drafts or tokens.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import socket
import socketserver
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from apps.response_service.config import (
    DEFAULT_MAX_REQUEST_BYTES,
    ServiceConfig,
    build_pipeline,
    resolve_token,
)
from packages.response.contracts import loads_strict
from packages.response.errors import UNKNOWN_ID, GenerationError, error_object, safe_id
from packages.response.hashing import canonical_json

LOGGER = logging.getLogger("tuturing.response_service")
GENERATE_PATH = "/internal/v1/generate"
HEALTH_PATHS = ("/internal/v1/health", "/healthz")
SERVER_VERSION = "tuturing-response-service/1.0"
MAX_SESSION_LOCKS = 4096
LINGER_SECONDS = 2.0
LINGER_MAX_BYTES = 8 * 1024 * 1024


class ResponseService:
    """Transport-independent request handling, shared by the HTTP handler and tests."""

    def __init__(
        self,
        pipeline,
        *,
        token: str | None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_concurrent: int = 4,
        idempotency_cache_size: int = 1024,
        clock=time.monotonic,
        logger: logging.Logger = LOGGER,
    ):
        if token is not None and not token:
            raise ValueError("token must be None or a nonempty string")
        self.pipeline = pipeline
        self._token = token
        self.max_request_bytes = int(max_request_bytes)
        self._admission = threading.BoundedSemaphore(max(1, int(max_concurrent)))
        self._cache_size = max(0, int(idempotency_cache_size))
        self._cache: OrderedDict[str, tuple[str, dict]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._session_locks: dict[str, threading.Lock] = {}
        self._session_guard = threading.Lock()
        self._clock = clock
        self._logger = logger
        self._config_errors = {"count": 0, "last": None, "last_seen_at": None}
        self._config_error_lock = threading.Lock()

    @property
    def authenticated(self) -> bool:
        return self._token is not None

    def health(self) -> dict:
        """Versions plus any provider configuration rejection seen since startup.

        ``status`` is ``degraded`` once the model provider has refused the configured request
        format: learners still get verified fallbacks, but no model text is being delivered.
        """
        info = self.pipeline.describe()
        with self._config_error_lock:
            config_errors = dict(self._config_errors) if self._config_errors["count"] else None
        return {
            "status": "degraded" if config_errors else "ok",
            "artifact_id": info["artifact_id"],
            "compatible_artifact_ids": info["compatible_artifact_ids"],
            "responder_kind": info["responder"].get("kind"),
            "decoding": info["limits"]["decoding"],
            "versions": info["versions"],
            "auth": "bearer" if self.authenticated else "none",
            "responder_config_errors": config_errors,
        }

    def _note_config_error(self, record: dict) -> bool:
        """Lift a provider config rejection out of the attempt trail into the log record."""
        detail = next(
            (
                attempt.get("detail")
                for attempt in record.get("attempts") or ()
                if attempt.get("outcome") == "config_error"
            ),
            None,
        )
        if detail is None:
            return False
        record["responder_config_error"] = detail
        with self._config_error_lock:
            self._config_errors["count"] += 1
            self._config_errors["last"] = detail
            self._config_errors["last_seen_at"] = (
                datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )
        return True

    def handle_generate(self, headers, read) -> tuple[int, dict, bool]:
        """Return ``(status, body, close_connection)`` for one generate request.

        ``headers`` is a case-insensitive mapping and ``read(n)`` returns up to ``n`` body
        bytes. The body is read only after authentication and the size check pass.
        """
        started = self._clock()
        record = {"event": "generate", "request_id": UNKNOWN_ID, "trace_id": UNKNOWN_ID}
        request = None
        close = True
        try:
            self._authorize(headers.get("Authorization"))
            length = self._declared_length(headers)
            raw = read(length)
            if len(raw) != length:
                raise GenerationError(
                    "INVALID_SCHEMA", "request body is shorter than Content-Length",
                    retryable=False,
                )
            close = False
            request = self._parse(raw)
            if isinstance(request, dict):  # correlate even when validation fails below
                record["request_id"] = safe_id(request.get("request_id"))
                record["trace_id"] = safe_id(request.get("trace_id"))
            self.pipeline.verify(request)
            plan = request["plan"]
            action = plan["teaching_action"]
            record.update(
                request_id=request["request_id"],
                trace_id=request["trace_id"],
                plan_id=plan["plan_id"],
                session_id=plan["session_id"],
                response_artifact_id=request["response_artifact_id"],
                move=action["move"],
                difficulty=action["difficulty"],
                pace=action["pace"],
                deadline_ms=request["deadline_ms"],
            )
            response, cached = self._generate(request, record)
            record.update(status=200, cached=cached)
            return 200, response, False
        except GenerationError as error:
            # GenerationError messages are fixed strings and schema paths, never values.
            record.update(status=error.http_status, code=error.code, message=error.message)
            if error.attempts:
                record["attempts"] = [attempt.redacted() for attempt in error.attempts]
            return error.http_status, self._error(error, request), close
        except Exception as error:  # never leak an exception message to the client
            record.update(status=503, code="GENERATION_UNAVAILABLE", exception=type(error).__name__)
            fallback = GenerationError("GENERATION_UNAVAILABLE", "internal error", retryable=True)
            return 503, self._error(fallback, request), True
        finally:
            record["elapsed_ms"] = max(0, round((self._clock() - started) * 1000))
            # A provider that refuses the configured format is an operator problem, not a
            # learner one: log it at WARNING so it never hides behind a working fallback.
            level = logging.WARNING if self._note_config_error(record) else logging.INFO
            self._logger.log(level, json.dumps(record, sort_keys=True))

    def _error(self, error: GenerationError, request) -> dict:
        request_id = trace_id = UNKNOWN_ID
        if isinstance(request, dict):
            request_id = safe_id(request.get("request_id"))
            trace_id = safe_id(request.get("trace_id"))
        return error_object(
            error.code,
            error.message,
            request_id=request_id,
            trace_id=trace_id,
            retryable=error.retryable,
        )

    def _authorize(self, header) -> None:
        if self._token is None:
            return
        provided = None
        if isinstance(header, str):
            scheme, _, value = header.strip().partition(" ")
            if scheme.lower() == "bearer":
                provided = value.strip()
        if provided is None or not hmac.compare_digest(
            provided.encode("utf-8"), self._token.encode("utf-8")
        ):
            raise GenerationError(
                "UNAUTHORIZED", "a valid service bearer token is required", retryable=False
            )

    def _declared_length(self, headers) -> int:
        if headers.get("Transfer-Encoding"):
            raise GenerationError(
                "INVALID_SCHEMA", "chunked request bodies are not accepted; send Content-Length",
                retryable=False,
            )
        raw_length = headers.get("Content-Length")
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise GenerationError(
                "INVALID_SCHEMA", "Content-Length is required", retryable=False
            ) from None
        if length < 0:
            raise GenerationError("INVALID_SCHEMA", "Content-Length is invalid", retryable=False)
        if length > self.max_request_bytes:
            raise GenerationError(
                "INVALID_SCHEMA",
                f"request body exceeds the {self.max_request_bytes} byte limit",
                retryable=False,
            )
        return length

    @staticmethod
    def _parse(raw: bytes):
        try:
            return loads_strict(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise GenerationError(
                "INVALID_SCHEMA", "request body is not valid JSON", retryable=False
            ) from None

    def _generate(self, request: dict, record: dict) -> tuple[dict, bool]:
        request_id = request["request_id"]
        fingerprint = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
        cached = self._cached(request_id, fingerprint)
        if cached is not None:
            return cached, True
        if not self._admission.acquire(blocking=False):
            raise GenerationError(
                "RATE_LIMITED", "the service is at its admission limit; retry shortly",
                retryable=True,
            )
        try:
            with self._session_lock(request["plan"]["session_id"]):
                cached = self._cached(request_id, fingerprint)
                if cached is not None:
                    return cached, True
                outcome = self.pipeline.run(request)
                self._remember(request_id, fingerprint, outcome.response)
        finally:
            self._admission.release()
        response = outcome.response
        record.update(
            delivery_mode=response["delivery_mode"],
            checks=response["checks"],
            usage=response["usage"],
            versions=response["versions"],
            attempts=[attempt.redacted() for attempt in outcome.attempts],
        )
        return response, False

    def _cached(self, request_id: str, fingerprint: str) -> dict | None:
        with self._cache_lock:
            entry = self._cache.get(request_id)
            if entry is None:
                return None
            self._cache.move_to_end(request_id)
        if entry[0] != fingerprint:
            raise GenerationError(
                "STATE_CONFLICT", "request_id was already used with different content",
                retryable=False,
            )
        return entry[1]

    def _remember(self, request_id: str, fingerprint: str, response: dict) -> None:
        if self._cache_size == 0:
            return
        with self._cache_lock:
            self._cache[request_id] = (fingerprint, response)
            self._cache.move_to_end(request_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._session_guard:
            lock = self._session_locks.get(session_id)
            if lock is None:
                if len(self._session_locks) >= MAX_SESSION_LOCKS:
                    idle = [key for key, held in self._session_locks.items() if not held.locked()]
                    for key in idle[: MAX_SESSION_LOCKS // 2]:
                        del self._session_locks[key]
                lock = self._session_locks[session_id] = threading.Lock()
            return lock


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer that skips HTTPServer's reverse DNS lookup when binding."""

    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def _handler_class(service: ResponseService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = SERVER_VERSION
        sys_version = ""

        def log_message(self, format, *args):  # the service writes its own redacted log
            return

        def do_GET(self):
            if self.path.split("?", 1)[0] in HEALTH_PATHS:
                self._reply(200, service.health(), close=False)
            else:
                self._reply(404, {"message": "not found"}, close=True)

        def do_POST(self):
            if self.path.split("?", 1)[0] != GENERATE_PATH:
                self._reply(404, {"message": "not found"}, close=True, unread=self._declared())
                return
            consumed = 0

            def read(size):
                nonlocal consumed
                data = self.rfile.read(size)
                consumed += len(data)
                return data

            status, body, close = service.handle_generate(self.headers, read)
            unread = self._declared() - consumed if close else 0
            self._reply(status, body, close=close, unread=unread)

        def _declared(self) -> int:
            try:
                return max(0, int(self.headers.get("Content-Length") or 0))
            except ValueError:
                return 0

        def _reply(self, status: int, body: dict, *, close: bool, unread: int = 0):
            payload = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(payload)
            if close and unread > 0:
                self._linger(unread)

        def _linger(self, remaining: int):
            """Lingering close for a refused body the client may still be sending.

            The response is already written; half-close, then discard incoming bytes for a
            bounded time so the client reads the error instead of a connection reset.
            """
            try:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
                self.connection.settimeout(0.2)
            except OSError:
                return
            deadline = time.monotonic() + LINGER_SECONDS
            remaining = min(remaining, LINGER_MAX_BYTES)
            while remaining > 0 and time.monotonic() < deadline:
                try:
                    chunk = self.connection.recv(min(65_536, remaining))
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                remaining -= len(chunk)

    return Handler


def build_service(config: ServiceConfig, environ=None) -> ResponseService:
    environ = os.environ if environ is None else environ
    pipeline = build_pipeline(config, environ)
    return ResponseService(
        pipeline,
        token=resolve_token(config, environ),
        max_request_bytes=config.max_request_bytes,
        max_concurrent=config.max_concurrent,
        idempotency_cache_size=config.idempotency_cache_size,
    )


def make_server(config: ServiceConfig, service: ResponseService) -> ThreadingHTTPServer:
    """Bind the HTTP server; ``server.server_address`` reveals the port when it was 0."""
    server = _Server((config.host, config.port), _handler_class(service))
    server.service = service
    return server


def describe_service(config: ServiceConfig, service: ResponseService) -> dict:
    """Startup summary with no secrets: config path, bind address, auth mode and versions."""
    return {
        "config": config.path,
        "host": config.host,
        "port": config.port,
        "auth": "bearer" if service.authenticated else "none (loopback only)",
        "max_request_bytes": service.max_request_bytes,
        **service.pipeline.describe(),
    }
