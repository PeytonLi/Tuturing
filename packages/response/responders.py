"""Responders: the text producers behind the response pipeline.

Every responder implements ``generate`` and ``repair`` and returns a ``ResponderOutput``.
The pipeline, not the responder, decides what is delivered: it validates every output,
allows one repair and otherwise uses a verified fallback.

- ``TemplateStubResponder`` renders the existing deterministic CPU stub templates
  (packages.response.stub) and repairs by deterministic trimming.
- ``OpenAICompatibleResponder`` calls any OpenAI-compatible ``/chat/completions``
  endpoint (a local llama.cpp ``llama-server``, vLLM or a hosted provider) with the
  standard library and the 384-token output cap. ``decoding`` sets the token field
  (``max_tokens`` or ``max_completion_tokens``), the temperature (0 for greedy decoding, or
  null to omit it for models that accept only their default) and an optional
  ``reasoning_effort``, in the vocabulary of tools/response/benchmark.py. Model-backed
  responders send only the compiled ``messages``; they never read the plan.

A provider that rejects the request format (HTTP 400 or 422 for a parameter, 401, 403 or 404)
raises ``ResponderConfigRejected``. The pipeline still falls back for the learner but records
the rejection as a ``config_error`` attempt, so a misconfigured model is never a silent
fallback.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from packages.response.repair import repair_messages, trim_to_limits
from packages.response.stub import render_template

MAX_OUTPUT_TOKENS = 384
MAX_RESPONSE_BYTES = 1_000_000
MAX_ERROR_BYTES = 65_536
TEMPLATE_STUB_VERSION = "template_stub_v1"
OPENAI_COMPATIBLE_VERSION = "openai_compatible_v2"
TOKEN_PARAMS = ("max_tokens", "max_completion_tokens")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EFFORT = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_DECODING_KEYS = frozenset({"token_param", "temperature", "reasoning_effort"})
# extra_body may add provider fields, never one the responder already sets.
_DECODING_OWNERS = {
    "temperature": "responder.decoding.temperature (null omits it)",
    "max_tokens": "responder.decoding.token_param with generation.max_output_tokens",
    "max_completion_tokens": "responder.decoding.token_param with generation.max_output_tokens",
    "reasoning_effort": "responder.decoding.reasoning_effort",
}
_PROTOCOL_FIELDS = frozenset({"model", "messages", "stream", "n"})
_SAMPLING_FIELDS = frozenset({"top_p", "seed"})
_REQUEST_SPECIFIC_CODES = frozenset(
    {"context_length_exceeded", "string_above_max_length", "content_policy_violation",
     "content_filter"}
)
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\[\]-]")
_KEYLIKE = re.compile(r"\b(?:sk|pk|rk|fw|key)[-_][A-Za-z0-9*._-]{4,}|\b[A-Za-z0-9*]{24,}\b")
PROBE_MESSAGES = (
    {"role": "system", "content": "Configuration probe. Reply with exactly one word: ready"},
    {"role": "user", "content": "Are you ready?"},
)


class ResponderError(RuntimeError):
    """A responder could not produce text (transport, protocol or server failure)."""


class ResponderConfigRejected(ResponderError):
    """The provider rejected the request format: a configuration problem, not a transient one.

    ``summary`` holds only the status, a reason and the provider's parameter name and error
    code, so it is safe for logs; ``provider_message`` is kept for an operator's probe only.
    """

    def __init__(self, status, reason, *, param=None, code=None, provider_message=None):
        self.status = status
        self.reason = reason
        self.param = param
        self.code = code
        self.provider_message = provider_message
        super().__init__(self.summary)

    @property
    def summary(self) -> str:
        parts = [f"HTTP {self.status} {self.reason}"]
        if self.param:
            parts.append(f"param={self.param}")
        if self.code:
            parts.append(f"code={self.code}")
        return " ".join(parts)


class ResponderConfigError(ValueError):
    """A responder configuration is unusable, for example a missing API key variable."""


@dataclass(frozen=True)
class Decoding:
    """Decoding fields sent with every request; defaults reproduce greedy max_tokens decoding."""

    token_param: str = "max_tokens"
    temperature: float | int | None = 0
    reasoning_effort: str | None = None

    @property
    def label(self) -> str:
        if self.temperature is None:
            return "provider_default_temperature"
        if self.temperature == 0:
            return "greedy"
        return f"temperature_{self.temperature}"

    def fields(self, max_tokens: int) -> dict:
        body = {self.token_param: max_tokens}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def describe(self) -> dict:
        return {
            "token_param": self.token_param,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "label": self.label,
        }


def parse_decoding(spec) -> Decoding:
    """Validate ``responder.decoding``; an absent block means greedy ``max_tokens`` decoding."""
    if spec is None:
        return Decoding()
    if not isinstance(spec, dict):
        raise ResponderConfigError("responder.decoding must be a mapping")
    unknown = sorted(set(spec) - _DECODING_KEYS)
    if unknown:
        raise ResponderConfigError(
            "responder.decoding has unknown keys: " + ", ".join(unknown)
            + "; it accepts token_param, temperature and reasoning_effort, and the output "
            "token cap is generation.max_output_tokens"
        )
    token_param = spec.get("token_param", "max_tokens")
    if token_param not in TOKEN_PARAMS:
        raise ResponderConfigError(
            "responder.decoding.token_param must be max_tokens or max_completion_tokens"
        )
    temperature = spec.get("temperature", 0)
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ResponderConfigError(
            "responder.decoding.temperature must be a number in [0, 2], or null to omit it"
        )
    effort = spec.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or not _EFFORT.fullmatch(effort)):
        raise ResponderConfigError(
            "responder.decoding.reasoning_effort must be a short lowercase name such as none, "
            "low or medium, or null to omit it"
        )
    return Decoding(token_param=token_param, temperature=temperature, reasoning_effort=effort)


def check_extra_body(extra_body) -> dict:
    """Reject provider fields that the responder already controls, naming where they belong."""
    extra_body = {} if extra_body is None else extra_body
    if not isinstance(extra_body, dict):
        raise ResponderConfigError("responder.extra_body must be a mapping")
    for key in sorted(extra_body):
        if key in _DECODING_OWNERS:
            raise ResponderConfigError(
                f"responder.extra_body.{key} would set a decoding field twice; set it once, "
                f"under {_DECODING_OWNERS[key]}"
            )
        if key in _PROTOCOL_FIELDS:
            raise ResponderConfigError(
                f"responder.extra_body.{key} is set by the responder itself and cannot be "
                "overridden"
            )
        if key in _SAMPLING_FIELDS:
            raise ResponderConfigError(
                f"responder.extra_body.{key} is a sampling control; serving decodes only as "
                "responder.decoding says (PRD 14.1)"
            )
    return dict(extra_body)


def _safe_token(value) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    return _UNSAFE.sub("", str(value))[:64] or None


def classify_http_error(status: int, raw: bytes, fields=()) -> ResponderError:
    """Map a provider error response to a config rejection or a transient responder error.

    ``fields`` are the decoding and extra_body field names that were sent; when the provider
    omits ``param`` the first of them quoted in its message is reported instead.
    """
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    details = error if isinstance(error, dict) else payload if isinstance(payload, dict) else {}
    message = error if isinstance(error, str) else details.get("message")
    message = message if isinstance(message, str) else None
    param = _safe_token(details.get("param"))
    code = _safe_token(details.get("code"))
    if param is None and message:
        lowered = message.lower()
        positions = [
            (lowered.find(name.lower()), name) for name in fields if name.lower() in lowered
        ]
        param = _safe_token(min(positions)[1]) if positions else None
    if status in (400, 422):
        if code in _REQUEST_SPECIFIC_CODES:
            return ResponderError(f"chat completions returned HTTP {status} ({code})")
        return ResponderConfigRejected(
            status, "invalid_parameter", param=param, code=code, provider_message=message
        )
    reasons = {401: "unauthorized", 403: "forbidden", 404: "not_found"}
    if status in reasons:
        return ResponderConfigRejected(
            status, reasons[status], code=code, provider_message=message
        )
    return ResponderError(f"chat completions returned HTTP {status}")


@dataclass(frozen=True)
class ResponderOutput:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False


@dataclass(frozen=True)
class RepairLimits:
    max_words: int
    max_questions: int


def _plan_anchors(plan: dict) -> tuple[str, ...]:
    content = plan.get("content", {})
    values = [
        content.get("problem_stem"),
        content.get("permitted_hint"),
        content.get("worked_example"),
        content.get("previous_step"),
        *(content.get("approved_facts") or []),
    ]
    return tuple(value for value in values if isinstance(value, str) and value)


class TemplateStubResponder:
    """Deterministic CPU responder built on the existing stub templates.

    It reports zero token usage: no model or tokenizer is involved.
    """

    kind = "template_stub"
    version = TEMPLATE_STUB_VERSION
    io_bound = False
    decoding_label = "deterministic"
    # Template output is a verified fallback realization, never model text (PRD 8.4);
    # decision recorded in decisions.md on 2026-11-02.
    delivery_mode = "fallback"

    def generate(self, messages, *, plan, max_tokens, timeout_s) -> ResponderOutput:
        return ResponderOutput(text=render_template(plan))

    def repair(self, messages, draft, *, plan, limits, max_tokens, timeout_s) -> ResponderOutput:
        return ResponderOutput(
            text=trim_to_limits(
                draft,
                max_words=limits.max_words,
                max_questions=limits.max_questions,
                anchors=_plan_anchors(plan),
            )
        )

    def describe(self) -> dict:
        return {"kind": self.kind, "version": self.version}


def strip_reasoning(text: str) -> str:
    """Drop ``<think>`` reasoning blocks that some chat models emit before the reply."""
    lowered = text.lower()
    end = lowered.rfind("</think>")
    if end != -1:
        text = text[end + len("</think>"):]
        lowered = text.lower()
    start = lowered.find("<think>")
    if start != -1:
        text = text[:start]
    return text.strip()


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
            and isinstance(part.get("text", ""), str)
        )
    return ""


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the Authorization header never follows them to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_base_url(base_url: object) -> str:
    if not isinstance(base_url, str) or not base_url:
        raise ResponderConfigError("base_url must be a nonempty http or https URL")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ResponderConfigError("base_url must be an http or https URL with a host")
    if parsed.username or parsed.password:
        raise ResponderConfigError("base_url must not embed credentials; use api_key_env")
    if parsed.query or parsed.fragment:
        raise ResponderConfigError("base_url must not have a query or fragment")
    return base_url.rstrip("/")


class OpenAICompatibleResponder:
    """Chat-completions client for any OpenAI-compatible server (standard library only)."""

    kind = "openai_compatible"
    version = OPENAI_COMPATIBLE_VERSION
    delivery_mode = "model"
    io_bound = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        request_timeout_s: float = 8.0,
        decoding: dict | Decoding | None = None,
        extra_body: dict | None = None,
        environ=None,
    ):
        # Shape checks come first so a bad config is reported even without credentials.
        self.base_url = validate_base_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ResponderConfigError("model must be a nonempty string")
        self.model = model
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env)
        ):
            raise ResponderConfigError("api_key_env must be an environment variable name")
        self.api_key_env = api_key_env
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, int | float)
            or not 0 < request_timeout_s <= 60
        ):
            raise ResponderConfigError("request_timeout_s must be a number in (0, 60]")
        self.request_timeout_s = float(request_timeout_s)
        self.decoding = decoding if isinstance(decoding, Decoding) else parse_decoding(decoding)
        self.extra_body = check_extra_body(extra_body)
        environ = os.environ if environ is None else environ
        self._api_key = None
        if api_key_env is not None:
            self._api_key = environ.get(api_key_env) or None
            if self._api_key is None:
                raise ResponderConfigError(f"environment variable {api_key_env} is not set")
        self.url = self.base_url + "/chat/completions"
        handlers = [_NoRedirect()]
        if _is_loopback(urllib.parse.urlsplit(self.url).hostname or ""):
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def __repr__(self) -> str:
        return f"OpenAICompatibleResponder(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def decoding_label(self) -> str:
        return self.decoding.label

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "decoding": self.decoding.describe(),
            "extra_body_keys": sorted(self.extra_body),
        }

    def probe(self, *, max_tokens: int = MAX_OUTPUT_TOKENS, timeout_s: float | None = None):
        """Send one minimal request in the configured format and report what the provider said.

        For operators (``tools.response.serve --check --probe``); never used in tests against
        a real provider. The prompt is synthetic, so no learner text is sent, and the
        provider's message is shown only after any key-like string is removed.
        """
        started = time.monotonic()
        try:
            output = self._complete(
                list(PROBE_MESSAGES),
                max_tokens=max_tokens,
                timeout_s=timeout_s or self.request_timeout_s,
            )
        except ResponderConfigRejected as error:
            return {
                "status": "rejected",
                "detail": error.summary,
                "provider_message": self._redact(error.provider_message),
            }
        except ResponderError as error:
            return {"status": "error", "detail": str(error)}
        return {
            "status": "accepted",
            "reply": output.text[:40],
            "truncated": output.truncated,
            "input_tokens": output.input_tokens,
            "output_tokens": output.output_tokens,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }

    def _redact(self, message) -> str | None:
        if not isinstance(message, str):
            return None
        if self._api_key:
            message = message.replace(self._api_key, "[redacted]")
        return " ".join(_KEYLIKE.sub("[redacted]", message).split())[:300]

    def generate(self, messages, *, plan, max_tokens, timeout_s) -> ResponderOutput:
        return self._complete(messages, max_tokens=max_tokens, timeout_s=timeout_s)

    def repair(self, messages, draft, *, plan, limits, max_tokens, timeout_s) -> ResponderOutput:
        return self._complete(
            repair_messages(
                messages,
                draft,
                max_words=limits.max_words,
                max_questions=limits.max_questions,
            ),
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    def _complete(self, messages, *, max_tokens, timeout_s) -> ResponderOutput:
        decoding = self.decoding.fields(min(int(max_tokens), MAX_OUTPUT_TOKENS))
        # check_extra_body guarantees the three parts share no key, so nothing is overridden.
        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            **decoding,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tuturing-response-service/1.0",
        }
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        timeout = max(0.001, min(float(timeout_s), self.request_timeout_s))
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            try:
                error_body = error.read(MAX_ERROR_BYTES)
            except OSError:
                error_body = b""
            sent = [*decoding, *self.extra_body]
            raise classify_http_error(error.code, error_body, sent) from None
        except (urllib.error.URLError, OSError) as error:
            reason = type(error).__name__
            raise ResponderError(f"chat completions request failed: {reason}") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ResponderError("chat completions response is too large")
        try:
            body = json.loads(raw.decode("utf-8"))
            choice = body["choices"][0]
            message = choice["message"]
        except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError):
            raise ResponderError("chat completions response is malformed") from None
        if not isinstance(message, dict):
            raise ResponderError("chat completions response is malformed")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return ResponderOutput(
            text=strip_reasoning(_message_text(message.get("content"))),
            input_tokens=_count(usage.get("prompt_tokens")),
            output_tokens=_count(usage.get("completion_tokens")),
            truncated=choice.get("finish_reason") == "length",
        )


RESPONDER_KINDS = ("template_stub", "openai_compatible")
_OPENAI_KEYS = frozenset(
    {"kind", "base_url", "model", "api_key_env", "request_timeout_s", "decoding", "extra_body"}
)


def build_responder(spec: dict, *, environ=None):
    """Build a responder from a config mapping such as ``{"kind": "template_stub"}``."""
    if not isinstance(spec, dict):
        raise ResponderConfigError("responder must be a mapping")
    kind = spec.get("kind")
    if kind == "template_stub":
        unknown = sorted(set(spec) - {"kind"})
        if unknown:
            raise ResponderConfigError("unknown template_stub keys: " + ", ".join(unknown))
        return TemplateStubResponder()
    if kind == "openai_compatible":
        unknown = sorted(set(spec) - _OPENAI_KEYS)
        if unknown:
            raise ResponderConfigError("unknown openai_compatible keys: " + ", ".join(unknown))
        return OpenAICompatibleResponder(
            base_url=spec.get("base_url"),
            model=spec.get("model"),
            api_key_env=spec.get("api_key_env"),
            request_timeout_s=spec.get("request_timeout_s", 8.0),
            decoding=spec.get("decoding"),
            extra_body=spec.get("extra_body"),
            environ=environ,
        )
    raise ResponderConfigError(
        "responder.kind must be one of " + ", ".join(RESPONDER_KINDS)
    )
