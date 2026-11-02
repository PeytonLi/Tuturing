"""Responders: the text producers behind the response pipeline.

Every responder implements ``generate`` and ``repair`` and returns a ``ResponderOutput``.
The pipeline, not the responder, decides what is delivered: it validates every output,
allows one repair and otherwise uses a verified fallback.

- ``TemplateStubResponder`` renders the existing deterministic CPU stub templates
  (packages.response.stub) and repairs by deterministic trimming.
- ``OpenAICompatibleResponder`` calls any OpenAI-compatible ``/chat/completions``
  endpoint (a local llama.cpp ``llama-server``, vLLM or a hosted provider) with the
  standard library, greedy decoding and the 384-token output cap. Model-backed
  responders send only the compiled ``messages``; they never read the plan.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from packages.response.repair import repair_messages, trim_to_limits
from packages.response.stub import render_template

MAX_OUTPUT_TOKENS = 384
MAX_RESPONSE_BYTES = 1_000_000
TEMPLATE_STUB_VERSION = "template_stub_v1"
OPENAI_COMPATIBLE_VERSION = "openai_compatible_v1"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED_BODY_KEYS = frozenset(
    {"model", "messages", "temperature", "max_tokens", "stream", "n", "top_p", "seed"}
)


class ResponderError(RuntimeError):
    """A responder could not produce text (transport, protocol or server failure)."""


class ResponderConfigError(ValueError):
    """A responder configuration is unusable, for example a missing API key variable."""


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
    io_bound = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        request_timeout_s: float = 8.0,
        extra_body: dict | None = None,
        environ=None,
    ):
        self.base_url = validate_base_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ResponderConfigError("model must be a nonempty string")
        self.model = model
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not _ENV_NAME.fullmatch(api_key_env)
        ):
            raise ResponderConfigError("api_key_env must be an environment variable name")
        self.api_key_env = api_key_env
        environ = os.environ if environ is None else environ
        self._api_key = None
        if api_key_env is not None:
            self._api_key = environ.get(api_key_env) or None
            if self._api_key is None:
                raise ResponderConfigError(f"environment variable {api_key_env} is not set")
        if (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, int | float)
            or not 0 < request_timeout_s <= 60
        ):
            raise ResponderConfigError("request_timeout_s must be a number in (0, 60]")
        self.request_timeout_s = float(request_timeout_s)
        extra_body = {} if extra_body is None else extra_body
        if not isinstance(extra_body, dict):
            raise ResponderConfigError("extra_body must be a mapping")
        reserved = sorted(_RESERVED_BODY_KEYS & set(extra_body))
        if reserved:
            raise ResponderConfigError(
                "extra_body cannot set decoding or routing fields: " + ", ".join(reserved)
            )
        self.extra_body = dict(extra_body)
        self.url = self.base_url + "/chat/completions"
        handlers = [_NoRedirect()]
        if _is_loopback(urllib.parse.urlsplit(self.url).hostname or ""):
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    def __repr__(self) -> str:
        return f"OpenAICompatibleResponder(base_url={self.base_url!r}, model={self.model!r})"

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
        }

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
        payload = {
            **self.extra_body,
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": min(int(max_tokens), MAX_OUTPUT_TOKENS),
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
            raise ResponderError(f"chat completions returned HTTP {error.code}") from None
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
    {"kind", "base_url", "model", "api_key_env", "request_timeout_s", "extra_body"}
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
            extra_body=spec.get("extra_body"),
            environ=environ,
        )
    raise ResponderConfigError(
        "responder.kind must be one of " + ", ".join(RESPONDER_KINDS)
    )
