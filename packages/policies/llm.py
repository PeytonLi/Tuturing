from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from packages.learning.actions import DIFFICULTIES, PACES, decode_strategist, encode_strategist
from packages.policies.rules import RulesStrategist

ROOT = Path(__file__).resolve().parents[2]
FALLBACK_REASONS = ("truncated", "parse_failure", "illegal", "http_error")
REQUIRED_CONTEXT_KEYS = {
    "active_item",
    "active_item_difficulty",
    "active_option_type",
    "active_target_concept_id",
    "current_certified",
    "current_concept_id",
    "explicit_faster",
    "explicit_skip",
    "explicit_slower",
    "goal_concept_id",
    "goal_stack",
    "hint_count",
    "hint_level",
    "independent_failure_streak",
    "independent_success_streak",
    "mastery_by_concept",
    "meaningful_error",
    "misconceptions",
    "pace_preference",
    "previous_move",
    "recent_independent_correct",
    "remaining_budget",
    "turn_index",
    "turns_in_option",
    "unsuccessful_hint_count",
}


def legal_action_text(mask: list[bool]) -> str:
    legal = [decode_strategist(index) for index, allowed in enumerate(mask) if allowed]
    lines = []
    for move in sorted({item[0] for item in legal}):
        group = [item for item in legal if item[0] == move]
        difficulties = sorted({item[1] for item in group}, key=DIFFICULTIES.index)
        paces = sorted({item[2] for item in group}, key=PACES.index)
        lines.append(
            f"{move}: difficulties={', '.join(difficulties)}; paces={', '.join(paces)}"
        )
    return "\n".join(lines)


def parse_first_json_object(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    leading = text.lstrip()
    if leading.startswith("<think>"):
        first_object = text.find("{")
        text = text[first_object:] if first_object >= 0 else ""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("reply contains no JSON object")


class CallLimitExceeded(RuntimeError):
    pass


class _ResponseParseError(ValueError):
    pass


class _HTTPFailure(RuntimeError):
    pass


def _misconception_summary(value) -> dict[str, float]:
    if isinstance(value, dict):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        summary = {}
        for index, (key, probability) in enumerate(ordered):
            if probability <= 0:
                continue
            name = str(key)
            if not re.fullmatch(r"M\d{2}", name):
                name = f"M{index + 1:02d}"
            summary[name] = round(probability, 2)
        return summary
    return {
        f"M{index + 1:02d}": round(probability, 2)
        for index, probability in enumerate(value or [])
        if probability > 0
    }


class LLMStrategist:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens_field: str,
        max_tokens: int,
        temperature: float | None,
        timeout: float = 120,
        sleep: Callable[[float], None] = time.sleep,
        cache_path: str | Path | None = None,
        max_calls: int = 2000,
        prompt_path: str | Path = "evaluation/learning/llm_strategist_prompt_v1.txt",
    ) -> None:
        if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("max_tokens_field must be max_tokens or max_completion_tokens")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if not model:
            raise ValueError("model must be non-empty")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens_field = max_tokens_field
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._sleep = sleep
        self.max_calls = max_calls
        prompt_path = Path(prompt_path)
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        if cache_path is None:
            model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-._") or "model"
            cache_path = ROOT / "experiments/learning/llm_cache" / f"{model_slug}.jsonl"
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_absolute():
            self.cache_path = ROOT / self.cache_path
        self._lock = threading.Lock()
        self._calls = 0
        self._cache_hits = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._decisions = 0
        self._fallbacks_by_reason = dict.fromkeys(FALLBACK_REASONS, 0)
        self._cache: dict[str, dict] = {}
        self._load_cache()
        self._rules = RulesStrategist()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(record, dict)
                and isinstance(record.get("cache_key"), str)
                and isinstance(record.get("content"), str)
            ):
                self._cache[record["cache_key"]] = {
                    "content": record["content"],
                    "finish_reason": record.get("finish_reason"),
                }

    @staticmethod
    def _cache_key(model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()

    @staticmethod
    def _state_summary(context: dict) -> dict:
        missing = REQUIRED_CONTEXT_KEYS - context.keys()
        if missing:
            raise ValueError(
                f"LLM context is missing required public keys: {sorted(missing)}"
            )
        return {
            "current_concept_id": context["current_concept_id"],
            "goal_concept_id": context["goal_concept_id"],
            "goal_stack": context.get("goal_stack", []),
            "option_type": context.get("active_option_type"),
            "option_target": context.get("active_target_concept_id"),
            "option_turns": context.get("turns_in_option", 0),
            "teaching_turns": context["turn_index"],
            "remaining_turns": context.get("remaining_budget"),
            "previous_move": context.get("previous_move") or "NONE",
            "pace_preference": context.get("pace_preference", "normal"),
            "explicit_request": (
                "skip"
                if context.get("explicit_skip")
                else "faster"
                if context.get("explicit_faster")
                else "slower"
                if context.get("explicit_slower")
                else "none"
            ),
            "concepts": {
                concept_id: {"mastery": round(mastery, 2)}
                for concept_id, mastery in sorted(context["mastery_by_concept"].items())
            },
            "current_concept_certified": bool(context.get("current_certified")),
            "misconceptions": _misconception_summary(context.get("misconceptions")),
            "active_item": (
                None
                if not context.get("active_item")
                else {
                    "difficulty": context.get("active_item_difficulty"),
                    "hint_level": context.get("hint_level", 0),
                }
            ),
            "last_answer_meaningful_error": bool(context.get("meaningful_error")),
            "recent_independent_correct": bool(context.get("recent_independent_correct")),
            "independent_success_streak": context.get("independent_success_streak", 0),
            "independent_failure_streak": context.get("independent_failure_streak", 0),
            "hints_given": context.get("hint_count", 0),
            "unsuccessful_hint_count": context.get("unsuccessful_hint_count", 0),
        }

    def build_prompt(self, context: dict, mask: list[bool]) -> str:
        summary = json.dumps(self._state_summary(context), indent=1, sort_keys=True)
        situation = (
            f"Turn {context['turn_index'] + 1} of a simulated probability tutoring session. "
            f"The learning goal is concept {context['goal_concept_id']}; the tutor is "
            f"currently teaching concept {context['current_concept_id']}."
        )
        return (
            self.prompt_template.replace("{situation}", situation)
            .replace("{state_summary}", summary)
            .replace("{recent_dialogue}", "(none; structured simulation, no dialogue text)")
            .replace("{legal_actions}", legal_action_text(mask))
        )

    def _claim_call(self) -> None:
        with self._lock:
            if self._calls >= self.max_calls:
                raise CallLimitExceeded(
                    f"LLM call limit of {self.max_calls} requests has been reached"
                )
            self._calls += 1

    def _save_cache(self, cache_key: str, completion: dict) -> None:
        record = {
            "cache_key": cache_key,
            "content": completion["content"],
            "finish_reason": completion.get("finish_reason"),
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if cache_key in self._cache:
                return
            with self.cache_path.open("a", encoding="utf-8") as cache_file:
                cache_file.write(json.dumps(record, sort_keys=True) + "\n")
            self._cache[cache_key] = completion

    def _record_usage(self, usage) -> None:
        if not isinstance(usage, dict):
            return
        with self._lock:
            self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self._completion_tokens += int(usage.get("completion_tokens") or 0)

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None = None) -> float:
        delay = min(30, 2**attempt)
        if retry_after is not None:
            try:
                retry_after_seconds = float(retry_after)
            except ValueError:
                retry_after_seconds = None
            if (
                retry_after_seconds is not None
                and math.isfinite(retry_after_seconds)
                and retry_after_seconds >= 0
            ):
                delay = max(delay, retry_after_seconds)
        return delay + random.random()

    def _request_completion(self, prompt: str) -> dict:
        cache_key = self._cache_key(self.model, prompt)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
        if cached is not None:
            return cached
        if not self.api_key:
            raise RuntimeError("an API key is required for an uncached LLM request")
        payload_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            self.max_tokens_field: self.max_tokens,
        }
        if self.temperature is not None:
            payload_data["temperature"] = self.temperature
        payload = json.dumps(payload_data).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "TutoringStrategistEval/1.0",
            },
            method="POST",
        )
        for attempt in range(6):
            self._claim_call()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
            except urllib.error.HTTPError as error:
                if error.code == 429 or 500 <= error.code <= 599:
                    if attempt < 5:
                        retry_after = (
                            error.headers.get("Retry-After") if error.headers else None
                        )
                        self._sleep(self._retry_delay(attempt, retry_after))
                        continue
                raise _HTTPFailure(f"LLM endpoint returned HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt < 5:
                    self._sleep(self._retry_delay(attempt))
                    continue
                raise _HTTPFailure("LLM endpoint request failed after retries") from error
            try:
                body = json.loads(response_body)
                choice = body["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason")
            except (
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                raise _ResponseParseError("LLM response has no parseable completion") from error
            if not isinstance(content, str):
                raise _ResponseParseError("LLM completion content is not text")
            completion = {"content": content, "finish_reason": finish_reason}
            self._save_cache(cache_key, completion)
            self._record_usage(body.get("usage"))
            return completion
        raise _HTTPFailure("LLM endpoint request failed after retries")

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        with self._lock:
            self._decisions += 1
        prompt = self.build_prompt(context, mask)
        try:
            completion = self._request_completion(prompt)
        except CallLimitExceeded:
            raise
        except _HTTPFailure:
            return self._fallback_action("http_error", observation, mask, context)
        except _ResponseParseError:
            return self._fallback_action("parse_failure", observation, mask, context)
        if completion.get("finish_reason") in {"length", "max_tokens"}:
            return self._fallback_action("truncated", observation, mask, context)
        try:
            parsed = parse_first_json_object(completion["content"])
            action_id = encode_strategist(
                str(parsed["move"]), str(parsed["difficulty"]), str(parsed["pace"])
            )
        except (KeyError, TypeError, ValueError):
            return self._fallback_action("parse_failure", observation, mask, context)
        if action_id >= len(mask) or not mask[action_id]:
            return self._fallback_action("illegal", observation, mask, context)
        return action_id

    def _fallback_action(
        self,
        reason: str,
        observation: list[float],
        mask: list[bool],
        context: dict,
    ) -> int:
        with self._lock:
            self._fallbacks_by_reason[reason] += 1
        return self._rules.act(observation, mask, context)

    def usage_summary(self) -> dict:
        with self._lock:
            fallback_count = sum(self._fallbacks_by_reason.values())
            return {
                "model": self.model,
                "calls": self._calls,
                "cache_hits": self._cache_hits,
                "fallbacks_by_reason": dict(self._fallbacks_by_reason),
                "fallback_rate": fallback_count / self._decisions if self._decisions else 0.0,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            }

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def cache_hits(self) -> int:
        with self._lock:
            return self._cache_hits

    @property
    def fallbacks_by_reason(self) -> dict[str, int]:
        with self._lock:
            return dict(self._fallbacks_by_reason)

    @property
    def prompt_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return self._completion_tokens
