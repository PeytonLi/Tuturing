"""Prompt-only response benchmark across deterministic, local and hosted responders.

For each case the harness compiles the TeachingPlan with ``packages.response.prompt``,
calls the configured model, runs the six ``packages.response.validate`` checks and records
latency, tokens and cost. The matrix lives in ``configs/response/benchmark_v1.yaml`` and the
runbook is ``docs/runbooks/response_benchmark.md``.

    uv run python -m tools.response.benchmark plan
    uv run python -m tools.response.benchmark run --models fallback
    uv run python -m tools.response.benchmark run --models gpt-6.1-sol --pilot 3
    uv run python -m tools.response.benchmark run --models gpt-6.1-sol
    uv run python -m tools.response.benchmark run --models qwen3.8-27b-fw-h100 \\
        --manage-deployment
    uv run python -m tools.response.benchmark rescore   # after a validator change
    uv run python -m tools.response.benchmark report

API keys are read only from the keys file named in the config (default
``~/.config/tuturing/llm.env``, ``KEY=value`` lines). Values are never printed or written.
Paid entries run only when that file sets ``TUTURING_BENCH_SPEND_CAP_USD``; every paid call
is checked against the cumulative ledger under the run directory before it is sent.
"""

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from packages.response.fallbacks import render_fallback
from packages.response.hashing import canonical_json
from packages.response.item_fidelity import item_fidelity
from packages.response.numbers import extract_numbers
from packages.response.prompt import PROMPT_VERSION, compile_prompt
from packages.response.shapes import is_valid
from packages.response.validate import (
    VALIDATOR_VERSION,
    _discloses_fingerprint,
    _fingerprint_number,
    run_checks,
    word_count,
)
from tools.response.evaluate import CHECK_NAMES

DEFAULT_CONFIG = "configs/response/benchmark_v1.yaml"
DEFAULT_KEYS_FILE = "~/.config/tuturing/llm.env"
USER_AGENT = "tuturing-response-benchmark/1.0"
HARNESS_VERSION = "benchmark_v1"
PRD_OUTPUT_TOKEN_CAP = 384
REPAIRABLE_CHECKS = frozenset({"length_valid", "question_count_valid"})
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_REQUIRED_CONTENT = {
    "HINT": "permitted_hint",
    "REFLECT": "previous_step",
    "WORKED_EXAMPLE": "worked_example",
    "PRACTICE": "problem_stem",
    "CHALLENGE": "problem_stem",
    "CHECK_MASTERY": "problem_stem",
}


class BenchmarkError(RuntimeError):
    """Configuration or precondition problem that stops a run."""


class BudgetExceeded(RuntimeError):
    """Raised before a call that could push committed spend past a cap."""


class CallError(RuntimeError):
    """A model call failed after retries."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    ids = [entry["id"] for entry in config.get("models", [])]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise BenchmarkError(f"duplicate model ids in config: {', '.join(duplicates)}")
    return config


def load_keys(path):
    """Parse ``KEY=value`` lines from the keys file. Returns {} when it is absent."""
    file_path = Path(os.path.expanduser(str(path)))
    if not file_path.is_file():
        return {}
    keys = {}
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ").strip()
        name, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name.strip() and value:
            keys[name.strip()] = value
    return keys


def spend_cap(keys, config):
    raw = keys.get(config.get("spend_cap_env", "TUTURING_BENCH_SPEND_CAP_USD"))
    try:
        cap = float(raw) if raw is not None else None
    except ValueError:
        return None
    return cap if cap is not None and cap > 0 else None


def redact(text, keys):
    for value in keys.values():
        if len(value) >= 8:
            text = text.replace(value, "<redacted>")
    return text


# ---------------------------------------------------------------------------
# Cases


@dataclass
class Case:
    case_set: str
    case_id: str
    plan: dict
    source: str
    reference: str | None = None

    @property
    def move(self):
        return self.plan["teaching_action"]["move"]

    @property
    def key(self):
        return f"{self.case_set}/{self.case_id}"


def stratified_sample(cases, per_move, seed=0, distinct=None):
    """Take ``per_move`` cases per move, preferring distinct ``plan[distinct]`` groups."""
    by_move = defaultdict(list)
    for case in sorted(cases, key=lambda item: item.case_id):
        by_move[case.move].append(case)
    chosen = []
    for move in sorted(by_move):
        pool = list(by_move[move])
        random.Random(f"{seed}:{move}").shuffle(pool)
        picked, seen = [], set()
        for case in pool:
            group = case.plan.get(distinct) if distinct else None
            if group is not None and group in seen:
                continue
            seen.add(group)
            picked.append(case)
            if len(picked) == per_move:
                break
        for case in pool:
            if len(picked) >= per_move:
                break
            if case not in picked:
                picked.append(case)
        chosen.extend(picked)
    return sorted(chosen, key=lambda item: (item.move, item.case_id))


def stratified_order(cases, seed=0):
    """Round-robin over (case set, move) strata so every prefix is spread evenly."""
    strata = defaultdict(list)
    for case in cases:
        strata[(case.case_set, case.move)].append(case)
    keys = sorted(strata)
    for key in keys:
        strata[key].sort(key=lambda item: item.case_id)
        random.Random(f"{seed}:{key[0]}:{key[1]}").shuffle(strata[key])
    ordered, depth = [], 0
    while len(ordered) < len(cases):
        for key in keys:
            if depth < len(strata[key]):
                ordered.append(strata[key][depth])
        depth += 1
    return ordered


def load_case_set(spec, root="."):
    """Return (cases, note). ``note`` explains why a set is empty or was filtered."""
    path = Path(root) / spec["path"]
    if not path.exists():
        if spec.get("optional"):
            return [], f"optional, not present at {spec['path']}"
        hint = spec.get("prepare") or "see docs/runbooks/response_benchmark.md"
        return [], f"missing {spec['path']}; prepare with: {hint}"
    fmt = spec.get("format", "jsonl")
    source = spec.get("source", spec["path"])
    cases, skipped = [], 0
    if fmt == "plan_dir":
        for file in sorted(path.glob(spec.get("glob", "*.json"))):
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "teaching_action" in data:
                if is_valid(data, "TeachingPlan"):
                    cases.append(Case(spec["name"], file.stem, data, source))
                else:
                    skipped += 1
    elif fmt == "jsonl":
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                record = json.loads(line)
                plan = record.get("plan")
                if not isinstance(plan, dict) or not is_valid(plan, "TeachingPlan"):
                    skipped += 1
                    continue
                completion = record.get("completion") or []
                reference = None
                if completion and isinstance(completion[0], dict):
                    reference = completion[0].get("content")
                case_id = str(record.get("example_id") or f"{spec['name']}:{index:05d}")
                cases.append(Case(spec["name"], case_id, plan, source, reference))
    else:
        raise BenchmarkError(f"unknown case set format: {fmt}")
    note = f"{skipped} records skipped (not a valid TeachingPlan)" if skipped else None
    sample = spec.get("sample")
    if sample:
        total = len(cases)
        cases = stratified_sample(
            cases,
            per_move=int(sample["per_move"]),
            seed=sample.get("seed", 0),
            distinct=sample.get("distinct"),
        )
        note = (note + "; " if note else "") + f"stratified sample {len(cases)} of {total}"
    return cases, note


def load_all_cases(config, names=None, root="."):
    selected, notes = [], {}
    for spec in config.get("case_sets", []):
        if names and spec["name"] not in names:
            continue
        cases, note = load_case_set(spec, root)
        selected.extend(cases)
        notes[spec["name"]] = {"n": len(cases), "note": note}
    return selected, notes


# ---------------------------------------------------------------------------
# Checks and diagnostics


def action_check_applicable(plan):
    """False when the plan lacks the content the move's action heuristic requires."""
    content = plan.get("content", {})
    move = plan.get("teaching_action", {}).get("move")
    if move == "EXPLAIN_CONCEPT":
        return bool(content.get("approved_facts"))
    field = _REQUIRED_CONTENT.get(move)
    return field is None or isinstance(content.get(field), str)


def _numbers(texts):
    return {token.value for token in extract_numbers(" ".join(t for t in texts if t))}


def _learner_texts(plan):
    context = plan.get("response_context", {})
    texts = [context.get("last_student_message") or ""]
    for entry in context.get("recent_dialogue") or []:
        if isinstance(entry, dict) and entry.get("role") == "student":
            texts.append(entry.get("text") or "")
    return [text for text in texts if isinstance(text, str)]


def number_diagnostics(plan, text):
    """Numbers the math check treats as unsupported, and which the learner wrote."""
    content = plan.get("content", {})
    sources = [
        content.get("problem_stem"),
        *(content.get("approved_facts") or []),
        content.get("permitted_hint"),
        content.get("worked_example"),
        content.get("previous_step"),
    ]
    if plan.get("constraints", {}).get("reveal_allowed"):
        sources.append(content.get("reveal_solution"))
    pool = _numbers([s for s in sources if isinstance(s, str)])
    learner = _numbers(_learner_texts(plan))
    unsupported, from_learner = [], []
    for token in extract_numbers(text):
        value = token.value
        if value in pool or (value.denominator == 1 and 0 <= value <= 10):
            continue
        unsupported.append(token.raw.strip())
        if value in learner:
            from_learner.append(token.raw.strip())
    return unsupported, from_learner


def disclosure_diagnostics(plan, text):
    """Forbidden fingerprints found in the text, and whether all also appear in context."""
    constraints = plan.get("constraints", {})
    if constraints.get("reveal_allowed"):
        return [], None
    tokens = extract_numbers(text)
    normalized = " ".join(text.casefold().split())
    matched = []
    for fingerprint in constraints.get("forbidden_answer_fingerprints", []):
        numeric = _fingerprint_number(fingerprint)
        if numeric is not None:
            if any(_discloses_fingerprint(token, numeric) for token in tokens):
                matched.append(fingerprint)
        else:
            phrase = " ".join(fingerprint.casefold().split())
            if phrase and phrase in normalized:
                matched.append(fingerprint)
    if not matched:
        return [], None
    context = _numbers([plan.get("content", {}).get("problem_stem") or "", *_learner_texts(plan)])
    in_context = all(
        (number := _fingerprint_number(fingerprint)) is not None and number.value in context
        for fingerprint in matched
    )
    return matched, in_context


def evaluate_text(plan, text):
    checks = run_checks(plan, text, plan["teaching_action"])
    applicable = {name: True for name in CHECK_NAMES}
    applicable["action_faithful"] = action_check_applicable(plan)
    failed = [name for name in CHECK_NAMES if applicable[name] and not checks[name]]
    failed_raw = [name for name in CHECK_NAMES if not checks[name]]
    return checks, applicable, failed, failed_raw


def delivery_for(failed, *, error=False):
    """What the serving pipeline (PRD 14.1) would deliver for this text."""
    if error:
        return "error"
    if not failed:
        return "model"
    if set(failed) <= REPAIRABLE_CHECKS:
        return "repair"
    return "fallback"


def fallback_available(plan):
    try:
        render_fallback(plan)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Spend ledger


class Ledger:
    """Append-only JSONL ledger of estimated and actual spend, kept outside Git."""

    def __init__(self, path, deployments_dir=None, clock=time.time):
        self.path = Path(path)
        self.deployments_dir = Path(deployments_dir) if deployments_dir else None
        self.clock = clock
        self.lock = threading.RLock()
        self.reserved = 0.0

    def events(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def append(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {"at": utc_now(), **event}
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def spent(self, model_id=None):
        return sum(
            float(event.get("actual_usd") or 0.0)
            for event in self.events()
            if model_id is None or event.get("model_id") == model_id
        )

    def open_deployment_cost(self):
        """Running GPU cost of deployments that are not yet verified deleted."""
        if not self.deployments_dir or not self.deployments_dir.exists():
            return 0.0
        total = 0.0
        for path in self.deployments_dir.glob("*.json"):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("deleted_verified") or not state.get("create_requested_epoch"):
                continue
            hours = max(self.clock() - state["create_requested_epoch"], 0) / 3600
            total += hours * float(state.get("usd_per_hour") or 0.0)
        return total

    def committed(self):
        with self.lock:
            return self.spent() + self.reserved + self.open_deployment_cost()

    def reserve(self, amount, cap, label):
        with self.lock:
            committed = self.committed()
            if committed + amount > cap + 1e-9:
                raise BudgetExceeded(
                    f"{label}: committed ${committed:.4f} + reserve ${amount:.4f} "
                    f"would pass the cap ${cap:.2f}"
                )
            self.reserved += amount

    def release(self, amount):
        with self.lock:
            self.reserved = max(self.reserved - amount, 0.0)


# ---------------------------------------------------------------------------
# Model calls


def cost_basis(entry):
    if entry.get("provider") == "fallback":
        return "free"
    return (entry.get("pricing") or {}).get("basis", "tokens")


def is_paid(entry):
    basis = cost_basis(entry)
    if basis == "gpu_time":
        return True
    pricing = entry.get("pricing") or {}
    return basis == "tokens" and (
        float(pricing.get("input_usd_per_mtok") or 0) > 0
        or float(pricing.get("output_usd_per_mtok") or 0) > 0
    )


def gpu_usd_per_hour(entry):
    spec = entry.get("deployment") or {}
    return float(spec.get("usd_per_gpu_hour") or 0.0) * int(spec.get("accelerator_count") or 1)


def token_cost(usage, pricing):
    cached = min(int(usage.get("cached_input_tokens") or 0), int(usage.get("input_tokens") or 0))
    uncached = int(usage.get("input_tokens") or 0) - cached
    cached_price = pricing.get("cached_input_usd_per_mtok")
    if cached_price is None:
        cached_price = pricing.get("input_usd_per_mtok", 0)
    return (
        uncached * float(pricing.get("input_usd_per_mtok") or 0)
        + cached * float(cached_price or 0)
        + int(usage.get("output_tokens") or 0) * float(pricing.get("output_usd_per_mtok") or 0)
    ) / 1_000_000


def max_output_tokens(entry):
    return int((entry.get("decoding") or {}).get("max_tokens", 384))


def worst_case_cost(entry, prompt_chars, chars_per_token):
    """Upper bound for one token-billed call: estimated input plus the full output cap."""
    pricing = entry.get("pricing") or {}
    input_tokens = prompt_chars / max(chars_per_token, 0.5) + 64
    return (
        input_tokens * float(pricing.get("input_usd_per_mtok") or 0)
        + max_output_tokens(entry) * float(pricing.get("output_usd_per_mtok") or 0)
    ) / 1_000_000


def build_body(entry, model, messages):
    decoding = entry.get("decoding") or {}
    body = {"model": model, "messages": messages}
    if decoding.get("temperature") is not None:
        body["temperature"] = decoding["temperature"]
    body[decoding.get("token_param", "max_tokens")] = max_output_tokens(entry)
    body.update(decoding.get("extra") or {})
    return body


def normalize_usage(usage):
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": int(prompt_details.get("cached_tokens") or 0),
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "visible_output_tokens": max(output_tokens - reasoning, 0),
    }


def _message_text(message):
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content if isinstance(content, str) else ""


def strip_thinking(text):
    """Remove <think> blocks a serving layer would hide; report whether any were found."""
    if "<think>" not in text and "</think>" not in text:
        return text, False
    stripped = _THINK_BLOCK.sub("", text)
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[1]
    if "<think>" in stripped:
        stripped = stripped.split("<think>", 1)[0]
    return stripped, True


def _error_detail(error, keys):
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return redact(" ".join(body.split())[:600], keys)


def call_model(entry, model, messages, keys, *, urlopen=None, sleep=time.sleep):
    """POST one chat completion with retries; returns text, usage and timing."""
    urlopen = urlopen or urllib.request.urlopen
    url = entry["base_url"].rstrip("/") + "/chat/completions"
    body = build_body(entry, model, messages)
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    key_name = entry.get("api_key_env")
    if key_name:
        headers["Authorization"] = f"Bearer {keys[key_name]}"
    retry = entry.get("retry") or {}
    max_attempts = int(retry.get("max_attempts", 4))
    scaling_attempts = int(retry.get("scaling_attempts", 0))
    base_delay = float(retry.get("base_delay_s", 2.0))
    timeout = float(entry.get("timeout_s", 120))
    payload = json.dumps(body).encode("utf-8")
    attempts = 0
    started = time.monotonic()
    while True:
        attempts += 1
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        attempt_started = time.monotonic()
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            detail = _error_detail(error, keys)
            scaling = error.code == 503 and "DEPLOYMENT_SCALING_UP" in detail
            limit = max(max_attempts, scaling_attempts) if scaling else max_attempts
            if error.code in RETRYABLE_STATUS and attempts < limit:
                sleep(min(base_delay * (2 ** min(attempts - 1, 5)), 60.0))
                continue
            raise CallError(f"HTTP {error.code}: {detail}", status=error.code) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if attempts < max_attempts:
                sleep(min(base_delay * (2 ** min(attempts - 1, 5)), 60.0))
                continue
            reason = getattr(error, "reason", error)
            raise CallError(f"{type(error).__name__}: {redact(str(reason), keys)}") from None
    choices = data.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    text, think_stripped = strip_thinking(_message_text(choice.get("message") or {}))
    return {
        "text": text.strip(),
        "finish_reason": choice.get("finish_reason"),
        "usage": normalize_usage(data.get("usage")),
        "latency_ms": int((time.monotonic() - attempt_started) * 1000),
        "total_ms": int((time.monotonic() - started) * 1000),
        "attempts": attempts,
        "served_model": data.get("model"),
        "think_stripped": think_stripped,
    }


# ---------------------------------------------------------------------------
# Results


def decoding_id(entry):
    identity = {
        "model": (entry.get("deployment") or {}).get("base_model") or entry.get("model"),
        "provider": entry.get("provider"),
        "decoding": entry.get("decoding") or {},
    }
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:12]


def prompt_digest(messages):
    return hashlib.sha256(canonical_json(messages).encode("utf-8")).hexdigest()


class ResultWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.Lock()

    def write(self, record):
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{record['case_set']}.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_results(run_dir, model_id):
    """Latest record per case key for one model entry."""
    latest = {}
    directory = Path(run_dir) / model_id
    if not directory.exists():
        return latest
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    latest[f"{record['case_set']}/{record['case_id']}"] = record
    return latest


def reusable(record, case, entry, messages):
    return bool(
        record
        and record.get("error") is None
        and record.get("prompt_sha256") == prompt_digest(messages)
        and record.get("decoding_id") == decoding_id(entry)
    )


def score_text(plan, text, *, error=None, provider=None):
    """Every check-derived field of a record, so stored texts can be re-scored later."""
    checks, applicable, failed, failed_raw = evaluate_text(plan, text)
    unsupported, from_learner = number_diagnostics(plan, text)
    disclosed, disclosed_in_context = disclosure_diagnostics(plan, text)
    delivery = delivery_for(failed, error=error is not None)
    if provider == "fallback" and error is not None:
        delivery = "unavailable"
    fidelity = item_fidelity(plan, text)
    return {
        "validator_version": VALIDATOR_VERSION,
        "word_count": word_count(text),
        "question_marks": text.count("?"),
        "checks": checks,
        "applicable": applicable,
        "failed_checks": failed,
        "failed_checks_raw": failed_raw,
        "all_applicable_pass": not failed and error is None,
        "delivery": delivery,
        "delivery_raw": delivery_for(failed_raw, error=error is not None),
        "fallback_needed": delivery != "model",
        "fallback_available": fallback_available(plan),
        "unsupported_numbers": unsupported,
        "unsupported_numbers_from_learner": from_learner,
        "disclosed_fingerprints": disclosed,
        "disclosed_fingerprints_in_context": disclosed_in_context,
        "item_fidelity_applicable": fidelity.applicable,
        "item_fidelity_pass": fidelity.passed,
        "item_fidelity_reasons": list(fidelity.reasons),
        "item_fidelity_new_objects": list(fidelity.new_objects),
        "item_fidelity_coverage": round(fidelity.coverage, 3),
    }


def run_one(entry, case, ctx, *, model, tag):
    plan = case.plan
    messages = compile_prompt(plan)
    action = plan["teaching_action"]
    constraints = plan["constraints"]
    record = {
        "benchmark_id": ctx.config.get("benchmark_id", "response_benchmark_v1"),
        "harness_version": HARNESS_VERSION,
        "model_id": entry["id"],
        "provider": entry["provider"],
        "model": model,
        "decoding_id": decoding_id(entry),
        "decoding": entry.get("decoding") or {},
        "case_set": case.case_set,
        "case_id": case.case_id,
        "source": case.source,
        "move": action["move"],
        "difficulty": action.get("difficulty"),
        "pace": action.get("pace"),
        "assessment_mode": plan.get("assessment_mode"),
        "reveal_allowed": bool(constraints.get("reveal_allowed")),
        "max_words": constraints.get("max_words"),
        "max_questions": constraints.get("max_questions"),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "prompt_sha256": prompt_digest(messages),
        "prompt_chars": sum(len(message["content"]) for message in messages),
        "run_tag": tag,
        "started_at": utc_now(),
    }
    text, error, served = "", None, None
    finish_reason, think_stripped, attempts = None, False, 1
    usage = normalize_usage({})
    if entry["provider"] == "fallback":
        started = time.perf_counter()
        try:
            text = render_fallback(plan)
        except Exception as exc:  # FallbackUnavailable or missing content
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)
    else:
        try:
            result = call_model(
                entry, model, messages, ctx.keys, urlopen=ctx.urlopen, sleep=ctx.sleep
            )
        except CallError as exc:
            error, latency_ms = str(exc), None
        else:
            text, usage = result["text"], result["usage"]
            finish_reason, latency_ms = result["finish_reason"], result["latency_ms"]
            attempts, served = result["attempts"], result["served_model"]
            think_stripped = result["think_stripped"]
    cost = token_cost(usage, entry.get("pricing") or {}) if cost_basis(entry) == "tokens" else 0.0
    record.update({
        "text": text,
        "error": error,
        "finish_reason": finish_reason,
        "think_stripped": think_stripped,
        "served_model": served,
        "attempts": attempts,
        "latency_ms": latency_ms,
        "usage": usage,
        "over_visible_token_cap": usage["visible_output_tokens"] > PRD_OUTPUT_TOKEN_CAP,
        "cost_usd": round(cost, 8),
        "cost_basis": cost_basis(entry),
    })
    record.update(score_text(plan, text, error=error, provider=entry["provider"]))
    return record


@dataclass
class RunContext:
    config: dict
    keys: dict
    cap: float | None
    ledger: Ledger
    run_dir: Path
    urlopen: object = None
    sleep: object = time.sleep
    out: object = print
    max_usd: float | None = None
    invocation_spent: float = 0.0


def entry_by_id(config, model_id):
    for entry in config.get("models", []):
        if entry["id"] == model_id:
            return entry
    raise BenchmarkError(f"unknown model id: {model_id}")


def readiness(entry, ctx, *, managing=False):
    """(ok, reason) for running an entry now, without calling any paid endpoint."""
    if not entry.get("enabled", True):
        return False, f"disabled: {entry.get('disabled_reason', 'enabled: false')}"
    if entry["provider"] == "fallback":
        return True, "deterministic, free"
    key_name = entry.get("api_key_env")
    keys_file = ctx.config.get("keys_file", DEFAULT_KEYS_FILE)
    if key_name and key_name not in ctx.keys:
        return False, f"skipped: {key_name} is not set in {keys_file}"
    if is_paid(entry) and ctx.cap is None:
        return False, (
            f"skipped: no {ctx.config.get('spend_cap_env', 'TUTURING_BENCH_SPEND_CAP_USD')} "
            f"in {keys_file}, so no paid calls"
        )
    if entry.get("deployment") and not managing and not active_deployment_model(entry, ctx):
        return False, "skipped: no active deployment; rerun with --manage-deployment"
    return True, "ready"


def deployment_state_path(ctx, entry):
    return ctx.run_dir / "deployments" / f"{entry['id']}.json"


def active_deployment_model(entry, ctx):
    path = deployment_state_path(ctx, entry)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("deleted_verified") or state.get("state") in {"DELETED", "DELETING", "FAILED"}:
        return None
    return state.get("name")


def resolve_model(entry, ctx):
    model = entry.get("model")
    if entry.get("deployment"):
        name = active_deployment_model(entry, ctx)
        if not name:
            raise BenchmarkError(f"{entry['id']}: no active deployment")
        return (model or "{deployment}").replace("{deployment}", name)
    return model


def local_server_reachable(entry, urlopen=None, timeout=3):
    urlopen = urlopen or urllib.request.urlopen
    url = entry["base_url"].rstrip("/") + "/models"
    try:
        with urlopen(urllib.request.Request(url, method="GET"), timeout=timeout):
            return True
    except Exception:
        return False


def execute(entry, cases, ctx, *, model, tag, deadline=None):
    """Run cases with the entry's concurrency, enforcing the spend guards per call."""
    writer = ResultWriter(ctx.run_dir / entry["id"])
    cap = ctx.cap
    stop = threading.Event()
    paid = is_paid(entry)
    if paid and cap is None:
        raise BenchmarkError(f"{entry['id']}: paid entry without a spend cap")
    basis = cost_basis(entry)
    chars_per_token = float(ctx.config.get("chars_per_token_estimate", 3.0))
    records, stopped = [], []

    def work(case):
        if stop.is_set():
            return None
        reserve = 0.0
        if paid:
            if deadline is not None and time.time() > deadline:
                stop.set()
                stopped.append("deployment time budget reached")
                return None
            if basis == "tokens":
                prompt_chars = sum(len(m["content"]) for m in compile_prompt(case.plan))
                reserve = worst_case_cost(entry, prompt_chars, chars_per_token)
            label = f"{entry['id']} {case.key}"
            try:
                ctx.ledger.reserve(reserve, cap, label)
                if ctx.max_usd is not None and ctx.invocation_spent + reserve > ctx.max_usd:
                    ctx.ledger.release(reserve)
                    raise BudgetExceeded(f"{label}: --max-usd {ctx.max_usd:.2f} reached")
            except BudgetExceeded as exc:
                stop.set()
                stopped.append(str(exc))
                return None
        try:
            record = run_one(entry, case, ctx, model=model, tag=tag)
        finally:
            if reserve:
                ctx.ledger.release(reserve)
        writer.write(record)
        if paid and basis == "tokens":
            with ctx.ledger.lock:
                ctx.invocation_spent += record["cost_usd"]
            ctx.ledger.append({
                "event": "call",
                "model_id": entry["id"],
                "case": case.key,
                "usage": record["usage"],
                "actual_usd": record["cost_usd"],
                "worst_case_usd": round(reserve, 6),
                "error": record["error"],
                "run_tag": tag,
            })
        return record

    concurrency = max(1, int(entry.get("concurrency", 1)))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, case) for case in cases]
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record is not None:
                records.append(record)
                if done % 10 == 0 or done == len(futures):
                    errors = sum(1 for item in records if item["error"])
                    ctx.out(
                        f"  {entry['id']}: {len(records)}/{len(futures)} done, {errors} errors"
                    )
    if stopped:
        ctx.out(f"  {entry['id']}: stopped early: {stopped[0]}")
    return records, stopped


def pending_cases(entry, cases, ctx):
    done = load_results(ctx.run_dir, entry["id"])
    todo = [
        case for case in cases
        if not reusable(done.get(case.key), case, entry, compile_prompt(case.plan))
    ]
    return stratified_order(todo, seed=ctx.config.get("order_seed", 0))


def calibration(entry, ctx):
    """Mean cost and token use per case from this entry's completed records."""
    records = [r for r in load_results(ctx.run_dir, entry["id"]).values() if not r["error"]]
    if not records:
        return None
    n = len(records)
    return {
        "n": n,
        "mean_cost_usd": sum(r["cost_usd"] for r in records) / n,
        "mean_output_tokens": sum(r["usage"]["output_tokens"] for r in records) / n,
        "mean_input_tokens": sum(r["usage"]["input_tokens"] for r in records) / n,
    }


def fit_to_budget(cases, per_case_usd, budget_usd, *, min_cases):
    """Largest evenly stratified prefix whose expected cost fits; None if below min_cases."""
    if per_case_usd <= 0:
        return cases
    affordable = int(budget_usd // per_case_usd)
    if affordable >= len(cases):
        return cases
    if affordable < min_cases:
        return None
    return cases[:affordable]


def run_entry(entry, cases, ctx, *, pilot=None, skip_pilot=False, model=None, deadline=None,
              seconds_per_case=None):
    """Run one entry. Paid entries need a pilot first and are subsampled to fit the cap."""
    model = model or resolve_model(entry, ctx)
    todo = pending_cases(entry, cases, ctx)
    ctx.out(f"{entry['id']}: {len(cases)} selected cases, {len(todo)} still to run")
    if not todo:
        return []
    if pilot:
        batch = todo[:pilot]
        records, _ = execute(entry, batch, ctx, model=model, tag="pilot", deadline=deadline)
        summary = pilot_summary(entry, ctx, len(todo) - len(batch), seconds_per_case)
        ctx.out(f"  pilot summary: {json.dumps(summary, sort_keys=True)}")
        return records
    if is_paid(entry) and cost_basis(entry) == "tokens":
        calib = calibration(entry, ctx)
        if calib is None and not skip_pilot:
            raise BenchmarkError(f"{entry['id']}: run a pilot first (--pilot 3)")
        per_case = (calib or {}).get("mean_cost_usd") or worst_case_cost(entry, 2000, 3.0)
        per_case *= float(ctx.config.get("estimate_safety_factor", 1.3))
        budget = ctx.cap - ctx.ledger.committed()
        if ctx.max_usd is not None:
            budget = min(budget, ctx.max_usd - ctx.invocation_spent)
        selected = fit_to_budget(
            todo, per_case, budget, min_cases=int(ctx.config.get("min_subsample_cases", 20))
        )
        decision = {
            "event": "estimate",
            "model_id": entry["id"],
            "pending_cases": len(todo),
            "expected_usd_per_case": round(per_case, 6),
            "expected_usd": round(per_case * len(todo), 4),
            "budget_usd": round(budget, 4),
            "selected_cases": None if selected is None else len(selected),
        }
        ctx.ledger.append(decision)
        ctx.out(f"  estimate: {json.dumps(decision, sort_keys=True)}")
        if selected is None:
            raise BenchmarkError(
                f"{entry['id']}: even a minimal subsample does not fit the remaining budget"
            )
        todo = selected
    records, _ = execute(entry, todo, ctx, model=model, tag="full", deadline=deadline)
    return records


def pilot_summary(entry, ctx, remaining, seconds_per_case=None):
    calib = calibration(entry, ctx) or {}
    summary = {
        "event": "pilot",
        "model_id": entry["id"],
        "completed_cases": calib.get("n", 0),
        "remaining_cases": remaining,
        "mean_output_tokens": round(calib.get("mean_output_tokens", 0.0), 1),
    }
    if cost_basis(entry) == "tokens":
        per_case = calib.get("mean_cost_usd", 0.0)
        summary["mean_usd_per_case"] = round(per_case, 6)
        summary["extrapolated_remaining_usd"] = round(per_case * remaining, 4)
    if seconds_per_case is not None:
        summary["seconds_per_case"] = round(seconds_per_case, 2)
        summary["extrapolated_remaining_usd"] = round(
            seconds_per_case * remaining / 3600 * gpu_usd_per_hour(entry), 4
        )
    summary["committed_usd"] = round(ctx.ledger.committed(), 4)
    summary["cap_usd"] = ctx.cap
    ctx.ledger.append(summary)
    return summary


def run_with_deployment(entry, cases, ctx, *, pilot=3):
    """Create the deployment, pilot, run the rest within budget, then always delete it."""
    from tools.response.fireworks_deployment import DeploymentError, DeploymentRun, FireworksAPI

    spec = entry["deployment"]
    rate = gpu_usd_per_hour(entry)
    max_minutes = float(spec.get("max_minutes", 60))
    reserve = rate * max_minutes / 60
    committed = ctx.ledger.committed()
    if ctx.cap is None or committed + reserve > ctx.cap:
        raise BenchmarkError(
            f"{entry['id']}: committed ${committed:.2f} + deployment reserve ${reserve:.2f} "
            f"({max_minutes:.0f} min at ${rate:.2f}/h) exceeds the cap {ctx.cap}"
        )
    api = FireworksAPI(ctx.keys[entry["api_key_env"]], urlopen=ctx.urlopen)
    run = DeploymentRun(
        api, entry, deployment_state_path(ctx, entry), ctx.ledger, out=ctx.out, sleep=ctx.sleep
    )
    try:
        run.create()
    except DeploymentError as exc:
        raise BenchmarkError(f"{entry['id']}: {exc}") from None
    try:
        run.wait_ready(timeout_s=float(spec.get("ready_timeout_s", 2400)))
        model = resolve_model(entry, ctx)
        run.warm_up(lambda: call_model(
            {**entry, "retry": {"max_attempts": 1}},
            model,
            [{"role": "user", "content": "Reply with the word ready."}],
            ctx.keys,
            urlopen=ctx.urlopen,
            sleep=ctx.sleep,
        ), timeout_s=float(spec.get("warmup_timeout_s", 2400)))
        deadline = run.create_requested_epoch + max_minutes * 60 - spec.get("delete_buffer_s", 300)
        todo = pending_cases(entry, cases, ctx)
        pilot_batch, remaining = todo[:pilot], todo[pilot:]
        ctx.out(f"{entry['id']}: {len(todo)} cases to run; pilot of {len(pilot_batch)}")
        pilot_records, stopped = execute(
            entry, pilot_batch, ctx, model=model, tag="pilot", deadline=deadline
        )
        if stopped or not remaining:
            return pilot_records
        if pilot_records and all(r["error"] for r in pilot_records):
            ctx.out(f"  {entry['id']}: every pilot call failed; stopping before the full run")
            return pilot_records
        latencies = [r["latency_ms"] for r in pilot_records if r.get("latency_ms")]
        mean_latency_s = (sum(latencies) / len(latencies) / 1000) if latencies else 30.0
        concurrency = max(1, int(entry.get("concurrency", 1)))
        safety = float(ctx.config.get("estimate_safety_factor", 1.3))
        effective = mean_latency_s / min(concurrency, len(remaining)) * safety
        summary = pilot_summary(entry, ctx, len(remaining), seconds_per_case=effective)
        ctx.out(f"  pilot summary: {json.dumps(summary, sort_keys=True)}")
        time_left = deadline - time.time()
        delete_reserve = rate * spec.get("delete_buffer_s", 300) / 3600
        budget_left = ctx.cap - ctx.ledger.committed() - delete_reserve
        cost_left_s = budget_left / rate * 3600 if rate else float("inf")
        affordable = int(max(min(time_left, cost_left_s), 0) // max(effective, 0.01))
        if affordable < len(remaining):
            min_cases = int(ctx.config.get("min_subsample_cases", 20))
            ctx.ledger.append({
                "event": "estimate", "model_id": entry["id"], "pending_cases": len(remaining),
                "selected_cases": affordable if affordable >= min_cases else 0,
                "reason": "deployment time or budget limit",
            })
            if affordable < min_cases:
                ctx.out(f"  {entry['id']}: only {affordable} more cases fit; stopping")
                return pilot_records
            remaining = remaining[:affordable]
        records, _ = execute(entry, remaining, ctx, model=model, tag="full", deadline=deadline)
        return pilot_records + records
    except DeploymentError as exc:
        raise BenchmarkError(f"{entry['id']}: {exc}") from None
    finally:
        run.delete_and_verify()


# ---------------------------------------------------------------------------
# CLI


def make_context(args, config):
    keys_file = args.keys_file or config.get("keys_file", DEFAULT_KEYS_FILE)
    config = {**config, "keys_file": keys_file}
    keys = load_keys(keys_file)
    run_dir = Path(config.get("run_dir", "experiments/response/benchmark_v1/runs"))
    ledger = Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments")
    return RunContext(
        config=config,
        keys=keys,
        cap=spend_cap(keys, config),
        ledger=ledger,
        run_dir=run_dir,
        max_usd=getattr(args, "max_usd", None),
    )


def selected_entries(config, names):
    entries = config.get("models", [])
    if not names:
        return [entry for entry in entries if entry.get("enabled", True)]
    return [entry_by_id(config, name) for name in names]


def command_plan(args, config, ctx):
    cases, notes = load_all_cases(config, args.case_sets)
    ctx.out(f"Keys file: {config['keys_file']} ({'found' if ctx.keys else 'absent'})")
    cap_text = f"${ctx.cap:.2f}" if ctx.cap is not None else "none (no paid calls)"
    ctx.out(f"Spend cap: {cap_text}; committed so far ${ctx.ledger.committed():.4f}")
    ctx.out("Case sets:")
    for name, info in notes.items():
        ctx.out(f"  {name}: {info['n']} cases" + (f" ({info['note']})" if info["note"] else ""))
    chars = sum(sum(len(m["content"]) for m in compile_prompt(c.plan)) for c in cases)
    ctx.out(f"Models ({len(cases)} cases, {chars} prompt characters in total):")
    chars_per_token = float(config.get("chars_per_token_estimate", 3.0))
    for entry in config.get("models", []):
        ok, reason = readiness(entry, ctx, managing=True)
        line = f"  {entry['id']:<28} {'RUN ' if ok else 'SKIP'} {reason}"
        if is_paid(entry) and cost_basis(entry) == "tokens":
            worst = sum(
                worst_case_cost(entry, sum(len(m["content"]) for m in compile_prompt(c.plan)),
                                chars_per_token)
                for c in cases
            )
            line += f"; worst-case ${worst:.2f} for all cases"
        elif cost_basis(entry) == "gpu_time":
            spec = entry["deployment"]
            line += (
                f"; GPU reserve ${gpu_usd_per_hour(entry) * spec.get('max_minutes', 60) / 60:.2f}"
                f" ({spec.get('max_minutes', 60)} min max)"
            )
        ctx.out(line)
    return 0


def command_run(args, config, ctx):
    cases, notes = load_all_cases(config, args.case_sets)
    for name, info in notes.items():
        ctx.out(f"case set {name}: {info['n']}" + (f" ({info['note']})" if info["note"] else ""))
    if args.limit:
        cases = stratified_order(cases, seed=config.get("order_seed", 0))[: args.limit]
    status = 0
    for entry in selected_entries(config, args.models):
        ok, reason = readiness(entry, ctx, managing=args.manage_deployment)
        if not ok:
            ctx.out(f"{entry['id']}: {reason}")
            continue
        if entry["provider"] != "fallback" and cost_basis(entry) == "free":
            if not local_server_reachable(entry, ctx.urlopen):
                ctx.out(f"{entry['id']}: skipped: server not reachable at {entry['base_url']}; "
                        f"start it with: {entry.get('server_command', 'see runbook')}")
                continue
        try:
            if entry.get("deployment"):
                if not args.manage_deployment:
                    run_entry(entry, cases, ctx, pilot=args.pilot, skip_pilot=args.skip_pilot)
                else:
                    run_with_deployment(entry, cases, ctx, pilot=args.pilot or 3)
            else:
                run_entry(entry, cases, ctx, pilot=args.pilot, skip_pilot=args.skip_pilot)
        except (BenchmarkError, BudgetExceeded) as exc:
            ctx.out(f"{entry['id']}: {exc}")
            status = 1
    ctx.out(f"Committed spend after run: ${ctx.ledger.committed():.4f}"
            + (f" of ${ctx.cap:.2f}" if ctx.cap is not None else ""))
    return status


def rescore(entry, cases, ctx, *, force=False):
    """Re-check this entry's stored texts under the current validator; no model calls."""
    case_by_key = {case.key: case for case in cases}
    latest = load_results(ctx.run_dir, entry["id"])
    if not latest:
        return 0, 0, 0
    writer = ResultWriter(ctx.run_dir / entry["id"])
    rescored = current = missing = 0
    for key, record in sorted(latest.items()):
        case = case_by_key.get(key)
        if case is None:
            missing += 1
            continue
        if record.get("validator_version") == VALIDATOR_VERSION and not force:
            current += 1
            continue
        updated = dict(record)
        updated.update(
            score_text(case.plan, record["text"], error=record["error"], provider=entry["provider"])
        )
        updated["run_tag"] = f"rescore_{VALIDATOR_VERSION}"
        updated["rescored_at"] = utc_now()
        updated["rescored_from_validator"] = record.get("validator_version")
        writer.write(updated)
        rescored += 1
    return rescored, current, missing


def command_rescore(args, config, ctx):
    cases, _ = load_all_cases(config, args.case_sets)
    for entry in config.get("models", []):
        if args.models and entry["id"] not in args.models:
            continue
        rescored, current, missing = rescore(entry, cases, ctx, force=args.force)
        if rescored or current or missing:
            ctx.out(
                f"{entry['id']}: rescored {rescored} under {VALIDATOR_VERSION}, "
                f"{current} already current, {missing} not in the loaded case sets"
            )
    return 0


def command_report(args, config, ctx):
    from tools.response.benchmark_report import build_report

    paths = build_report(config, ctx)
    for path in paths:
        ctx.out(f"wrote {path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--keys-file", help="override the keys file named in the config")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="show cases, readiness and cost estimates; no calls")
    plan.add_argument("--case-sets", type=lambda s: s.split(","))
    run = sub.add_parser("run", help="run model entries")
    run.add_argument("--models", type=lambda s: s.split(","))
    run.add_argument("--case-sets", type=lambda s: s.split(","))
    run.add_argument("--pilot", type=int, help="run only this many new cases, then estimate")
    run.add_argument("--limit", type=int, help="evenly stratified subset of the cases")
    run.add_argument("--skip-pilot", action="store_true")
    run.add_argument("--max-usd", type=float, help="extra cap for this invocation")
    run.add_argument(
        "--manage-deployment",
        action="store_true",
        help="create the entry's GPU deployment, run, and always delete it",
    )
    sub.add_parser("report", help="aggregate runs into CSVs, report.md and the review sheet")
    rescore_parser = sub.add_parser(
        "rescore", help="re-check stored texts under the current validator, no model calls"
    )
    rescore_parser.add_argument("--models", type=lambda s: s.split(","))
    rescore_parser.add_argument("--case-sets", type=lambda s: s.split(","))
    rescore_parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ctx = make_context(args, config)
    commands = {
        "plan": command_plan, "run": command_run, "report": command_report,
        "rescore": command_rescore,
    }
    return commands[args.command](args, ctx.config, ctx)


if __name__ == "__main__":
    raise SystemExit(main())
