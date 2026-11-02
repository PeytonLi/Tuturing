"""Label every usable MathDial teacher turn with one of the eight moves or EXCLUDE.

The labels come from an OpenAI chat model through the Chat Completions API,
never from a person. Three stages share one config:

    python -m tools.response.label_mathdial_moves pilot --model gpt-5-nano
    python -m tools.response.label_mathdial_moves full --model gpt-5-nano
    python -m tools.response.label_mathdial_moves report --model gpt-5-nano

``pilot`` labels the calibration turns (the 200 drafted labels) and measures
agreement; ``full`` labels every usable turn, resumably and under the spend
cap, fitting a subset to the budget when the estimate exceeds it; ``report``
writes the committed labels file (ids and labels only) and the manifest. The
API key is read from the key file named in the config and is never printed or
written. Reasons, raw outputs and the ledger stay in the ignored run directory.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import yaml

from tools.response.prepare_sft import (
    MOVE_ORDER,
    assign_qid_splits,
    compute_raw_counts,
    iter_teacher_turns,
    load_config,
    load_source,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL_VERSION = "tools.response.label_mathdial_moves v1"
LABELS = (*MOVE_ORDER, "EXCLUDE")
DEFAULT_CONFIG = "configs/response/mathdial_moves_v1.yaml"
RETRY_NUDGE = (
    "Your reply did not follow the format. Reply again with exactly two lines: "
    "LABEL: <label> and REASON: <one short line>."
)
_LABEL_LINE = re.compile(r"^\W*LABEL\W*:?\W*([A-Z_]+)\W*$", re.MULTILINE)
_REASON_LINE = re.compile(r"^\W*REASON\W*:?\W*(.+?)\s*$", re.MULTILINE)
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


# --------------------------------------------------------------------------- prompt


def build_messages(turn, system_prompt):
    student = (turn.get("context_last_student") or "").strip() or "(no student message yet)"
    teacher = (turn.get("teacher_text") or "").strip()
    user = (
        f"<student_last_message>\n{student}\n</student_last_message>\n\n"
        f"<tutor_turn>\n{teacher}\n</tutor_turn>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def parse_label(text):
    """Return (label, reason) from a strict two-line reply, or (None, None)."""
    if not isinstance(text, str):
        return None, None
    label_match = _LABEL_LINE.search(text)
    if not label_match or label_match.group(1) not in LABELS:
        return None, None
    reason_match = _REASON_LINE.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    return label_match.group(1), reason


def prompt_sha256(system_prompt):
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def few_shot_examples(calibration, per_label, seed, max_words=40):
    """Pick drafted rows as examples, clear ones first; return (text block, ids held out)."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for example_id, entry in calibration.items():
        if entry.get("teacher_text") and len(entry["teacher_text"].split()) <= max_words:
            by_label[entry["draft"]].append((example_id, entry))
    lines = ["Examples of labeled tutor turns, with the reason that applies the rubric:"]
    ids = []
    for label in LABELS:
        bucket = by_label.get(label, [])
        clear = [item for item in bucket if item[1].get("ambiguous") == "no"]
        unclear = [item for item in bucket if item[1].get("ambiguous") != "no"]
        rng.shuffle(clear)
        rng.shuffle(unclear)
        for example_id, entry in (clear + unclear)[:per_label]:
            text = " ".join(entry["teacher_text"].split())
            lines.append(f'- "{text}" -> {label} ({entry.get("reason") or "see rubric"})')
            ids.append(example_id)
    return "\n".join(lines), ids


def load_prompt(config, calibration=None):
    """Return (prompt_version, system prompt, held-out ids) from the versioned prompt file.

    With ``few_shot`` in the config, drafted calibration rows are inserted as examples
    and their ids are held out of every agreement computation.
    """
    path = ROOT / config["prompt_file"]
    version = path.stem
    text = path.read_text(encoding="utf-8").strip()
    few_shot = config.get("few_shot") or {}
    ids = []
    if few_shot.get("per_label") and calibration:
        seed = int(few_shot.get("seed", 20261102))
        block, ids = few_shot_examples(calibration, int(few_shot["per_label"]), seed)
        text = text.replace("{examples}", block) if "{examples}" in text else f"{text}\n\n{block}"
        version = f"{version}-fs{few_shot['per_label']}-s{seed}"
    else:
        text = text.replace("{examples}\n\n", "")
    return version, text, ids


# --------------------------------------------------------------------------- key and client


def load_api_key(path, name):
    """Read NAME=value from a dotenv-style file; the value is returned, never logged."""
    path = Path(path).expanduser()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, separator, value = line.partition("=")
        if separator and key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise KeyError(f"{name} is not set in {path}")


class ChatClient:
    """Minimal Chat Completions client that adapts to parameters a model rejects."""

    def __init__(self, base_url, api_key, model, *, temperature=0, reasoning_effort="minimal",
                 max_completion_tokens=120, timeout=60, max_attempts=6, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.params = {}
        if temperature is not None:
            self.params["temperature"] = temperature
        if reasoning_effort is not None:
            self.params["reasoning_effort"] = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.adjustments = []
        self._lock = threading.Lock()

    def complete(self, messages):
        """Return (content, usage); raise RuntimeError after the retry budget."""
        attempt = 0
        while True:
            attempt += 1
            with self._lock:
                params = dict(self.params)
            body = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_completion_tokens,
                **params,
            }
            request = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "tuturing-mathdial-move-labeler/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                if error.code == 400 and self._adapt_parameters(detail):
                    continue
                if error.code in _RETRYABLE_STATUS and attempt < self.max_attempts:
                    self.sleep(min(30, 2 ** attempt))
                    continue
                raise RuntimeError(f"HTTP {error.code}: {detail[:200]}") from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt < self.max_attempts:
                    self.sleep(min(30, 2 ** attempt))
                    continue
                raise RuntimeError(f"request failed: {error}") from None
            choice = payload["choices"][0]
            return (choice["message"].get("content") or ""), (payload.get("usage") or {})

    def _adapt_parameters(self, detail):
        """Drop or change a parameter the model rejects; return True when the request changed."""
        message = detail
        try:
            message = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        rejected = re.search(r"does not support '?([A-Za-z0-9.]+)'?", message)
        rejected_value = rejected.group(1) if rejected else None
        with self._lock:
            if "'temperature'" in message:
                if "temperature" in self.params:
                    self.params.pop("temperature")
                    self.adjustments.append(f"temperature dropped: {message}")
                return True  # a concurrent request already dropped it: just retry
            if "'reasoning_effort'" in message:
                current = self.params.get("reasoning_effort")
                if current is None or (rejected_value and rejected_value != str(current)):
                    return True  # already changed by a concurrent request: just retry
                supported = []
                if "Supported values" in message:
                    supported = re.findall(r"'([a-z]+)'", message.split("Supported values", 1)[1])
                for candidate in ("none", "minimal", "low"):
                    if candidate in supported and candidate != current:
                        self.params["reasoning_effort"] = candidate
                        self.adjustments.append(f"reasoning_effort set to {candidate}: {message}")
                        return True
                self.params.pop("reasoning_effort")
                self.adjustments.append(f"reasoning_effort dropped: {message}")
                return True
        return False


# --------------------------------------------------------------------------- spend and storage


def request_cost(usage, prices):
    prompt = int(usage.get("prompt_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return (
        (prompt - cached) * prices["input_per_m"]
        + cached * prices["cached_input_per_m"]
        + completion * prices["output_per_m"]
    ) / 1_000_000


class SpendTracker:
    """Append-only USD ledger with a hard cap, shared by worker threads."""

    def __init__(self, ledger_path, cap_usd, prices_by_model):
        self.path = Path(ledger_path)
        self.cap = float(cap_usd)
        self.prices_by_model = prices_by_model
        self.spent = 0.0
        self.requests = 0
        self.by_stage_model = defaultdict(float)
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                self.spent += float(entry.get("cost_usd") or 0.0)
                self.requests += 1
                self.by_stage_model[f"{entry.get('stage')}:{entry.get('model')}"] += float(
                    entry.get("cost_usd") or 0.0
                )

    def remaining(self):
        return self.cap - self.spent

    def allows(self, estimate_usd):
        with self._lock:
            return self.spent + estimate_usd <= self.cap

    def record(self, stage, model, example_id, usage, attempt=1):
        cost = request_cost(usage, self.prices_by_model[model])
        details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        with self._lock:
            self.spent += cost
            self.requests += 1
            self.by_stage_model[f"{stage}:{model}"] += cost
            entry = {
                "time": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "stage": stage,
                "model": model,
                "example_id": example_id,
                "attempt": attempt,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "cached_tokens": int(details.get("cached_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
                "cost_usd": round(cost, 8),
                "cumulative_usd": round(self.spent, 6),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
        return cost

    def summary(self):
        return {
            "cap_usd": self.cap,
            "spent_usd": round(self.spent, 6),
            "remaining_usd": round(self.remaining(), 6),
            "requests": self.requests,
            "by_stage_model": {
                key: round(value, 6) for key, value in sorted(self.by_stage_model.items())
            },
        }


class LabelStore:
    """JSONL store of label records keyed by example_id; appends survive restarts."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["example_id"]] = record

    def add(self, record):
        with self._lock:
            self.records[record["example_id"]] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- labeling


def label_one(client, turn, system_prompt):
    """Label one turn; a malformed reply is retried once, then becomes EXCLUDE."""
    messages = build_messages(turn, system_prompt)
    content, usage = client.complete(messages)
    usages = [usage]
    outputs = [content]
    label, reason = parse_label(content)
    status = "labeled"
    if label is None:
        retry = [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": RETRY_NUDGE},
        ]
        content, usage = client.complete(retry)
        usages.append(usage)
        outputs.append(content)
        label, reason = parse_label(content)
        status = "labeled_after_retry"
        if label is None:
            label, reason, status = "EXCLUDE", "malformed output after one retry", "malformed"
    return {"label": label, "reason": reason, "status": status}, usages, outputs


def _sum_usage(usages):
    total = Counter()
    for usage in usages:
        total["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or {}
        total["cached_tokens"] += int(prompt_details.get("cached_tokens") or 0)
        total["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        total["reasoning_tokens"] += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        )
    return dict(total)


def label_turns(turns, client, store, tracker, *, stage, estimate_per_turn, system_prompt,
                prompt_version, concurrency=8, chunk_size=100, log=print):
    """Label the turns not yet in the store; stop cleanly when the cap would be exceeded."""
    pending = [turn for turn in turns if turn["example_id"] not in store.records]
    outcome = {"requested": len(turns), "already_done": len(turns) - len(pending),
               "labeled": 0, "errors": 0, "skipped_for_cap": 0, "error_messages": []}
    stop = threading.Event()

    def work(turn):
        if stop.is_set():
            return "skipped"
        estimate = estimate_per_turn(turn)
        if not tracker.allows(estimate):
            stop.set()
            return "skipped"
        result, usages, outputs = label_one(client, turn, system_prompt)
        for attempt, usage in enumerate(usages, 1):
            tracker.record(stage, client.model, turn["example_id"], usage, attempt)
        store.add({
            "example_id": turn["example_id"],
            "split": turn["split"],
            "source_act": turn["source_act"],
            "label": result["label"],
            "status": result["status"],
            "reason": result["reason"],
            "model": client.model,
            "prompt_version": prompt_version,
            "usage": _sum_usage(usages),
            "outputs": outputs,
            "words": len((turn.get("context_last_student") or "").split())
            + len((turn.get("teacher_text") or "").split()),
        })
        return "labeled"

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for start in range(0, len(pending), chunk_size):
            if stop.is_set():
                break
            chunk = pending[start:start + chunk_size]
            futures = [pool.submit(work, turn) for turn in chunk]
            for future in as_completed(futures):
                try:
                    state = future.result()
                except RuntimeError as error:
                    outcome["errors"] += 1
                    if len(outcome["error_messages"]) < 10:
                        outcome["error_messages"].append(str(error)[:200])
                    continue
                outcome[{"labeled": "labeled", "skipped": "skipped_for_cap"}[state]] += 1
            log(
                f"[{stage}:{client.model}] done "
                f"{min(start + chunk_size, len(pending))}/{len(pending)} "
                f"labeled={outcome['labeled']} errors={outcome['errors']} "
                f"spent=${tracker.spent:.4f} of ${tracker.cap:.2f}"
            )
    if stop.is_set():
        outcome["skipped_for_cap"] = len(pending) - outcome["labeled"] - outcome["errors"]
    return outcome


# --------------------------------------------------------------------------- calibration


def load_calibration(path, label_column):
    """Return {example_id: {"draft": label, "source_act": act}} from the drafted audit sheet."""
    calibration = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            calibration[row["example_id"]] = {
                "draft": row[label_column],
                "source_act": row.get("source_act", ""),
                "teacher_text": row.get("teacher_text", ""),
                "reason": row.get("model_rationale", ""),
                "ambiguous": row.get("model_ambiguous", ""),
            }
    return calibration


def act_mapping_agreement(calibration, move_map):
    """Agreement between a per-act map and the drafted turn-level labels, per source act."""
    mappings = move_map.get("mappings") or {}
    per_act = defaultdict(lambda: {"agree": 0, "n": 0})
    for entry in calibration.values():
        mapping = mappings.get(entry["source_act"]) or {}
        mapped = "EXCLUDE" if mapping.get("exclude") else mapping.get("move")
        bucket = per_act[entry["source_act"] or "none"]
        bucket["n"] += 1
        bucket["agree"] += int(mapped == entry["draft"])
    total_agree = sum(item["agree"] for item in per_act.values())
    total = sum(item["n"] for item in per_act.values())
    return {
        "overall": {
            "agree": total_agree, "n": total,
            "rate": round(total_agree / total, 4) if total else None,
        },
        "per_act": {
            act: {**item, "rate": round(item["agree"] / item["n"], 4) if item["n"] else None}
            for act, item in sorted(per_act.items())
        },
    }


def label_agreement(records, calibration):
    """Agreement of model labels with the drafted labels, overall, per act and as a crosstab."""
    per_act = defaultdict(lambda: {"agree": 0, "n": 0})
    crosstab = defaultdict(Counter)
    for example_id, entry in calibration.items():
        record = records.get(example_id)
        if record is None:
            continue
        bucket = per_act[entry["source_act"] or "none"]
        bucket["n"] += 1
        bucket["agree"] += int(record["label"] == entry["draft"])
        crosstab[entry["draft"]][record["label"]] += 1
    total_agree = sum(item["agree"] for item in per_act.values())
    total = sum(item["n"] for item in per_act.values())
    return {
        "overall": {
            "agree": total_agree, "n": total,
            "rate": round(total_agree / total, 4) if total else None,
        },
        "per_act": {
            act: {**item, "rate": round(item["agree"] / item["n"], 4) if item["n"] else None}
            for act, item in sorted(per_act.items())
        },
        "draft_vs_model": {
            draft: dict(sorted(counts.items())) for draft, counts in sorted(crosstab.items())
        },
    }


def wilson_interval(agree, n, z=1.96):
    """Wilson score interval for an agreement rate, as [low, high], or None when n is 0."""
    if not n:
        return None
    rate = agree / n
    denominator = 1 + z * z / n
    centre = rate + z * z / (2 * n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return [round((centre - half) / denominator, 4), round((centre + half) / denominator, 4)]


def repeat_consistency(first, second, ids):
    """How often two label stores give the same label on the turns both labeled."""
    shared = [example_id for example_id in ids if example_id in first and example_id in second]
    same = sum(first[example_id]["label"] == second[example_id]["label"] for example_id in shared)
    return {
        "rows": len(shared),
        "same_label": same,
        "rate": round(same / len(shared), 4) if shared else None,
    }


def fit_tokens(records):
    """Least-squares fit prompt_tokens = a + b * words over labeled records, plus mean output."""
    points = [
        (record["words"], record["usage"]["prompt_tokens"])
        for record in records.values()
        if record.get("usage") and record["usage"].get("prompt_tokens")
    ]
    if not points:
        return {"intercept": 600.0, "per_word": 1.6, "completion_mean": 80.0, "n": 0}
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var if var else 1.5
    slope = max(slope, 1.0)
    intercept = max(mean_y - slope * mean_x, 0.0)
    completion = [
        record["usage"]["completion_tokens"] for record in records.values() if record.get("usage")
    ]
    prompt_total = sum(record["usage"]["prompt_tokens"] for record in records.values())
    cached_total = sum(record["usage"].get("cached_tokens", 0) for record in records.values())
    return {
        "intercept": round(intercept, 2),
        "per_word": round(slope, 4),
        "completion_mean": round(sum(completion) / len(completion), 2),
        "cached_fraction": round(cached_total / prompt_total, 4) if prompt_total else 0.0,
        "n": n,
    }


def make_estimator(fit, prices, safety=1.15):
    def estimate(turn):
        words = len((turn.get("context_last_student") or "").split()) + len(
            (turn.get("teacher_text") or "").split()
        )
        prompt_tokens = fit["intercept"] + fit["per_word"] * words
        cached = prompt_tokens * fit.get("cached_fraction", 0.0)
        usage = {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens": fit["completion_mean"],
        }
        return request_cost(usage, prices) * safety
    return estimate


def select_within_budget(turns, estimate_per_turn, budget_usd, seed=20261102,
                         train_per_validation=4):
    """Choose turns to label when the estimate exceeds the budget.

    Order: every test-split turn first; then train and validation turns in a
    per-source-act round robin (stratified), taking ``train_per_validation``
    train picks for every validation pick, until the budget is used.
    """
    rng = random.Random(seed)
    buckets = {"train": defaultdict(list), "validation": defaultdict(list)}
    ordered = []
    for turn in turns:
        if turn["split"] == "test":
            ordered.append(turn)
        elif turn["split"] in buckets:
            buckets[turn["split"]][turn["source_act"] or "none"].append(turn)
    cycles = {}
    for split, by_act in buckets.items():
        for bucket in by_act.values():
            rng.shuffle(bucket)
        cycles[split] = sorted(by_act)
    position = {"train": 0, "validation": 0}

    def pick(split):
        by_act = buckets[split]
        acts = cycles[split]
        for _ in range(len(acts)):
            act = acts[position[split] % len(acts)]
            position[split] += 1
            if by_act[act]:
                return by_act[act].pop()
        return None

    while any(by_act[act] for by_act in buckets.values() for act in by_act):
        for _ in range(train_per_validation):
            turn = pick("train")
            if turn is not None:
                ordered.append(turn)
        turn = pick("validation")
        if turn is not None:
            ordered.append(turn)
    selected = []
    total = 0.0
    for turn in ordered:
        cost = estimate_per_turn(turn)
        if total + cost > budget_usd:
            break
        selected.append(turn)
        total += cost
    return selected, total


# --------------------------------------------------------------------------- stages


def source_turns(config):
    """Every labelable MathDial teacher turn: non-empty and within the word cap."""
    source_config = load_config(ROOT / config["source_config"])
    rows_by_official, files, _ = load_source(source_config)
    raw_counts = compute_raw_counts(rows_by_official)
    assignments = assign_qid_splits(rows_by_official, source_config.get("split_salt", "mathdial"))
    turns = []
    skipped = Counter()
    for turn in iter_teacher_turns(rows_by_official, source_config, assignments):
        if not turn["teacher_text"]:
            skipped["empty_text"] += 1
            continue
        if len(turn["teacher_text"].split()) > source_config["max_words"]:
            skipped["over_max_words"] += 1
            continue
        turns.append({
            key: turn[key] for key in (
                "example_id", "split", "source_act", "teacher_text", "context_last_student",
            )
        })
    universe = {
        "teacher_turns_total": raw_counts["teacher_turns"],
        "skipped": dict(sorted(skipped.items())),
        "labelable": len(turns),
        "by_split_and_act": {
            split: dict(sorted(Counter(
                turn["source_act"] or "none" for turn in turns if turn["split"] == split
            ).items()))
            for split in ("train", "validation", "test")
        },
    }
    source = {
        "source_id": source_config["source_id"],
        "revision": source_config["revision"],
        "config": {
            "path": config["source_config"],
            "sha256": sha256_file(ROOT / config["source_config"]),
        },
        "files": files,
    }
    return turns, universe, source


def prices_for(config, model):
    for candidate in config["candidates"]:
        if candidate["model"] == model:
            return candidate
    raise KeyError(f"{model} is not a configured candidate")


def make_client(config, model, api_key, base_url=None):
    return ChatClient(
        base_url or config["base_url"],
        api_key,
        model,
        temperature=config.get("temperature", 0),
        reasoning_effort=config.get("reasoning_effort", "minimal"),
        max_completion_tokens=config.get("max_completion_tokens", 120),
        timeout=config.get("timeout_seconds", 60),
    )


def run_dir(config):
    path = ROOT / config["run_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def tracker_for(config):
    prices = {candidate["model"]: candidate for candidate in config["candidates"]}
    return SpendTracker(run_dir(config) / "ledger.jsonl", config["spend_cap_usd"], prices)


def pilot(config, model, turns, api_key, *, base_url=None, log=print):
    """Label the calibration turns with one model and report agreement with the drafts."""
    calibration = load_calibration(
        ROOT / config["calibration_csv"], config["calibration_label_column"]
    )
    by_id = {turn["example_id"]: turn for turn in turns}
    missing = [example_id for example_id in calibration if example_id not in by_id]
    if missing:
        raise KeyError(f"{len(missing)} calibration turns are not labelable, e.g. {missing[:3]}")
    prompt_version, system_prompt, few_shot_ids = load_prompt(config, calibration)
    evaluated = {
        example_id: entry for example_id, entry in calibration.items()
        if example_id not in set(few_shot_ids)
    }
    selected = [by_id[example_id] for example_id in evaluated]
    tag = config.get("run_tag") or model
    client = make_client(config, model, api_key, base_url)
    store = LabelStore(run_dir(config) / f"pilot_{prompt_version}_{tag}.jsonl")
    tracker = tracker_for(config)
    prices = prices_for(config, model)
    estimator = make_estimator(
        {"intercept": 1400.0, "per_word": 1.6, "completion_mean": 100.0, "cached_fraction": 0.0},
        prices,
    )
    outcome = label_turns(
        selected, client, store, tracker, stage="pilot", estimate_per_turn=estimator,
        system_prompt=system_prompt, prompt_version=prompt_version,
        concurrency=config.get("concurrency", 8), log=log,
    )
    agreement = label_agreement(store.records, evaluated)
    cost = sum(request_cost(record["usage"], prices) for record in store.records.values())
    summary = {
        "stage": "pilot",
        "model": model,
        "run_tag": tag,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(system_prompt),
        "few_shot_ids": few_shot_ids,
        "n_evaluated": len(evaluated),
        "prices_usd_per_m": {
            key: prices[key] for key in ("input_per_m", "cached_input_per_m", "output_per_m")
        },
        "n_calibration": len(calibration),
        "labeled": len([example_id for example_id in evaluated if example_id in store.records]),
        "outcome": outcome,
        "agreement": agreement,
        "status_counts": dict(sorted(
            Counter(record["status"] for record in store.records.values()).items()
        )),
        "label_counts": dict(sorted(
            Counter(record["label"] for record in store.records.values()).items()
        )),
        "token_fit": fit_tokens(store.records),
        "cost_usd": round(cost, 6),
        "request_parameters": {
            **client.params, "max_completion_tokens": client.max_completion_tokens,
        },
        "adjustments": client.adjustments,
        "meets_threshold": bool(
            agreement["overall"]["rate"] is not None
            and agreement["overall"]["rate"] >= config["agreement_threshold"]
        ),
        "updated": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    (run_dir(config) / f"pilot_{prompt_version}_{tag}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(json.dumps({
        "model": model, "agreement": agreement["overall"], "per_act": agreement["per_act"],
        "cost_usd": summary["cost_usd"], "adjustments": client.adjustments,
    }, sort_keys=True))
    return summary


def full(config, model, turns, api_key, *, base_url=None, log=print, seed=20261102):
    """Label every labelable turn with the chosen model, within the remaining budget."""
    calibration = load_calibration(
        ROOT / config["calibration_csv"], config["calibration_label_column"]
    )
    prompt_version, system_prompt, _ = load_prompt(config, calibration)
    tag = config.get("run_tag") or model
    pilot_path = run_dir(config) / f"pilot_{prompt_version}_{tag}.json"
    if not pilot_path.exists():
        raise FileNotFoundError(f"run the pilot for {model} first: {pilot_path}")
    pilot_summary = json.loads(pilot_path.read_text(encoding="utf-8"))
    prices = prices_for(config, model)
    client = make_client(config, model, api_key, base_url)
    store = LabelStore(run_dir(config) / f"full_{prompt_version}_{tag}.jsonl")
    tracker = tracker_for(config)
    estimator = make_estimator(pilot_summary["token_fit"], prices)
    pending = [turn for turn in turns if turn["example_id"] not in store.records]
    already_done = len(turns) - len(pending)
    estimate_total = sum(estimator(turn) for turn in pending)
    budget = tracker.remaining() - config.get("budget_reserve_usd", 0.1)
    selection = {
        "pending": len(pending),
        "estimate_usd": round(estimate_total, 4),
        "budget_usd": round(budget, 4),
        "rule": "all pending turns",
    }
    if estimate_total > budget:
        pending, planned = select_within_budget(pending, estimator, budget, seed)
        selection.update({
            "rule": "estimate exceeds the budget: every test turn, then train and validation "
                    "turns in a per-act round robin (four train picks per validation pick) "
                    "until the budget is used",
            "selected": len(pending),
            "planned_usd": round(planned, 4),
            "selected_by_split": dict(sorted(Counter(turn["split"] for turn in pending).items())),
        })
    log(json.dumps({"selection": selection}, sort_keys=True))
    started = datetime.now(UTC).replace(microsecond=0).isoformat()
    outcome = label_turns(
        pending, client, store, tracker, stage="full", estimate_per_turn=estimator,
        system_prompt=system_prompt, prompt_version=prompt_version,
        concurrency=config.get("concurrency", 8), log=log,
    )
    outcome["already_done"] = already_done
    summary_path = run_dir(config) / f"full_{prompt_version}_{tag}.json"
    invocations = []
    if summary_path.exists():
        invocations = json.loads(summary_path.read_text(encoding="utf-8")).get("invocations", [])
    invocations.append({
        "started": started, "selection": selection, "outcome": outcome,
        "spent_after_usd": round(tracker.spent, 6),
    })
    summary = {
        "stage": "full",
        "model": model,
        "run_tag": tag,
        "prompt_version": prompt_version,
        "selection": selection,
        "outcome": outcome,
        "invocations": invocations,
        "labeled_total": len(store.records),
        "request_parameters": {
            **client.params, "max_completion_tokens": client.max_completion_tokens,
        },
        "adjustments": client.adjustments,
        "spend": tracker.summary(),
        "updated": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps({"outcome": outcome, "spend": tracker.summary()}, sort_keys=True))
    return summary


def report(config, model, turns, universe, source, *, log=print):
    """Write the committed labels file (ids and labels only) and the manifest."""
    directory = run_dir(config)
    calibration = load_calibration(
        ROOT / config["calibration_csv"], config["calibration_label_column"]
    )
    prompt_version, system_prompt, few_shot_ids = load_prompt(config, calibration)
    evaluated = {
        example_id: entry for example_id, entry in calibration.items()
        if example_id not in set(few_shot_ids)
    }
    tag = config.get("run_tag") or model
    store = LabelStore(directory / f"full_{prompt_version}_{tag}.jsonl")
    move_map_v1 = yaml.safe_load((ROOT / config["act_move_map"]).read_text(encoding="utf-8"))
    pilots = {}
    for path in sorted(directory.glob("pilot_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        entry = {
            key: summary.get(key) for key in (
                "model", "run_tag", "prompt_version", "agreement", "n_evaluated", "cost_usd",
                "prices_usd_per_m", "labeled", "meets_threshold", "adjustments",
                "request_parameters", "token_fit", "status_counts",
            )
        }
        overall = summary["agreement"]["overall"]
        entry["agreement_wilson_95"] = wilson_interval(overall["agree"], overall["n"])
        pilots[f"{summary.get('run_tag', summary['model'])}@{summary['prompt_version']}"] = entry
    full_summary = None
    full_path = directory / f"full_{prompt_version}_{tag}.json"
    if full_path.exists():
        full_summary = json.loads(full_path.read_text(encoding="utf-8"))
    tracker = tracker_for(config)

    labels_path = ROOT / config["labels_path"]
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for turn in sorted(turns, key=lambda item: item["example_id"]):
        record = store.records.get(turn["example_id"])
        if record is None:
            continue
        rows.append({
            "example_id": turn["example_id"],
            "split": turn["split"],
            "source_act": turn["source_act"],
            "label": record["label"],
            "status": record["status"],
            "model": record["model"],
            "prompt_version": record["prompt_version"],
        })
    with labels_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    labeled_ids = {row["example_id"] for row in rows}
    unlabeled = [turn for turn in turns if turn["example_id"] not in labeled_ids]
    crosstab = defaultdict(Counter)
    per_split = defaultdict(Counter)
    for row in rows:
        crosstab[row["source_act"] or "none"][row["label"]] += 1
        per_split[row["split"]][row["label"]] += 1
    manifest = {
        "dataset_id": config["dataset_id"],
        "responsible_owner": "Person A",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "tool_version": TOOL_VERSION,
        "prompt_file": config["prompt_file"],
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(system_prompt),
        "label_source": (
            "model (OpenAI Chat Completions), never human; action_label_source stays 'mapped'"
        ),
        "source": source,
        "universe": universe,
        "labels_file": {
            "path": config["labels_path"],
            "sha256": sha256_file(labels_path),
            "rows": len(rows),
        },
        "labeled": len(rows),
        "unlabeled": {
            "count": len(unlabeled),
            "by_split": dict(sorted(Counter(turn["split"] for turn in unlabeled).items())),
            "by_act": dict(sorted(
                Counter(turn["source_act"] or "none" for turn in unlabeled).items()
            )),
        },
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "counts_per_label": dict(sorted(Counter(row["label"] for row in rows).items())),
        "counts_per_source_act_and_label": {
            act: dict(sorted(counts.items())) for act, counts in sorted(crosstab.items())
        },
        "counts_per_split_and_label": {
            split: dict(sorted(counts.items())) for split, counts in sorted(per_split.items())
        },
        "calibration": {
            "path": config["calibration_csv"],
            "sha256": sha256_file(ROOT / config["calibration_csv"]),
            "label_column": config["calibration_label_column"],
            "n": len(calibration),
            "few_shot_ids_held_out": few_shot_ids,
            "n_evaluated": len(evaluated),
            "drafted_by": "claude-fable-5-1 (model draft, audit_draft_v1); not human labels",
            "act_mapping_agreement": {
                "move_map": config["act_move_map"],
                **act_mapping_agreement(calibration, move_map_v1),
            },
            "chosen_model_agreement_in_full_run": label_agreement(store.records, evaluated),
            "repeat_consistency_pilot_vs_full": repeat_consistency(
                LabelStore(directory / f"pilot_{prompt_version}_{tag}.jsonl").records,
                store.records,
                sorted(evaluated),
            ),
        },
        "model_selection": {
            "threshold": config["agreement_threshold"],
            "rule": (
                "cheapest configured candidate whose pilot agreement with the drafts meets "
                "the threshold"
            ),
            "pilots": pilots,
            "chosen_model": model,
            "chosen_run_tag": tag,
            "chosen_prompt_version": prompt_version,
        },
        "pricing": config.get("pricing_note"),
        "request_parameters": (full_summary or {}).get("request_parameters"),
        "adjustments": (full_summary or {}).get("adjustments"),
        "selection": (full_summary or {}).get("selection"),
        "invocations": (full_summary or {}).get("invocations"),
        "spend": {
            **tracker.summary(),
            "ledger": config["run_dir"] + "/ledger.jsonl (ignored by Git)",
        },
        "notes": [
            "Every label is model-made; D09's human audit was waived for the hackathon by "
            "Xueming on 2026-11-02.",
            "The labels file holds ids and labels only; MathDial text is reproducible from "
            "the pinned revision.",
            "Person B's Strategist may not imitate these labels without Peyton's acceptance.",
        ],
    }
    manifest_path = ROOT / config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log(json.dumps({
        "labeled": len(rows), "unlabeled": len(unlabeled),
        "counts_per_label": manifest["counts_per_label"], "spend": tracker.summary(),
    }, sort_keys=True))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "full", "report"))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default=None,
                        help="override the configured effort; the run files get a -<effort> tag")
    parser.add_argument("--few-shot-per-label", type=int, default=None,
                        help="override few_shot.per_label; the prompt version records it")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.reasoning_effort:
        config["reasoning_effort"] = args.reasoning_effort
        config["run_tag"] = f"{args.model}-{args.reasoning_effort}"
    if args.few_shot_per_label is not None:
        few_shot = dict(config.get("few_shot") or {})
        few_shot["per_label"] = args.few_shot_per_label
        config["few_shot"] = few_shot
    turns, universe, source = source_turns(config)
    if args.stage == "report":
        report(config, args.model, turns, universe, source)
        return
    api_key = load_api_key(config["key_file"], config["key_name"])
    if args.stage == "pilot":
        pilot(config, args.model, turns, api_key)
    else:
        full(config, args.model, turns, api_key)


if __name__ == "__main__":
    main()
