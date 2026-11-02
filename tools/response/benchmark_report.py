"""Aggregate response benchmark runs into CSVs, report.md and a blinded review sheet.

Reads the latest record per (model, case) from the run directory, writes per-model,
per-move and per-case-set tables with explicit denominators, quotes failure examples, and
builds a review sheet whose response labels are shuffled per case with the key in a
separate file. ``experiments/response/benchmark_v1/notes.md`` is included verbatim when
present so hand-written findings survive a rerun.
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from packages.response.prompt import PROMPT_VERSION
from packages.response.validate import VALIDATOR_VERSION
from tools.response.benchmark import (
    HARNESS_VERSION,
    cost_basis,
    load_all_cases,
    load_results,
    stratified_order,
    utc_now,
)
from tools.response.evaluate import CHECK_NAMES

DELIVERIES = ("model", "repair", "fallback", "error", "unavailable")
SHEET_FILENAME = "review_sheet_blinded.csv"
SUPPLEMENT_FILENAME = "review_sheet_blinded_supplement.csv"
KEY_FILENAME = "review_key_do_not_open_before_rating.csv"
KEY_COLUMNS = ("item", "case_set", "case_id", "response_label", "model_id")
REVIEW_COLUMNS = (
    "item", "case_set", "case_id", "move", "difficulty", "pace", "max_words",
    "max_questions", "reveal_allowed", "must_request_attempt", "problem_stem",
    "approved_content", "last_learner_message", "response_label", "response_text",
    "math_correct_yn", "faithful_to_move_yn", "no_early_answer_yn", "actionable_yn",
    "tone_1to5", "notes",
)


def _rate(num, den):
    return f"{num / den:.3f}" if den else "n/a"


def _pct(num, den):
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "0/0"


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(records):
    """Counts with denominators for one group of records."""
    n = len(records)
    errors = sum(1 for r in records if r["error"])
    summary = {
        "n": n,
        "errors": errors,
        "all_applicable_pass": sum(1 for r in records if r["all_applicable_pass"]),
        "all_raw_pass": sum(1 for r in records if not r["failed_checks_raw"] and not r["error"]),
    }
    for name in CHECK_NAMES:
        applicable = [r for r in records if r["applicable"].get(name, True)]
        summary[f"{name}_applicable"] = len(applicable)
        summary[f"{name}_pass"] = sum(1 for r in applicable if r["checks"][name])
    for delivery in DELIVERIES:
        summary[f"delivery_{delivery}"] = sum(1 for r in records if r["delivery"] == delivery)
    summary["fallback_needed"] = sum(1 for r in records if r["fallback_needed"])
    summary["fallback_needed_unavailable"] = sum(
        1 for r in records if r["fallback_needed"] and not r["fallback_available"]
    )
    ok = [r for r in records if not r["error"]]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    summary.update({
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "mean_input_tokens": round(_mean([r["usage"]["input_tokens"] for r in ok]), 1),
        "mean_output_tokens": round(_mean([r["usage"]["output_tokens"] for r in ok]), 1),
        "mean_reasoning_tokens": round(_mean([r["usage"]["reasoning_tokens"] for r in ok]), 1),
        "mean_visible_output_tokens": round(
            _mean([r["usage"]["visible_output_tokens"] for r in ok]), 1
        ),
        "mean_words": round(_mean([r["word_count"] for r in ok]), 1),
        "over_visible_token_cap": sum(1 for r in ok if r["over_visible_token_cap"]),
        "think_stripped": sum(1 for r in ok if r["think_stripped"]),
        "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "cost_usd_per_case": round(sum(r["cost_usd"] for r in records) / n, 6) if n else 0.0,
        "math_unsupported_only_learner_numbers": sum(
            1 for r in records
            if "math_supported" in r["failed_checks"]
            and r["unsupported_numbers"]
            and set(r["unsupported_numbers"]) <= set(r["unsupported_numbers_from_learner"])
        ),
        "disclosure_fingerprint_in_context": sum(
            1 for r in records
            if "disclosure_allowed" in r["failed_checks"]
            and r["disclosed_fingerprints_in_context"]
        ),
    })
    return summary


def _write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _model_table(entries, runs):
    lines = [
        "| Model | Provider | Model ID | Reasoning | Temp. | Price in/out ($/MTok) | Cases |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in entries:
        model_id = entry["id"]
        pricing = entry.get("pricing") or {}
        if cost_basis(entry) == "tokens":
            price = f"{pricing.get('input_usd_per_mtok')} / {pricing.get('output_usd_per_mtok')}"
        elif cost_basis(entry) == "gpu_time":
            spec = entry.get("deployment") or {}
            price = f"GPU {spec.get('usd_per_gpu_hour')}/h x{spec.get('accelerator_count', 1)}"
        else:
            price = "free"
        decoding = entry.get("decoding") or {}
        temp = decoding.get("temperature")
        reasoning = (entry.get("reasoning") or {}).get("setting", "n/a")
        model = (entry.get("deployment") or {}).get("base_model") or entry.get("model") or "-"
        lines.append(
            f"| {entry.get('label', model_id)} | {entry.get('provider')} | `{model}` | "
            f"{reasoning} | {'default' if temp is None else temp} | {price} | "
            f"{len(runs.get(model_id, {}))} |"
        )
    return lines


def _summary_rows(groups):
    rows = []
    for key, records in groups:
        summary = summarize(records)
        row = {**key, **summary}
        for name in CHECK_NAMES:
            row[f"{name}_rate"] = _rate(summary[f"{name}_pass"], summary[f"{name}_applicable"])
        rows.append(row)
    return rows


def _check_table(rows, label):
    header = f"| {label} | N | All applicable pass | Delivered as model | Fallback needed | "
    header += " | ".join(CHECK_NAMES) + " |"
    lines = [header, "|---|---:|---:|---:|---:|" + "---:|" * len(CHECK_NAMES)]
    for row in rows:
        cells = " | ".join(
            _pct(row[f"{name}_pass"], row[f"{name}_applicable"]) for name in CHECK_NAMES
        )
        lines.append(
            f"| {row['label']} | {row['n']} | {_pct(row['all_applicable_pass'], row['n'])} | "
            f"{_pct(row['delivery_model'], row['n'])} | {_pct(row['fallback_needed'], row['n'])} | "
            f"{cells} |"
        )
    return lines


def _failure_examples(records, model_label, per_check=2, width=320):
    lines = []
    for name in CHECK_NAMES:
        failed = [r for r in records if name in r["failed_checks"]]
        if not failed:
            continue
        lines.append(f"**{model_label}, `{name}` ({len(failed)} failures):**")
        lines.append("")
        for record in failed[:per_check]:
            text = " ".join(record["text"].split())
            if len(text) > width:
                text = text[:width] + "..."
            detail = ""
            if name == "math_supported":
                detail = (
                    f" Unsupported numbers: {record['unsupported_numbers']}; from the "
                    f"learner's own message: {record['unsupported_numbers_from_learner']}."
                )
            elif name == "disclosure_allowed":
                detail = (
                    f" Fingerprints: {record['disclosed_fingerprints']}; already in the stem "
                    f"or learner text: {record['disclosed_fingerprints_in_context']}."
                )
            lines.append(
                f"- `{record['case_set']}/{record['case_id']}` ({record['move']}): "
                f"\"{text}\"{detail}"
            )
        lines.append("")
    return lines


def _ledger_summary(ledger_path):
    if not Path(ledger_path).exists():
        return {}, []
    spend = defaultdict(float)
    gpu = []
    with Path(ledger_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            spend[event.get("model_id", "?")] += float(event.get("actual_usd") or 0.0)
            if event.get("event") == "gpu":
                gpu.append(event)
    return dict(spend), gpu


def _deployment_records(run_dir):
    records = []
    directory = Path(run_dir) / "deployments"
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _review_row(item, case, label, text):
    plan = case.plan
    content = plan["content"]
    constraints = plan["constraints"]
    approved = []
    for name, field in (
        ("facts", "approved_facts"), ("hint", "permitted_hint"),
        ("worked example", "worked_example"), ("previous step", "previous_step"),
        ("solution (reveal allowed)", "reveal_solution"),
    ):
        value = content.get(field)
        if field == "reveal_solution" and not constraints.get("reveal_allowed"):
            continue
        if isinstance(value, list):
            value = " | ".join(value)
        if value:
            approved.append(f"{name}: {value}")
    return {
        "item": item,
        "case_set": case.case_set,
        "case_id": case.case_id,
        "move": case.move,
        "difficulty": plan["teaching_action"].get("difficulty"),
        "pace": plan["teaching_action"].get("pace"),
        "max_words": constraints.get("max_words"),
        "max_questions": constraints.get("max_questions"),
        "reveal_allowed": constraints.get("reveal_allowed"),
        "must_request_attempt": constraints.get("must_request_student_attempt"),
        "problem_stem": content.get("problem_stem") or "",
        "approved_content": "\n".join(approved),
        "last_learner_message": plan["response_context"].get("last_student_message"),
        "response_label": label,
        "response_text": text,
    }


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_review_supplement(cases, entries, runs, out_dir):
    """Add models absent from the published sheet without touching its items or labels.

    The supplement keeps the same items and columns; each added model's response takes
    the next unused label for that item and its mapping is appended to the key. Raters can
    tell these rows were added later, which weakens blinding between the added model and
    the earlier ones, but not among the earlier ones.
    """
    review_dir = Path(out_dir) / "review"
    key_path = review_dir / KEY_FILENAME
    key_rows = _read_csv(key_path)
    present = {row["model_id"] for row in key_rows}
    new_models = [e["id"] for e in entries if runs.get(e["id"]) and e["id"] not in present]
    if not new_models:
        return None, 0
    items = {}
    labels = defaultdict(set)
    for row in key_rows:
        items[int(row["item"])] = (row["case_set"], row["case_id"])
        labels[int(row["item"])].add(row["response_label"])
    supplement_path = review_dir / SUPPLEMENT_FILENAME
    sheet = _read_csv(supplement_path) if supplement_path.exists() else []
    case_by_key = {case.key: case for case in cases}
    additions = []
    for item in sorted(items):
        case_set, case_id = items[item]
        case = case_by_key.get(f"{case_set}/{case_id}")
        if case is None:
            continue
        for model_id in new_models:
            record = runs[model_id].get(case.key)
            if not record or record["error"] or not record["text"].strip():
                continue
            number = 1
            while f"R{number}" in labels[item]:
                number += 1
            label = f"R{number}"
            labels[item].add(label)
            sheet.append(_review_row(item, case, label, record["text"]))
            additions.append({
                "item": item, "case_set": case_set, "case_id": case_id,
                "response_label": label, "model_id": model_id,
            })
    _write_csv(supplement_path, sheet, REVIEW_COLUMNS)
    _write_csv(key_path, key_rows + additions, KEY_COLUMNS)
    with (review_dir / "README.md").open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n## Supplement added {utc_now()}\n\n`{SUPPLEMENT_FILENAME}` adds "
            f"{len(additions)} responses from {len(new_models)} model(s) that ran after the "
            "sheet above was published, for the same items and columns, each under the next "
            "unused label of its item (R2 or R3); the mappings were appended to the key. A "
            "rater can tell these rows were added later, which weakens the blinding between "
            "the added model(s) and the earlier ones but not among the earlier ones.\n"
        )
    return supplement_path, len(additions)


def build_review(cases, entries, runs, config, out_dir):
    """Blinded sheet: per case, responses shuffled and labelled R1..Rk; key kept apart."""
    review = config.get("review") or {}
    wanted = int(review.get("cases", 40))
    seed = review.get("seed", 23)
    rng = random.Random(seed)
    model_ids = [entry["id"] for entry in entries if runs.get(entry["id"])]
    sheet, key = [], []
    item = 0
    for case in stratified_order(cases, seed=seed):
        responses = []
        for model_id in model_ids:
            record = runs[model_id].get(case.key)
            if record and record["text"].strip() and not record["error"]:
                responses.append((model_id, record["text"]))
        if not responses:
            continue
        item += 1
        rng.shuffle(responses)
        for index, (model_id, text) in enumerate(responses, start=1):
            label = f"R{index}"
            sheet.append(_review_row(item, case, label, text))
            key.append({
                "item": item, "case_set": case.case_set, "case_id": case.case_id,
                "response_label": label, "model_id": model_id,
            })
        if item >= wanted:
            break
    review_dir = Path(out_dir) / "review"
    _write_csv(review_dir / SHEET_FILENAME, sheet, REVIEW_COLUMNS)
    _write_csv(review_dir / KEY_FILENAME, key, KEY_COLUMNS)
    per_model = Counter(row["model_id"] for row in key)
    (review_dir / "README.md").write_text(
        "# Blinded response review\n\n"
        f"`review_sheet_blinded.csv` holds {item} cases and {len(sheet)} responses. Within "
        "each item the responses are shuffled and labelled R1, R2, ...; the mapping to models "
        "is in `review_key_do_not_open_before_rating.csv` and nowhere else in the sheet. "
        "Fill the `*_yn`, `tone_1to5` and `notes` columns per response (PRD 13.3 criteria: "
        "mathematical correctness, action fidelity, disclosure permission, relevance and a "
        "clear next step), then join on item and response_label.\n\n"
        "Responses per model (seen only after unblinding): "
        + ", ".join(f"{count}" for _, count in sorted(per_model.items()))
        + f" across {len(per_model)} models.\n",
        encoding="utf-8",
    )
    return review_dir, item, len(sheet)


def build_report(config, ctx):
    run_dir = Path(ctx.run_dir)
    out_dir = Path(config.get("report_dir", "experiments/response/benchmark_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, notes = load_all_cases(config)
    case_by_key = {case.key: case for case in cases}
    entries = list(config.get("models", []))
    runs = {}
    for entry in entries:
        records = {
            key: record for key, record in load_results(run_dir, entry["id"]).items()
            if key in case_by_key
        }
        if records:
            runs[entry["id"]] = records
    ran = [entry for entry in entries if entry["id"] in runs]
    labels = {entry["id"]: entry.get("label", entry["id"]) for entry in entries}
    written = []

    spend, gpu_events = _ledger_summary(run_dir / "ledger.jsonl")
    by_model = _summary_rows(
        [({"model_id": e["id"], "label": labels[e["id"]]}, list(runs[e["id"]].values()))
         for e in ran]
    )
    for entry, row in zip(ran, by_model):
        if cost_basis(entry) == "gpu_time":
            # GPU-hour entries have no per-call cost; attribute the ledger's GPU spend.
            row["cost_usd"] = round(spend.get(entry["id"], 0.0), 4)
            row["cost_usd_per_case"] = round(row["cost_usd"] / row["n"], 6) if row["n"] else 0.0
    by_move_rows, by_set_rows = [], []
    for entry in ran:
        records = list(runs[entry["id"]].values())
        for move in sorted({r["move"] for r in records}):
            group = [r for r in records if r["move"] == move]
            by_move_rows += _summary_rows(
                [({"model_id": entry["id"], "label": labels[entry["id"]], "move": move}, group)]
            )
        for case_set in sorted({r["case_set"] for r in records}):
            group = [r for r in records if r["case_set"] == case_set]
            by_set_rows += _summary_rows([(
                {"model_id": entry["id"], "label": labels[entry["id"]], "case_set": case_set},
                group,
            )])
    summary_columns = ["model_id", "label", "n", "errors", "all_applicable_pass", "all_raw_pass"]
    summary_columns += [f"{name}_{suffix}" for name in CHECK_NAMES
                        for suffix in ("pass", "applicable", "rate")]
    summary_columns += [f"delivery_{d}" for d in DELIVERIES]
    summary_columns += [
        "fallback_needed", "fallback_needed_unavailable", "latency_p50_ms", "latency_p95_ms",
        "mean_input_tokens", "mean_output_tokens", "mean_reasoning_tokens",
        "mean_visible_output_tokens", "mean_words", "over_visible_token_cap", "think_stripped",
        "cost_usd", "cost_usd_per_case", "math_unsupported_only_learner_numbers",
        "disclosure_fingerprint_in_context",
    ]
    for name, rows, extra in (
        ("results_by_model.csv", by_model, []),
        ("results_by_move.csv", by_move_rows, ["move"]),
        ("results_by_case_set.csv", by_set_rows, ["case_set"]),
    ):
        columns = summary_columns[:2] + extra + summary_columns[2:]
        _write_csv(out_dir / name, rows, columns)
        written.append(out_dir / name)

    case_rows = []
    for entry in ran:
        for key, record in sorted(runs[entry["id"]].items()):
            row = {
                "model_id": entry["id"], "case_set": record["case_set"],
                "case_id": record["case_id"], "move": record["move"],
                "difficulty": record["difficulty"], "pace": record["pace"],
                "delivery": record["delivery"], "error": record["error"] or "",
                "failed_checks": ";".join(record["failed_checks"]),
                "failed_checks_raw": ";".join(record["failed_checks_raw"]),
                "word_count": record["word_count"], "question_marks": record["question_marks"],
                "latency_ms": record["latency_ms"],
                "input_tokens": record["usage"]["input_tokens"],
                "output_tokens": record["usage"]["output_tokens"],
                "reasoning_tokens": record["usage"]["reasoning_tokens"],
                "cost_usd": record["cost_usd"], "finish_reason": record["finish_reason"] or "",
                "unsupported_numbers": ";".join(record["unsupported_numbers"]),
                "unsupported_numbers_from_learner": ";".join(
                    record["unsupported_numbers_from_learner"]),
                "disclosed_fingerprints": ";".join(record["disclosed_fingerprints"]),
                "run_tag": record["run_tag"], "started_at": record["started_at"],
            }
            for name in CHECK_NAMES:
                row[name] = int(record["checks"][name])
                row[f"{name}_applicable"] = int(record["applicable"].get(name, True))
            case_rows.append(row)
    case_columns = list(case_rows[0].keys()) if case_rows else ["model_id"]
    _write_csv(out_dir / "results_by_case.csv", case_rows, case_columns)
    written.append(out_dir / "results_by_case.csv")

    deployments = _deployment_records(run_dir)
    review_dir = out_dir / "review"
    if (review_dir / SHEET_FILENAME).exists():
        # A published sheet is frozen: raters may be working on it. New models go to a
        # supplement with the same items and the next unused labels.
        supplement_path, added = build_review_supplement(cases, ran, runs, out_dir)
        if supplement_path is not None:
            written.append(supplement_path)
        key_rows = _read_csv(review_dir / KEY_FILENAME)
        review_items = len({row["item"] for row in key_rows})
        review_rows = len(key_rows)
        review_note = (
            f" The published sheet is frozen; `review/{SUPPLEMENT_FILENAME}` holds responses "
            "from models that ran later, under the same items." if (
                review_dir / SUPPLEMENT_FILENAME).exists() else ""
        )
    else:
        review_dir, review_items, review_rows = build_review(cases, ran, runs, config, out_dir)
        written.append(review_dir / SHEET_FILENAME)
        review_note = ""

    lines = [
        "# Response benchmark v1: prompt-only models on the tutoring response task",
        "",
        f"Generated {utc_now()} by `tools.response.benchmark report` "
        f"({HARNESS_VERSION}, {PROMPT_VERSION}, {VALIDATOR_VERSION}). Every number below is an "
        "automated check from `packages.response.validate` on one sampled response per case; "
        "no human rating is included. Read the limitations before using any table.",
        "",
        "## Models",
        "",
        *_model_table(entries, runs),
        "",
        "Entries with 0 cases did not run: hosted entries wait for an API key in the keys "
        "file, local entries wait for weights, and disabled entries record why. Prices are "
        "standard-tier list prices per million tokens on the date in the config.",
        "",
        "## Case sets",
        "",
        "| Case set | Cases used | Note |",
        "|---|---:|---|",
    ]
    for name, info in notes.items():
        lines.append(f"| {name} | {info['n']} | {info['note'] or ''} |")
    lines += [
        "",
        f"Total: {len(cases)} cases. A case is one TeachingPlan; the model sees only the "
        "compiled prompt (plan projection plus recent dialogue).",
        "",
        "## Results by model",
        "",
        "Denominator per cell is shown explicitly. `action_faithful` is counted only where "
        "the plan carries the content its heuristic needs (MathDial HINT and EXPLAIN_CONCEPT "
        "plans have no permitted hint or approved facts, so they are not applicable). "
        "\"Delivered as model\" means all applicable checks passed; \"fallback needed\" means "
        "the serving pipeline (PRD 14.1) would repair, fall back or error instead.",
        "",
        *_check_table(by_model, "Model"),
        "",
        "| Model | Errors | Repair | Fallback | Unavailable | p50 ms | p95 ms | In tok | "
        "Out tok | Reasoning tok | Words | Over 384 visible | <think> leaks | Cost $ | "
        "$/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_model:
        lines.append(
            f"| {row['label']} | {row['errors']} | {row['delivery_repair']} | "
            f"{row['delivery_fallback']} | {row['delivery_unavailable']} | "
            f"{row['latency_p50_ms']} | {row['latency_p95_ms']} | {row['mean_input_tokens']} | "
            f"{row['mean_output_tokens']} | {row['mean_reasoning_tokens']} | {row['mean_words']} "
            f"| {row['over_visible_token_cap']} | {row['think_stripped']} | {row['cost_usd']} | "
            f"{row['cost_usd_per_case']} |"
        )
    lines += [
        "",
        "Diagnostics on the strict checks (counts of failing cases): ",
        "",
        "| Model | math failures where every flagged number came from the learner's message | "
        "disclosure flags whose fingerprint already appears in the stem or learner text |",
        "|---|---:|---:|",
    ]
    for row in by_model:
        lines.append(
            f"| {row['label']} | {row['math_unsupported_only_learner_numbers']} of "
            f"{row['math_supported_applicable'] - row['math_supported_pass']} | "
            f"{row['disclosure_fingerprint_in_context']} of "
            f"{row['disclosure_allowed_applicable'] - row['disclosure_allowed_pass']} |"
        )
    lines += ["", "## Results by move", "", *_check_table(
        [{**row, "label": f"{row['label']} / {row['move']}"} for row in by_move_rows],
        "Model / move",
    )]
    lines += ["", "## Results by case set", "", *_check_table(
        [{**row, "label": f"{row['label']} / {row['case_set']}"} for row in by_set_rows],
        "Model / case set",
    )]
    lines += ["", "## Failure examples (automated checks)", ""]
    for entry in ran:
        lines += _failure_examples(list(runs[entry["id"]].values()), labels[entry["id"]])
    lines += ["## Spend", ""]
    if spend:
        lines += ["| Ledger entry | Actual USD |", "|---|---:|"]
        for model_id, amount in sorted(spend.items()):
            lines.append(f"| {model_id} | {amount:.4f} |")
        lines.append(f"| **Total** | **{sum(spend.values()):.4f}** |")
        cap_text = f"${ctx.cap:.2f}" if ctx.cap is not None else "none"
        lines.append("")
        lines.append(
            f"Cap: {cap_text}; token costs are computed "
            "from each response's `usage` times list prices; GPU costs are wall time from the "
            "create request to the verified deletion times the list price (an upper bound)."
        )
    else:
        lines.append("No paid calls recorded.")
    lines += ["", "## Deployments", ""]
    if deployments:
        for state in deployments:
            precision = state.get("precision") or (state.get("create_response") or {}).get(
                "precision") or "precision n/a"
            lines.append(
                f"- `{state.get('name') or state.get('deployment_id')}`: "
                f"{state.get('accelerator_count')}x {state.get('accelerator_type')} "
                f"({precision}), shape `{state.get('deployment_shape')}`, "
                f"created {state.get('create_requested_at')}, ended {state.get('ended_at')}, "
                f"final state {state.get('state')}, deletion verified "
                f"{state.get('deleted_verified')}, estimated GPU cost "
                f"${state.get('estimated_gpu_usd', 0):.2f}."
            )
    else:
        lines.append("No deployment was created in the recorded runs.")
    notes_path = out_dir / "notes.md"
    if notes_path.exists():
        lines += ["", notes_path.read_text(encoding="utf-8").rstrip()]
    lines += [
        "",
        "## Limitations",
        "",
        "- Automated checks only. `math_supported` accepts a number only if it appears in the "
        "plan's approved content (or is a small integer); numbers the learner wrote, "
        "intermediate results and LaTeX-escaped percentages count as unsupported, so it "
        "over-reports math failures on dialogue data and never proves reasoning correct.",
        "- `disclosure_allowed` matches answer fingerprints anywhere in the text, including "
        "values that the problem stem or the learner already stated.",
        "- `action_faithful` is a heuristic: a PROBE must restate every stem number, a "
        "WORKED_EXAMPLE every example number, and a HINT must stay within 25 words of the "
        "permitted hint. Natural tutor replies fail it without being unfaithful.",
        "- MathDial is GSM8K-style arithmetic with LLM-simulated students, not probability; "
        "its plans carry mapped (unaudited) moves and no approved hint, facts or example.",
        "- Samples are small (see the case-set table); differences of a few cases are noise. "
        "Hosted models with a default temperature of 1 are not deterministic, so a rerun "
        "gives different texts.",
        "- Latency is one client on a home connection with concurrency per entry as "
        "configured, not the PRD 14.3 serving measurement.",
        "",
        f"Review sheet: `review/{SHEET_FILENAME}` ({review_items} cases, {review_rows} "
        f"responses including any supplement); key in `review/{KEY_FILENAME}`.{review_note}",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    written.append(out_dir / "report.md")
    return written
