"""Measure the evidence extractor on evaluation/response/extraction_cases_v1.jsonl.

Usage:
    # Score the live extractor and write the report.
    python -m tools.response.evaluate_extraction

    # Save the live extractor's predictions (used to freeze a "before" column).
    python -m tools.response.evaluate_extraction --predictions-out <file.json>

    # Compare stored predictions ("before") with the live extractor ("after").
    python -m tools.response.evaluate_extraction --before <file.json>

The report lists, per split, accuracy with denominators and confusion matrices
for explicit_request, requested_concept_id, assistance and normalized_answer
(exact match, plus match under B's grader normalization), misconception
precision and recall at the id level and at the id-plus-span level, and two
guards: the extractor must never emit a correctness grade, and its normalized
answer should let B's grader reach the gold verdict. The test split is the
held-out number; the dev split is the tuning set and is reported for reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from packages.evidence_extractor.extract import EXTRACTOR_VERSION, extract
from packages.learning.content.templates.families import normalize_answer

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = ROOT / "evaluation/response/extraction_cases_v1.jsonl"
CONTEXTS_PATH = ROOT / "evaluation/response/extraction_contexts_v1.json"
MANIFEST_PATH = ROOT / "evaluation/response/extraction_cases_v1.manifest.json"
REPORT_PATH = ROOT / "experiments/response/extraction_report_v1.md"
SCHEMA_PATH = ROOT / "contracts/learning_v1.schema.json"

REQUESTS = ("none", "skip", "faster", "slower", "hint", "answer", "stop", "change_goal")
ASSISTANCE = ("none", "hint", "example", "answer_shown", "unknown")
SPLITS = ("test", "dev")
METRIC_ORDER = (
    ("explicit_request", "explicit_request accuracy"),
    ("requested_concept_id", "requested_concept_id accuracy (gold skip or change_goal)"),
    ("concept_false_alarm", "requested_concept_id emitted when gold has none"),
    ("assistance", "assistance accuracy"),
    ("answer_exact", "normalized_answer exact match"),
    ("answer_grader", "normalized_answer match under B's grader normalization"),
    ("answer_presence", "answer presence (null versus non-null) agreement"),
    ("misconception_precision", "misconception precision (id level)"),
    ("misconception_recall", "misconception recall (id level)"),
    ("misconception_span_precision", "misconception precision (id and span overlap)"),
    ("misconception_span_recall", "misconception recall (id and span overlap)"),
    ("misconception_case_exact", "misconception set exact match per case"),
    ("grade_guard", "cases where the extractor emits no correct or incorrect grade"),
    ("grader_agreement", "B's grader on the extracted answer reaches the gold verdict"),
    ("schema_valid", "Evidence validates against learning_v1 with spans in range"),
)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_extractor(cases, contexts):
    predictions = {}
    for case in cases:
        plan = contexts[case["context"]["context_id"]]["plan"]
        evidence = extract(
            case["message"],
            plan,
            f"response-{case['case_id']}",
            event_id=f"event-{case['case_id']}",
            observed_at="2026-11-02T00:00:00Z",
            raw_message_ref=f"raw-{case['case_id']}",
        )
        predictions[case["case_id"]] = evidence
    return predictions


def _evidence_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(
        {"$ref": "#/$defs/Evidence", "$defs": schema["$defs"]},
        format_checker=FormatChecker(),
    )


def _grader_value(answer):
    if answer is None:
        return None
    return normalize_answer(answer)


def _spans_overlap(predicted, gold_spans):
    return any(
        span["start"] < gold["end"] and gold["start"] < span["end"]
        for span in predicted for gold in gold_spans
    )


def _ratio(numerator, denominator):
    return {"n": numerator, "d": denominator,
            "rate": (numerator / denominator) if denominator else None}


def score(cases, predictions, validator=None):
    """Return per-split metrics with denominators and confusion matrices."""
    validator = validator or _evidence_validator()
    per_split = {}
    for split in SPLITS:
        selected = [case for case in cases if case["split"] == split]
        counts = Counter()
        confusion = {
            "explicit_request": defaultdict(Counter),
            "assistance": defaultdict(Counter),
            "answer_presence": defaultdict(Counter),
            "misconception": defaultdict(Counter),
        }
        per_category = defaultdict(Counter)
        failures = []
        for case in selected:
            gold = case["gold"]
            pred = predictions[case["case_id"]]
            category = case["category"]
            per_category[category]["n"] += 1
            wrong = []

            # explicit_request
            confusion["explicit_request"][gold["explicit_request"]][pred["explicit_request"]] += 1
            if gold["explicit_request"] == pred["explicit_request"]:
                counts["explicit_request"] += 1
                per_category[category]["explicit_request"] += 1
            else:
                wrong.append(
                    f"explicit_request gold={gold['explicit_request']} "
                    f"pred={pred['explicit_request']}"
                )

            # requested_concept_id
            if gold["explicit_request"] in {"skip", "change_goal"}:
                counts["concept_d"] += 1
                if gold["requested_concept_id"] == pred["requested_concept_id"]:
                    counts["requested_concept_id"] += 1
                else:
                    wrong.append(
                        f"requested_concept_id gold={gold['requested_concept_id']} "
                        f"pred={pred['requested_concept_id']}"
                    )
            else:
                counts["concept_fa_d"] += 1
                if pred["requested_concept_id"] is not None:
                    counts["concept_false_alarm"] += 1
                    wrong.append(
                        f"requested_concept_id emitted {pred['requested_concept_id']}"
                    )

            # assistance
            confusion["assistance"][gold["assistance"]][pred["assistance"]] += 1
            if gold["assistance"] == pred["assistance"]:
                counts["assistance"] += 1
                per_category[category]["assistance"] += 1
            else:
                wrong.append(f"assistance gold={gold['assistance']} pred={pred['assistance']}")

            # normalized_answer
            gold_answer, pred_answer = gold["normalized_answer"], pred["normalized_answer"]
            presence_gold = "answer" if gold_answer is not None else "null"
            presence_pred = "answer" if pred_answer is not None else "null"
            confusion["answer_presence"][presence_gold][presence_pred] += 1
            if presence_gold == presence_pred:
                counts["answer_presence"] += 1
            if gold_answer == pred_answer:
                counts["answer_exact"] += 1
                per_category[category]["answer_exact"] += 1
            else:
                wrong.append(f"normalized_answer gold={gold_answer!r} pred={pred_answer!r}")
            gold_value, pred_value = _grader_value(gold_answer), _grader_value(pred_answer)
            if gold_answer == pred_answer or (
                gold_value is not None and gold_value == pred_value
            ):
                counts["answer_grader"] += 1

            # misconceptions
            gold_ids = {item["misconception_id"]: item["evidence_spans"]
                        for item in gold["misconception_hypotheses"]}
            pred_ids = {item["misconception_id"]: item["evidence_spans"]
                        for item in pred["misconception_hypotheses"]}
            counts["mis_gold"] += len(gold_ids)
            counts["mis_pred"] += len(pred_ids)
            hits = set(gold_ids) & set(pred_ids)
            counts["mis_hit"] += len(hits)
            counts["mis_span_hit"] += sum(
                1 for item in hits if _spans_overlap(pred_ids[item], gold_ids[item])
            )
            for item in gold_ids:
                confusion["misconception"][item][item if item in pred_ids else "missed"] += 1
            for item in pred_ids:
                if item not in gold_ids:
                    confusion["misconception"]["none"][item] += 1
            if not gold_ids and not pred_ids:
                confusion["misconception"]["none"]["none"] += 1
            if set(gold_ids) == set(pred_ids):
                counts["misconception_case_exact"] += 1
                per_category[category]["misconception_case_exact"] += 1
            else:
                wrong.append(
                    f"misconceptions gold={sorted(gold_ids)} pred={sorted(pred_ids)}"
                )

            # guards
            if pred["correctness"] not in {"correct", "incorrect"} and (
                pred["grade_source"] != "deterministic"
            ):
                counts["grade_guard"] += 1
            else:
                wrong.append(
                    f"emitted grade correctness={pred['correctness']} "
                    f"grade_source={pred['grade_source']}"
                )
            if gold["correctness"] is not None:
                counts["grader_d"] += 1
                reference = case["context"]["reference_answer"]
                verdict = None
                if pred_value is not None and reference is not None:
                    verdict = "correct" if pred_value == Fraction(reference) else "incorrect"
                if verdict == gold["correctness"]:
                    counts["grader_agreement"] += 1
            errors = [error.message for error in validator.iter_errors(pred)]
            length = len(case["message"])
            for item in pred["misconception_hypotheses"]:
                for span in item["evidence_spans"]:
                    if not 0 <= span["start"] < span["end"] <= length:
                        errors.append(f"span out of range: {span}")
            if not errors:
                counts["schema_valid"] += 1
            else:
                wrong.append("schema: " + "; ".join(errors)[:160])

            if wrong:
                failures.append({
                    "case_id": case["case_id"], "category": category,
                    "message": case["message"], "errors": wrong,
                })

        total = len(selected)
        metrics = {
            "explicit_request": _ratio(counts["explicit_request"], total),
            "requested_concept_id": _ratio(counts["requested_concept_id"], counts["concept_d"]),
            "concept_false_alarm": _ratio(counts["concept_false_alarm"], counts["concept_fa_d"]),
            "assistance": _ratio(counts["assistance"], total),
            "answer_exact": _ratio(counts["answer_exact"], total),
            "answer_grader": _ratio(counts["answer_grader"], total),
            "answer_presence": _ratio(counts["answer_presence"], total),
            "misconception_precision": _ratio(counts["mis_hit"], counts["mis_pred"]),
            "misconception_recall": _ratio(counts["mis_hit"], counts["mis_gold"]),
            "misconception_span_precision": _ratio(counts["mis_span_hit"], counts["mis_pred"]),
            "misconception_span_recall": _ratio(counts["mis_span_hit"], counts["mis_gold"]),
            "misconception_case_exact": _ratio(counts["misconception_case_exact"], total),
            "grade_guard": _ratio(counts["grade_guard"], total),
            "grader_agreement": _ratio(counts["grader_agreement"], counts["grader_d"]),
            "schema_valid": _ratio(counts["schema_valid"], total),
        }
        per_split[split] = {
            "n": total,
            "metrics": metrics,
            "confusion": {
                name: {gold: dict(row) for gold, row in table.items()}
                for name, table in confusion.items()
            },
            "per_category": {
                name: dict(row) for name, row in sorted(per_category.items())
            },
            "failures": failures,
        }
    return per_split


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _fmt(ratio):
    if ratio is None or ratio["d"] == 0:
        return "n/a"
    return f"{ratio['rate']:.1%} ({ratio['n']}/{ratio['d']})"


def _delta(before, after):
    if before is None or after is None or before["d"] == 0 or after["d"] == 0:
        return ""
    return f"{(after['rate'] - before['rate']) * 100:+.1f} pt"


def _confusion_table(table, labels, title, extra_columns=()):
    predicted = list(labels) + [
        label for label in sorted({col for row in table.values() for col in row})
        if label not in labels
    ]
    predicted += [column for column in extra_columns if column not in predicted]
    gold_rows = list(labels) + [row for row in sorted(table) if row not in labels]
    lines = [f"{title}", "", "| gold \\ predicted | " + " | ".join(predicted) + " | total |",
             "|---|" + "---|" * (len(predicted) + 1)]
    for gold in gold_rows:
        row = table.get(gold, {})
        total = sum(row.values())
        if total == 0 and gold not in labels:
            continue
        cells = [str(row.get(column, 0)) if row.get(column, 0) else "." for column in predicted]
        lines.append(f"| {gold} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")
    return lines


def render_report(results, *, before=None, before_meta=None, manifest, after_meta):
    lines = [
        "# Evidence extraction report v1",
        "",
        f"Generated by `python -m tools.response.evaluate_extraction` at commit "
        f"`{after_meta['commit']}` with extractor `{after_meta['extractor_version']}`.",
        "",
        f"Evaluation set: `{CASES_PATH.relative_to(ROOT)}` "
        f"(dataset `{manifest['dataset_id']}`, {manifest['counts']['total']} cases: "
        f"{manifest['counts']['test']} test, {manifest['counts']['dev']} dev; "
        f"test split sha256 `{manifest['test_sha256'][:16]}...`).",
        "",
        "Every message and gold label in the set is synthetic and model-authored "
        f"({manifest['provenance']['generator']}); reviewer status is "
        f"`{manifest['provenance']['reviewer_status']}` for Xueming Xu. Rates below "
        "are agreement with those pending labels, not with human judgments. The test "
        "split was frozen before any extractor change and was not used for tuning; "
        "the dev split was.",
        "",
    ]
    if before is not None:
        lines += [
            f"The **before** column is the extractor `{before_meta['extractor_version']}` at "
            f"commit `{before_meta['commit']}` (stored predictions); the **after** column "
            "is the live extractor.",
            "",
        ]
    for split in SPLITS:
        result = results[split]
        lines += [f"## {split} split (n = {result['n']})", ""]
        if before is not None:
            lines += ["| metric | before | after | change |", "|---|---|---|---|"]
            for key, label in METRIC_ORDER:
                prior = before[split]["metrics"].get(key)
                current = result["metrics"][key]
                lines.append(
                    f"| {label} | {_fmt(prior)} | {_fmt(current)} | {_delta(prior, current)} |"
                )
        else:
            lines += ["| metric | value |", "|---|---|"]
            for key, label in METRIC_ORDER:
                lines.append(f"| {label} | {_fmt(result['metrics'][key])} |")
        lines += [
            "",
            "Lower is better for the false-alarm row; every other row is higher is better. "
            "Denominators: all cases in the split unless the metric names its subset; "
            "misconception precision divides by predicted hypotheses, recall by gold "
            "hypotheses (id level counts a hit when the id matches; span level also "
            "requires a predicted span to overlap a gold span).",
            "",
        ]
        if before is not None:
            lines += ["### Confusion matrices, before", ""]
            lines += _confusion_table(
                before[split]["confusion"]["explicit_request"], REQUESTS,
                "explicit_request (rows: gold, columns: predicted)",
            )
            lines += _confusion_table(
                before[split]["confusion"]["assistance"], ASSISTANCE, "assistance",
            )
            lines += _confusion_table(
                before[split]["confusion"]["answer_presence"], ("answer", "null"),
                "normalized_answer presence",
            )
            lines += _confusion_table(
                before[split]["confusion"]["misconception"],
                ("none", "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"),
                "misconception hypotheses (rows: gold id or none; columns: predicted id, "
                "`missed` when a gold id was not proposed)",
                extra_columns=("missed",),
            )
        lines += [f"### Confusion matrices, {'after' if before is not None else 'live'}", ""]
        lines += _confusion_table(
            result["confusion"]["explicit_request"], REQUESTS,
            "explicit_request (rows: gold, columns: predicted)",
        )
        lines += _confusion_table(result["confusion"]["assistance"], ASSISTANCE, "assistance")
        lines += _confusion_table(
            result["confusion"]["answer_presence"], ("answer", "null"),
            "normalized_answer presence",
        )
        lines += _confusion_table(
            result["confusion"]["misconception"],
            ("none", "M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08"),
            "misconception hypotheses (rows: gold id or none; columns: predicted id, "
            "`missed` when a gold id was not proposed)",
            extra_columns=("missed",),
        )
        lines += ["### Per category, " + ("after" if before is not None else "live"), "",
                  "| category | n | explicit_request | normalized_answer exact | "
                  "assistance | misconception set |", "|---|---|---|---|---|---|"]
        for category, row in result["per_category"].items():
            n = row["n"]
            lines.append(
                f"| {category} | {n} | {row.get('explicit_request', 0)}/{n} | "
                f"{row.get('answer_exact', 0)}/{n} | {row.get('assistance', 0)}/{n} | "
                f"{row.get('misconception_case_exact', 0)}/{n} |"
            )
        lines += ["", f"### Remaining {split} failures, "
                  + ("after" if before is not None else "live")
                  + f" ({len(result['failures'])} cases)", ""]
        for failure in result["failures"]:
            lines.append(
                f"- `{failure['case_id']}` ({failure['category']}) "
                f"{json.dumps(failure['message'], ensure_ascii=False)}: "
                + "; ".join(failure["errors"])
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--contexts", default=str(CONTEXTS_PATH))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--before", help="stored predictions JSON for the before column")
    parser.add_argument("--predictions-out", help="write the live predictions JSON here")
    parser.add_argument("--no-report", action="store_true", help="do not write the report")
    parser.add_argument("--split", choices=SPLITS, help="print failures for this split")
    args = parser.parse_args(argv)

    cases = load_jsonl(args.cases)
    contexts = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    predictions = run_extractor(cases, contexts)
    after_meta = {"commit": _git_commit(), "extractor_version": EXTRACTOR_VERSION,
                  "cases_sha256": manifest["cases_sha256"]}
    results = score(cases, predictions)

    if args.predictions_out:
        payload = {"meta": after_meta, "predictions": predictions}
        Path(args.predictions_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote predictions for {len(predictions)} cases to {args.predictions_out}")

    before = before_meta = None
    if args.before:
        stored = json.loads(Path(args.before).read_text(encoding="utf-8"))
        before_meta = stored["meta"]
        if before_meta.get("cases_sha256") != manifest["cases_sha256"]:
            raise SystemExit("stored predictions were made on a different evaluation set")
        before = score(cases, stored["predictions"])

    for split in SPLITS:
        metrics = results[split]["metrics"]
        print(
            f"{split}: n={results[split]['n']} "
            f"request={_fmt(metrics['explicit_request'])} "
            f"answer={_fmt(metrics['answer_exact'])} "
            f"assistance={_fmt(metrics['assistance'])} "
            f"mis P={_fmt(metrics['misconception_precision'])} "
            f"R={_fmt(metrics['misconception_recall'])} "
            f"grade_guard={_fmt(metrics['grade_guard'])}"
        )
    if args.split:
        for failure in results[args.split]["failures"]:
            print(f"{failure['case_id']} ({failure['category']}) {failure['message']!r}")
            for error in failure["errors"]:
                print(f"    {error}")
    if not args.no_report:
        report = render_report(
            results, before=before, before_meta=before_meta, manifest=manifest,
            after_meta=after_meta,
        )
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return results


if __name__ == "__main__":
    main()
