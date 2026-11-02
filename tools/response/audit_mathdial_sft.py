"""Audit prepared MathDial SFT records for leakage from the raw source.

Reads the pinned raw JSONL and the prepared split files named by the MathDial
config and prints JSON counts. Checks: context equals the preceding turns,
prompt equals the compiled plan, no later-turn or private-field text that the
past dialogue cannot explain, and no reference solution or stated final answer
in reveal-forbidden examples.

    python -m tools.response.audit_mathdial_sft --config configs/response/mathdial_v1.yaml
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from packages.response.prompt import compile_prompt
from packages.response.validate import run_checks
from tools.response.prepare_sft import _teacher_turns, load_config, parse_turn

PRIVATE_FIELDS = ("self-correctness", "teacher_described_confusion", "student_profile")
MIN_MATCH_CHARS = 25


def _normalized(text):
    return " ".join(str(text).casefold().split())


def _raw_rows(config):
    raw_dir = Path(config["raw_dir"])
    rows = {}
    for name in config["files"]:
        official = name.removesuffix(".jsonl")
        lines = (raw_dir / name).read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip():
                rows[f"mathdial:{official}:{index:04d}"] = json.loads(line)
    return rows


def audit_example(example, row, include_opening=True):
    """Return the set of finding names for one prepared example."""
    findings = set()
    turn_index = example["extensions"]["mathdial"]["turn_index"]
    profile = (row.get("student_profile") or "").split()
    name = profile[0] if profile else ""
    turns = [parse_turn(turn, name) for turn in _teacher_turns(row)]
    opening = (row.get("student_incorrect_solution") or "").strip()
    prefix = [{"role": "student", "text": opening}] if include_opening and opening else []
    expected = prefix + [
        {"role": turn["role"], "text": turn["text"]} for turn in turns[:turn_index] if turn["text"]
    ]
    plan = example["plan"]
    if plan["response_context"]["recent_dialogue"] != expected:
        findings.add("context_differs_from_preceding_turns")
    if example["prompt"] != compile_prompt(plan):
        findings.add("prompt_differs_from_compiled_plan")
    completion = example["completion"][0]["content"]
    if completion != turns[turn_index]["text"]:
        findings.add("completion_differs_from_target_turn")
    prompt_text = _normalized("\n".join(message["content"] for message in example["prompt"]))
    past = [_normalized(row.get("question") or ""), _normalized(opening)]
    past.extend(_normalized(turn["text"]) for turn in turns[:turn_index])
    if plan["constraints"]["reveal_allowed"]:
        past.append(_normalized(row.get("ground_truth") or ""))

    def unexplained(text):
        return (
            len(text) >= MIN_MATCH_CHARS
            and text in prompt_text
            and not any(text in item for item in past)
        )

    if any(unexplained(_normalized(turn["text"])) for turn in turns[turn_index + 1:]):
        findings.add("later_turn_text_in_prompt")
    for field in PRIVATE_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and len(value) > 3 and _normalized(value) in prompt_text:
            explained = any(_normalized(value) in item for item in past)
            findings.add(
                f"{field}_text_in_prompt_via_dialogue" if explained
                else f"{field}_text_in_prompt"
            )
    record_text = json.dumps(
        {key: value for key, value in example.items() if key != "completion"},
        ensure_ascii=False,
    )
    if any(f'"{field}"' in record_text for field in PRIVATE_FIELDS):
        findings.add("private_field_name_in_record")
    if name and re.search(rf"\b{re.escape(name)}\b", completion, re.IGNORECASE):
        if not re.search(rf"\b{re.escape(name)}\b", prompt_text, re.IGNORECASE):
            findings.add("target_uses_profile_name_absent_from_prompt")
    if not plan["constraints"]["reveal_allowed"]:
        findings.add("reveal_forbidden")
        solution = row.get("ground_truth") or ""
        if _normalized(solution) in prompt_text:
            findings.add("protected_prompt_has_full_reference_solution")
        lines = [
            _normalized(line) for line in solution.splitlines()
            if len(line.strip()) >= MIN_MATCH_CHARS
        ]
        if any(line in prompt_text for line in lines):
            findings.add("protected_prompt_has_reference_solution_line_via_dialogue")
        if not run_checks(plan, completion, plan["teaching_action"])["disclosure_allowed"]:
            findings.add("protected_target_states_final_answer")
        tutor_text = " ".join(
            item["text"] for item in plan["response_context"]["recent_dialogue"]
            if item["role"] == "tutor"
        )
        if tutor_text and not run_checks(
            plan, tutor_text, plan["teaching_action"]
        )["disclosure_allowed"]:
            findings.add("protected_prompt_prior_tutor_turn_stated_final_answer")
    return findings


def audit(config_path):
    config = load_config(config_path)
    rows = _raw_rows(config)
    include_opening = bool(config.get("include_student_solution_as_opening_turn"))
    counts = Counter()
    by_split = Counter()
    for split in ("train", "validation", "test"):
        path = Path(config["out_dir"]) / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            example = json.loads(line)
            by_split[split] += 1
            findings = audit_example(example, rows[example["dialogue_id"]], include_opening)
            counts.update(findings)
    return {
        "examples": sum(by_split.values()),
        "examples_per_split": dict(sorted(by_split.items())),
        "findings": dict(sorted(counts.items())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/response/mathdial_v1.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
