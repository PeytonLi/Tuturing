from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from packages.learning.actions import (
    decode_strategist,
    encode_strategist,
    strategist_mask,
)
from packages.learning.features import build_features
from packages.learning.scheduler import skip_diagnostic_due
from packages.policies.llm import legal_action_text, parse_first_json_object
from packages.policies.network import NetworkStrategist
from packages.policies.rules import (
    AlwaysHintStrategist,
    AlwaysWorkedExampleStrategist,
    RandomStrategist,
    RulesStrategist,
)

ROOT = Path(__file__).resolve().parents[2]
STATE_FIELDS = {
    "active_item",
    "concepts",
    "current_concept_id",
    "explicit_request",
    "goal_concept_id",
    "goal_stack",
    "hint_count",
    "hint_level",
    "independent_success_streak",
    "misconceptions",
    "option_target",
    "option_turns",
    "option_type",
    "pace_preference",
    "previous_move",
    "previous_step",
    "teaching_turns",
}


def case_state_to_inputs(state: dict) -> tuple[dict, list[float], list[bool], dict]:
    unknown = set(state) - STATE_FIELDS
    if unknown:
        raise ValueError(f"unmapped decision state fields: {sorted(unknown)}")
    request = state.get("explicit_request", "none")
    if request == "hint":
        raise ValueError(
            "explicit_request='hint' has no corresponding feature or mask input; "
            "cannot score this case without inventing semantics"
        )
    if request not in {"none", "faster", "slower", "skip"}:
        raise ValueError(f"unmapped explicit_request value: {request!r}")
    item = state.get("active_item")
    if item is not None:
        item_fields = {"template_family_id", "difficulty", "assessment_mode", "answered"}
        unknown_item_fields = set(item) - item_fields
        if unknown_item_fields:
            raise ValueError(
                f"unmapped active_item fields: {sorted(unknown_item_fields)}"
            )
        if item.get("assessment_mode") not in {"practice", "independent"}:
            raise ValueError(
                f"unmapped active_item assessment_mode: {item.get('assessment_mode')!r}"
            )
        if item.get("answered") not in {"none", "correct", "incorrect"}:
            raise ValueError(f"unmapped active_item answer: {item.get('answered')!r}")
    concepts = state.get("concepts", {})
    current_concept = concepts.get(state["current_concept_id"], {})
    turn = int(state.get("teaching_turns", 0))
    recent_correct = int(state.get("independent_success_streak", 0)) > 0 or bool(
        item
        and item.get("assessment_mode") == "independent"
        and item.get("answered") == "correct"
    )
    mapped = {
        "concepts": concepts,
        "misconceptions": state.get("misconceptions", {}),
        "current_concept_id": state["current_concept_id"],
        "goal_concept_id": state["goal_concept_id"],
        "active_target_concept_id": state.get("option_target"),
        "active_option_type": state.get("option_type"),
        "previous_move": state.get("previous_move") or "NONE",
        "turn": turn,
        "turns_in_option": int(state.get("option_turns", 0)),
        "hint_count": int(state.get("hint_count", 0)),
        "independent_success_streak": int(state.get("independent_success_streak", 0)),
        "remaining_budget": max(0, 16 - turn),
        "explicit_skip": request == "skip",
        "explicit_faster": request == "faster",
        "explicit_slower": request == "slower",
        "response_failure": False,
        "stack_depth": len(state.get("goal_stack", [])),
    }
    observation = build_features(mapped)
    difficulty = item.get("difficulty", "medium") if item else "medium"
    mask_state = {
        "mastery": float(current_concept.get("mastery", 0.35)),
        "recent_independent_correct": recent_correct,
        "active_item": item is not None,
        "remaining_hints": int(state.get("hint_level", 0)) < 3,
        "previous_step": bool(state.get("previous_step", False)),
        "active_item_difficulty": difficulty,
        "problem_instance_id": item.get("template_family_id") if item else None,
        "hint_level": int(state.get("hint_level", 0)),
    }
    mask = strategist_mask(mask_state)
    skip_option = state.get("option_type") == "SKIP_TO_TARGET"
    skip_deadline = turn - int(state.get("option_turns", 0)) + 2 if skip_option else None
    skip_done = skip_option and state.get("previous_move") == "CHECK_MASTERY"
    context = {
        **mask_state,
        "misconceptions": state.get("misconceptions", {}),
        "previous_move": state.get("previous_move") or "NONE",
        "pace_preference": state.get("pace_preference", "normal"),
        "explicit_request": request,
        "explicit_skip": request == "skip",
        "explicit_faster": request == "faster",
        "explicit_slower": request == "slower",
        "independent_success_streak": int(state.get("independent_success_streak", 0)),
        "meaningful_error": request == "hint"
        or bool(item and item.get("answered") == "incorrect"),
        "no_diagnostic_yet": turn == 0,
        "unsupported_advancement": False,
        "wants_certification": False,
        "skip_diagnostic_due": skip_diagnostic_due(
            {
                "skip_diagnostic_deadline": skip_deadline,
                "skip_diagnostic_done": skip_done,
            },
            turn,
        ),
        "turn_index": turn,
    }
    if sum(mask) == 0:
        raise ValueError("decision case produced an empty strategist mask")
    return mapped, observation, mask, context


def llm_action(case: dict, mask: list[bool]) -> tuple[int | None, bool, bool]:
    base_url = os.environ.get("TUTOR_LLM_BASE_URL")
    api_key = os.environ.get("TUTOR_LLM_API_KEY")
    model = os.environ.get("TUTOR_LLM_MODEL")
    if not base_url or not api_key or not model:
        raise RuntimeError(
            "LLM mode requires TUTOR_LLM_BASE_URL, TUTOR_LLM_API_KEY, and TUTOR_LLM_MODEL"
        )
    template = (ROOT / "evaluation/learning/llm_strategist_prompt_v1.txt").read_text(
        encoding="utf-8"
    )
    dialogue = "\n".join(
        f"{item['role']}: {item['text']}" for item in case.get("recent_dialogue", [])
    )
    prompt = (
        template.replace("{situation}", case["situation"])
        .replace("{state_summary}", json.dumps(case["state"], indent=1, sort_keys=True))
        .replace("{recent_dialogue}", dialogue)
        .replace("{legal_actions}", legal_action_text(mask))
    )
    payload_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if os.environ.get("TUTOR_LLM_OMIT_TEMPERATURE") != "1":
        payload_data["temperature"] = 0
    token_field = os.environ.get("TUTOR_LLM_MAX_TOKENS_FIELD", "max_tokens")
    token_limit = int(os.environ.get("TUTOR_LLM_MAX_TOKENS", "2048"))
    if not token_field or token_limit <= 0:
        raise ValueError("LLM token-limit field must be non-empty and limit must be positive")
    payload_data[token_field] = token_limit
    payload = json.dumps(payload_data).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "TutoringDecisionScorer/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        truncated = choice.get("finish_reason") in {"length", "max_tokens"}
        if truncated:
            return None, True, True
        content = choice["message"]["content"]
        parsed = parse_first_json_object(content)
        action_id = encode_strategist(
            str(parsed["move"]), str(parsed["difficulty"]), str(parsed["pace"])
        )
        return action_id, False, False
    except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError):
        return None, True, False


def score_action(
    case: dict,
    action_id: int | None,
    mask: list[bool],
    parse_failure: bool,
    truncated: bool = False,
):
    move, difficulty, pace = decode_strategist(action_id) if action_id is not None else (None,) * 3
    illegal = action_id is None or not mask[action_id]
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "move": move,
        "difficulty": difficulty,
        "pace": pace,
        "move_ok": move in case["acceptable_moves"] and not illegal and not parse_failure,
        "forbidden": move in case["forbidden_moves"] if move is not None else False,
        "pace_ok": (
            pace in case["acceptable_paces"] if case["acceptable_paces"] is not None else None
        ),
        "difficulty_ok": (
            difficulty in case["acceptable_difficulties"]
            if case["acceptable_difficulties"] is not None
            else None
        ),
        "illegal": illegal,
        "parse_failure": parse_failure,
        "truncated": truncated,
    }


def rates(rows: list[dict]) -> dict:
    fields = ("move_ok", "forbidden", "pace_ok", "difficulty_ok", "illegal", "parse_failure")
    result = {
        "cases": len(rows),
        "truncated_output_count": sum(bool(row.get("truncated")) for row in rows),
    }
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[field + "_rate"] = sum(values) / len(values) if values else None
        if field in {"forbidden", "illegal", "parse_failure"}:
            result[field + "_count"] = sum(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evaluation/learning/decision_cases_v1.jsonl")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=20250115)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in (ROOT / args.cases).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapped_cases = []
    for case in cases:
        try:
            inputs = case_state_to_inputs(case["state"])
        except ValueError as error:
            raise ValueError(f"decision case {case['case_id']}: {error}") from error
        mapped_cases.append((case, *inputs[1:]))
    if args.policy == "rules":
        policy = RulesStrategist()
    elif args.policy == "random":
        policy = RandomStrategist(args.seed)
    elif args.policy == "always_hint":
        policy = AlwaysHintStrategist()
    elif args.policy == "always_worked_example":
        policy = AlwaysWorkedExampleStrategist()
    elif args.policy.startswith("checkpoint:"):
        policy = NetworkStrategist(args.policy.split(":", 1)[1], greedy=True)
    elif args.policy == "llm":
        policy = None
    else:
        parser.error(
            "policy must be rules, random, always_hint, always_worked_example, "
            "llm, or checkpoint:PATH"
        )
    rows = []
    for case, observation, mask, context in mapped_cases:
        if args.policy == "llm":
            action_id, parse_failure, truncated = llm_action(case, mask)
        else:
            action_id = policy.act(observation, mask, context)
            parse_failure = False
            truncated = False
        rows.append(score_action(case, action_id, mask, parse_failure, truncated))
    categories = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
    report = {
        "result_label": "one-step judged results, not learning outcomes",
        "policy": args.policy,
        "model_id": os.environ.get("TUTOR_LLM_MODEL") if args.policy == "llm" else None,
        "overall": rates(rows),
        "categories": {key: rates(value) for key, value in sorted(categories.items())},
        "cases": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(
        "\n| Category | Cases | Move OK | Forbidden | Pace OK | Difficulty OK "
        "| Illegal | Parse failures | Truncated |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for category, summary in [("overall", report["overall"]), *report["categories"].items()]:
        print(
            f"| {category} | {summary['cases']} | {summary['move_ok_rate']:.3f} "
            f"| {summary['forbidden_rate']:.3f} | {summary['pace_ok_rate']} "
            f"| {summary['difficulty_ok_rate']} | {summary['illegal_rate']:.3f} "
            f"| {summary['parse_failure_rate']:.3f} "
            f"| {summary['truncated_output_count']} |"
        )


if __name__ == "__main__":
    main()
