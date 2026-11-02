from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = [
    item["name"]
    for item in json.loads((ROOT / "contracts/feature_spec_v1.json").read_text(encoding="utf-8"))[
        "features"
    ]
]


def _get(source: dict, key: str, default: object = None) -> object:
    return source.get(key, default)


def build_features(state: dict, *, navigator: bool = False) -> list[float]:
    concepts = state.get("concepts", [])
    if isinstance(concepts, dict):
        concepts = [concepts.get(f"C{index:02d}", {}) for index in range(1, 16)]
    concepts = list(concepts)
    concepts += [{}] * max(0, 15 - len(concepts))
    features: list[float] = []
    features.extend(float(concept.get("mastery", 0.0)) for concept in concepts[:15])
    features.extend(float(concept.get("uncertainty", 0.0)) for concept in concepts[:15])
    features.extend(
        min(
            10,
            int(concept.get("independent_correct", 0))
            + int(concept.get("independent_incorrect", 0)),
        )
        / 10
        for concept in concepts[:15]
    )
    misconception_values = state.get("misconceptions", [])
    if isinstance(misconception_values, dict):
        misconception_values = [
            misconception_values.get(f"M{index:02d}", 0.0) for index in range(1, 26)
        ]
    features.extend(
        float(value.get("probability", 0.0) if isinstance(value, dict) else value)
        for value in list(misconception_values)[:25]
    )
    features.extend(_one_hot(state.get("current_concept_id"), "C", 15))
    features.extend(_one_hot(state.get("goal_concept_id"), "C", 15))
    target = 0.0 if navigator else state.get("active_target_concept_id")
    features.extend(_one_hot(target, "C", 15))
    option_type = None if navigator else state.get("active_option_type")
    features.extend(
        float(option_type == name)
        for name in (
            "STAY",
            "ADVANCE_ONE",
            "SKIP_TO_TARGET",
            "REVIEW_PREREQUISITE",
            "RETURN_TO_GOAL",
            "END_SESSION",
        )
    )
    previous_move = state.get("previous_move", "NONE")
    features.extend(
        float(previous_move == name)
        for name in (
            "PROBE",
            "HINT",
            "EXPLAIN_CONCEPT",
            "WORKED_EXAMPLE",
            "PRACTICE",
            "CHALLENGE",
            "CHECK_MASTERY",
            "REFLECT",
            "NONE",
        )
    )
    scalars = (
        min(1.0, max(0.0, float(_get(state, "turn", 0)) / 16)),
        min(1.0, max(0.0, float(_get(state, "turns_in_option", 0)) / 4)),
        min(1.0, max(0.0, float(_get(state, "hint_count", 0)) / 4)),
        min(1.0, max(0.0, float(_get(state, "independent_success_streak", 0)) / 3)),
        min(1.0, max(0.0, float(_get(state, "remaining_budget", 16)) / 16)),
        float(bool(_get(state, "explicit_skip", False))),
        float(bool(_get(state, "explicit_faster", False))),
        float(bool(_get(state, "explicit_slower", False))),
        float(bool(_get(state, "response_failure", False))),
        min(1.0, max(0.0, float(_get(state, "stack_depth", 0)) / 4)),
    )
    features.extend(scalars)
    if len(features) != 140:
        raise AssertionError(f"feature vector width is {len(features)}, expected 140")
    return features


def _one_hot(value: object, prefix: str, width: int) -> list[float]:
    result = [0.0] * width
    if isinstance(value, str) and value.startswith(prefix):
        try:
            index = int(value[len(prefix) :]) - 1
        except ValueError:
            return result
        if 0 <= index < width:
            result[index] = 1.0
    return result


def feature_order() -> list[str]:
    return list(FEATURE_NAMES)
