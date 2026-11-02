from __future__ import annotations

import random

from packages.learning.actions import (
    DIFFICULTIES,
    MOVES,
    PACES,
    encode_navigator,
    encode_strategist,
)


def _valid(mask: list[bool], action_id: int) -> bool:
    return 0 <= action_id < len(mask) and bool(mask[action_id])


def _first_valid(mask: list[bool], preferences: list[int]) -> int:
    for action_id in preferences:
        if _valid(mask, action_id):
            return action_id
    return next((index for index, allowed in enumerate(mask) if allowed), 0)


def _target_index(concept_id: str | None) -> int | None:
    if not concept_id or not concept_id.startswith("C"):
        return None
    try:
        return int(concept_id[1:]) - 1
    except ValueError:
        return None


class RulesNavigator:
    policy_version = "rules_navigator_v1"

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        goal = context.get("goal_concept_id")
        current = context.get("current_concept_id")
        goal_index = _target_index(goal)
        current_index = _target_index(current)
        if context.get("learner_stop") and _valid(mask, 75):
            return 75
        if context.get("goal_certified") and _valid(mask, 75):
            return 75
        if context.get("remediation_certified") and context.get("goal_stack"):
            target = _target_index(context["goal_stack"][-1])
            if target is not None:
                action = encode_navigator("RETURN_TO_GOAL", target)
                if _valid(mask, action):
                    return action
        if context.get("explicit_skip") and goal_index is not None:
            action = encode_navigator("SKIP_TO_TARGET", goal_index)
            if _valid(mask, action):
                context["skip_diagnostic_deadline"] = int(context.get("turn_index", 0)) + 2
                context["skip_diagnostic_done"] = False
                return action
        if int(context.get("independent_failure_streak", 0)) >= 2:
            weakest = context.get("weakest_ancestor")
            target = _target_index(weakest)
            if target is not None:
                action = encode_navigator("REVIEW_PREREQUISITE", target)
                if _valid(mask, action):
                    return action
        if int(context.get("independent_success_streak", 0)) >= 2 or context.get(
            "current_certified"
        ):
            successors = sorted(context.get("successors", {}).get(current, []))
            on_path = [
                item
                for item in successors
                if item in context.get("goal_ancestors", set()) or item == goal
            ]
            for successor in on_path or successors:
                target = _target_index(successor)
                if target is not None:
                    action = encode_navigator("ADVANCE_ONE", target)
                    if _valid(mask, action):
                        return action
        if current_index is not None:
            stay = encode_navigator("STAY", current_index)
            if _valid(mask, stay):
                return stay
        return _first_valid(mask, [75])


class RulesStrategist:
    policy_version = "rules_strategist_v2"

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        difficulty = str(context.get("active_item_difficulty", "medium"))
        mastery = float(context.get("mastery", 0.35))
        difficulty = "easy" if mastery < 0.4 else "hard" if mastery >= 0.75 else "medium"
        pace = str(context.get("pace_preference", "normal"))
        if pace not in PACES:
            pace = "normal"
        misconceptions = context.get("misconceptions", {})
        if isinstance(misconceptions, dict):
            misconception_values = misconceptions.values()
        elif isinstance(misconceptions, (list, tuple)) and len(misconceptions) == 25:
            misconception_values = misconceptions
        else:
            misconception_values = ()
        probability = max((float(value) for value in misconception_values), default=0.0)
        likely_misconception = probability >= 0.5
        confirmed_misconception = probability >= 0.75 or (
            likely_misconception
            and context.get("previous_move") in {"PROBE", "REFLECT", "HINT"}
            and context.get("meaningful_error")
        )
        if context.get("skip_diagnostic_due"):
            move = "CHECK_MASTERY"
        elif context.get("unsupported_advancement"):
            move = "PROBE"
        elif (
            context.get("wants_certification")
            or int(context.get("independent_success_streak", 0)) >= 2
        ):
            move = "CHECK_MASTERY"
        elif (
            context.get("repeated_unsuccessful_hints")
            or int(context.get("unsuccessful_hint_count", 0)) >= 2
        ):
            move = "WORKED_EXAMPLE"
        elif likely_misconception and not confirmed_misconception:
            move = "PROBE"
        elif confirmed_misconception and context.get("previous_move") not in {
            "EXPLAIN_CONCEPT",
            "WORKED_EXAMPLE",
        }:
            move = "WORKED_EXAMPLE" if mastery < 0.4 else "EXPLAIN_CONCEPT"
        elif context.get("meaningful_error") or context.get("independent_failure_streak", 0):
            move = "HINT"
        elif context.get("no_diagnostic_yet"):
            move = "PROBE"
        else:
            move = "PRACTICE"
        if move in {"HINT", "REFLECT"} and context.get("active_item_difficulty") in DIFFICULTIES:
            difficulty = context["active_item_difficulty"]
        action = encode_strategist(move, difficulty, pace)
        if _valid(mask, action):
            return action
        preference = [
            encode_strategist(candidate, difficulty, pace)
            for candidate in (
                "PROBE",
                "CHECK_MASTERY",
                "PRACTICE",
                "HINT",
                "WORKED_EXAMPLE",
                "EXPLAIN_CONCEPT",
                "REFLECT",
            )
        ]
        return _first_valid(mask, preference)


class RandomStrategist:
    policy_version = "random_strategist_v1"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        valid = [index for index, allowed in enumerate(mask) if allowed]
        return self.rng.choice(valid) if valid else encode_strategist("PROBE", "easy", "normal")


class AlwaysHintStrategist:
    policy_version = "always_hint_strategist_v1"

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        difficulty = str(context.get("active_item_difficulty", "medium"))
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        pace = str(context.get("pace_preference", "normal"))
        if pace not in PACES:
            pace = "normal"
        hint = encode_strategist("HINT", difficulty, pace)
        if _valid(mask, hint):
            return hint
        fallback = encode_strategist("PROBE", "easy", pace)
        return _first_valid(mask, [fallback])


class AlwaysWorkedExampleStrategist:
    policy_version = "always_worked_example_strategist_v1"

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        difficulty = str(context.get("active_item_difficulty", "medium"))
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        pace = str(context.get("pace_preference", "normal"))
        if pace not in PACES:
            pace = "normal"
        worked_example = encode_strategist("WORKED_EXAMPLE", difficulty, pace)
        if _valid(mask, worked_example):
            return worked_example
        fallback = encode_strategist("PROBE", "easy", pace)
        return _first_valid(mask, [fallback])


def action_label(action_id: int) -> tuple[str, str, str]:
    move_index, rest = divmod(action_id, 9)
    difficulty_index, pace_index = divmod(rest, 3)
    return MOVES[move_index], DIFFICULTIES[difficulty_index], PACES[pace_index]
