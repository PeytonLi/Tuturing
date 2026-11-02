from __future__ import annotations

from dataclasses import dataclass

MOVES = (
    "PROBE",
    "HINT",
    "EXPLAIN_CONCEPT",
    "WORKED_EXAMPLE",
    "PRACTICE",
    "CHALLENGE",
    "CHECK_MASTERY",
    "REFLECT",
)
DIFFICULTIES = ("easy", "medium", "hard")
PACES = ("concise", "normal", "detailed")
OPERATIONS = ("STAY", "ADVANCE_ONE", "SKIP_TO_TARGET", "REVIEW_PREREQUISITE", "RETURN_TO_GOAL")
NAVIGATOR_ACTIONS = 76
STRATEGIST_ACTIONS = 72


def encode_strategist(move: str, difficulty: str, pace: str) -> int:
    return MOVES.index(move) * 9 + DIFFICULTIES.index(difficulty) * 3 + PACES.index(pace)


def decode_strategist(action_id: int) -> tuple[str, str, str]:
    if not 0 <= action_id < STRATEGIST_ACTIONS:
        raise ValueError("strategist action id outside [0, 71]")
    move, within_move = divmod(action_id, 9)
    difficulty, pace = divmod(within_move, 3)
    return MOVES[move], DIFFICULTIES[difficulty], PACES[pace]


def encode_navigator(operation: str, concept_index: int | None = None) -> int:
    if operation == "END_SESSION":
        if concept_index is not None:
            raise ValueError("END_SESSION has no target concept")
        return 75
    if concept_index is None or not 0 <= concept_index < 15:
        raise ValueError("navigator concept index must be in [0, 14]")
    return OPERATIONS.index(operation) * 15 + concept_index


def decode_navigator(action_id: int) -> tuple[str, int | None]:
    if action_id == 75:
        return "END_SESSION", None
    if not 0 <= action_id < 75:
        raise ValueError("navigator action id outside [0, 75]")
    operation, concept_index = divmod(action_id, 15)
    return OPERATIONS[operation], concept_index


def _index(concept_id: str | None) -> int | None:
    if not concept_id or not concept_id.startswith("C"):
        return None
    try:
        number = int(concept_id[1:])
    except ValueError:
        return None
    return number - 1 if 1 <= number <= 15 else None


def strategist_mask(state: dict) -> list[bool]:
    mask = [True] * STRATEGIST_ACTIONS
    mastery = float(state.get("mastery", 0.35))
    recent_correct = bool(state.get("recent_independent_correct", False))
    active = bool(state.get("active_item", state.get("problem_instance_id")))
    remaining_hints = bool(state.get("remaining_hints", state.get("hint_level", 0) < 3))
    previous_step = bool(state.get("previous_step", False))
    active_difficulty = state.get("active_item_difficulty", "medium")
    for pace_index, pace in enumerate(PACES):
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            for move_index, move in enumerate(MOVES):
                action_id = move_index * 9 + difficulty_index * 3 + pace_index
                if move == "CHALLENGE" and (
                    mastery < 0.65 or not recent_correct or difficulty != "hard"
                ):
                    mask[action_id] = False
                if move in {"HINT", "REFLECT"} and (
                    not active
                    or (move == "HINT" and not remaining_hints)
                    or (move == "REFLECT" and not previous_step)
                    or difficulty != active_difficulty
                ):
                    mask[action_id] = False
    if not any(mask):
        mask[encode_strategist("PROBE", "easy", "normal")] = True
    return mask


def navigator_mask(
    state: dict, graph: dict, boundary_reasons: list[str] | None = None
) -> list[bool]:
    mask = [False] * NAVIGATOR_ACTIONS
    current = state.get("current_concept_id")
    goal = state.get("goal_concept_id")
    current_index = _index(current)
    goal_index = _index(goal)
    enabled = set(state.get("enabled_concepts", []))
    if not enabled:
        enabled = set(graph.get("enabled_concepts", []))
    mastery = state.get("mastery_by_concept", {})
    operations = graph.get("operations", {})
    if current in enabled and current_index is not None:
        mask[encode_navigator("STAY", current_index)] = True
    for successor in operations.get("successors", {}).get(current, []):
        index = _index(successor)
        if successor in enabled and index is not None:
            mask[encode_navigator("ADVANCE_ONE", index)] = True
    if goal in enabled and goal_index is not None and state.get("explicit_skip"):
        mask[encode_navigator("SKIP_TO_TARGET", goal_index)] = True
    admissible_path = set(state.get("goal_ancestors", [])) | {goal}
    stack = list(state.get("goal_stack", []))
    if stack:
        index = _index(state.get("weakest_ancestor") or operations.get("weakest_ancestor"))
        if index is not None and f"C{index + 1:02d}" in enabled:
            mask[encode_navigator("REVIEW_PREREQUISITE", index)] = True
        return_index = _index(stack[-1])
        if return_index is not None and stack[-1] in enabled:
            mask[encode_navigator("RETURN_TO_GOAL", return_index)] = True
    else:
        for ancestor in operations.get("ancestors", {}).get(current, []):
            index = _index(ancestor)
            if (
                index is not None
                and ancestor in enabled
                and float(mastery.get(ancestor, 1.0)) < 0.8
            ):
                mask[encode_navigator("REVIEW_PREREQUISITE", index)] = True
    can_advance = int(state.get("independent_success_streak", 0)) >= 2 or bool(
        state.get("current_certified")
    )
    if not can_advance:
        for successor in operations.get("successors", {}).get(current, []):
            index = _index(successor)
            if index is not None:
                mask[encode_navigator("ADVANCE_ONE", index)] = False
    for successor in operations.get("successors", {}).get(current, []):
        if successor not in admissible_path:
            index = _index(successor)
            if index is not None:
                mask[encode_navigator("ADVANCE_ONE", index)] = False
    if state.get("goal_certified") or state.get("allow_policy_stop") or state.get("learner_stop"):
        mask[75] = True
    if not any(mask):
        if current_index is not None:
            mask[encode_navigator("STAY", current_index)] = True
        else:
            mask[75] = True
    return mask


@dataclass(frozen=True)
class ForcedAction:
    action_id: int
    actor_eligible: bool = False
    reveal_allowed: bool = True
    assistance: str = "answer_shown"


def forced_answer_action(*, difficulty: str = "medium", pace: str = "normal") -> ForcedAction:
    return ForcedAction(encode_strategist("EXPLAIN_CONCEPT", difficulty, pace))
