from __future__ import annotations

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


def validate_plan_rules(plan: dict) -> list[str]:
    errors: list[str] = []
    action = plan.get("teaching_action", {})
    move = action.get("move")
    difficulty = action.get("difficulty")
    pace = action.get("pace")
    content = plan.get("content", {})
    constraints = plan.get("constraints", {})
    if move in MOVES and difficulty in DIFFICULTIES and pace in PACES:
        expected = MOVES.index(move) * 9 + DIFFICULTIES.index(difficulty) * 3 + PACES.index(pace)
        if action.get("action_id") != expected:
            errors.append("teaching action id disagrees with its encoded fields")
    else:
        errors.append("unknown teaching move, difficulty, or pace")
    navigation = plan.get("navigation_action", {})
    operation = navigation.get("operation")
    target = navigation.get("target_concept_id")
    if operation in OPERATIONS:
        try:
            target_index = int(str(target)[1:]) - 1 if str(target).startswith("C") else -1
        except ValueError:
            target_index = -1
        expected = (
            OPERATIONS.index(operation) * 15 + target_index if 0 <= target_index < 15 else None
        )
        if expected is None or navigation.get("action_id") != expected:
            errors.append("navigation action id disagrees with its encoded fields")
    elif operation == "END_SESSION":
        if navigation.get("action_id") != 75 or target is not None:
            errors.append("END_SESSION must use action 75 and a null target")
    else:
        errors.append("unknown navigation operation")
    independent = plan.get("assessment_mode") == "independent"
    if independent and (
        constraints.get("reveal_allowed") is not False
        or plan.get("assistance") != "none"
        or content.get("reveal_solution") is not None
    ):
        errors.append("independent plans must forbid reveals and assistance")
    if content.get("reveal_solution") is not None and constraints.get("reveal_allowed") is not True:
        errors.append("reveal_solution requires reveal_allowed")
    if move in {"PRACTICE", "CHALLENGE", "CHECK_MASTERY"}:
        if not plan.get("problem_instance_id") or not plan.get("template_family_id"):
            errors.append(f"{move} requires an item and template family")
    if move == "PROBE" and bool(plan.get("problem_instance_id")) != bool(
        plan.get("template_family_id")
    ):
        errors.append("PROBE item and template family must be set together")
    if move == "HINT":
        if not plan.get("problem_instance_id") or not content.get("permitted_hint"):
            errors.append("HINT requires an active item and permitted_hint")
        if plan.get("assistance") != "hint":
            errors.append("HINT must mark hint assistance")
    if move == "WORKED_EXAMPLE" and not content.get("worked_example"):
        errors.append("WORKED_EXAMPLE requires an approved worked example")
    if move == "EXPLAIN_CONCEPT" and not content.get("approved_facts"):
        errors.append("EXPLAIN_CONCEPT requires approved facts")
    if move == "REFLECT" and not content.get("previous_step"):
        errors.append("REFLECT requires a previous step")
    if move == "WORKED_EXAMPLE" and plan.get("assistance") != "example":
        errors.append("WORKED_EXAMPLE must mark example assistance")
    if move == "CHALLENGE" and difficulty != "hard":
        errors.append("CHALLENGE requires hard difficulty")
    pace_cap = {"concise": 80, "normal": 150, "detailed": 250}.get(pace)
    if pace_cap is not None and constraints.get("max_words", pace_cap) > pace_cap:
        errors.append("max_words exceeds the pace cap")
    if constraints.get("max_questions", 0) > 1:
        errors.append("max_questions exceeds one")
    return errors
