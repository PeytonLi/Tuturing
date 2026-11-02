"""Plan verification before rendering (PRD 9.3 discriminated content rules).

``response_plan_errors`` holds the rules the response side needs in order to realize a
plan faithfully. ``plan_errors`` adds Person B's executable plan rules
(packages.learning.plan_rules), which also cover action-index consistency, the pace word
caps and the one-question maximum. Every message is a fixed string: no plan content,
fingerprint or learner text is ever quoted.
"""

from __future__ import annotations

from packages.learning.plan_rules import validate_plan_rules

ITEM_MOVES = frozenset({"PRACTICE", "CHALLENGE", "CHECK_MASTERY"})


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def response_plan_errors(plan: dict) -> list[str]:
    """Return rule violations for a schema-valid TeachingPlan."""
    errors = []
    move = plan["teaching_action"]["move"]
    content = plan["content"]
    constraints = plan["constraints"]
    has_item = bool(plan.get("problem_instance_id"))
    has_family = bool(plan.get("template_family_id"))

    if plan["assessment_mode"] == "independent":
        if content.get("reveal_solution") is not None:
            errors.append("independent plans must not carry a reveal_solution")
        if constraints["reveal_allowed"]:
            errors.append("independent plans must set reveal_allowed to false")
        if plan["assistance"] != "none":
            errors.append("independent plans must set assistance to none")
    if content.get("reveal_solution") is not None and not constraints["reveal_allowed"]:
        errors.append("reveal_solution requires reveal_allowed")

    if move in ITEM_MOVES:
        if not (has_item and has_family):
            errors.append(f"{move} requires a problem instance and a template family")
        if not _has_text(content.get("problem_stem")):
            errors.append(f"{move} requires a problem stem")
    elif move == "PROBE" and (has_item or has_family):
        if not (has_item and has_family):
            errors.append("PROBE with an item requires a problem instance and a template family")
        if not _has_text(content.get("problem_stem")):
            errors.append("PROBE with an item requires a problem stem")
    elif move == "HINT":
        if not has_item:
            errors.append("HINT requires an active item")
        if not _has_text(content.get("permitted_hint")):
            errors.append("HINT requires a permitted hint")
    elif move == "WORKED_EXAMPLE":
        if not _has_text(content.get("worked_example")):
            errors.append("WORKED_EXAMPLE requires an approved worked example")
    elif move == "EXPLAIN_CONCEPT":
        if not any(_has_text(fact) for fact in content.get("approved_facts") or []):
            errors.append("EXPLAIN_CONCEPT requires approved facts")
    elif move == "REFLECT":
        if not _has_text(content.get("previous_step")):
            errors.append("REFLECT requires a previous step")
    return errors


def plan_errors(plan: dict) -> list[str]:
    """Response-side rules plus Person B's plan rules, deduplicated in order."""
    return list(dict.fromkeys(response_plan_errors(plan) + validate_plan_rules(plan)))
