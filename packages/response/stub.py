from __future__ import annotations

import hashlib
import json

from packages.learning.plan_rules import validate_plan_rules
from packages.response.types import GenerateRequest, GenerateResponse


class ResponseValidationError(ValueError):
    pass


PACE_CAPS = {"concise": 80, "normal": 150, "detailed": 250}


def _render(plan: dict) -> str:
    action = plan["teaching_action"]
    content = plan["content"]
    move = action["move"]
    stem = content.get("problem_stem") or "the current probability idea"
    if move == "PROBE":
        return f"Let's check your thinking about this: {stem} Show how you reason it out."
    if move == "HINT":
        return f"Here's a hint: {content['permitted_hint']} Try the problem again."
    if move == "EXPLAIN_CONCEPT":
        text = " ".join(content.get("approved_facts", []))
        if plan["constraints"]["reveal_allowed"] and content.get("reveal_solution") is not None:
            text += f" The solution is {content['reveal_solution']}."
        return text
    if move == "WORKED_EXAMPLE":
        return content["worked_example"]
    if move == "PRACTICE":
        return f"Try this: {stem} Share your answer and reasoning."
    if move == "CHALLENGE":
        return f"Here is a challenge: {stem} Explain your approach."
    if move == "CHECK_MASTERY":
        return f"Work this one independently: {stem} What is your answer?"
    if move == "REFLECT":
        return f"Think about this previous step: {content['previous_step']} What led you to it?"
    raise ResponseValidationError(f"unsupported move: {move}")


def render_template(plan: dict) -> str:
    """Return the stub's raw template text for a plan, before word and question limiting.

    The response pipeline uses this as its deterministic template responder and runs the
    full response checks on it. ``generate`` below is unchanged for existing callers.
    """
    return _render(plan)


def _limit_questions(text: str, maximum: int) -> str:
    if maximum <= 0:
        return text.replace("?", ".")
    pieces = text.split("?")
    if len(pieces) <= maximum + 1:
        return text
    return "?".join(pieces[:maximum]) + "."


def _limit_words(text: str, maximum: int) -> str:
    words = text.split()
    return " ".join(words[:maximum])


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def generate(request: GenerateRequest) -> GenerateResponse:
    plan = request["plan"]
    if (
        plan.get("assessment_mode") == "independent"
        and plan.get("content", {}).get("reveal_solution") is not None
    ):
        raise ResponseValidationError("independent plans cannot contain a reveal_solution")
    errors = validate_plan_rules(plan)
    if errors:
        raise ResponseValidationError("; ".join(errors))
    action = dict(plan["teaching_action"])
    maximum = min(int(plan["constraints"]["max_words"]), PACE_CAPS[action["pace"]])
    text = _limit_words(
        _limit_questions(_render(plan), int(plan["constraints"]["max_questions"])),
        maximum,
    )
    checks = {
        "schema_valid": True,
        "action_faithful": action == plan["teaching_action"],
        "math_supported": bool(
            plan["content"].get("approved_facts")
            or plan["content"].get("problem_stem")
            or plan["content"].get("permitted_hint")
            or plan["content"].get("worked_example")
            or plan["content"].get("previous_step")
        ),
        "disclosure_allowed": plan["constraints"]["reveal_allowed"]
        or plan["content"].get("reveal_solution") is None,
        "length_valid": len(text.split()) <= maximum,
        "question_count_valid": text.count("?") <= int(plan["constraints"]["max_questions"]),
    }
    if not all(checks.values()):
        raise ResponseValidationError("stub failed one or more response checks")
    canonical = _canonical_json(action)
    content_hash = hashlib.sha256(f"{text}\n{canonical}".encode()).hexdigest()
    response_hash = hashlib.sha256(request["request_id"].encode("utf-8")).hexdigest()[:24]
    return {
        "schema_version": "1.0",
        "request_id": request["request_id"],
        "plan_id": plan["plan_id"],
        "response_id": f"response-{response_hash}",
        "text": text,
        "effective_move": action,
        "delivery_mode": "fallback",
        "content_hash": content_hash,
        "checks": checks,
        "usage": {"input_tokens": 0, "output_tokens": len(text.split()), "elapsed_ms": 0},
        "versions": {
            "response_artifact": request["response_artifact_id"],
            "prompt": "stub_prompt_v1",
            "validator": "stub_validator_v1",
        },
    }
