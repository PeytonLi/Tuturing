"""Deterministic plan-grounded fallback responses."""

from packages.response.validate import run_checks

FALLBACK_VERSION = "fallback_v1"


class FallbackUnavailable(RuntimeError):
    """Raised when no deterministic fallback can safely realize the plan."""


def render_fallback(plan):
    content = plan["content"]
    constraints = plan["constraints"]
    move = plan["teaching_action"]["move"]
    stem = content.get("problem_stem")

    def append_question(text, question):
        if text.count("?") >= constraints["max_questions"]:
            return text
        return text + question

    if move == "HINT":
        text = append_question(
            f"Here is a hint: {content['permitted_hint']}",
            " What is your next step?",
        )
    elif move in {"PRACTICE", "CHECK_MASTERY", "CHALLENGE"}:
        prefix = "On your own, without hints: " if move == "CHECK_MASTERY" else ""
        suffix = (
            " Give it a try and tell me your answer."
            if "?" in stem else " What do you get?"
        )
        text = prefix + stem + suffix
    elif move == "WORKED_EXAMPLE":
        text = (
            f"Here is a worked example. {content['worked_example']} "
            "Try the same steps on your problem."
        )
    elif move == "EXPLAIN_CONCEPT":
        text = f"{content['concept_name']}: " + " ".join(content["approved_facts"])
        solution = content.get("reveal_solution")
        if constraints["reveal_allowed"] and solution:
            text += f" Here is the solution: {solution}"
        if "?" not in text:
            text = append_question(text, " Does that make sense?")
    elif move == "REFLECT":
        text = append_question(
            f"Look back at this step: {content['previous_step']}",
            " Can you explain why it works?",
        )
    elif move == "PROBE":
        if stem:
            text = stem
            if "?" not in text:
                text = append_question(text, " What do you think?")
        else:
            text = f"In your own words, what does {content['concept_name']} mean?"
    else:
        raise FallbackUnavailable(f"unsupported move: {move}")

    if not all(run_checks(plan, text, plan["teaching_action"]).values()):
        raise FallbackUnavailable("rendered fallback did not pass all response checks")
    return text
