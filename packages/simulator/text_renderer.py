from __future__ import annotations

from packages.simulator.student import StudentOutcome


def render_student_outcome(outcome: StudentOutcome) -> str:
    if outcome.explicit_request != "none":
        return "Could I have a hint?"
    if outcome.correctness == "correct":
        return f"I think the answer is {outcome.normalized_answer}."
    if outcome.correctness == "incorrect":
        return f"My answer is {outcome.normalized_answer}."
    return "I am not sure yet."
