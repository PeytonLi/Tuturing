from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from fractions import Fraction

from packages.evidence_extractor.extract import extract
from packages.response.numbers import extract_numbers
from packages.simulator.student import StudentOutcome

RENDERER_VERSION = "student_templates_bridge_v1"
ANSWER_TEMPLATES = (
    "My answer is {answer}.",
    "I got {answer}.",
    "The final answer is {answer}.",
    "After thinking it through, my answer is {answer}.",
)
REQUEST_TEMPLATES = {
    "hint": ("Could I have a hint?", "Please give me a hint.", "I need a hint."),
    "answer": ("Please tell me the answer.",),
    "stop": ("Please stop.",),
    "skip": ("Please skip this.",),
    "faster": ("Please go faster.",),
    "slower": ("Please slow down.",),
}


def _signature(message: str, assistance: str) -> tuple:
    evidence = extract(message, {"assistance": assistance}, "renderer-check")
    answer = evidence["normalized_answer"]
    try:
        value = Fraction(answer) if answer is not None else None
    except (ValueError, ZeroDivisionError):
        value = answer
    return value, evidence["explicit_request"], evidence["assistance"]


def render_frozen(outcome: StudentOutcome, *, key: str = "") -> str:
    """Phrase a sampled outcome without sampling another student event."""
    digest = hashlib.sha256((RENDERER_VERSION + key).encode()).digest()
    index = int.from_bytes(digest[:4], "little")
    if outcome.explicit_request != "none":
        templates = REQUEST_TEMPLATES[outcome.explicit_request]
        request = templates[index % len(templates)]
    else:
        request = ""
    if outcome.normalized_answer is not None:
        answer = ANSWER_TEMPLATES[index % len(ANSWER_TEMPLATES)].format(
            answer=outcome.normalized_answer
        )
        return f"{answer} {request}".strip()
    return request or ("I am not sure yet.", "I don't know.", "I am unsure.")[index % 3]


def render_checked(
    outcome: StudentOutcome, *, key: str = "", verbalizer: Callable[[str], str] | None = None
) -> tuple[str, bool]:
    frozen = render_frozen(outcome, key=key)
    if verbalizer is None:
        return frozen, False
    allowed = {render_frozen(outcome, key=str(index)) for index in range(64)}
    prompt = (
        "Choose exactly one of these frozen student phrasings. Return that message only. "
        "Do not solve, add reasoning, facts or numbers. " + json.dumps(sorted(allowed))
    )
    candidate = verbalizer(prompt)
    # No numeric additions, omissions or request/assistance reinterpretations are accepted.
    if (
        isinstance(candidate, str)
        and candidate in allowed
        and _signature(candidate, outcome.assistance) == _signature(frozen, outcome.assistance)
        and [n.value for n in extract_numbers(candidate)]
        == [n.value for n in extract_numbers(frozen)]
    ):
        return candidate, False
    return frozen, True


def render_student_outcome(outcome: StudentOutcome, *, key: str = "") -> str:
    return render_frozen(outcome, key=key)
