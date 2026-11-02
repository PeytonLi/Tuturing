from __future__ import annotations

from fractions import Fraction

from packages.learning.content.templates.families import TemplateFamily, normalize_answer


def grade_answer(
    answer: str | int | float | Fraction | None,
    instance: dict[str, object],
    family: TemplateFamily,
) -> str:
    if answer is None:
        return "unknown"
    normalized = family.answer_normalizer(answer, 1e-9)
    expected = instance.get("answer")
    if normalized is None or not isinstance(expected, Fraction):
        return "unknown"
    return "correct" if normalized == expected else "incorrect"


def normalize_for_instance(answer: str | int | float | Fraction | None) -> Fraction | None:
    return normalize_answer(answer) if answer is not None else None
