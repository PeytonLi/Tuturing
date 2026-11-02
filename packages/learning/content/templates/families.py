from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

MISCONCEPTIONS = tuple(f"M{index:02d}" for index in range(1, 9))
SPLIT_BY_VARIANT = ("train", "train", "validation", "test")
ASSESMENT_SPLITS = ("train", "validation", "test")


def normalize_answer(
    answer: str | int | float | Fraction, tolerance: float = 1e-9
) -> Fraction | None:
    if isinstance(answer, Fraction):
        return answer
    text = str(answer).strip().replace(",", "")
    try:
        if text.endswith("%"):
            value = Fraction(Decimal(text[:-1].strip())) / 100
        elif re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", text):
            numerator, denominator = re.split(r"\s*/\s*", text)
            if int(denominator) == 0:
                return None
            value = Fraction(int(numerator), int(denominator))
        else:
            value = Fraction(Decimal(text))
    except (ArithmeticError, ValueError):
        return None
    approximation = value.limit_denominator(1_000_000)
    return approximation if abs(float(value - approximation)) <= tolerance else value


def _solver(parameters: dict[str, int]) -> Fraction:
    return Fraction(parameters["numerator"], parameters["denominator"])


def _sampler(concept_index: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    if concept_index == 0:
        denominator = rng.choice((4, 6, 8, 10, 12))
        numerator = rng.randint(1, denominator - 1)
    elif concept_index == 1:
        denominator = rng.randint(5, 15)
        numerator = rng.randint(1, denominator - 1)
    elif concept_index == 2:
        denominator = rng.randint(4, 14)
        numerator = rng.randint(1, denominator - 1)
    elif concept_index == 3:
        denominator = rng.randint(4, 12)
        numerator = rng.randint(1, denominator)
    else:
        denominator = rng.randint(2, 6) * rng.randint(2, 6)
        numerator = rng.randint(1, denominator - 1)
    return {"numerator": numerator, "denominator": denominator}


def _error_factory(misconception_id: str) -> Callable[[dict[str, int]], Fraction]:
    def generate(parameters: dict[str, int]) -> Fraction:
        answer = _solver(parameters)
        number = int(misconception_id[1:])
        candidates = (
            Fraction(1, parameters["denominator"]),
            Fraction(parameters["numerator"] + 1, parameters["denominator"]),
            Fraction(parameters["numerator"], parameters["denominator"] + 1),
            1 - answer,
            Fraction(parameters["denominator"], parameters["numerator"]),
            Fraction(max(1, parameters["numerator"] - 1), max(1, parameters["denominator"] - 1)),
            min(Fraction(1), answer + Fraction(1, 7)),
            answer * Fraction(1, 2),
        )
        wrong = candidates[number - 1]
        if wrong == answer:
            wrong = Fraction(
                (answer.numerator + 1) % (answer.denominator + 1), answer.denominator + 1
            )
        if wrong == answer:
            wrong = Fraction(1, 2) if answer != Fraction(1, 2) else Fraction(1, 3)
        return wrong

    return generate


def _example_generator(seed: int, correct: Fraction) -> dict[str, str]:
    rng = random.Random(seed ^ 0xA11CE)
    denominator = rng.randint(3, 12)
    numerator = rng.randint(1, denominator - 1)
    candidates = [Fraction(numerator, denominator), Fraction(numerator + 1, denominator + 1)]
    candidates.extend(Fraction(1, divisor) for divisor in range(2, 32))
    answer = next(candidate for candidate in candidates if candidate != correct and candidate <= 1)
    return {
        "prompt": (
            f"A separate spinner has {answer.numerator} favorable sections "
            f"out of {answer.denominator}."
        ),
        "answer": f"{answer.numerator}/{answer.denominator}",
    }


FACTS = {
    "C01": ["A sample space lists all possible outcomes.", "An event is a set of outcomes."],
    "C02": [
        "A probability is between zero and one.",
        "Equally likely outcomes have equal probability.",
    ],
    "C03": ["An event and its complement have probabilities that sum to one."],
    "C06": ["Conditional probability restricts the denominator to the condition event."],
    "C08": ["A path probability is the product of its conditional branch probabilities."],
}
HINTS = {
    "C01": (
        "List every outcome.",
        "Mark the outcomes in the event.",
        "Divide favorable outcomes by all equally likely outcomes.",
    ),
    "C02": (
        "Check that the probability is a ratio.",
        "Count favorable and total outcomes.",
        "Reduce favorable outcomes over total outcomes.",
    ),
    "C03": (
        "Name the event that is not wanted.",
        "Use the full probability as one.",
        "Subtract the event probability from one.",
    ),
    "C06": (
        "Identify what is already known.",
        "Count only outcomes satisfying the condition.",
        "Divide the intersection count by the condition count.",
    ),
    "C08": (
        "Trace one complete path.",
        "Write each branch probability.",
        "Multiply the probabilities along that path.",
    ),
}


@dataclass(frozen=True)
class TemplateFamily:
    template_family_id: str
    concept_id: str
    difficulty: str
    split: str
    role: str
    pool: str
    sampler: Callable[[int], dict[str, int]]
    solver: Callable[[dict[str, int]], Fraction]
    answer_normalizer: Callable[[str | int | float | Fraction, float], Fraction | None]
    error_generators: dict[str, Callable[[dict[str, int]], Fraction]]
    hint_ladder: tuple[str, str, str]
    worked_example_generator: Callable[[int, Fraction], dict[str, str]]
    approved_facts: tuple[str, ...]
    item_kind: str

    def sample(self, seed: int) -> dict[str, int | Fraction]:
        parameters = self.sampler(seed)
        answer = self.solver(parameters)
        return {**parameters, "answer": answer, "seed": seed}

    def generate_error(self, misconception_id: str, parameters: dict[str, int]) -> Fraction:
        if misconception_id not in self.error_generators:
            raise KeyError(misconception_id)
        return self.error_generators[misconception_id](parameters)

    def worked_example(self, seed: int, instance_answer: Fraction) -> dict[str, str]:
        return self.worked_example_generator(seed, instance_answer)


def _families() -> dict[str, TemplateFamily]:
    result: dict[str, TemplateFamily] = {}
    for concept_index, concept_id in enumerate(("C01", "C02", "C03", "C06", "C08")):
        for variant, split in enumerate(SPLIT_BY_VARIANT):
            family_id = f"{concept_id.lower()}_teach_{variant + 1:02d}"
            item_kind = ("dice", "coins", "marbles", "spinners")[variant]
            result[family_id] = TemplateFamily(
                family_id,
                concept_id,
                ("easy", "medium", "medium", "hard")[variant],
                split,
                "teaching",
                "teaching",
                lambda seed, i=concept_index: _sampler(i, seed),
                _solver,
                normalize_answer,
                {
                    misconception_id: _error_factory(misconception_id)
                    for misconception_id in MISCONCEPTIONS
                },
                HINTS[concept_id],
                _example_generator,
                tuple(FACTS[concept_id]),
                item_kind,
            )
        for variant, split in enumerate(ASSESMENT_SPLITS):
            family_id = f"{concept_id.lower()}_assess_{variant + 1:02d}"
            result[family_id] = TemplateFamily(
                family_id,
                concept_id,
                ("easy", "medium", "hard")[variant],
                split,
                "assessment",
                "train-reward" if split == "train" else split,
                lambda seed, i=concept_index: _sampler(i, seed ^ 0xA55E55),
                _solver,
                normalize_answer,
                {
                    misconception_id: _error_factory(misconception_id)
                    for misconception_id in MISCONCEPTIONS
                },
                HINTS[concept_id],
                _example_generator,
                tuple(FACTS[concept_id]),
                ("cards", "two_way_table", "tree")[variant],
            )
    return result


FAMILIES = _families()


def families_for(
    concept_id: str, *, role: str | None = None, split: str | None = None
) -> list[TemplateFamily]:
    return [
        family
        for family in FAMILIES.values()
        if family.concept_id == concept_id
        and (role is None or family.role == role)
        and (split is None or family.split == split)
    ]
