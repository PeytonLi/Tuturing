from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConceptBelief:
    mastery: float = 0.35
    independent_correct: int = 0
    independent_incorrect: int = 0
    assisted_attempts: int = 0
    last_evidence_event_id: str | None = None
    recent_independent: deque[tuple[bool, str, str | None]] = field(
        default_factory=lambda: deque(maxlen=5)
    )

    @property
    def uncertainty(self) -> float:
        return 1.0 / math.sqrt(1 + self.independent_correct + self.independent_incorrect)

    @property
    def certified(self) -> bool:
        recent = list(self.recent_independent)
        if self.mastery < 0.80 or len(recent) < 2 or any(not item[0] for item in recent):
            return False
        latest = recent[-2:]
        return all(item[0] for item in latest) and len({item[1] for item in latest}) == 2


def bkt_posterior(
    mastery: float, correctness: str, guess: float = 0.20, slip: float = 0.10
) -> float:
    if correctness not in {"correct", "incorrect"}:
        return mastery
    if not 0 <= mastery <= 1 or not 0 <= guess or not 0 <= slip or guess + slip >= 1:
        raise ValueError("BKT parameters are outside their valid range")
    if correctness == "correct":
        numerator = mastery * (1 - slip)
        denominator = numerator + (1 - mastery) * guess
    else:
        numerator = mastery * slip
        denominator = numerator + (1 - mastery) * (1 - guess)
    return numerator / denominator if denominator else mastery


def update_belief(
    belief: ConceptBelief,
    correctness: str,
    *,
    assistance: str = "none",
    move: str | None = None,
    family_id: str | None = None,
    event_id: str | None = None,
    extractor_correctness: str | None = None,
    grader_correctness: str | None = None,
    conflict_logger: Callable[[dict[str, str | None]], None] | None = None,
    guess: float = 0.20,
    slip: float = 0.10,
    learning_probability: float = 0.08,
) -> ConceptBelief:
    if grader_correctness is not None:
        if (
            extractor_correctness is not None
            and extractor_correctness != grader_correctness
            and conflict_logger
        ):
            conflict_logger(
                {
                    "extractor_correctness": extractor_correctness,
                    "grader_correctness": grader_correctness,
                    "event_id": event_id,
                }
            )
        correctness = grader_correctness
    if assistance != "none":
        if correctness in {"correct", "incorrect"}:
            belief.assisted_attempts += 1
        return belief
    if correctness not in {"correct", "incorrect"}:
        return belief
    belief.mastery = bkt_posterior(belief.mastery, correctness, guess, slip)
    if correctness == "correct":
        belief.independent_correct += 1
    else:
        belief.independent_incorrect += 1
    if move in {"PRACTICE", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE"}:
        belief.mastery += (1 - belief.mastery) * learning_probability
    if family_id is not None:
        belief.recent_independent.append((correctness == "correct", family_id, event_id))
    if event_id is not None:
        belief.last_evidence_event_id = event_id
    return belief


def update_from_evidence(
    belief: ConceptBelief,
    evidence: dict[str, object],
    *,
    move: str | None = None,
    family_id: str | None = None,
    conflict_logger: Callable[[dict[str, str | None]], None] | None = None,
) -> ConceptBelief:
    correctness = str(evidence.get("correctness", "unknown"))
    if evidence.get("confidence", 1.0) < 0.80 and correctness in {"correct", "incorrect"}:
        correctness = "unknown"
    return update_belief(
        belief,
        correctness,
        assistance=str(evidence.get("assistance", "none")),
        move=move,
        family_id=family_id,
        event_id=str(evidence.get("event_id")) if evidence.get("event_id") is not None else None,
        extractor_correctness=str(evidence.get("extractor_correctness"))
        if evidence.get("extractor_correctness") is not None
        else None,
        grader_correctness=str(evidence.get("grader_correctness"))
        if evidence.get("grader_correctness") is not None
        else None,
        conflict_logger=conflict_logger,
    )
