from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from packages.learning.content.templates.families import FAMILIES
from packages.simulator.student import PrivateStudentState, probability_correct

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RewardComponents:
    terminal_gain: float = 0.0
    teaching_cost: float = 0.0
    disclosure_penalty: float = 0.0
    repetition_penalty: float = 0.0
    dropout_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    shaping: float = 0.0

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def assessment_split(split: str) -> str:
    return "train-reward" if split == "train" else split


def proto_ancestors(concept_id: str) -> list[str]:
    concepts = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    edges = concepts["graphs"]["graph_proto_v1"]
    ancestors: set[str] = set()
    frontier = [concept_id]
    while frontier:
        child = frontier.pop()
        for edge in edges:
            parent = edge["source_concept_id"]
            if edge["dependent_concept_id"] == child and parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)
    return sorted(ancestors)


def _proto_graph() -> dict:
    concepts = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    prerequisites: dict[str, list[str]] = {}
    for edge in concepts["graphs"]["graph_proto_v1"]:
        prerequisites.setdefault(edge["dependent_concept_id"], []).append(edge["source_concept_id"])
    return {"prerequisites": prerequisites}


def expected_score(student: PrivateStudentState, goal_concept_id: str, split: str) -> float:
    ancestors = proto_ancestors(goal_concept_id)
    weights = {goal_concept_id: 0.5 if ancestors else 1.0}
    if ancestors:
        weights.update({ancestor: 0.5 / len(ancestors) for ancestor in ancestors})
    graph = _proto_graph()
    simulator_config = yaml.safe_load(
        (ROOT / "configs/learning/simulator_proto_v1.yaml").read_text(encoding="utf-8")
    )
    misconception_effect = float(
        simulator_config["misconceptions"]["susceptible_incorrect_probability"]
    )
    score = 0.0
    selected_split = assessment_split(split)
    for concept_id, weight in weights.items():
        if concept_id not in student.mastery:
            continue
        probability = probability_correct(student, concept_id, "medium", graph)
        susceptible = any(
            concept_id in student.susceptible_concepts.get(misconception, set())
            for misconception in student.active_misconceptions
        )
        if susceptible:
            probability *= 1 - misconception_effect
        assessment_families = [
            family
            for family in FAMILIES.values()
            if family.concept_id == concept_id
            and family.role == "assessment"
            and family.pool == selected_split
        ]
        if not assessment_families:
            raise ValueError(f"missing assessment family for {concept_id} in {split}")
        score += weight * probability
    return score


def terminal_components(
    pre_score: float,
    post_score: float,
    *,
    dropout: bool = False,
    config_path: str | Path = "configs/learning/reward_v1.yaml",
) -> RewardComponents:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    return RewardComponents(
        terminal_gain=float(config["terminal_gain_multiplier"]) * (post_score - pre_score),
        dropout_penalty=float(config["terminal_dropout"]) if dropout else 0.0,
        shaping=float(config["shaping"]),
    )


def turn_components(
    *,
    disclosure: bool = False,
    unproductive_repetition: bool = False,
    config_path: str | Path = "configs/learning/reward_v1.yaml",
) -> RewardComponents:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    return RewardComponents(
        teaching_cost=float(config["teaching_turn_cost"]),
        disclosure_penalty=float(config["unauthorized_disclosure"]) if disclosure else 0.0,
        repetition_penalty=float(config["unproductive_repetition"])
        if unproductive_repetition
        else 0.0,
        invalid_action_penalty=float(config["invalid_action_penalty"]),
        shaping=float(config["shaping"]),
    )
