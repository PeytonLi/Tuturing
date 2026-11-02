from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import yaml

from packages.learning.content.templates.families import TemplateFamily
from packages.simulator.randomness import EventRandomBank

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIMULATOR_CONFIG = "configs/learning/simulator_proto_v2.yaml"
DIFFICULTY_FACTORS = {"easy": 1.0, "medium": 0.85, "hard": 0.65}


@dataclass
class PrivateStudentState:
    mastery: dict[str, float]
    learning_rates: dict[str, float]
    guess: dict[str, float]
    slip: dict[str, float]
    profile_id: str
    active_misconceptions: set[str] = field(default_factory=set)
    surfaced_misconceptions: set[str] = field(default_factory=set)
    susceptible_concepts: dict[str, set[str]] = field(default_factory=dict)
    preferred_pace: str = "normal"
    help_seeking_probability: float = 0.0
    consecutive_independent_incorrect: int = 0
    last_answer_correct: bool | None = None
    last_answer_independent: bool = False
    last_answer_item_id: str | None = None
    previous_step: str | None = None
    move_history: list[tuple[str, str]] = field(default_factory=list)
    active_item_id: str | None = None


@dataclass(frozen=True)
class StudentOutcome:
    correctness: str
    normalized_answer: str | None
    assistance: str
    explicit_request: str
    misconception_used: str | None
    independent: bool


def _range_sample(rng, values: list[float]) -> float:
    return float(rng.uniform(float(values[0]), float(values[1])))


def _proto_prerequisites() -> dict[str, list[str]]:
    catalogue = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    result: dict[str, list[str]] = {}
    for edge in catalogue["graphs"]["graph_proto_v1"]:
        result.setdefault(edge["dependent_concept_id"], []).append(edge["source_concept_id"])
    return result


def sample_student(
    scenario: dict,
    episode_seed: int,
    *,
    config_path: str | Path = DEFAULT_SIMULATOR_CONFIG,
    bank: EventRandomBank | None = None,
) -> PrivateStudentState:
    random_bank = bank or EventRandomBank()
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    profile_id = scenario["profile_id"]
    profile = config["profiles"][profile_id]
    rng = random_bank.rng(scenario["scenario_id"], episode_seed, -1, "profile_initialization")
    concept_ids = [f"C{index:02d}" for index in range(1, 16)]
    goal = scenario["goal_concept_id"]
    if profile_id == "novice_missing_prereqs":
        mastery = {concept_id: float(rng.uniform(0.35, 0.70)) for concept_id in concept_ids}
        prerequisites = _proto_prerequisites()
        related = {scenario["start_concept_id"]}
        frontier = [scenario["start_concept_id"]]
        while frontier:
            concept = frontier.pop()
            for prerequisite in prerequisites.get(concept, []):
                if prerequisite not in related:
                    related.add(prerequisite)
                    frontier.append(prerequisite)
        for concept_id in related:
            mastery[concept_id] = _range_sample(rng, profile["concept_mastery"])
    else:
        low, high = profile["concept_mastery"]
        mastery = {concept_id: float(rng.uniform(low, high)) for concept_id in concept_ids}
        if profile_id == "knowledgeable_concise":
            mastery[goal] = _range_sample(rng, profile["goal_mastery"])
    rates = {concept_id: _range_sample(rng, profile["ell"]) for concept_id in concept_ids}
    guesses = {
        concept_id: float(rng.uniform(*config["observation"]["guess"]))
        for concept_id in concept_ids
    }
    slips = {
        concept_id: float(rng.uniform(*config["observation"]["slip"])) for concept_id in concept_ids
    }
    active = {profile["active_misconception"]} if profile.get("active_misconception") else set()
    susceptible = (
        {profile["active_misconception"]: set(profile.get("susceptible_concepts", []))}
        if profile.get("active_misconception")
        else {}
    )
    return PrivateStudentState(
        mastery=mastery,
        learning_rates=rates,
        guess=guesses,
        slip=slips,
        profile_id=profile_id,
        active_misconceptions=active,
        susceptible_concepts=susceptible,
        preferred_pace=profile["preferred_pace"],
        help_seeking_probability=float(profile["help_seeking_probability"]),
    )


def prerequisite_factor(concept_id: str, mastery: dict[str, float], graph: dict) -> float:
    prerequisites = graph.get("prerequisites", {}).get(concept_id, [])
    if not prerequisites:
        return 1.0
    return 0.5 + 0.5 * min(mastery[prerequisite] for prerequisite in prerequisites)


def probability_correct(
    student: PrivateStudentState,
    concept_id: str,
    difficulty: str,
    graph: dict,
) -> float:
    factor = prerequisite_factor(concept_id, student.mastery, graph)
    effective_mastery = min(
        1.0, max(0.0, student.mastery[concept_id] * factor * DIFFICULTY_FACTORS[difficulty])
    )
    guess = student.guess[concept_id]
    slip = student.slip[concept_id]
    if guess < 0 or slip < 0 or guess + slip >= 1:
        raise ValueError("guess/slip parameters violate the observation model")
    return guess + (1 - guess - slip) * effective_mastery


def sample_student_outcome(
    student: PrivateStudentState,
    *,
    scenario: dict,
    episode_seed: int,
    turn: int,
    concept_id: str,
    difficulty: str,
    item_id: str | None,
    assistance: str,
    family: TemplateFamily | None,
    instance: dict | None,
    graph: dict,
    bank: EventRandomBank,
    config: dict,
) -> StudentOutcome:
    if instance is None or family is None:
        return StudentOutcome("not_applicable", None, assistance, "none", None, False)
    help_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "help_request")
    if assistance != "answer_shown" and float(help_rng.random()) < student.help_seeking_probability:
        return StudentOutcome("not_applicable", None, "none", "hint", None, False)
    raw_answer = instance["answer"]
    expected = raw_answer if isinstance(raw_answer, Fraction) else Fraction(raw_answer)
    answer_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "student_answer")
    active_errors = [
        misconception
        for misconception in sorted(student.active_misconceptions)
        if concept_id in student.susceptible_concepts.get(misconception, set())
        and misconception in family.error_generators
    ]
    misconception_used = None
    if assistance == "answer_shown":
        submitted = expected
        correctness = "correct"
    elif active_errors:
        incorrect_probability = float(config["misconceptions"]["susceptible_incorrect_probability"])
        if float(answer_rng.random()) < incorrect_probability:
            misconception_used = active_errors[0]
            submitted = family.generate_error(misconception_used, instance)
            correctness = "incorrect"
        else:
            submitted = expected
            correctness = "correct"
    else:
        p_correct = probability_correct(student, concept_id, difficulty, graph)
        if float(answer_rng.random()) < p_correct:
            submitted = expected
            correctness = "correct"
        else:
            error_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "arithmetic_error")
            submitted = expected + Fraction(1, int(error_rng.integers(2, 8)))
            if submitted == expected:
                submitted = expected + Fraction(1, 3)
            correctness = "incorrect"
    return StudentOutcome(
        correctness,
        f"{submitted.numerator}/{submitted.denominator}",
        assistance,
        "none",
        misconception_used,
        assistance == "none",
    )


def apply_teaching_effect(
    student: PrivateStudentState,
    *,
    concept_id: str,
    move: str,
    difficulty: str,
    turn: int,
    scenario: dict,
    episode_seed: int,
    graph: dict,
    bank: EventRandomBank,
    config: dict,
    content_realized: bool,
    item_id: str | None,
) -> float:
    mastery = student.mastery[concept_id]
    effects = config["action_effect"]
    if move == "PROBE":
        action_effect = float(effects["PROBE"])
    elif move == "CHECK_MASTERY":
        action_effect = float(effects["CHECK_MASTERY"])
    elif move == "PRACTICE":
        action_effect = float(effects["PRACTICE"])
    elif move == "HINT":
        action_effect = float(
            effects["HINT"]["incorrect"]
            if student.last_answer_correct is False and student.last_answer_item_id == item_id
            else effects["HINT"]["otherwise"]
        )
    elif move == "EXPLAIN_CONCEPT":
        bands = effects["EXPLAIN_CONCEPT"]
        action_effect = float(
            bands["below_0_4"]
            if mastery < 0.4
            else bands["from_0_4_to_0_75"]
            if mastery <= 0.75
            else bands["above_0_75"]
        )
    elif move == "WORKED_EXAMPLE":
        bands = effects["WORKED_EXAMPLE"]
        action_effect = float(
            bands["below_0_4"]
            if mastery < 0.4
            else bands["from_0_4_to_0_75"]
            if mastery <= 0.75
            else bands["above_0_75"]
        )
    elif move == "CHALLENGE":
        bands = effects["CHALLENGE"]
        action_effect = float(bands["at_least_0_6"] if mastery >= 0.6 else bands["below_0_6"])
    else:
        bands = effects["REFLECT"]
        action_effect = float(
            bands["previous_and_incorrect"]
            if student.previous_step and student.last_answer_correct is False
            else bands["otherwise"]
        )
    prerequisite = prerequisite_factor(concept_id, student.mastery, graph)
    fit_config = config["readiness"]["difficulty_fit"][difficulty]
    if difficulty == "easy":
        fit = float(fit_config["below_0_5"] if mastery < 0.5 else fit_config["otherwise"])
    elif difficulty == "medium":
        fit = float(fit_config)
    else:
        fit = float(fit_config["at_least_0_6"] if mastery >= 0.6 else fit_config["otherwise"])
    readiness = prerequisite * fit
    repeated_moves = 0
    for previous_move, previous_concept in reversed(student.move_history):
        if (previous_move, previous_concept) != (move, concept_id):
            break
        repeated_moves += 1
    novelty = 0.6**repeated_moves
    fidelity_config = config["content_fidelity"]
    failure_probability = float(fidelity_config["failure_probability"])
    failure_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "content_fidelity_failure")
    fidelity = (
        0.0
        if not content_realized or float(failure_rng.random()) < failure_probability
        else float(fidelity_config["stub_realized"])
    )
    learning_rate = student.learning_rates[concept_id]
    delta = (1 - mastery) * learning_rate * action_effect * readiness * novelty * fidelity
    student.mastery[concept_id] = min(1.0, max(0.0, mastery + delta))
    student.move_history.append((move, concept_id))
    if len(student.move_history) > 16:
        del student.move_history[:-16]
    if move in {"PROBE", "REFLECT"} and item_id:
        for misconception in student.active_misconceptions:
            if concept_id in student.susceptible_concepts.get(misconception, set()):
                student.surfaced_misconceptions.add(misconception)
    if fidelity > 0 and move in {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"}:
        deactivation_rng = bank.rng(
            scenario["scenario_id"], episode_seed, turn, "misconception_deactivation"
        )
        for misconception in sorted(student.active_misconceptions):
            deactivation = config["misconceptions"]["deactivate_probability"]
            probability = float(
                deactivation["surfaced"]
                if misconception in student.surfaced_misconceptions
                else deactivation["not_surfaced"]
            )
            if float(deactivation_rng.random()) < probability:
                student.active_misconceptions.remove(misconception)
    student.previous_step = f"{move} on {concept_id}"
    return delta


def dropout_hazard(student: PrivateStudentState, pace: str, config: dict | None = None) -> float:
    return min(1.0, sum(dropout_hazard_components(student, pace, config).values()))


def dropout_hazard_components(
    student: PrivateStudentState, pace: str, config: dict | None = None
) -> dict[str, float]:
    dropout_config = (config or {}).get("dropout", {})
    base = float(dropout_config.get("base_per_turn", 0.01))
    per_failure = float(dropout_config.get("per_independent_incorrect_after_two", 0.02))
    per_mismatch = float(dropout_config.get("per_pace_mismatch_turn", 0.02))
    extra_incorrect = max(0, student.consecutive_independent_incorrect - 2)
    return {
        "base": base,
        "incorrect_streak": per_failure * extra_incorrect,
        "pace_mismatch": per_mismatch if pace != student.preferred_pace else 0.0,
    }
