import json
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from packages.learning.actions import decode_strategist, encode_strategist
from packages.learning.belief import ConceptBelief
from packages.learning.content.templates.families import families_for
from packages.learning.episode import (
    _public_state,
    load_graph,
    update_public_misconceptions,
)
from packages.policies.rules import RulesStrategist
from packages.simulator.scenarios import load_scenarios
from packages.simulator.student import sample_student
from tools.learning.move_diagnostic import move_value
from tools.learning.score_decisions import case_state_to_inputs


def _sample_family_and_instance():
    family = families_for("C03", role="teaching", split="validation")[0]
    instance = family.sample(173)
    return family, instance


def test_public_misconception_updates_likely_confirmed_and_decays():
    family, instance = _sample_family_and_instance()
    misconception = sorted(family.error_generators)[0]
    error_answer = family.generate_error(misconception, instance)

    likely = update_public_misconceptions(
        {},
        family=family,
        instance=instance,
        normalized_answer=error_answer,
        correctness="incorrect",
        move="PRACTICE",
    )
    assert likely[misconception] == 0.6

    confirmed = update_public_misconceptions(
        likely,
        family=family,
        instance=instance,
        normalized_answer=error_answer,
        correctness="incorrect",
        move="PRACTICE",
    )
    assert confirmed[misconception] == 0.85

    probe_confirmed = update_public_misconceptions(
        {},
        family=family,
        instance=instance,
        normalized_answer=error_answer,
        correctness="incorrect",
        move="PROBE",
    )
    assert probe_confirmed[misconception] == 0.85

    decayed = update_public_misconceptions(
        confirmed,
        family=family,
        instance=instance,
        normalized_answer=Fraction(0),
        correctness="correct",
        move="PRACTICE",
    )
    assert decayed[misconception] == pytest.approx(0.425)
    nearly_cleared = update_public_misconceptions(
        {misconception: 0.3},
        family=family,
        instance=instance,
        normalized_answer=Fraction(0),
        correctness="correct",
        move="PRACTICE",
    )
    assert misconception not in nearly_cleared


def _rules_move(**overrides):
    context = {
        "mastery": 0.5,
        "pace_preference": "normal",
        "misconceptions": {},
        "meaningful_error": False,
        "independent_failure_streak": 0,
        "no_diagnostic_yet": False,
    }
    context.update(overrides)
    action = RulesStrategist().act([0.0] * 140, [True] * 72, context)
    return decode_strategist(action)[0]


def test_rules_strategist_responds_to_likely_and_confirmed_misconceptions():
    assert _rules_move(misconceptions={"M05": 0.6}) == "PROBE"
    assert (
        _rules_move(
            misconceptions={"M05": 0.85},
            mastery=0.3,
            previous_move="PRACTICE",
        )
        == "WORKED_EXAMPLE"
    )
    assert (
        _rules_move(
            misconceptions={"M05": 0.85},
            mastery=0.5,
            previous_move="PRACTICE",
        )
        == "EXPLAIN_CONCEPT"
    )
    assert (
        _rules_move(
            misconceptions={"M05": 0.85},
            previous_move="EXPLAIN_CONCEPT",
        )
        == "PRACTICE"
    )


def test_rules_strategist_legacy_contexts_and_mask_fallback():
    assert _rules_move(no_diagnostic_yet=True) == "PROBE"
    assert _rules_move(meaningful_error=True) == "HINT"
    assert _rules_move(wants_certification=True) == "CHECK_MASTERY"
    assert _rules_move(repeated_unsuccessful_hints=True) == "WORKED_EXAMPLE"
    assert _rules_move(misconceptions=[0.0] * 25, meaningful_error=True) == "HINT"
    dict_move = _rules_move(misconceptions={"M05": 0.6})
    list_values = [0.0] * 25
    list_values[4] = 0.6
    assert _rules_move(misconceptions=list_values) == dict_move == "PROBE"

    context = {
        "mastery": 0.5,
        "pace_preference": "normal",
        "misconceptions": {"M05": 0.6},
    }
    practice = encode_strategist("PRACTICE", "medium", "normal")
    mask = [False] * 72
    mask[practice] = True
    action = RulesStrategist().act([0.0] * 140, mask, context)
    assert action == practice
    assert RulesStrategist.policy_version == "rules_strategist_v2"


def test_decision_benchmark_passes_public_misconceptions_to_rules_strategist():
    cases = [
        json.loads(line)
        for line in (
            Path(__file__).resolve().parents[1]
            / "evaluation/learning/decision_cases_v1.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_moves = {
        "DC06": "PROBE",
        "DC07": "EXPLAIN_CONCEPT",
        "DC08": "PROBE",
        "DC09": "WORKED_EXAMPLE",
        "DC10": "PROBE",
        "DC28": "HINT",
    }
    for case in cases:
        if case["case_id"] not in expected_moves:
            continue
        _, observation, mask, context = case_state_to_inputs(case["state"])
        assert context["misconceptions"] == case["state"]["misconceptions"]
        assert context["previous_move"] == (
            case["state"].get("previous_move") or "NONE"
        )
        action = RulesStrategist().act(observation, mask, context)
        assert decode_strategist(action)[0] == expected_moves[case["case_id"]]


def test_public_state_keeps_zero_default_and_maps_misconceptions_by_id():
    beliefs = {f"C{index:02d}": ConceptBelief() for index in range(1, 16)}
    state = _public_state(
        beliefs,
        current="C01",
        goal="C01",
        turn=0,
        active_item=None,
        active_option_type=None,
        active_target=None,
        previous_move="NONE",
        option_turns=0,
        success_streak=0,
        failure_streak=0,
        hint_count=0,
        explicit_skip=False,
        explicit_faster=False,
        explicit_slower=False,
        stack=[],
    )
    assert state["misconceptions"] == [0.0] * 25
    assert "public_misconceptions" not in state

    state = _public_state(
        beliefs,
        current="C01",
        goal="C01",
        turn=0,
        active_item=None,
        active_option_type=None,
        active_target=None,
        previous_move="NONE",
        option_turns=0,
        success_streak=0,
        failure_streak=0,
        hint_count=0,
        explicit_skip=False,
        explicit_faster=False,
        explicit_slower=False,
        stack=[],
        misconceptions={"M05": 0.6, "M25": 0.85},
    )
    assert len(state["misconceptions"]) == 25
    assert state["misconceptions"][4] == 0.6
    assert state["misconceptions"][24] == 0.85
    assert "public_misconceptions" not in state


def test_move_diagnostic_probe_and_worked_example_values_for_validation_student():
    scenario = next(
        row
        for row in load_scenarios()
        if row["split"] == "validation"
        and row["profile_id"] == "novice_missing_prereqs"
    )
    config_path = "configs/learning/simulator_proto_v2.yaml"
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / config_path).read_text()
    )
    student = sample_student(scenario, 23, config_path=config_path)
    shared = {
        "concept": scenario["start_concept_id"],
        "goal": scenario["goal_concept_id"],
        "split": scenario["split"],
        "scenario": scenario,
        "graph": load_graph(),
        "config": config,
        "config_path": config_path,
        "last_answer_incorrect": False,
    }
    probe_mastery_delta, _ = move_value(student, move="PROBE", **shared)
    example_mastery_delta, _ = move_value(student, move="WORKED_EXAMPLE", **shared)
    assert probe_mastery_delta == pytest.approx(0.0)
    assert example_mastery_delta > 0.0
