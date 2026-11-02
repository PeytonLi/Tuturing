from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from packages.simulator.pace_cues import (
    DEFAULT_PACE_CUES,
    detail_drive,
    pace_cue_parameters,
    sample_pace_cues,
)
from packages.simulator.randomness import EventRandomBank
from packages.simulator.scenarios import load_scenarios
from packages.simulator.student import (
    PrivateStudentState,
    help_seeking_probability,
    sample_student,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "configs/learning/simulator_proto_v1.yaml",
    "configs/learning/simulator_proto_v2.yaml",
    "configs/learning/simulator_proto_alt.yaml",
)
SCENARIO = {"scenario_id": "pace-test"}
DRAWS = 400


def _config(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def _student(preferred_pace: str) -> PrivateStudentState:
    return PrivateStudentState(
        mastery={},
        learning_rates={},
        guess={},
        slip={},
        profile_id="fixture",
        preferred_pace=preferred_pace,
    )


def _draws(preferred_pace: str, tutor_pace: str, config: dict) -> list[dict]:
    bank = EventRandomBank()
    student = _student(preferred_pace)
    return [
        sample_pace_cues(
            student,
            tutor_pace,
            scenario=SCENARIO,
            episode_seed=seed,
            turn=3,
            bank=bank,
            config=config,
        )
        for seed in range(DRAWS)
    ]


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def test_pace_cues_are_deterministic_and_keyed_by_turn():
    config = _config(CONFIGS[1])
    bank = EventRandomBank()
    student = _student("detailed")
    kwargs = {"scenario": SCENARIO, "episode_seed": 7, "bank": bank, "config": config}
    first = sample_pace_cues(student, "normal", turn=2, **kwargs)
    assert first == sample_pace_cues(student, "normal", turn=2, **kwargs)
    assert set(first) == {"reply_seconds", "message_words", "skim", "confusion", "why_question"}
    assert isinstance(first["reply_seconds"], float) and first["reply_seconds"] > 0
    assert isinstance(first["message_words"], int) and first["message_words"] >= 1
    others = [sample_pace_cues(student, "normal", turn=turn, **kwargs) for turn in range(3, 9)]
    assert any(row != first for row in others)


def test_pace_cue_stream_leaves_existing_streams_unchanged():
    bank = EventRandomBank()
    before = bank.rng("pace-test", 7, 2, "help_request").random()
    sample_pace_cues(
        _student("concise"),
        "detailed",
        scenario=SCENARIO,
        episode_seed=7,
        turn=2,
        bank=bank,
        config={},
    )
    assert bank.rng("pace-test", 7, 2, "help_request").random() == before


def test_cues_separate_by_preferred_pace_and_mismatch():
    config = _config(CONFIGS[1])
    concise = _draws("concise", "normal", config)
    normal = _draws("normal", "normal", config)
    detailed = _draws("detailed", "normal", config)
    assert _mean(concise, "reply_seconds") < _mean(normal, "reply_seconds")
    assert _mean(normal, "reply_seconds") < _mean(detailed, "reply_seconds")
    assert _mean(concise, "message_words") < _mean(detailed, "message_words")
    assert _mean(concise, "skim") > _mean(normal, "skim") > _mean(detailed, "skim")
    assert _mean(detailed, "confusion") > _mean(concise, "confusion")
    assert _mean(detailed, "why_question") > _mean(concise, "why_question")
    # Noise keeps the signal imperfect: the reply-time ranges overlap.
    assert max(row["reply_seconds"] for row in concise) > min(
        row["reply_seconds"] for row in detailed
    )
    too_long = _draws("concise", "detailed", config)
    assert _mean(too_long, "skim") > _mean(_draws("concise", "concise", config), "skim")
    too_short = _draws("detailed", "concise", config)
    matched = _draws("detailed", "detailed", config)
    assert _mean(too_short, "confusion") > _mean(matched, "confusion")
    assert _mean(too_short, "reply_seconds") > _mean(matched, "reply_seconds")


def test_strength_scales_separation():
    full = _config(CONFIGS[1])
    half = {"pace_cues": {**full["pace_cues"], "strength": 0.5}}
    zero = {"pace_cues": {**full["pace_cues"], "strength": 0.0}}

    def gap(config: dict) -> tuple[float, float]:
        concise = _draws("concise", "normal", config)
        detailed = _draws("detailed", "normal", config)
        return (
            _mean(detailed, "reply_seconds") - _mean(concise, "reply_seconds"),
            _mean(concise, "skim") - _mean(detailed, "skim"),
        )

    full_gap, half_gap = gap(full), gap(half)
    assert 0 < half_gap[0] < full_gap[0]
    assert 0 < half_gap[1] < full_gap[1]
    assert _draws("concise", "detailed", zero) == _draws("detailed", "concise", zero)
    alt = _config(CONFIGS[2])
    assert pace_cue_parameters(alt)["strength"] == pytest.approx(0.5)


def test_missing_pace_cues_config_uses_defaults():
    assert pace_cue_parameters({}) == pace_cue_parameters({"pace_cues": DEFAULT_PACE_CUES})
    partial = pace_cue_parameters({"pace_cues": {"reply_seconds": {"sigma": 0.9}}})
    assert partial["reply_seconds"]["sigma"] == 0.9
    assert partial["reply_seconds"]["median"] == DEFAULT_PACE_CUES["reply_seconds"]["median"]
    rows = _draws("detailed", "concise", {})
    assert len(rows) == DRAWS
    assert _mean(rows, "reply_seconds") > _mean(_draws("concise", "detailed", {}), "reply_seconds")


def test_unknown_pace_is_rejected():
    assert detail_drive("detailed", "concise", 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        detail_drive("normal", "fast", 0.5)


@pytest.mark.parametrize("config_path", CONFIGS)
def test_careful_detailed_samples_in_every_config(config_path):
    config = _config(config_path)
    profile = config["profiles"]["careful_detailed"]
    assert profile["preferred_pace"] == "detailed"
    assert (
        profile["help_seeking_probability"]
        > config["profiles"]["knowledgeable_concise"]["help_seeking_probability"]
    )
    pace_scenarios = [
        row
        for row in load_scenarios(include_pace_scenarios=True)
        if row["profile_id"] == "careful_detailed"
    ]
    assert {row["split"] for row in pace_scenarios} == {"train", "validation", "test"}
    for scenario in pace_scenarios:
        student = sample_student(scenario, 0, config_path=config_path)
        assert student.preferred_pace == "detailed"
        assert all(0 <= value <= 1 for value in student.mastery.values())
        low, high = profile["concept_mastery"]
        assert all(low <= value <= high for value in student.mastery.values())


def test_pace_scenarios_are_opt_in_and_keep_frozen_ids():
    frozen = load_scenarios()
    combined = load_scenarios(include_pace_scenarios=True)
    assert len(frozen) == 30
    assert combined[:30] == frozen
    assert [row["scenario_id"] for row in combined[30:]] == [
        f"proto-scenario-{number}" for number in range(31, 37)
    ]


def test_help_seeking_rises_only_for_detailed_learner_taught_too_concisely():
    config = _config(CONFIGS[1])
    detailed = _student("detailed")
    detailed.help_seeking_probability = 0.15
    bonus = config["help_seeking"]["detailed_too_concise_bonus"]
    assert help_seeking_probability(detailed, None, config) == 0.15
    assert help_seeking_probability(detailed, "detailed", config) == 0.15
    assert help_seeking_probability(detailed, "concise", config) == pytest.approx(0.15 + bonus)
    assert help_seeking_probability(detailed, "concise", {}) == 0.15
    concise = _student("concise")
    concise.help_seeking_probability = 0.05
    assert help_seeking_probability(concise, "detailed", config) == 0.05
