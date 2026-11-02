import json

import pytest

import packages.learning.episode as episode
from packages.learning.features import build_features
from packages.learning.pace import (
    choose_pace,
    ensure_state,
    initial_state,
    lean_faster,
    lean_slower,
    select,
    step_pace,
    update,
)
from packages.simulator.scenarios import load_scenarios

FAST_CUES = {"reply_seconds": 3.0, "message_words": 1, "skim": True}
SLOW_CUES = {"reply_seconds": 90.0, "message_words": 40, "confusion": True, "why_question": True}


def test_step_pace_moves_one_step_and_clamps():
    assert step_pace("normal", "faster") == "concise"
    assert step_pace("concise", "faster") == "concise"
    assert step_pace("concise", "slower") == "normal"
    assert step_pace("normal", "slower") == "detailed"
    assert step_pace("detailed", "slower") == "detailed"
    assert step_pace("detailed", "faster") == "normal"
    assert step_pace("normal", None) == "normal"
    assert step_pace("normal", "hint") == "normal"


def test_normal_to_concise_and_back_to_normal():
    pace, state = "normal", initial_state("normal")
    state = update(state, request="faster", current=pace)
    pace, state = select(state, pace)
    assert pace == "concise"
    state = update(state, request="slower", current=pace)
    pace, state = select(state, pace)
    assert pace == "normal"


def test_explicit_request_applies_immediately_with_full_lean():
    state = update(initial_state("normal"), request="slower", current="normal")
    assert lean_slower(state) == 1.0
    assert choose_pace(state, "normal") == "detailed"
    state = update(initial_state("normal"), request="faster", current="normal")
    assert lean_faster(state) == 1.0
    assert choose_pace(state, "normal") == "concise"


def test_request_does_not_persist_to_the_next_turn():
    pace, state = "normal", initial_state("normal")
    state = update(state, request="faster", current=pace)
    pace, state = select(state, pace)
    state = update(state, current=pace)
    assert state["request"] is None
    assert lean_faster(state) < 1.0
    assert choose_pace(state, pace) == "concise"


def test_one_weak_turn_does_not_flip_two_do():
    pace, state = "normal", initial_state("normal")
    state = update(state, correct=True, success_streak=2, cues=FAST_CUES, current=pace)
    pace, state = select(state, pace)
    assert pace == "normal"
    state = update(state, correct=True, success_streak=3, cues=FAST_CUES, current=pace)
    pace, state = select(state, pace)
    assert pace == "concise"
    state = update(state, correct=True, success_streak=4, cues=FAST_CUES, current=pace)
    assert choose_pace(state, pace) == "concise"


def test_slow_evidence_leans_detailed_after_two_turns():
    pace, state = "normal", initial_state("normal")
    for expected, failures in (("normal", 2), ("detailed", 3)):
        state = update(
            state, correct=False, failure_streak=failures, used_hint=True, cues=SLOW_CUES,
            current=pace,
        )
        pace, state = select(state, pace)
        assert pace == expected


def test_alternating_weak_evidence_never_switches():
    pace, state = "normal", initial_state("normal")
    for index in range(8):
        cues = FAST_CUES if index % 2 == 0 else SLOW_CUES
        state = update(state, cues=cues, current=pace)
        pace, state = select(state, pace)
        assert pace == "normal"


def test_switch_steps_at_most_one_and_resets_hold():
    pace, state = "detailed", initial_state("detailed")
    for _ in range(2):
        state = update(state, correct=True, success_streak=5, cues=FAST_CUES, current=pace)
        pace, state = select(state, pace)
    assert pace == "normal"
    assert state["lead_count"] == 0


def test_decay_keeps_belief_from_saturating():
    state = initial_state("normal")
    for _ in range(50):
        state = update(state, correct=True, success_streak=9, cues=FAST_CUES, current="concise")
    assert max(state["belief"]) < 0.99
    assert min(state["belief"]) > 0.0


def test_lean_scores_in_range_and_json_serializable():
    state = initial_state("normal")
    for cues in (FAST_CUES, SLOW_CUES, None, {"reply_seconds": "bad"}, {}):
        state = update(state, cues=cues, used_hint=True, asked_answer=True)
        assert 0.0 <= lean_faster(state) <= 1.0
        assert 0.0 <= lean_slower(state) <= 1.0
        assert abs(sum(state["belief"]) - 1.0) < 1e-9
    assert json.loads(json.dumps(state)) == state


def test_ensure_state_handles_missing_and_partial_state():
    assert ensure_state(None, "detailed")["pace"] == "detailed"
    partial = ensure_state({"belief": [1.0]}, "normal")
    assert len(partial["belief"]) == 3
    assert update(None, request="faster")["lean_faster"] == 1.0


def test_features_width_and_lean_slots():
    base = {"explicit_faster": True, "explicit_slower": False, "misconceptions": [0.0] * 25}
    features = build_features(base)
    assert len(features) == 140
    assert features[136] == 1.0 and features[137] == 0.0
    lean = build_features({**base, "pace_lean_faster": 0.25, "pace_lean_slower": 0.7})
    assert len(lean) == 140
    assert lean[136] == 0.25 and lean[137] == 0.7
    clamped = build_features({**base, "pace_lean_faster": 2.0, "pace_lean_slower": -1.0})
    assert clamped[136] == 1.0 and clamped[137] == 0.0
    assert build_features({"misconceptions": [0.0] * 25})[136:138] == [0.0, 0.0]


def _scenario():
    return dict(load_scenarios()[0])


def test_episode_records_pace_diagnostics_without_cues(monkeypatch):
    monkeypatch.setattr(episode, "sample_pace_cues", None)
    result = episode.run_episode(_scenario(), episode_seed=3, horizon=6)
    assert result.private_diagnostics
    for row in result.private_diagnostics:
        assert row["pace_chosen"] in {"concise", "normal", "detailed"}
        assert row["pace_cues"] is None
        assert set(row["pace_belief"]) == {"concise", "normal", "detailed"}
        assert 0.0 <= row["pace_lean_faster"] <= 1.0
    for transition in result.strategist_transitions:
        assert len(transition["observation"]) == 140


def test_episode_feeds_cues_and_steps_from_initial_request(monkeypatch):
    calls = []

    def fake_cues(student, pace, *, scenario, episode_seed, turn, bank, config):
        calls.append((pace, turn))
        return dict(FAST_CUES)

    monkeypatch.setattr(episode, "sample_pace_cues", fake_cues)
    scenario = _scenario()
    scenario["public_history"] = [{"explicit_request": "slower"}]
    result = episode.run_episode(scenario, episode_seed=3, horizon=6)
    assert calls and [turn for _, turn in calls] == list(range(len(calls)))
    assert result.strategist_transitions[0]["observation"][137] == 1.0
    first = result.private_diagnostics[0]
    assert first["pace_before"] == "detailed"
    assert first["pace_cues"] == FAST_CUES


@pytest.mark.parametrize("seed", [0, 1])
def test_episode_is_reproducible(seed):
    first = episode.run_episode(_scenario(), episode_seed=seed, horizon=6)
    second = episode.run_episode(_scenario(), episode_seed=seed, horizon=6)
    assert first.private_diagnostics == second.private_diagnostics
