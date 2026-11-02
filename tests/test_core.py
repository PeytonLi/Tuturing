import hashlib
import json

from packages.learning.actions import (
    decode_navigator,
    decode_strategist,
    encode_navigator,
    encode_strategist,
    navigator_mask,
    strategist_mask,
)
from packages.learning.belief import ConceptBelief, bkt_posterior, update_belief
from packages.learning.episode import run_episode, simulate_episodes, write_run
from packages.learning.features import build_features, feature_order
from packages.policies.returns import duration_aware_gae
from packages.simulator.reward import RewardComponents
from packages.simulator.scenarios import ROOT, load_scenarios


def test_action_encodings_round_trip_and_masks_are_nonempty():
    for action_id in range(72):
        move, difficulty, pace = decode_strategist(action_id)
        assert encode_strategist(move, difficulty, pace) == action_id
    for action_id in range(76):
        operation, concept = decode_navigator(action_id)
        assert encode_navigator(operation, concept) == action_id
    for seed in range(1000):
        mastery = (seed % 101) / 100
        strategist_state = {
            "mastery": mastery,
            "recent_independent_correct": bool(seed % 2),
            "active_item": bool(seed % 3),
            "remaining_hints": bool(seed % 5),
            "previous_step": bool(seed % 7),
            "active_item_difficulty": ("easy", "medium", "hard")[seed % 3],
        }
        assert any(strategist_mask(strategist_state))
        navigator_state = {
            "current_concept_id": "C02",
            "goal_concept_id": "C08",
            "enabled_concepts": ["C01", "C02", "C03", "C06", "C08"],
            "goal_ancestors": ["C01", "C02", "C06"],
            "goal_stack": [],
            "mastery_by_concept": {"C01": mastery, "C02": mastery},
            "independent_success_streak": seed % 3,
        }
        graph = {
            "enabled_concepts": navigator_state["enabled_concepts"],
            "operations": {
                "successors": {"C02": ["C03", "C06"]},
                "ancestors": {"C02": ["C01"]},
                "weakest_ancestor": "C01",
            },
        }
        assert any(navigator_mask(navigator_state, graph))


def test_feature_width_order_and_navigator_zeroing():
    state = {
        "concepts": [{"mastery": 0.5, "uncertainty": 0.7}] * 15,
        "misconceptions": [0.2] * 25,
        "current_concept_id": "C02",
        "goal_concept_id": "C08",
        "active_target_concept_id": "C06",
        "active_option_type": "ADVANCE_ONE",
        "previous_move": "PRACTICE",
        "turn": 2,
        "turns_in_option": 1,
    }
    spec = json.loads((ROOT / "contracts/feature_spec_v1.json").read_text(encoding="utf-8"))
    assert len(build_features(state)) == len(spec["features"]) == 140
    assert feature_order() == [item["name"] for item in spec["features"]]
    nav = build_features(state, navigator=True)
    assert nav[100:121] == [0.0] * 21
    assert not any("private" in name.lower() for name in feature_order())


def test_bkt_assisted_unknown_and_certification():
    belief = ConceptBelief()
    update_belief(belief, "correct", family_id="a")
    assert abs(belief.mastery - 0.7078651685393258) < 1e-12
    prior = belief.mastery
    update_belief(belief, "correct", assistance="copied", move="PRACTICE", family_id="b")
    update_belief(belief, "unknown", move="PRACTICE", family_id="c")
    assert belief.mastery == prior
    assert belief.independent_correct == 1
    assert belief.assisted_attempts == 1
    assert bkt_posterior(prior, "unknown") == prior
    update_belief(belief, "correct", move="PRACTICE", family_id="practice")
    for family in ("b", "c"):
        update_belief(belief, "correct", move="PRACTICE", family_id=family)
    assert belief.certified
    update_belief(belief, "incorrect", move="PRACTICE", family_id="d")
    assert not belief.certified


def test_duration_aware_gae_boundaries():
    for duration in (0, 1, 3, 4):
        result = duration_aware_gae(
            [2.0], [1.0], [duration], [4.0], [False], gamma=0.5, lam=0.9
        )
        assert result.deltas == [2.0 + 0.5**duration * 4.0 - 1.0]
        assert result.advantages == result.deltas
    cut = duration_aware_gae(
        [2.0, 1.0],
        [1.0, 0.5],
        [3, 1],
        [4.0, 9.0],
        [False, False],
        [True, False],
        [True, False],
        gamma=0.5,
        lam=0.9,
    )
    assert cut.deltas == [1.5, 5.0]
    assert cut.advantages == cut.deltas
    terminal = duration_aware_gae([4.0], [1.0], [0], [99.0], [True])
    assert terminal.deltas == [3.0]
    assert terminal.advantages == [3.0]


def test_transition_reward_components_sum_and_replay_is_deterministic(tmp_path):
    components = RewardComponents(terminal_gain=1.2, teaching_cost=-0.02, shaping=0.0)
    assert components.total == sum(components.to_dict().values())
    first = simulate_episodes(
        episodes=1, split="train", seed=1717, strategist_name="rules", run_id="test-replay"
    )
    second = simulate_episodes(
        episodes=1, split="train", seed=1717, strategist_name="rules", run_id="test-replay"
    )
    assert first[0].transitions == second[0].transitions
    assert first[0].terminal_components == second[0].terminal_components
    write_run(first, tmp_path / "first", manifest={"run_id": "test-replay"})
    write_run(second, tmp_path / "second", manifest={"run_id": "test-replay"})
    first_hash = hashlib.sha256((tmp_path / "first/transitions.jsonl").read_bytes()).hexdigest()
    second_hash = hashlib.sha256((tmp_path / "second/transitions.jsonl").read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_reward_components_sum_across_fifty_episodes():
    results = simulate_episodes(
        episodes=50, split="train", seed=20250116, strategist_name="rules", run_id="reward-sum"
    )
    for result in results:
        for transition in result.strategist_transitions:
            components = transition["reward_components"]
            assert abs(transition["reward"] - sum(components.values())) < 1e-12


def test_grader_wins_extractor_conflict_and_logs_it():
    belief = ConceptBelief()
    conflicts = []
    update_belief(
        belief,
        "unknown",
        extractor_correctness="correct",
        grader_correctness="incorrect",
        conflict_logger=conflicts.append,
        family_id="family-a",
        event_id="conflict-1",
    )
    assert belief.independent_incorrect == 1
    assert belief.independent_correct == 0
    assert conflicts == [
        {
            "extractor_correctness": "correct",
            "grader_correctness": "incorrect",
            "event_id": "conflict-1",
        }
    ]


def test_transition_records_keep_policy_ownership_and_terminal_once():
    result = simulate_episodes(
        episodes=1, split="train", seed=19, strategist_name="rules", run_id="test-terminal"
    )[0]
    assert all(item["actor_eligible"] is False for item in result.navigator_transitions)
    terminal_rewards = [
        item.get("discounted_option_reward", 0.0)
        for item in result.transitions
        if item.get("terminated") is True
    ]
    assert len(terminal_rewards) >= 1
    assert sum(item["terminated"] is True for item in result.transitions) >= 1


def test_explicit_stop_terminates_without_teaching_turn():
    scenario = dict(load_scenarios()[0])
    scenario["public_history"] = [{"explicit_request": "stop"}]
    result = run_episode(scenario, episode_seed=23, strategist_name="rules")
    assert result.end_reason == "learner_stop"
    assert result.turns == 0
