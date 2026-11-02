import csv
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from packages.learning.actions import encode_strategist
from packages.learning.episode import load_scenarios, run_episode, simulate_episodes
from packages.policies.network import (
    NetworkStrategist,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from packages.policies.rules import AlwaysWorkedExampleStrategist, RulesStrategist
from tools.learning import evaluate, score_decisions, train_bc, train_ppo

ROOT = Path(__file__).resolve().parents[1]


def test_always_worked_example_uses_legal_move_and_falls_back_when_masked():
    policy = AlwaysWorkedExampleStrategist()
    worked_example = encode_strategist("WORKED_EXAMPLE", "hard", "detailed")
    mask = [False] * 72
    mask[worked_example] = True
    assert (
        policy.act(
            [0.0] * 140,
            mask,
            {"active_item_difficulty": "hard", "pace_preference": "detailed"},
        )
        == worked_example
    )

    fallback = encode_strategist("PROBE", "easy", "normal")
    fallback_mask = [False] * 72
    fallback_mask[fallback] = True
    assert (
        policy.act(
            [0.0] * 140,
            fallback_mask,
            {"active_item_difficulty": "medium", "pace_preference": "normal"},
        )
        == fallback
    )


def test_always_worked_example_runs_validation_episode_without_masking(monkeypatch):
    from packages.learning import episode

    scenarios = [row for row in load_scenarios() if row["split"] == "validation"]
    worked_example_legal = []
    original_mask = episode.strategist_mask

    def record_mask(context):
        mask = original_mask(context)
        expected = encode_strategist(
            "WORKED_EXAMPLE",
            context["active_item_difficulty"],
            context["pace_preference"],
        )
        worked_example_legal.append(mask[expected])
        return mask

    monkeypatch.setattr(episode, "strategist_mask", record_mask)
    result = simulate_episodes(
        episodes=1,
        split="validation",
        seed=17,
        strategist_name="always_worked_example",
        scenarios=scenarios,
        run_id="always-worked-example-test",
    )[0]
    assert result.turns > 0
    assert result.move_histogram == {"WORKED_EXAMPLE": result.turns}
    assert worked_example_legal
    assert all(worked_example_legal)


def test_evaluation_paired_seed_and_simulator_override_are_reported(
    monkeypatch, tmp_path
):
    scenarios = [row for row in load_scenarios() if row["split"] == "validation"]
    monkeypatch.setattr(evaluate, "load_scenarios", lambda: scenarios[:1])
    output_name = str(tmp_path / "paired-evaluation")
    simulator_config = "configs/learning/simulator_proto_v1.yaml"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--config",
            "configs/learning/smoke.yaml",
            "--split",
            "validation",
            "--policies",
            "rules,always_hint",
            "--episodes-per-scenario",
            "1",
            "--paired-seeds",
            "--simulator-config",
            simulator_config,
            "--name",
            output_name,
        ],
    )
    evaluate.main()
    report = json.loads((Path(output_name) / "report.json").read_text(encoding="utf-8"))
    assert report["paired_seeds"] is True
    assert report["simulator_config"] == simulator_config
    assert set(report["policies"]) == {"rules", "always_hint"}
    rules_report = report["policies"]["rules"]
    hint_report = report["policies"]["always_hint"]
    assert "paired_gain_difference_vs_first" not in rules_report
    difference = hint_report["paired_gain_difference_vs_first"]
    expected_difference = (
        hint_report["overall"]["mean_gain"] - rules_report["overall"]["mean_gain"]
    )
    assert difference["mean"] == pytest.approx(expected_difference)
    assert difference["ci_95"] == pytest.approx(
        [expected_difference, expected_difference]
    )
    for policy_report in report["policies"].values():
        assert sum(policy_report["overall"]["pace_frequency"].values()) == pytest.approx(1)
        for profile in policy_report["profiles"].values():
            assert sum(profile["pace_frequency"].values()) == pytest.approx(1)
    assert "Pace frequency" in (Path(output_name) / "report.md").read_text(
        encoding="utf-8"
    )
    assert evaluate.policy_seed(42, 0, paired=False) == 42
    assert evaluate.policy_seed(42, 1, paired=False) == 100_042
    assert evaluate.policy_seed(42, 1, paired=True) == 42


def test_masked_actor_never_samples_an_illegal_action():
    model = StrategistActorCritic(seed=0)
    generator = random.Random(0)
    masks = []
    for _ in range(1000):
        mask = [generator.random() < 0.2 for _ in range(72)]
        mask[generator.randrange(72)] = True
        masks.append(mask)
    observations = [[0.0] * 140 for _ in masks]
    actions, _, _ = model.act(observations, masks)
    assert all(mask[action] for mask, action in zip(masks, actions, strict=True))


def test_network_strategist_loads_versioned_checkpoint(tmp_path):
    model = StrategistActorCritic(seed=0)
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(
        model,
        checkpoint,
        {
            "feature_spec_version": "features_v1",
            "action_spec_version": "actions_v1",
            "seed": 0,
            "init": "scratch",
        },
    )
    policy = NetworkStrategist(checkpoint)
    action = policy.act([0.0] * 140, [True] * 72, {})
    assert 0 <= action < 72


def test_dagger_labels_match_rules_and_teacher_mix_replays_rules_actions():
    scenario = next(
        row for row in load_scenarios() if row["split"] == "train"
    )
    episode_seed = 31
    strategist = train_bc.DaggerStrategist(
        RulesStrategist(),
        teacher_prob=1.0,
        seed=episode_seed,
        record_label_inputs=True,
    )
    dagger_result = run_episode(
        scenario,
        run_id="dagger-rules-test",
        episode_seed=episode_seed,
        strategist=strategist,
    )
    rules_result = run_episode(
        scenario,
        run_id="plain-rules-test",
        episode_seed=episode_seed,
        strategist=RulesStrategist(),
    )

    assert len(strategist.labels) == len(dagger_result.strategist_transitions)
    assert len(strategist.label_inputs) == len(dagger_result.strategist_transitions)
    assert strategist.labels == [
        RulesStrategist().act(observation, mask, context)
        for observation, mask, context in strategist.label_inputs
    ]
    assert strategist.actions == strategist.labels
    assert [
        transition["action_id"] for transition in dagger_result.strategist_transitions
    ] == [transition["action_id"] for transition in rules_result.strategist_transitions]


def test_one_behavior_cloning_epoch_reduces_small_batch_loss():
    model = StrategistActorCritic(seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    observations = [[0.0] * 140 for _ in range(8)]
    masks = [[True, True] + [False] * 70 for _ in range(8)]
    actions = [0] * 8
    targets = [0.0] * 8

    def loss():
        policy, value = train_bc._batch_losses(model, observations, masks, actions, targets)
        return float((policy + value).detach())

    before = loss()
    train_bc.train_epoch(
        model, optimizer, observations, masks, actions, targets, 8, random.Random(0)
    )
    assert loss() < before


def test_ppo_smoke_runs_two_updates_and_writes_metrics(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo",
            "--config",
            "configs/learning/ppo_smoke.yaml",
            "--seed",
            "0",
            "--sessions",
            "32",
        ],
    )
    train_ppo.main()
    metrics = ROOT / "experiments/learning/runs/ppo_smoke_seed0_32/metrics.csv"
    assert metrics.is_file()
    with metrics.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(float(row["anchor_kl"]) == 0.0 for row in rows)
    manifest_path = metrics.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["command"].startswith("python -m tools.learning.train_ppo")
    assert manifest["git_sha"] is None or isinstance(manifest["git_sha"], str)
    assert manifest["resolved_config"]["sessions"] == 32


def test_zero_anchor_coefficient_does_not_create_a_reference(monkeypatch):
    model = StrategistActorCritic(seed=0)
    monkeypatch.setattr(
        train_ppo.copy,
        "deepcopy",
        lambda value: pytest.fail("zero anchor coefficient must not copy the model"),
    )
    assert train_ppo._make_reference_model(model, 0.0, None) is None


def test_anchor_coefficient_requires_an_initial_checkpoint():
    with pytest.raises(ValueError, match="requires --init"):
        train_ppo._make_reference_model(StrategistActorCritic(seed=0), 0.1, None)


def test_ppo_anchor_kl_is_zero_at_initialization_and_positive_after_perturbation():
    observation = [0.0] * 140
    transitions = [
        {
            "observation": observation,
            "action_mask": [True, True] + [False] * 70,
            "action_id": 0,
            "actor_eligible": True,
            "reward": 1.0,
            "terminated": True,
            "next_observation": observation,
            "reward_components": {"disclosure_penalty": 0.0},
        },
        {
            "observation": [0.1] * 140,
            "action_mask": [True, True] + [False] * 70,
            "action_id": 1,
            "actor_eligible": False,
            "reward": 0.0,
            "terminated": True,
            "next_observation": [0.1] * 140,
            "reward_components": {"disclosure_penalty": 0.0},
        },
    ]
    sample = SimpleNamespace(
        strategist_transitions=transitions,
        navigator_transitions=[],
        terminal_components={},
        gain=0.0,
        dropout=False,
    )
    config = {
        "gamma": 0.99,
        "lam": 0.95,
        "epochs": 1,
        "minibatch_size": 2,
        "clip": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.0,
        "max_grad_norm": 0.5,
        "bc_anchor_coef": 0.1,
    }
    model = StrategistActorCritic(seed=0)
    reference = train_ppo._make_reference_model(model, 0.1, "init.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    initial_metrics = train_ppo._ppo_update(
        model, optimizer, [sample], config, random.Random(0), reference_model=reference
    )
    assert initial_metrics["anchor_kl"] == pytest.approx(0.0, abs=1e-7)

    with torch.no_grad():
        model.policy_head.bias[0] += 2.0
    perturbed_metrics = train_ppo._ppo_update(
        model, optimizer, [sample], config, random.Random(0), reference_model=reference
    )
    assert perturbed_metrics["anchor_kl"] > 0.0


def test_evaluation_refuses_sealed_test_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--config", "configs/learning/smoke.yaml", "--split", "test"],
    )
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2


def test_all_decision_cases_map_to_observations_and_legal_masks():
    cases = [
        json.loads(line)
        for line in (ROOT / "evaluation/learning/decision_cases_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(cases) == 30
    for case in cases:
        _, observation, mask, _ = score_decisions.case_state_to_inputs(case["state"])
        assert len(observation) == 140
        assert any(mask)
    assert len(cases) == 30


def test_rules_context_maps_dc20_error_and_skip_diagnostic_state():
    cases = {
        json.loads(line)["case_id"]: json.loads(line)
        for line in (ROOT / "evaluation/learning/decision_cases_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    dc20 = cases["DC20"]
    _, observation, mask, context = score_decisions.case_state_to_inputs(dc20["state"])
    assert context["meaningful_error"] is True
    action_id = score_decisions.RulesStrategist().act(observation, mask, context)
    assert score_decisions.decode_strategist(action_id)[0] == "HINT"

    _, _, _, after_skip = score_decisions.case_state_to_inputs(cases["DC12"]["state"])
    _, _, _, after_diagnostic = score_decisions.case_state_to_inputs(
        cases["DC13"]["state"]
    )
    assert after_skip["skip_diagnostic_due"] is True
    assert after_diagnostic["skip_diagnostic_due"] is False


def test_llm_parser_strips_complete_and_unterminated_think_blocks():
    assert score_decisions.parse_first_json_object(
        '<think>internal notes {ignored:true}</think>\n{"move":"HINT"}'
    ) == {"move": "HINT"}
    assert score_decisions.parse_first_json_object(
        '<think>unfinished reasoning with brace {json starts here\n{"move":"PROBE"}'
    ) == {"move": "PROBE"}


def test_llm_truncated_output_is_reported_as_parse_failure(monkeypatch):
    monkeypatch.setenv("TUTOR_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TUTOR_LLM_API_KEY", "test-key")
    monkeypatch.setenv("TUTOR_LLM_MODEL", "test/model")
    response = io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {"message": {"content": '{"move":"HINT"}'}, "finish_reason": "length"}
                ]
            }
        ).encode()
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return response

    monkeypatch.setattr(score_decisions.urllib.request, "urlopen", fake_urlopen)
    case = {"situation": "test", "state": {}, "recent_dialogue": []}
    action_id, parse_failure, truncated = score_decisions.llm_action(case, [True] * 72)
    assert action_id is None
    assert parse_failure is True
    assert truncated is True
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 2048


def test_llm_openai_payload_switches_are_optional_and_network_free(monkeypatch):
    monkeypatch.setenv("TUTOR_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TUTOR_LLM_API_KEY", "test-key")
    monkeypatch.setenv("TUTOR_LLM_MODEL", "test/reasoning-model")
    monkeypatch.setenv("TUTOR_LLM_OMIT_TEMPERATURE", "1")
    monkeypatch.setenv("TUTOR_LLM_MAX_TOKENS_FIELD", "max_completion_tokens")
    monkeypatch.setenv("TUTOR_LLM_MAX_TOKENS", "4000")
    response = io.BytesIO(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"move":"HINT","difficulty":"easy","pace":"normal"}'
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        return response

    monkeypatch.setattr(score_decisions.urllib.request, "urlopen", fake_urlopen)
    case = {"situation": "test", "state": {}, "recent_dialogue": []}
    action_id, parse_failure, truncated = score_decisions.llm_action(case, [True] * 72)
    assert action_id is not None
    assert parse_failure is False
    assert truncated is False
    assert "temperature" not in captured["payload"]
    assert captured["payload"]["max_completion_tokens"] == 4000
    assert "max_tokens" not in captured["payload"]
    assert "reasoning_effort" not in captured["payload"]


def test_rules_scorer_scores_all_cases(monkeypatch, tmp_path):
    output = tmp_path / "rules_scores.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["score_decisions", "--policy", "rules", "--out", str(output)],
    )
    score_decisions.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["cases"]) == 30
    assert report["overall"]["cases"] == 30
