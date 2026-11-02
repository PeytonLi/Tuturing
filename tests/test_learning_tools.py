import json
import random
import sys
from pathlib import Path

import pytest

from packages.policies.network import (
    NetworkStrategist,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from tools.learning import evaluate, score_decisions, train_bc, train_ppo

ROOT = Path(__file__).resolve().parents[1]


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
    assert len(metrics.read_text(encoding="utf-8").splitlines()) == 3


def test_evaluation_refuses_sealed_test_split(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate", "--config", "configs/learning/smoke.yaml", "--split", "test"],
    )
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2


def test_decision_cases_map_or_report_unmapped_request():
    cases = [
        json.loads(line)
        for line in (ROOT / "evaluation/learning/decision_cases_v1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(cases) == 30
    failed = []
    mapped = 0
    for case in cases:
        try:
            _, observation, mask, _ = score_decisions.case_state_to_inputs(case["state"])
        except ValueError as error:
            failed.append((case["case_id"], str(error)))
            continue
        assert len(observation) == 140
        assert any(mask)
        mapped += 1
    assert mapped == 29
    assert failed == [
        (
            "DC20",
            "explicit_request='hint' has no corresponding feature or mask input; "
            "cannot score this case without inventing semantics",
        )
    ]


def test_rules_scorer_stops_on_unmapped_case(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["score_decisions", "--policy", "rules"])
    with pytest.raises(ValueError, match="explicit_request='hint'"):
        score_decisions.main()
