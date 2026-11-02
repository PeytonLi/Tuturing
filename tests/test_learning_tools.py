import io
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
