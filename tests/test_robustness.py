import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from packages.policies.network import StrategistActorCritic, save_checkpoint
from packages.simulator.student import sample_student
from tools.learning import robustness

ROOT = Path(__file__).resolve().parents[1]


def test_alt_config_only_changes_intended_assumptions():
    v2_path, alt_path = robustness.DEFAULT_CONFIGS
    v2 = yaml.safe_load((ROOT / v2_path).read_text(encoding="utf-8"))
    alt_text = (ROOT / alt_path).read_text(encoding="utf-8")
    assert alt_text.startswith(
        "# robustness-only alternate assumptions, never used for training or checkpoint selection\n"
    )
    alt = yaml.safe_load(alt_text)
    restored = deepcopy(alt)

    def check_halved(original, alternate):
        if isinstance(original, dict):
            assert original.keys() == alternate.keys()
            for key in original:
                check_halved(original[key], alternate[key])
        else:
            assert alternate == pytest.approx(original * 0.5)

    check_halved(v2["action_effect"], alt["action_effect"])
    restored["action_effect"] = v2["action_effect"]
    for parameter in ("guess", "slip"):
        assert alt["observation"][parameter] == pytest.approx(
            [min(1.0, max(0.0, value + 0.05)) for value in v2["observation"][parameter]]
        )
        restored["observation"][parameter] = v2["observation"][parameter]
    assert max(alt["observation"]["guess"]) + max(alt["observation"]["slip"]) < 1
    for profile_id, profile in v2["profiles"].items():
        for key in ("concept_mastery", "goal_mastery"):
            if key in profile:
                assert alt["profiles"][profile_id][key] == pytest.approx(
                    [max(0.05, value - 0.10) for value in profile[key]]
                )
                restored["profiles"][profile_id][key] = profile[key]
    assert alt["pace_cues"]["strength"] < v2["pace_cues"]["strength"]
    assert alt["pace_cues"]["reply_seconds"]["sigma"] > v2["pace_cues"]["reply_seconds"]["sigma"]
    restored["pace_cues"] = v2["pace_cues"]
    assert restored == v2
    for scenario in robustness.load_scenarios():
        if scenario["split"] == "validation":
            student = sample_student(scenario, 0, config_path=alt_path)
            assert all(0 <= value <= 1 for value in student.mastery.values())
            assert all(
                student.guess[concept] + student.slip[concept] < 1
                for concept in student.mastery
            )


def test_robustness_cli_smoke_uses_paired_seeds(monkeypatch, tmp_path, capsys):
    calls = []
    simulate = robustness.simulate_episodes

    def capture(**kwargs):
        results = simulate(**kwargs)
        calls.append((kwargs, [row.episode_id for row in results]))
        return results

    monkeypatch.setattr(robustness, "simulate_episodes", capture)
    prefix = tmp_path / "robustness"
    monkeypatch.setattr(sys, "argv", [
        "robustness", "--episodes", "3", "--policies", "rules,random",
        "--output-prefix", str(prefix),
    ])
    robustness.main()
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    markdown = prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert "SIMULATED" in capsys.readouterr().out
    assert report["split"] == "validation"
    assert report["episodes"] == 3
    assert report["skipped_policies"] == []
    assert len(report["episode_schedule"]) == 3
    assert set(report["simulator_configs"]) == set(robustness.DEFAULT_CONFIGS)
    assert len(calls) == 4
    assert {kwargs["seed"] for kwargs, _ in calls} == {20250115}
    assert all(ids == calls[0][1] for _, ids in calls)
    assert all(kwargs["split"] == "validation" for kwargs, _ in calls)
    for values in report["simulator_configs"].values():
        assert set(values["policies"]) == {"rules", "random"}
        for policy, groups in values["policies"].items():
            summary = groups["overall"]
            assert summary["episodes"] == 3
            assert len(summary["mean_gain_ci_95"]) == 2
            assert len(summary["paired_difference_vs_rules_ci_95"]) == 2
            assert summary["disclosure_count"] >= 0
            assert 0 <= summary["dropout_rate"] <= 1
            assert 0 < summary["mean_turns"] <= 16
            assert sum(summary["move_frequency"].values()) == pytest.approx(1)
            assert sum(group["episodes"] for group in groups["profiles"].values()) == 3
            if policy == "rules":
                assert summary["paired_difference_vs_rules"] == 0
                assert summary["paired_difference_vs_rules_ci_95"] == [0, 0]
    assert "Per-profile gain" in markdown
    assert "Move frequencies" in markdown
    assert "Paired difference vs rules" in markdown


@pytest.mark.parametrize("split", ["test", "train"])
def test_robustness_cli_refuses_non_validation_split(monkeypatch, split, capsys):
    monkeypatch.setattr(sys, "argv", ["robustness", "--split", split])
    with pytest.raises(SystemExit) as error:
        robustness.main()
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_checkpoint_flag_reuses_evaluate_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "strategist.pt"
    save_checkpoint(StrategistActorCritic(seed=0), checkpoint, {
        "feature_spec_version": "features_v1", "action_spec_version": "actions_v1",
        "seed": 0, "init": "scratch",
    })
    prefix = tmp_path / "checkpoint_run"
    monkeypatch.setattr(sys, "argv", [
        "robustness", "--episodes", "1", "--policies", "rules",
        "--checkpoint", str(checkpoint),
        "--simulator-config", robustness.DEFAULT_CONFIGS[1],
        "--output-prefix", str(prefix),
    ])
    robustness.main()
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    policies = report["simulator_configs"][robustness.DEFAULT_CONFIGS[1]]["policies"]
    assert policies[f"checkpoint:{checkpoint}"]["overall"]["episodes"] == 1


def test_optional_baseline_skips_gracefully_and_rules_is_always_reference(monkeypatch):
    original_policy_from_name = robustness.policy_from_name

    def policy_from_name_without_optional_baseline(name):
        if name == "always_worked_example":
            raise ValueError("unknown policy")
        return original_policy_from_name(name)

    monkeypatch.setattr(
        robustness, "policy_from_name", policy_from_name_without_optional_baseline
    )
    monkeypatch.delattr(robustness.rules, "AlwaysWorkedExampleStrategist", raising=False)
    policies, skipped = robustness.load_policies(["random", "always_worked_example", "random"])
    assert list(policies) == ["rules", "random"]
    assert skipped == ["always_worked_example"]
    monkeypatch.setattr(
        robustness.rules, "AlwaysWorkedExampleStrategist", object, raising=False
    )
    policies, skipped = robustness.load_policies(["always_worked_example"])
    assert policies["always_worked_example"][0] is not None
    assert skipped == []
    with pytest.raises(ValueError, match="unknown policy"):
        robustness.load_policies(["invalid_policy"])
