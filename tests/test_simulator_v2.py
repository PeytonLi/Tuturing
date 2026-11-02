from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from packages.learning.actions import PACES, encode_strategist
from packages.learning.content.templates.families import FAMILIES, families_for
from packages.learning.episode import _next_item, load_scenarios, run_episode
from packages.simulator.randomness import EventRandomBank

ROOT = Path(__file__).resolve().parents[1]
ENABLED_CONCEPTS = ("C01", "C02", "C03", "C06", "C08")
SPLITS = ("train", "validation", "test")


def test_v2_config_only_changes_base_dropout():
    v1 = yaml.safe_load(
        (ROOT / "configs/learning/simulator_proto_v1.yaml").read_text(encoding="utf-8")
    )
    v2 = yaml.safe_load(
        (ROOT / "configs/learning/simulator_proto_v2.yaml").read_text(encoding="utf-8")
    )
    assert v2["dropout"]["base_per_turn"] == 0.004
    restored = deepcopy(v2)
    restored["dropout"]["base_per_turn"] = v1["dropout"]["base_per_turn"]
    assert restored == v1
    for config_name in ("smoke", "bc", "ppo_strategist", "ppo_smoke"):
        config = yaml.safe_load(
            (ROOT / f"configs/learning/{config_name}.yaml").read_text(encoding="utf-8")
        )
        assert config["simulator_config"] == "configs/learning/simulator_proto_v2.yaml"


def test_two_assessment_families_per_concept_per_split():
    for concept_id in ENABLED_CONCEPTS:
        for split in SPLITS:
            family_ids = {
                family.template_family_id
                for family in families_for(concept_id, role="assessment", split=split)
            }
            assert len(family_ids) == 2

    for scenario in load_scenarios():
        allowed = set(scenario["template_family_allowlist"])
        for concept_id in ENABLED_CONCEPTS:
            family_ids = {
                family_id
                for family_id in allowed
                if FAMILIES[family_id].concept_id == concept_id
                and FAMILIES[family_id].role == "assessment"
                and FAMILIES[family_id].split == scenario["split"]
            }
            assert len(family_ids) == 2
        if scenario["split"] in {"train", "validation"}:
            assert all(FAMILIES[family_id].split != "test" for family_id in allowed)


def test_collector_prefers_fresh_assessment_family():
    scenario = next(row for row in load_scenarios() if row["split"] == "train")
    assessment_family_uses: Counter[str] = Counter()
    assessment_family_reuse_counts: Counter[str] = Counter()
    bank = EventRandomBank()
    family_ids = []

    for turn in range(3):
        family, _, _ = _next_item(
            "C06",
            "CHECK_MASTERY",
            scenario,
            12,
            turn,
            bank,
            assessment_family_uses,
            assessment_family_reuse_counts,
            set(),
        )
        family_ids.append(family.template_family_id)
        if turn < 2:
            assert sum(assessment_family_reuse_counts.values()) == 0

    assert family_ids == ["c06_assess_01", "c06_assess_04", "c06_assess_01"]
    assert sum(assessment_family_reuse_counts.values()) == 1


class ScriptedMasteryStrategist:
    def act(self, observation, mask, context):
        preferred_pace = context.get("pace_preference", "normal")
        paces = (
            (preferred_pace, *(pace for pace in PACES if pace != preferred_pace))
            if preferred_pace in PACES
            else PACES
        )
        for pace in paces:
            action = encode_strategist("CHECK_MASTERY", "medium", pace)
            if mask[action]:
                return action
        return next(action for action, allowed in enumerate(mask) if allowed)


def test_certification_reachable_in_simulated_episode():
    scenario = next(
        row
        for row in load_scenarios()
        if row["split"] == "train" and row["profile_id"] == "knowledgeable_concise"
    )
    certified_evidence = None

    for seed in range(10):
        result = run_episode(scenario, episode_seed=seed, strategist=ScriptedMasteryStrategist())
        assert result.transitions
        assert result.transitions[0]["environment_bundle_id"] == "simulator_proto_v2"
        certified = [
            result.final_beliefs[concept_id]
            for concept_id in ENABLED_CONCEPTS
            if result.final_beliefs[concept_id].certified
        ]
        if certified:
            certified_evidence = list(certified[0].recent_independent)[-2:]
            break

    assert certified_evidence is not None
    family_ids = [family_id for _, family_id, _ in certified_evidence]
    assert len(set(family_ids)) == 2
    assert all(FAMILIES[family_id].role == "assessment" for family_id in family_ids)
    assert all(FAMILIES[family_id].split == "train" for family_id in family_ids)
