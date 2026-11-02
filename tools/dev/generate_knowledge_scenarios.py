"""Write the hidden-knowledge adaptation test set (scenarios and simulator profiles)."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/learning/simulator_proto_v2.yaml"
CONFIG_OUT = ROOT / "configs/learning/simulator_knowledge_v1.yaml"
SCENARIOS_OUT = ROOT / "data/scenarios/knowledge_v1.yaml"
TAUGHT = ("C01", "C02", "C03", "C06", "C08")

# learner_type: (start, goal, profile, what is hidden, what a tutor that adapts should do)
CASES = {
    "missing_basics": (
        "C06", "C08",
        {"concept_mastery": [0.05, 0.30], "weak_prerequisites_of_start": True,
         "ell": [0.06, 0.12], "help_seeking_probability": 0.2},
        "Starts on conditional probability but never learned sample spaces and basic probability.",
        "Notice repeated failure despite hints, drop difficulty, explain or show a worked example.",
    ),
    "already_knows": (
        "C06", "C08",
        {"concept_mastery": [0.80, 0.95], "goal_mastery": [0.80, 0.95],
         "ell": [0.04, 0.10], "help_seeking_probability": 0.02},
        "Already knows the goal topic.",
        "Check mastery or challenge quickly; do not spend turns re-teaching.",
    ),
    "reverses_conditionals": (
        "C06", "C08",
        {"concept_mastery": [0.40, 0.60], "active_misconception": "M05",
         "susceptible_concepts": ["C06", "C08"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.05},
        "Confidently computes P(B|A) when asked for P(A|B).",
        "Probe to surface the wrong idea, then target it with a worked example or explanation.",
    ),
    "wrong_denominator": (
        "C06", "C08",
        {"concept_mastery": [0.40, 0.60], "active_misconception": "M06",
         "susceptible_concepts": ["C06"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.05},
        "Divides by the whole population instead of the conditioning event.",
        "Probe, then target the denominator error.",
    ),
    "assumes_independence": (
        "C06", "C08",
        {"concept_mastery": [0.40, 0.60], "active_misconception": "M07",
         "susceptible_concepts": ["C08"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.05},
        "Multiplies plain probabilities on trees, as if events were independent.",
        "Probe, then target the independence assumption.",
    ),
    "tree_paths": (
        "C06", "C08",
        {"concept_mastery": [0.40, 0.60], "active_misconception": "M08",
         "susceptible_concepts": ["C08"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.05},
        "Multiplies along branches that do not form one path.",
        "Probe, then target the tree-path error.",
    ),
    "complement_slip": (
        "C02", "C03",
        {"concept_mastery": [0.40, 0.60], "active_misconception": "M04",
         "susceptible_concepts": ["C03"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.05},
        "Subtracts the wrong complement.",
        "Probe, then target the complement error.",
    ),
    "event_vs_outcome": (
        "C01", "C02",
        {"concept_mastery": [0.30, 0.50], "active_misconception": "M02",
         "susceptible_concepts": ["C01", "C02"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.1},
        "Treats an event as a single outcome.",
        "Probe, then target the event and outcome confusion.",
    ),
    "strong_but_reverses": (
        "C06", "C08",
        {"concept_mastery": [0.70, 0.85], "active_misconception": "M05",
         "susceptible_concepts": ["C06", "C08"], "ell": [0.04, 0.10],
         "help_seeking_probability": 0.02},
        "Strong overall and gets easy items right, but still reverses conditionals.",
        "Not be fooled by early streaks: probe and fix the hidden error before moving on.",
    ),
}


def allowlist(start: str, goal: str) -> list[str]:
    names = [f"{c.lower()}_teach_03" for c in TAUGHT]
    return names + [f"{c.lower()}_assess_0{i}" for c in TAUGHT for i in (2, 5)]


def main():
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config["profiles"] = {}
    scenarios = []
    learner_types = []
    for index, (learner, (start, goal, profile, hidden, adaptive)) in enumerate(
        CASES.items(), start=1
    ):
        config["profiles"][f"k_{learner}"] = {**profile, "preferred_pace": "normal"}
        scenarios.append(
            {
                "scenario_id": f"knowledge-{index:02d}",
                "scenario_family_id": f"knowledge-family-{learner}",
                "split": "validation",
                "goal_concept_id": goal,
                "start_concept_id": start,
                "profile_id": f"k_{learner}",
                "simulator_config_id": "simulator_knowledge_v1",
                "private_initial_state_ref": f"knowledge-init-{index:02d}",
                "public_history": [],
                "template_family_allowlist": allowlist(start, goal),
                "teaching_horizon": 16,
                "provenance": {"author": "Person B", "reviewed": False, "revision": "knowledge_v1"},
            }
        )
        learner_types.append(
            {"scenario_id": f"knowledge-{index:02d}", "learner_type": learner,
             "hidden_knowledge": hidden, "adaptive_behavior": adaptive}
        )
    header = "# Hidden-knowledge test set, from tools/dev/generate_knowledge_scenarios.py.\n"
    CONFIG_OUT.write_text(
        "# Copy of simulator_proto_v2 with test-only profiles; every learner prefers normal pace.\n"
        + yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    SCENARIOS_OUT.write_text(
        header + yaml.safe_dump({"schema_version": "scenarios_v1", "learner_types": learner_types,
                                "scenarios": scenarios},
                                sort_keys=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
