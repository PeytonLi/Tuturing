"""Write data/scenarios/proto_v1_pace.yaml: careful_detailed scenarios 31-36.

They reuse each split's template allowlist from the frozen proto_v1.yaml, which is left untouched
so existing scenario ids, split order and recorded hashes stay reproducible.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/scenarios/proto_v1.yaml"
TARGET = ROOT / "data/scenarios/proto_v1_pace.yaml"
FIRST_NUMBER = 31
SPLIT_COUNTS = (("train", 4), ("validation", 1), ("test", 1))


def build() -> dict:
    frozen = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))["scenarios"]
    allowlists = {}
    for scenario in frozen:
        allowlists.setdefault(scenario["split"], list(scenario["template_family_allowlist"]))
    scenarios = []
    number = FIRST_NUMBER
    for split, count in SPLIT_COUNTS:
        for _ in range(count):
            goal = "C06" if number % 3 == 0 else "C08"
            scenarios.append(
                {
                    "scenario_id": f"proto-scenario-{number:02d}",
                    "scenario_family_id": f"proto-family-detailed-{number:02d}",
                    "split": split,
                    "goal_concept_id": goal,
                    "start_concept_id": "C02" if goal == "C06" else "C06",
                    "profile_id": "careful_detailed",
                    "simulator_config_id": "simulator_proto_v1",
                    "private_initial_state_ref": f"private-init-{number:02d}",
                    "public_history": [],
                    "template_family_allowlist": allowlists[split],
                    "teaching_horizon": 16,
                    "provenance": {
                        "author": "Person B",
                        "reviewed": False,
                        "revision": "proto_v1_pace",
                    },
                }
            )
            number += 1
    return {"schema_version": "scenarios_v1", "scenarios": scenarios}


def main() -> None:
    TARGET.write_text(
        yaml.safe_dump(build(), sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
