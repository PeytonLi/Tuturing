from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


def load_scenarios(path: str | Path = "data/scenarios/proto_v1.yaml") -> list[dict]:
    scenarios_path = ROOT / path
    document = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    scenarios = document["scenarios"]
    schema = json.loads((ROOT / "contracts/learning_v1.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for scenario in scenarios:
        errors = list(validator.iter_errors(scenario))
        if errors:
            raise ValueError(f"invalid scenario {scenario.get('scenario_id')}: {errors}")
    return scenarios
