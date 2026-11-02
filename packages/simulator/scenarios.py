from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_PATH = "data/scenarios/proto_v1.yaml"
# Extra detail-preferring scenarios kept in a separate file so the frozen 30-scenario set, its
# split order and its recorded hash stay unchanged; callers opt in with include_pace_scenarios.
PACE_SCENARIOS_PATH = "data/scenarios/proto_v1_pace.yaml"
# Train-only strong-learner scenarios (strong_all, strong_goal profiles), opt in with
# include_strong_scenarios; same reasons as the pace file.
STRONG_SCENARIOS_PATH = "data/scenarios/proto_v1_strong.yaml"


def load_scenarios(
    path: str | Path = DEFAULT_SCENARIOS_PATH,
    *,
    include_pace_scenarios: bool = False,
    include_strong_scenarios: bool = False,
) -> list[dict]:
    paths = [path]
    if include_pace_scenarios:
        paths.append(PACE_SCENARIOS_PATH)
    if include_strong_scenarios:
        paths.append(STRONG_SCENARIOS_PATH)
    scenarios = []
    for scenario_path in paths:
        document = yaml.safe_load((ROOT / scenario_path).read_text(encoding="utf-8"))
        scenarios.extend(document["scenarios"])
    ids = [scenario["scenario_id"] for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario ids must be unique across scenario files")
    schema = json.loads((ROOT / "contracts/learning_v1.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for scenario in scenarios:
        errors = list(validator.iter_errors(scenario))
        if errors:
            raise ValueError(f"invalid scenario {scenario.get('scenario_id')}: {errors}")
    return scenarios
