from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from packages.learning.plan_rules import validate_plan_rules

ROOT = Path(__file__).resolve().parents[2]


def finite_json(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(key) and finite_json(item) for key, item in value.items())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def load_json(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def main() -> None:
    schema_path = ROOT / "contracts/learning_v1.schema.json"
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    fixtures = ROOT / "contracts/fixtures/learning"
    valid_count = invalid_count = 0
    for path in sorted((fixtures / "valid").glob("*.json")):
        instance = load_json(path)
        errors = list(validator.iter_errors(instance))
        if isinstance(instance, dict) and "teaching_action" in instance:
            errors.extend(validate_plan_rules(instance))
        if errors:
            raise SystemExit(f"invalid valid-fixture {path.name}: {errors}")
        valid_count += 1
    for path in sorted((fixtures / "invalid").glob("*.json")):
        try:
            instance = load_json(path)
            errors = list(validator.iter_errors(instance))
            if isinstance(instance, dict) and "teaching_action" in instance:
                errors.extend(validate_plan_rules(instance))
        except (ValueError, json.JSONDecodeError):
            errors = ["invalid JSON"]
        if not errors:
            raise SystemExit(f"invalid fixture unexpectedly passed: {path.name}")
        invalid_count += 1
    print(
        f"learning contracts: {valid_count} valid fixtures and "
        f"{invalid_count} invalid fixtures passed"
    )


if __name__ == "__main__":
    main()
