from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from packages.learning.plan_rules import validate_plan_rules
from tools.learning.validate_contracts import load_json

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    schema = load_json(ROOT / "contracts/response_v1.schema.json")
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    fixtures = ROOT / "contracts/fixtures/response"
    valid_count = invalid_count = 0

    def validate(instance: object, path: Path) -> list[str]:
        errors = [error.message for error in validator.iter_errors(instance)]
        if isinstance(instance, dict) and "teaching_action" in instance:
            errors.extend(validate_plan_rules(instance))
        if isinstance(instance, dict) and "plan" in instance:
            errors.extend(validate_plan_rules(instance["plan"]))
        if (
            isinstance(instance, dict)
            and "checks" in instance
            and not all(instance["checks"].values())
        ):
            errors.append("response checks must all pass")
        return errors

    for path in sorted((fixtures / "valid").glob("*.json")):
        instance = load_json(path)
        errors = validate(instance, path)
        if errors:
            raise SystemExit(f"invalid valid-fixture {path.name}: {errors}")
        valid_count += 1
    for path in sorted((fixtures / "invalid").glob("*.json")):
        try:
            instance = load_json(path)
            errors = validate(instance, path)
        except (ValueError, json.JSONDecodeError):
            errors = ["invalid JSON"]
        if not errors:
            raise SystemExit(f"invalid fixture unexpectedly passed: {path.name}")
        invalid_count += 1
    print(
        f"response contracts: {valid_count} valid fixtures and "
        f"{invalid_count} invalid fixtures passed"
    )


if __name__ == "__main__":
    main()
