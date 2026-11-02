"""Validate records against contracts/response_v1.schema.json with value-free messages.

jsonschema's own messages quote the offending value, which could be an answer
fingerprint or learner text. ``schema_errors`` reports only the JSON path and the failed
keyword, so its output is safe for clients and logs.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
RESPONSE_SCHEMA_PATH = ROOT / "contracts" / "response_v1.schema.json"
SUPPORTED_MAJOR_VERSION = "1"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def loads_strict(data: bytes | str) -> object:
    """Parse JSON and reject NaN and Infinity (PRD 9.1)."""
    return json.loads(data, parse_constant=_reject_constant)


@cache
def response_schema() -> dict:
    return loads_strict(RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8"))


@cache
def validator(kind: str) -> Draft202012Validator:
    definitions = response_schema()["$defs"]
    if kind not in definitions:
        raise ValueError(f"response_v1 has no record named {kind}")
    return Draft202012Validator(
        {"$ref": f"#/$defs/{kind}", "$defs": definitions}, format_checker=FormatChecker()
    )


def _path(error) -> str:
    text = ""
    for part in error.absolute_path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else str(part)
    return text or "$"


def schema_errors(instance: object, kind: str) -> list[str]:
    """Return sorted ``path: keyword`` messages that never echo instance values."""
    messages = set()
    for error in validator(kind).iter_errors(instance):
        detail = str(error.validator)
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = [name for name in error.validator_value if name not in error.instance]
            detail = "required " + ",".join(missing)
        messages.add(f"{_path(error)}: {detail}")
    return sorted(messages)


def major_version(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value.split(".", 1)[0]
    return None
