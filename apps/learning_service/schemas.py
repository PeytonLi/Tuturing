"""Strict internal records, sharing the frozen learning contract definitions."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[2] / "contracts/learning_v1.schema.json").read_text()
)
ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"^[\u0000-\u007F]+$"}
VERSION = {"const": "1.0"}
NULL = {"type": "null"}


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def record(**fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


SCHEMAS = dict(CONTRACT["$defs"])
SCHEMAS.update(
    {
        "IdempotencyKey": ID,
        "CreateSessionRequest": record(
            schema_version=VERSION,
            request_id=ID,
            goal_concept_id=ID,
            pace_preference={"enum": ["concise", "normal", "detailed"]},
        ),
        "CreateSessionResponse": record(
            schema_version=VERSION, session_id=ID, public_state=ref("PublicState")
        ),
        "ObserveRequest": record(
            schema_version=VERSION,
            request_id=ID,
            session_id=ID,
            expected_state_version={"type": "integer", "minimum": 0},
            evidence={"anyOf": [ref("Evidence"), NULL]},
        ),
        "ObserveResponse": {
            "oneOf": [
                record(kind={"const": "plan"}, teaching_plan=ref("TeachingPlan")),
                record(kind={"const": "session_end"}, session_end=ref("SessionEnd")),
            ]
        },
        "CancelRequest": record(schema_version=VERSION, request_id=ID, session_id=ID, plan_id=ID),
        "PlanStatus": record(
            schema_version=VERSION,
            plan_id=ID,
            session_id=ID,
            status={"enum": ["pending", "committed", "canceled"]},
            content_hash={"anyOf": [{"type": "string", "pattern": "^[a-f0-9]{64}$"}, NULL]},
            response_id={"anyOf": [ID, NULL]},
        ),
        "GradeRequest": record(
            schema_version=VERSION,
            request_id=ID,
            problem_instance_id=ID,
            normalized_answer={"type": "string"},
        ),
        "GradeResponse": record(
            schema_version=VERSION,
            problem_instance_id=ID,
            correctness={"enum": ["correct", "incorrect", "unknown"]},
            grade_source={"enum": ["deterministic", "none"]},
        ),
    }
)


def validate(name, body):
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": SCHEMAS,
        **SCHEMAS[name],
    }
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(body)
