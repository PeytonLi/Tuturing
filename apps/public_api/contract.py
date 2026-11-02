"""Public contract public_v1 (Person A): loading, validation and fixture checks.

`contracts/public_v1.openapi.yaml` references Person B's `contracts/learning_v1.schema.json` for
PublicState and SessionEnd (decision D11). Both documents, plus A's `response_v1.schema.json`, are
registered in one `referencing` registry so `$ref`s resolve exactly as they do on disk.

Validation messages are sanitized: they name the JSON path and the failed keyword, never the
offending value, because values can contain learner text.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts"
PUBLIC_CONTRACT_PATH = CONTRACTS / "public_v1.openapi.yaml"
LEARNING_SCHEMA_PATH = CONTRACTS / "learning_v1.schema.json"
RESPONSE_SCHEMA_PATH = CONTRACTS / "response_v1.schema.json"
PUBLIC_FIXTURES = CONTRACTS / "fixtures/response/public"
PUBLIC_BASE_URI = "https://example.invalid/contracts/public_v1.openapi.yaml"

# PRD 9.5, "Codes and HTTP statuses". The contract's x-tutor-http-status must equal this table.
PRD_ERROR_STATUS = {
    "INVALID_SCHEMA": 422,
    "UNSUPPORTED_VERSION": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "STATE_CONFLICT": 409,
    "PLAN_EXPIRED": 409,
    "SESSION_ENDED": 409,
    "RATE_LIMITED": 429,
    "GENERATION_UNAVAILABLE": 503,
    "LEARNING_UNAVAILABLE": 503,
    "DEADLINE_EXCEEDED": 504,
}

# PRD 9.5 public endpoints and the operationIds the contract gives them.
PRD_OPERATIONS = {
    ("post", "/v1/sessions"): "createSession",
    ("post", "/v1/sessions/{session_id}/turns"): "postTurn",
    ("get", "/v1/sessions/{session_id}"): "getSession",
    ("post", "/v1/sessions/{session_id}/stop"): "stopSession",
}

# Keys that belong to internal records (GenerateResponse, TeachingPlan, CommitPlanRequest) and
# must never appear anywhere in a public result (PRD 9.5: no checks, fingerprints, hidden content
# or private versions).
INTERNAL_FIELD_NAMES = frozenset(
    {
        "checks",
        "content_hash",
        "effective_move",
        "teaching_action",
        "navigation_action",
        "forbidden_answer_fingerprints",
        "reveal_solution",
        "permitted_hint",
        "worked_example",
        "approved_facts",
        "constraints",
        "response_context",
        "versions",
        "usage",
        "plan",
        "evidence",
        "learner_pseudonym",
        "mastery",
        "misconceptions",
    }
)

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")


class ContractError(ValueError):
    """The public contract or one of its fixtures is invalid."""


class _StrictLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ContractError(f"duplicate YAML key {key!r} at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def parse_json_strict(text: str) -> Any:
    """Parse JSON, rejecting NaN, Infinity and duplicate object keys."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def find_internal_fields(value: Any, path: str = "$") -> list[str]:
    """Return JSON paths of internal-only keys anywhere inside a public body."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in INTERNAL_FIELD_NAMES:
                found.append(child)
            found.extend(find_internal_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_internal_fields(item, f"{path}[{index}]"))
    return found


@dataclass(frozen=True)
class FixtureSummary:
    valid: int
    invalid: int
    operations: int
    error_codes: int


class PublicContract:
    """The public_v1 OpenAPI document plus the schema documents it references."""

    def __init__(self, document: dict, learning: dict, response: dict) -> None:
        self.document = document
        self.learning = learning
        self.response = response
        self.registry = Registry().with_resources(
            [
                (PUBLIC_BASE_URI, DRAFT202012.create_resource(document)),
                (learning["$id"], DRAFT202012.create_resource(learning)),
                (response["$id"], DRAFT202012.create_resource(response)),
            ]
        )
        self._validators: dict[str, Draft202012Validator] = {}
        self._lock = threading.Lock()

    # Validators -------------------------------------------------------------------------------

    def validator(self, uri: str) -> Draft202012Validator:
        with self._lock:
            validator = self._validators.get(uri)
            if validator is None:
                validator = Draft202012Validator(
                    {"$ref": uri}, registry=self.registry, format_checker=FormatChecker()
                )
                self._validators[uri] = validator
            return validator

    def public_uri(self, name: str) -> str:
        return f"{PUBLIC_BASE_URI}#/components/schemas/{name}"

    def learning_uri(self, name: str) -> str:
        return f"{self.learning['$id']}#/$defs/{name}"

    def response_uri(self, name: str) -> str:
        return f"{self.response['$id']}#/$defs/{name}"

    def issues_for(self, uri: str, instance: Any) -> list[str]:
        """Sanitized validation issues: JSON path and failed keyword, never the value."""
        errors = self.validator(uri).iter_errors(instance)
        return sorted({f"{error.json_path}: {error.validator}" for error in errors})

    def public_issues(self, name: str, instance: Any) -> list[str]:
        return self.issues_for(self.public_uri(name), instance)

    def learning_issues(self, name: str, instance: Any) -> list[str]:
        return self.issues_for(self.learning_uri(name), instance)

    def response_issues(self, name: str, instance: Any) -> list[str]:
        return self.issues_for(self.response_uri(name), instance)

    # Document navigation ----------------------------------------------------------------------

    def resolve(self, ref: str) -> Any:
        return self.registry.resolver(base_uri=PUBLIC_BASE_URI).lookup(ref).contents

    def deref(self, node: Any) -> Any:
        """Follow Reference Objects (`{"$ref": ...}` without siblings used as references)."""
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 16:
            node = self.resolve(node["$ref"])
            seen += 1
        return node

    @property
    def error_status(self) -> dict[str, int]:
        schema = self.document["components"]["schemas"]["ErrorCode"]
        return {str(code): int(status) for code, status in schema["x-tutor-http-status"].items()}

    def operations(self) -> dict[str, tuple[str, str, dict]]:
        found: dict[str, tuple[str, str, dict]] = {}
        for path, item in self.document.get("paths", {}).items():
            for method in HTTP_METHODS:
                operation = item.get(method)
                if isinstance(operation, dict):
                    found[operation["operationId"]] = (method, path, operation)
        return found

    def operation(self, operation_id: str) -> tuple[str, str, dict]:
        try:
            return self.operations()[operation_id]
        except KeyError as exc:
            raise ContractError(f"unknown operation {operation_id!r}") from exc

    def request_schema_uri(self, operation_id: str) -> str | None:
        _, _, operation = self.operation(operation_id)
        body = operation.get("requestBody")
        if body is None:
            return None
        schema = self.deref(body)["content"]["application/json"]["schema"]
        return _absolute(schema["$ref"])

    def response_schema_uri(self, operation_id: str, status: int) -> str | None:
        _, _, operation = self.operation(operation_id)
        response = operation.get("responses", {}).get(str(status))
        if response is None:
            return None
        schema = self.deref(response)["content"]["application/json"]["schema"]
        return _absolute(schema["$ref"])


def _absolute(ref: str) -> str:
    return PUBLIC_BASE_URI + ref if ref.startswith("#") else ref


def load_yaml_strict(path: Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)  # noqa: S506


def load_json_file(path: Path) -> Any:
    return parse_json_strict(path.read_text(encoding="utf-8"))


@cache
def load_contract() -> PublicContract:
    return PublicContract(
        load_yaml_strict(PUBLIC_CONTRACT_PATH),
        load_json_file(LEARNING_SCHEMA_PATH),
        load_json_file(RESPONSE_SCHEMA_PATH),
    )


# Structural checks ------------------------------------------------------------------------------


def _walk_refs(node: Any, path: str = "$") -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            refs.append((path, node["$ref"]))
        for key, value in node.items():
            refs.extend(_walk_refs(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            refs.extend(_walk_refs(value, f"{path}[{index}]"))
    return refs


def contract_problems(contract: PublicContract) -> list[str]:
    """Structural problems of the OpenAPI document; an empty list means it is usable."""
    document = contract.document
    problems: list[str] = []
    if not str(document.get("openapi", "")).startswith("3.1."):
        problems.append("openapi must be a 3.1.x version")
    if not document.get("info", {}).get("version"):
        problems.append("info.version is required")
    for json_path, ref in _walk_refs(document):
        try:
            contract.resolve(ref)
        except Unresolvable:
            problems.append(f"unresolvable $ref {ref!r} at {json_path}")
    for name, schema in document.get("components", {}).get("schemas", {}).items():
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            problems.append(f"component schema {name} is not valid JSON Schema 2020-12: {exc}")
    operations = contract.operations()
    for (method, path), operation_id in PRD_OPERATIONS.items():
        operation = document.get("paths", {}).get(path, {}).get(method)
        if not isinstance(operation, dict):
            problems.append(f"missing PRD 9.5 endpoint {method.upper()} {path}")
        elif operation.get("operationId") != operation_id:
            problems.append(f"{method.upper()} {path} must have operationId {operation_id}")
    status_map = contract.error_status
    if status_map != PRD_ERROR_STATUS:
        problems.append("x-tutor-http-status differs from the PRD 9.5 error table")
    enum = document["components"]["schemas"]["ErrorCode"]["enum"]
    if sorted(enum) != sorted(PRD_ERROR_STATUS):
        problems.append("ErrorCode enum differs from the PRD 9.5 error codes")
    for operation_id, (method, path, operation) in operations.items():
        if method == "post":
            uri = contract.request_schema_uri(operation_id)
            schema = contract.resolve(uri.removeprefix(PUBLIC_BASE_URI)) if uri else None
            if not schema or "request_id" not in schema.get("required", []):
                problems.append(f"{operation_id} must require request_id (the idempotency key)")
        for required_status in (401, 422, 429):
            if str(required_status) not in operation.get("responses", {}):
                problems.append(f"{operation_id} must declare a {required_status} response")
        for status_text in operation.get("responses", {}):
            status = int(status_text)
            if status < 400:
                continue
            uri = contract.response_schema_uri(operation_id, status)
            for code, code_status in PRD_ERROR_STATUS.items():
                sample = {
                    "schema_version": "1.0",
                    "request_id": "request-sample",
                    "trace_id": "trace-sample",
                    "code": code,
                    "message": "sample",
                    "retryable": False,
                    "current_state_version": None,
                }
                accepted = not contract.issues_for(uri, sample)
                if accepted != (code_status == status):
                    problems.append(
                        f"{operation_id} {status} response {'accepts' if accepted else 'rejects'}"
                        f" {code}, which PRD 9.5 maps to {code_status}"
                    )
    return problems


# Fixture checks ----------------------------------------------------------------------------------


def _exchange_problems(
    contract: PublicContract,
    operation_id: str,
    path_params: dict,
    headers: dict,
    request: Any,
    response: dict,
) -> list[str]:
    problems: list[str] = []
    try:
        _, path, _ = contract.operation(operation_id)
    except ContractError as exc:
        return [str(exc)]
    status = response.get("status")
    body = response.get("body")
    if not isinstance(status, int):
        return ["response.status must be an integer"]
    success = 200 <= status < 300
    for name in [part[1:-1] for part in path.split("/") if part.startswith("{")]:
        if name not in path_params:
            problems.append(f"missing path parameter {name}")
        else:
            problems.extend(
                f"path.{name}: {issue}" for issue in contract.public_issues("Id", path_params[name])
            )
    request_uri = contract.request_schema_uri(operation_id)
    if request_uri is None and request is not None:
        problems.append(f"{operation_id} takes no request body")
    if request_uri is not None and success:
        problems.extend(f"request {issue}" for issue in contract.issues_for(request_uri, request))
    if success and isinstance(request, dict):
        if "session_id" in request and request["session_id"] != path_params.get("session_id"):
            problems.append("request session_id disagrees with the path session ID")
        key = headers.get("Idempotency-Key")
        if key is not None and key != request.get("request_id"):
            problems.append("Idempotency-Key disagrees with request_id")
    response_uri = contract.response_schema_uri(operation_id, status)
    if response_uri is None:
        problems.append(f"{operation_id} does not declare status {status}")
    else:
        problems.extend(f"response {issue}" for issue in contract.issues_for(response_uri, body))
    if isinstance(body, dict):
        if isinstance(request, dict) and isinstance(request.get("request_id"), str):
            if "request_id" in body and body["request_id"] != request["request_id"]:
                problems.append("response request_id differs from the request's request_id")
        code = body.get("code")
        if not success and code in PRD_ERROR_STATUS and PRD_ERROR_STATUS[code] != status:
            problems.append(f"{code} must use HTTP {PRD_ERROR_STATUS[code]}, not {status}")
    if success:
        problems.extend(f"internal field {path}" for path in find_internal_fields(body))
    return problems


def fixture_problems(contract: PublicContract, fixture: Any) -> list[str]:
    """Problems of one public fixture; a valid fixture returns an empty list."""
    if not isinstance(fixture, dict):
        return ["fixture must be a JSON object"]
    kind = fixture.get("kind")
    if kind == "schema":
        name = fixture.get("schema")
        if name not in contract.document["components"]["schemas"]:
            return [f"unknown component schema {name!r}"]
        problems = contract.public_issues(name, fixture.get("instance"))
        if name not in {"Error", "ErrorCode"} and not name.startswith("Error"):
            problems += [f"internal field {p}" for p in find_internal_fields(fixture["instance"])]
        return problems
    if kind == "exchange":
        return _exchange_problems(
            contract,
            fixture.get("operation", ""),
            fixture.get("path_params", {}),
            fixture.get("headers", {}),
            fixture.get("request"),
            fixture.get("response", {}),
        )
    if kind == "retry":
        attempts = fixture.get("attempts", [])
        problems: list[str] = []
        if len(attempts) < 2:
            problems.append("a retry fixture needs at least two attempts")
        for index, attempt in enumerate(attempts):
            problems.extend(
                f"attempt {index}: {problem}"
                for problem in _exchange_problems(
                    contract,
                    fixture.get("operation", ""),
                    fixture.get("path_params", {}),
                    fixture.get("headers", {}),
                    fixture.get("request"),
                    attempt,
                )
            )
        if any(attempt != attempts[0] for attempt in attempts[1:]):
            problems.append("retry attempts must return the identical status and body")
        return problems
    return [f"unknown fixture kind {kind!r}"]


def validate_public_fixtures(
    contract: PublicContract | None = None, root: Path = PUBLIC_FIXTURES
) -> FixtureSummary:
    """Check the contract and every public fixture; raise ContractError on the first failure."""
    contract = contract or load_contract()
    problems = contract_problems(contract)
    if problems:
        raise ContractError("public contract: " + "; ".join(problems))
    valid = invalid = 0
    for path in sorted((root / "valid").glob("*.json")):
        found = fixture_problems(contract, load_json_file(path))
        if found:
            raise ContractError(f"invalid valid-fixture {path.name}: {found}")
        valid += 1
    for path in sorted((root / "invalid").glob("*.json")):
        try:
            found = fixture_problems(contract, load_json_file(path))
        except (ValueError, json.JSONDecodeError):
            found = ["invalid JSON"]
        if not found:
            raise ContractError(f"invalid fixture unexpectedly passed: {path.name}")
        invalid += 1
    if valid == 0 or invalid == 0:
        raise ContractError("public fixtures need at least one valid and one invalid fixture")
    return FixtureSummary(valid, invalid, len(contract.operations()), len(contract.error_status))
