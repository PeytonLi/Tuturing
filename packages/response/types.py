from __future__ import annotations

from typing import TypedDict


class GenerateRequest(TypedDict):
    schema_version: str
    request_id: str
    trace_id: str
    plan: dict
    response_artifact_id: str
    generation_seed: int
    deadline_ms: int


class GenerateResponse(TypedDict):
    schema_version: str
    request_id: str
    plan_id: str
    response_id: str
    text: str
    effective_move: dict
    delivery_mode: str
    content_hash: str
    checks: dict[str, bool]
    usage: dict[str, int]
    versions: dict[str, str]
