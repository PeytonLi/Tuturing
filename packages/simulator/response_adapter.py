from __future__ import annotations

from packages.response.stub import generate
from packages.response.types import GenerateRequest, GenerateResponse


def realize_plan(plan: dict, *, request_id: str, generation_seed: int = 0) -> GenerateResponse:
    request: GenerateRequest = {
        "schema_version": "1.0",
        "request_id": request_id,
        "trace_id": plan["trace_id"],
        "plan": plan,
        "response_artifact_id": "response_stub_v1",
        "generation_seed": generation_seed,
        "deadline_ms": 1000,
    }
    return generate(request)
