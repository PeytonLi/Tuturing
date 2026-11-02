"""Shared helpers for response pipeline and service tests (not a test module)."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from packages.response.pipeline import STUB_PIPELINE_ARTIFACT_ID
from packages.response.responders import ResponderOutput

ROOT = Path(__file__).resolve().parents[3]
LEARNING_VALID = ROOT / "contracts" / "fixtures" / "learning" / "valid"
LEARNING_INVALID = ROOT / "contracts" / "fixtures" / "learning" / "invalid"
RESPONSE_INVALID = ROOT / "contracts" / "fixtures" / "response" / "invalid"
LANE_PLANS = Path(__file__).resolve().parent / "fixtures" / "plans"
CHECK_NAMES = frozenset(
    {
        "schema_valid",
        "action_faithful",
        "math_supported",
        "disclosure_allowed",
        "length_valid",
        "question_count_valid",
    }
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def learning_plans() -> dict[str, dict]:
    """Person B's TeachingPlan fixtures (contracts/fixtures/learning/valid)."""
    plans = {}
    for path in sorted(LEARNING_VALID.glob("*.json")):
        value = load_json(path)
        if isinstance(value, dict) and "teaching_action" in value:
            plans[path.stem] = value
    return plans


def lane_plan(name: str) -> dict:
    """Person A's richer unit plans, which carry forbidden answer fingerprints."""
    return load_json(LANE_PLANS / f"{name}.json")


def make_request(
    plan: dict,
    request_id: str = "request-1",
    *,
    artifact: str = STUB_PIPELINE_ARTIFACT_ID,
    deadline_ms: int = 8000,
    seed: int = 0,
) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "trace_id": plan.get("trace_id", "trace-1"),
        "plan": copy.deepcopy(plan),
        "response_artifact_id": artifact,
        "generation_seed": seed,
        "deadline_ms": deadline_ms,
    }


class ScriptedResponder:
    """A responder that replays scripted drafts and repairs and records its calls."""

    kind = "scripted"

    def __init__(self, drafts=(), repairs=(), *, io_bound=False, delay_s=0.0, error=None):
        self.drafts = list(drafts)
        self.repairs = list(repairs)
        self.io_bound = io_bound
        self.delay_s = delay_s
        self.error = error
        self.generate_calls = []
        self.repair_calls = []

    def describe(self):
        return {"kind": self.kind}

    @staticmethod
    def _next(queue):
        item = queue.pop(0)
        if isinstance(item, ResponderOutput) or not isinstance(item, str):
            return item
        return ResponderOutput(text=item, input_tokens=10, output_tokens=5)

    def generate(self, messages, *, plan, max_tokens, timeout_s):
        self.generate_calls.append(
            {"messages": messages, "max_tokens": max_tokens, "timeout_s": timeout_s}
        )
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return self._next(self.drafts)

    def repair(self, messages, draft, *, plan, limits, max_tokens, timeout_s):
        self.repair_calls.append({"draft": draft, "limits": limits, "timeout_s": timeout_s})
        return self._next(self.repairs)
