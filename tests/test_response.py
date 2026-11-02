import hashlib
import json

import pytest

from packages.learning.actions import encode_strategist
from packages.response.stub import ResponseValidationError, generate
from packages.simulator.scenarios import ROOT


def _plan(name: str) -> dict:
    return json.loads(
        (ROOT / f"contracts/fixtures/learning/valid/{name}.json").read_text(encoding="utf-8")
    )


def test_hash_pin_matches_stub_and_canonical_formula():
    plan = _plan("teaching_plan_probe")
    response = generate(
        {"request_id": "request-probe", "plan": plan, "response_artifact_id": "stub_v1"}
    )
    pin = json.loads(
        (ROOT / "contracts/fixtures/response/valid/generate_response_hash_pin.json").read_text(
            encoding="utf-8"
        )
    )
    assert response["content_hash"] == pin["content_hash"]
    canonical = json.dumps(
        response["effective_move"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = hashlib.sha256(f"{response['text']}\n{canonical}".encode()).hexdigest()
    assert response["content_hash"] == expected


def test_rejects_independent_reveal_and_respects_pace_word_cap():
    independent = _plan("protected_check_mastery")
    independent["content"]["reveal_solution"] = "1/2"
    with pytest.raises(ResponseValidationError):
        generate(
            {
                "request_id": "request-independent",
                "plan": independent,
                "response_artifact_id": "stub_v1",
            }
        )
    practice = _plan("teaching_plan_probe")
    practice["teaching_action"]["pace"] = "concise"
    practice["teaching_action"]["action_id"] = encode_strategist("PROBE", "medium", "concise")
    practice["constraints"]["max_words"] = 80
    response = generate(
        {"request_id": "request-concise", "plan": practice, "response_artifact_id": "stub_v1"}
    )
    assert len(response["text"].split()) <= 80
