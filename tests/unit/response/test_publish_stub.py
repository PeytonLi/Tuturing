import hashlib
import json

from packages.response.contracts import schema_errors
from tests.unit.response.pipeline_helpers import ROOT
from tools.response import publish_stub

CREATED_AT = "2026-11-02T00:00:00Z"


def test_manifest_is_deterministic_and_covers_every_move():
    manifest, responses = publish_stub.build_manifest(ROOT, created_at=CREATED_AT, source=None)
    again, _ = publish_stub.build_manifest(ROOT, created_at=CREATED_AT, source=None)
    assert manifest == again
    assert manifest["artifact_id"] == "response_stub_pipeline_v1"
    assert manifest["accepts_response_artifact_ids"] == [
        "response_stub_pipeline_v1", "response_stub_v1",
    ]
    assert manifest["versions"] == {
        "pipeline": "pipeline_v1",
        "prompt": "prompt_v1",
        "validator": "validator_v2",
        "fallback": "fallback_v1",
        "repair": "repair_v1",
        "template_stub": "template_stub_v1",
    }
    assert manifest["fallback_coverage"] == dict.fromkeys(publish_stub.MOVES, True)
    assert len(manifest["fixtures"]) == len(responses) == 9
    for fixture in manifest["fixtures"]:
        assert all(fixture["checks"].values())
        response = responses[fixture["name"]]
        assert schema_errors(response, "GenerateResponse") == []
        assert fixture["content_hash"] == response["content_hash"]
        assert fixture["legacy_stub"]["delivery_mode"] == "fallback"
    assert manifest["rejected_fixtures"] == [
        {
            "plan_fixture": publish_stub.REJECTED_FIXTURE,
            "code": "INVALID_SCHEMA",
            "http_status": 422,
            "retryable": False,
        }
    ]
    assert len(manifest["fallback_templates_hash"]) == 64
    assert len(manifest["fallback_rendering_hash"]) == 64
    assert set(manifest["source_hashes"]) == set(
        publish_stub.TEMPLATE_SOURCES + publish_stub.PIPELINE_SOURCES
    )
    assert manifest["commands"]["serve"].endswith("configs/response/serve_stub.yaml")
    assert "limitations" in manifest


def test_write_artifact_lays_out_manifest_hash_and_fixtures(tmp_path):
    manifest, responses = publish_stub.build_manifest(ROOT, created_at=CREATED_AT, source=None)
    path = publish_stub.write_artifact(manifest, responses, tmp_path)
    target = tmp_path / "response_stub_pipeline_v1"
    assert path == target / "manifest.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == manifest
    digest, name = (target / "manifest.sha256").read_text(encoding="utf-8").split()
    assert name == "manifest.json"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    fixture_files = sorted(file.stem for file in (target / "fixtures").glob("*.json"))
    assert fixture_files == sorted(responses)
    for name, response in responses.items():
        assert json.loads((target / "fixtures" / f"{name}.json").read_text()) == response


def test_main_publishes_into_the_requested_directory(tmp_path, capsys):
    assert publish_stub.main(["--out", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "published response_stub_pipeline_v1" in output
    assert "9 fixtures" in output
    manifest = json.loads(
        (tmp_path / "response_stub_pipeline_v1" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["created_at"].endswith("Z")
    assert manifest["source"] is None or set(manifest["source"]) == {"commit", "dirty"}
