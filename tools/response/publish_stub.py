"""Publish the CPU response stub as an immutable artifact manifest for Person B.

    uv run python -m tools.response.publish_stub --out artifacts/response

Writes ``<out>/<artifact_id>/manifest.json``, ``manifest.sha256`` and one deterministic
GenerateResponse per TeachingPlan fixture (PRD 20.2, 20.4 and 21.1 item 7). ``artifacts/`` is
ignored by Git. Loading instructions are in docs/runbooks/response_service.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from packages.response import stub as legacy_stub
from packages.response.contracts import loads_strict, schema_errors
from packages.response.errors import GenerationError
from packages.response.fallbacks import FallbackUnavailable, render_fallback
from packages.response.hashing import canonical_json
from packages.response.pipeline import STUB_PIPELINE_ARTIFACT_ID, stub_pipeline
from packages.response.responders import TEMPLATE_STUB_VERSION

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_VERSION = "1.0"
MOVES = (
    "PROBE", "HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE", "PRACTICE", "CHALLENGE",
    "CHECK_MASTERY", "REFLECT",
)
TEMPLATE_SOURCES = ("packages/response/stub.py", "packages/response/fallbacks.py")
PIPELINE_SOURCES = (
    "packages/response/contracts.py",
    "packages/response/errors.py",
    "packages/response/hashing.py",
    "packages/response/numbers.py",
    "packages/response/pipeline.py",
    "packages/response/plan_checks.py",
    "packages/response/prompt.py",
    "packages/response/repair.py",
    "packages/response/responders.py",
    "packages/response/shapes.py",
    "packages/response/validate.py",
    "contracts/response_v1.schema.json",
)
VALID_FIXTURES = "contracts/fixtures/learning/valid"
REJECTED_FIXTURE = "contracts/fixtures/learning/invalid/independent_reveal_solution.json"
LEGACY_ARTIFACT_ID = "response_stub_v1"
PUBLISH_COMMAND = "uv run python -m tools.response.publish_stub --out artifacts/response"
SERVE_COMMAND = "uv run python -m tools.response.serve --config configs/response/serve_stub.yaml"
LIMITATIONS = (
    "Template text only: no language model, so wording is fixed per move and never adapts "
    "to the learner message.",
    "The six checks are deterministic heuristics over approved content; they do not prove "
    "free-form mathematical reasoning correct (PRD 14.1).",
    "Fallback coverage is measured on the committed fixtures only.",
    "response_id values differ between the legacy stub (packages.response.stub.generate) and "
    "the verified pipeline; content hashes agree whenever the text agrees.",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_source(root: Path) -> dict | None:
    """The source commit and dirty flag, or None when Git is unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if commit.returncode != 0 or status.returncode != 0:
        return None
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def load_plans(root: Path) -> dict[str, dict]:
    plans = {}
    for path in sorted((root / VALID_FIXTURES).glob("*.json")):
        value = loads_strict(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "teaching_action" in value:
            plans[path.stem] = value
    return plans


def fixture_request(name: str, plan: dict, artifact_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": f"publish-{name}",
        "trace_id": "trace-publish",
        "plan": plan,
        "response_artifact_id": artifact_id,
        "generation_seed": 0,
        "deadline_ms": 8000,
    }


def build_manifest(
    root: Path = ROOT, *, created_at: str, source: dict | None
) -> tuple[dict, dict[str, dict]]:
    """Return the manifest and the full GenerateResponse per fixture."""
    pipeline = stub_pipeline()
    info = pipeline.describe()
    plans = load_plans(root)
    responses = {}
    fixtures = []
    renderings = {}
    for name, plan in plans.items():
        response = pipeline.generate(fixture_request(name, plan, STUB_PIPELINE_ARTIFACT_ID))
        if schema_errors(response, "GenerateResponse") or not all(response["checks"].values()):
            raise RuntimeError(f"fixture {name} did not produce a valid response")
        legacy = legacy_stub.generate(fixture_request(name, plan, LEGACY_ARTIFACT_ID))
        renderings[name] = render_fallback(plan)
        responses[name] = response
        fixtures.append(
            {
                "name": name,
                "plan_fixture": f"{VALID_FIXTURES}/{name}.json",
                "move": plan["teaching_action"]["move"],
                "assessment_mode": plan["assessment_mode"],
                "request_id": response["request_id"],
                "response_id": response["response_id"],
                "content_hash": response["content_hash"],
                "delivery_mode": response["delivery_mode"],
                "checks": response["checks"],
                "text": response["text"],
                "legacy_stub": {
                    "response_id": legacy["response_id"],
                    "content_hash": legacy["content_hash"],
                    "delivery_mode": legacy["delivery_mode"],
                },
            }
        )
    rejected_plan = loads_strict((root / REJECTED_FIXTURE).read_text(encoding="utf-8"))
    try:
        pipeline.generate(fixture_request("rejected", rejected_plan, STUB_PIPELINE_ARTIFACT_ID))
    except GenerationError as error:
        rejected = {
            "plan_fixture": REJECTED_FIXTURE,
            "code": error.code,
            "http_status": error.http_status,
            "retryable": error.retryable,
        }
    else:
        raise RuntimeError("the independent reveal fixture was not rejected")

    coverage = {}
    for move in MOVES:
        examples = [plan for plan in plans.values() if plan["teaching_action"]["move"] == move]
        covered = bool(examples)
        for plan in examples:
            try:
                render_fallback(plan)
            except FallbackUnavailable:
                covered = False
        coverage[move] = covered

    template_hash = hashlib.sha256()
    source_hashes = {}
    for relative in TEMPLATE_SOURCES + PIPELINE_SOURCES:
        data = (root / relative).read_bytes()
        source_hashes[relative] = sha256_bytes(data)
        if relative in TEMPLATE_SOURCES:
            template_hash.update(relative.encode("utf-8") + b"\0" + data + b"\0")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "artifact_id": STUB_PIPELINE_ARTIFACT_ID,
        "kind": "cpu_response_stub",
        "owner": "Person A",
        "created_at": created_at,
        "source": source,
        "schema_versions": {"response_v1": "1.0", "learning_v1_teaching_plan": "1.0"},
        "versions": {**info["versions"], "template_stub": TEMPLATE_STUB_VERSION},
        "accepts_response_artifact_ids": info["compatible_artifact_ids"],
        "limits": info["limits"],
        "fallback_templates_hash": template_hash.hexdigest(),
        "fallback_rendering_hash": sha256_bytes(canonical_json(renderings).encode("utf-8")),
        "source_hashes": source_hashes,
        "interface": {
            "python_adapter": (
                "packages.simulator.response_adapter.realize_plan -> "
                "packages.response.stub.generate (unchanged; delivery_mode is always fallback)"
            ),
            "python_pipeline": (
                "packages.response.pipeline.stub_pipeline().generate(GenerateRequest)"
            ),
            "http": {
                "generate": "POST /internal/v1/generate",
                "health": "GET /internal/v1/health",
                "auth": "Authorization: Bearer $TUTOR_RESPONSE_SERVICE_TOKEN",
                "max_request_bytes": 262144,
                "config": "configs/response/serve_stub.yaml",
            },
            "request_schema": "contracts/response_v1.schema.json#/$defs/GenerateRequest",
            "response_schema": "contracts/response_v1.schema.json#/$defs/GenerateResponse",
            "error_schema": "contracts/learning_v1.schema.json#/$defs/Error",
        },
        "fallback_coverage": coverage,
        "fixtures": fixtures,
        "rejected_fixtures": [rejected],
        "limitations": list(LIMITATIONS),
        "commands": {"publish": PUBLISH_COMMAND, "serve": SERVE_COMMAND},
        "runbook": "docs/runbooks/response_service.md",
    }
    return manifest, responses


def write_artifact(manifest: dict, responses: dict[str, dict], out_dir: Path) -> Path:
    target = Path(out_dir) / manifest["artifact_id"]
    (target / "fixtures").mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path = target / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    (target / "manifest.sha256").write_text(
        sha256_bytes(payload.encode("utf-8")) + "  manifest.json\n", encoding="utf-8"
    )
    for name, response in responses.items():
        (target / "fixtures" / f"{name}.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default="artifacts/response", help="artifact root directory")
    args = parser.parse_args(argv)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest, responses = build_manifest(ROOT, created_at=created_at, source=git_source(ROOT))
    path = write_artifact(manifest, responses, Path(args.out))
    digest = (path.parent / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    print(
        f"published {manifest['artifact_id']} to {path.parent} "
        f"({len(responses)} fixtures, manifest sha256 {digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
