"""The CPU smoke workflow (PRD 11.2, gate G4) against B's real learning service and the stand-in.

B's service is started in-process on a free localhost port with a temporary database and a
throwaway token set only in this test's environment. The test is skipped when no port can be
bound. Scripted messages are synthetic; the report holds metadata only.
"""

from __future__ import annotations

import json
import secrets
from threading import Thread

import pytest
import yaml

from apps.public_api.clients import HttpLearningClient, InProcessResponseClient
from apps.public_api.config import LearningConfig, PublicApiConfig
from apps.public_api.contract import ROOT
from apps.public_api.service import PublicApi
from apps.public_api.storage import RequestStore
from tests.integration.response.learning_standin import LearningStandIn
from tools.response.smoke_sessions import SCRIPTS, run_smoke

TOKEN_ENV = "TUTOR_LEARNING_SERVICE_TOKEN"


@pytest.fixture
def learning_service(tmp_path, monkeypatch):
    from apps.learning_service.engine import LearningEngine, build_strategist
    from apps.learning_service.http import LearningHTTPServer

    monkeypatch.setenv(TOKEN_ENV, "smoke-" + secrets.token_hex(8))
    config = yaml.safe_load((ROOT / "configs/learning/rules.yaml").read_text(encoding="utf-8"))
    engine = LearningEngine(
        config, tmp_path / "learning" / "learning.sqlite3", strategist=build_strategist(config)
    )
    try:
        server = LearningHTTPServer(("127.0.0.1", 0), engine)
    except OSError as error:
        engine.close()
        pytest.skip(f"cannot bind a localhost port for B's learning service: {error}")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        engine.close()


def _assert_report(report: dict, path) -> None:
    summary = report["summary"]
    assert summary["sessions"] == len(SCRIPTS) == 10
    assert summary["sessions_completed"] == 10
    assert summary["errors"] == 0, report["errors"]
    assert summary["retries"]["exercised"] == 10
    assert summary["retries"]["identical"] == 10
    # the immediate_stop script's retried turn ends the session, so it has no hash to compare
    assert summary["retries"]["hash_checked"] == 9
    assert summary["retries"]["hash_matches_learning"] == summary["retries"]["hash_checked"]
    assert summary["turns"] >= 20
    assert set(summary["delivery_modes"]) <= {"model", "repaired", "fallback"}
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == report["run_id"]
    text = path.read_text(encoding="utf-8")
    assert "so I was thinking" not in text  # no learner text in the report


def test_ten_sessions_with_retry_against_b_real_service(tmp_path, learning_service):
    api = PublicApi(
        RequestStore(tmp_path / "requests.sqlite"),
        HttpLearningClient(learning_service, service_token_env=TOKEN_ENV, timeout_s=10),
        InProcessResponseClient(),
        config=PublicApiConfig(learning=LearningConfig(mode="http", base_url=learning_service)),
    )
    report_path = tmp_path / "smoke_report.json"
    report = run_smoke(api, report_path=report_path, learning_label="learning_service")
    _assert_report(report, report_path)
    reasons = report["summary"]["end_reasons"]
    assert reasons.get("learner_stop", 0) >= 8
    immediate = next(s for s in report["sessions_detail"] if s["script"] == "immediate_stop")
    assert immediate["end_reason"] == "learner_stop" and immediate["ended_by"] == "turn"
    assert immediate["teaching_turns"] == 1


def test_ten_sessions_with_retry_against_the_stand_in(tmp_path):
    api = PublicApi(
        RequestStore(tmp_path / "requests.sqlite"), LearningStandIn(), InProcessResponseClient()
    )
    report_path = tmp_path / "smoke_report.json"
    report = run_smoke(api, report_path=report_path, learning_label="person_a_standin")
    _assert_report(report, report_path)
