from __future__ import annotations

import gzip
import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from packages.evidence_extractor.extract import extract
from packages.learning.episode import load_scenarios, run_episode
from packages.simulator.student import StudentOutcome
from packages.simulator.text_path import (
    BudgetedResponder,
    CallBudget,
    CallLimitExceeded,
    DegradedRealizer,
    ServiceRealizer,
    TextPath,
)
from packages.simulator.text_renderer import render_checked, render_student_outcome
from tools.learning.bridge import (
    MODEL_CONFIG,
    STUB_CONFIG,
    g8_gate,
    main,
    requires_review,
    run_bridge,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan():
    return json.loads(
        (ROOT / "contracts/fixtures/learning/valid/teaching_plan_probe.json").read_text()
    )


def test_stub_bridge_is_exactly_paired():
    report, records = run_bridge(episodes=4)
    assert report["paths"]["structured"]["gain"] == report["paths"]["text"]["gain"]
    assert report["paired_text_minus_structured"] == {"mean": 0.0, "ci_95": [0.0, 0.0]}
    assert all(row["difference"] == 0 for row in report["pairs"])
    assert report["call_estimate"]["actual_model_calls"] == 0
    assert report["g8"]["status"] == "FAIL"
    assert report["g8"]["eligible_test_budget"] is False
    assert records and all("sampled_outcome" in row for row in records)


def test_stub_review_set_is_only_extractor_disagreements():
    report, records = run_bridge(episodes=4)
    required = [row for row in records if requires_review(row)]
    disagreements = [row for row in records if row["flags"]["extractor_disagreement"]]
    assert required == disagreements
    assert any(row["flags"]["fallback_used"] for row in records)
    assert report["g8"]["flagged_traces"] == len(disagreements) < len(records)
    reviews = [
        {
            "path": row["path"],
            "trace_id": row["trace_id"],
            "review_hash": row["review_hash"],
            "reviewed": True,
            "reviewed_by": "Fixture reviewer",
            "critical_failure": False,
        }
        for row in required
    ]
    assert g8_gate(report, records, reviews)["review_complete"]


@pytest.mark.parametrize("seed", [7, 20250115])
def test_default_structured_episode_matches_pre_fidelity_guard(monkeypatch, seed):
    import packages.learning.episode as episode
    import packages.simulator.student as student

    guarded = student.apply_teaching_effect
    original = inspect.getsource(guarded)
    legacy = original.replace(
        'if fidelity > 0 and move in {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"}:',
        'if move in {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"}:',
    )
    assert legacy != original
    namespace = dict(vars(student))
    exec(compile(legacy, "<pre-fidelity-guard>", "exec"), namespace)
    scenario = next(row for row in load_scenarios() if row["split"] == "validation")
    monkeypatch.setattr(episode, "apply_teaching_effect", namespace["apply_teaching_effect"])
    before = run_episode(scenario, episode_seed=seed, text_path=None)

    observed = []

    def checked(*args, **kwargs):
        assert kwargs["content_realized"] is True
        assert kwargs["config"]["content_fidelity"] == {
            "stub_realized": 1.0,
            "failure_probability": 0.0,
        }
        observed.append(kwargs["move"])
        return guarded(*args, **kwargs)

    monkeypatch.setattr(episode, "apply_teaching_effect", checked)
    after = run_episode(scenario, episode_seed=seed, text_path=None)
    assert observed
    assert {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"} & set(observed)
    assert before.gain == after.gain
    assert before.move_histogram == after.move_histogram
    assert before.private_diagnostics == after.private_diagnostics


@pytest.mark.parametrize("answer", [None, "0.6", "3/5", "-2", "100"])
@pytest.mark.parametrize(
    "request_type", ["none", "hint", "answer", "stop", "skip", "faster", "slower"]
)
def test_renderer_only_phrases_sampled_outcome(answer, request_type):
    sampled = StudentOutcome("correct", answer, "none", request_type, None, bool(answer))
    original = asdict(sampled)
    messages = set()
    for key in map(str, range(12)):
        message = render_student_outcome(sampled, key=key)
        assert message == render_student_outcome(sampled, key=key)
        assert asdict(sampled) == original
        evidence = extract(message, {"assistance": "none"}, key)
        assert evidence["normalized_answer"] == answer
        assert evidence["explicit_request"] == request_type
        messages.add(message)
    if request_type in {"none", "hint"} or answer is not None:
        assert len(messages) > 1


def test_optional_verbalizer_cannot_replace_outcome():
    sampled = StudentOutcome("incorrect", "2", "none", "none", None, True)
    frozen, _ = render_checked(sampled, key="sample")
    for replacement in ("My answer is 3.", "Please tell me the answer.", "2 because I guessed."):
        assert render_checked(sampled, key="sample", verbalizer=lambda _: replacement) == (
            frozen,
            True,
        )
    allowed = render_student_outcome(sampled, key="0")
    assert render_checked(sampled, verbalizer=lambda _: allowed) == (allowed, False)


def test_fake_realizer_extractor_disagreement_is_audited(plan):
    # The public extractor inherits assistance from the delivered plan. A hint request
    # sampled with no assistance therefore exposes the A/B interpretation difference.
    plan["assistance"] = "hint"
    realized = ServiceRealizer(STUB_CONFIG, budget=CallBudget())(
        plan,
        request_id="fake-hint",
    )
    path = TextPath(lambda *_args, **_kwargs: realized)
    realized = path.realize(plan, request_id="fake-hint")
    path.check_content(plan, realized, item=None)
    sampled = StudentOutcome("not_applicable", None, "none", "hint", None, False)
    observed = path.observe(sampled, plan, realized, turn=0, item=None)
    assert observed.assistance == "hint"
    assert path.records[0]["flags"]["extractor_disagreement"]
    assert path.records[0]["disagreement_fields"] == ["assistance"]


def test_wrong_move_uses_independent_checks_not_realizer_claims(plan):
    base = ServiceRealizer(STUB_CONFIG, budget=CallBudget())
    fake = DegradedRealizer(base, rate=1)
    path = TextPath(fake)
    realized = path.realize(plan, request_id="wrong-move")
    realized["checks"] = dict.fromkeys(realized["checks"], True)
    original = deepcopy(plan)
    checked = path.check_content(plan, realized, item=None)
    assert plan == original
    assert not checked["checks"]["action_faithful"]
    assert path.current["flags"]["wrong_move"]
    assert path.current["flags"]["validator_failure"]
    assert not path.current["content_fidelity_verified"]


def test_public_item_validator_detects_digit_free_substitution(plan):
    base = ServiceRealizer(STUB_CONFIG, budget=CallBudget())
    path = TextPath(base)
    realized = path.realize(plan, request_id="changed-item")
    realized["text"] = (
        "A fair six-sided die is rolled once. What is the probability of an even number? "
        "Show how you reason it out."
    )
    checked = path.check_content(plan, realized)
    assert not checked["checks"]["item_unchanged"]
    assert path.current["flags"]["item_changed"]
    assert path.current["item_fidelity"]["new_objects"]


def test_disclosed_protected_answer_fails_fidelity(plan):
    from fractions import Fraction

    base = ServiceRealizer(STUB_CONFIG, budget=CallBudget())
    path = TextPath(base)
    realized = path.realize(plan, request_id="leaked-answer")
    realized["text"] += " The final answer is 0.5."
    checked = path.check_content(plan, realized, item={"instance": {"answer": Fraction(1, 2)}})
    assert not checked["checks"]["disclosure_allowed"]
    assert path.current["flags"]["answer_disclosed"]
    assert not path.current["content_fidelity_verified"]


def test_zero_fidelity_prevents_mastery_gain_and_misconception_learning():
    scenario = next(row for row in load_scenarios() if row["split"] == "validation")
    path = TextPath(DegradedRealizer(ServiceRealizer(STUB_CONFIG, budget=CallBudget()), rate=1))
    episode = run_episode(scenario, text_path=path, horizon=4)
    assert episode.gain == 0
    assert episode.private_diagnostics
    for turn in episode.private_diagnostics:
        assert turn["private_mastery_after"] == turn["private_mastery_before"]


def test_fidelity_zero_blocks_misconception_deactivation():
    import yaml

    from packages.learning.episode import load_graph
    from packages.simulator.randomness import EventRandomBank
    from packages.simulator.student import PrivateStudentState, apply_teaching_effect

    config = yaml.safe_load((ROOT / "configs/learning/simulator_proto_v2.yaml").read_text())
    config["misconceptions"]["deactivate_probability"] = {"surfaced": 1, "not_surfaced": 1}
    graph = load_graph()
    before = PrivateStudentState(
        mastery={f"C{i:02d}": 0.2 for i in range(1, 16)},
        learning_rates={"C01": 0.1},
        guess={},
        slip={},
        profile_id="fixture",
        active_misconceptions={"fixture_misconception"},
    )
    options = dict(
        concept_id="C01",
        move="EXPLAIN_CONCEPT",
        difficulty="medium",
        turn=0,
        scenario={"scenario_id": "fidelity-fixture"},
        episode_seed=0,
        graph=graph,
        bank=EventRandomBank(),
        config=config,
        item_id=None,
    )
    invalid = deepcopy(before)
    assert apply_teaching_effect(invalid, content_realized=False, **options) == 0
    assert invalid.active_misconceptions == before.active_misconceptions
    valid = deepcopy(before)
    assert apply_teaching_effect(valid, content_realized=True, **options) > 0
    assert valid.active_misconceptions == set()


def test_test_split_is_sealed_even_for_dry_run(tmp_path):
    with pytest.raises(ValueError, match="unseal-test"):
        run_bridge(episodes=1, split="test", dry_run=True)
    with pytest.raises(SystemExit):
        main(["--split", "test", "--dry-run", "--output", str(tmp_path / "sealed")])
    assert not list(tmp_path.iterdir())


def test_dry_run_does_not_construct_model_or_make_network_calls(monkeypatch):
    import socket

    import packages.response.responders as responders

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry-run attempted a model/network operation")

    monkeypatch.setattr(responders.OpenAICompatibleResponder, "generate", forbidden)
    monkeypatch.setattr(responders.OpenAICompatibleResponder, "__init__", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    report, records = run_bridge(
        episodes=2,
        reply="openai",
        dry_run=True,
        verbalizer_config=MODEL_CONFIG,
    )
    assert records and all(row["path"] == "structured" for row in records)
    calls = report["call_estimate"]
    assert calls["actual_model_calls"] == 0
    assert calls["planned_primary_calls_at_stub_trajectory"] == len(records)
    assert calls["maximum_primary_calls"] == 32
    assert calls["maximum_repair_calls"] == 32
    assert calls["maximum_verbalizer_calls"] == 32
    assert "g8" not in report


def test_call_budget_covers_generate_and_repair():
    class FakeResponder:
        def generate(self):
            return "primary"

        def repair(self):
            return "repair"

    budget = CallBudget(2)
    responder = BudgetedResponder(FakeResponder(), budget)
    assert responder.generate() == "primary"
    assert responder.repair() == "repair"
    with pytest.raises(CallLimitExceeded):
        responder.generate()
    assert budget.calls == 2 and budget.exceeded
    with pytest.raises(ValueError, match="positive --max-calls"):
        run_bridge(episodes=1, reply="openai")
    with pytest.raises(ValueError, match="not template_stub"):
        run_bridge(episodes=1, reply="openai", response_config=STUB_CONFIG, max_calls=1)


def test_service_does_not_turn_budget_exhaustion_into_a_fallback_score(monkeypatch):
    import apps.response_service.config as service_config
    from packages.response.responders import TemplateStubResponder

    monkeypatch.setenv("TUTOR_LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("TUTOR_LLM_MODEL", "fake-no-network")
    monkeypatch.setattr(
        service_config, "build_responder", lambda *_args, **_kwargs: TemplateStubResponder()
    )
    with pytest.raises(CallLimitExceeded, match="run aborted"):
        run_bridge(episodes=1, reply="openai", max_calls=1)


def test_low_confidence_extraction_is_not_independent_answer_evidence(plan):
    base = ServiceRealizer(STUB_CONFIG, budget=CallBudget())
    path = TextPath(base, renderer=lambda *_args, **_kwargs: ("Maybe 2 or 3", False))
    realized = path.realize(plan, request_id="low-confidence")
    path.check_content(plan, realized)
    sampled = StudentOutcome("correct", "3", "none", "none", None, True)
    observed = path.observe(sampled, plan, realized, turn=0, item=None)
    assert path.records[0]["extracted_evidence"]["confidence"] < 0.8
    assert observed.normalized_answer is None and not observed.independent
    assert path.records[0]["flags"]["extractor_disagreement"]


def test_g8_requires_test_pairs_real_replies_and_human_review():
    record = {
        "path": "text",
        "trace_id": "trace-1",
        "review_hash": "hash-1",
        "flags": {"extractor_disagreement": True},
    }
    report = {
        "split": "test",
        "episodes": 300,
        "reply": "openai",
        "paths": {"text": {"model_delivered_episodes": 300}},
        "paired_text_minus_structured": {"mean": -0.02, "ci_95": (-0.03, -0.01)},
    }
    review = {
        "path": "text",
        "trace_id": "trace-1",
        "review_hash": "hash-1",
        "reviewed": True,
        "reviewed_by": "Human reviewer",
        "critical_failure": False,
    }
    assert g8_gate(report, [record])["status"] == "FAIL"
    assert g8_gate(report, [record], [review])["status"] == "PASS"
    assert g8_gate(report, [record], [{**review, "critical_failure": True}])["status"] == "FAIL"
    assert g8_gate({**report, "split": "validation"}, [record], [review])["status"] == "FAIL"
    assert g8_gate({**report, "reply": "stub"}, [record], [review])["status"] == "FAIL"
    assert g8_gate({**report, "episodes": 299}, [record], [review])["status"] == "FAIL"
    assert (
        g8_gate({**report, "paths": {"text": {"model_delivered_episodes": 0}}}, [record], [review])[
            "status"
        ]
        == "FAIL"
    )
    with pytest.raises(ValueError, match="review_hash"):
        g8_gate(report, [record], [{**review, "review_hash": "stale"}])
    informational = {
        **record,
        "trace_id": "informational",
        "flags": {"fallback_used": True, "draft_validator_failure": True},
    }
    gate = g8_gate(report, [informational])
    assert gate["flagged_traces"] == 0
    assert gate["review_complete"] and gate["status"] == "PASS"
    assert g8_gate(report, [record, informational], [review])["status"] == "PASS"
    with pytest.raises(ValueError, match="flagged trace"):
        g8_gate(report, [informational], [{**review, "trace_id": "informational"}])


def test_report_saves_only_critical_turns_and_keeps_full_dump_outside_output(tmp_path, monkeypatch):
    import tools.learning.bridge as bridge

    monkeypatch.setattr(bridge, "AUDIT_DIR", tmp_path / "audit")
    report, records = run_bridge(episodes=1, reply="degraded", degradation_rate=1)
    output = tmp_path / "smoke"
    write_report(output, report, records)
    with gzip.open(output.with_suffix(".traces.jsonl.gz"), "rt") as source:
        traces = [json.loads(line) for line in source]
    assert len(traces) == sum(requires_review(row) for row in records)
    assert all(requires_review(row) for row in traces)
    assert any(row["flags"]["wrong_move"] for row in traces)
    assert all(row["plan"] and row["response"] and row["review_hash"] for row in traces)
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["label"] == "SIMULATED"
    assert saved["g8"]["status"] == "FAIL"
    turns = saved["artifacts"]["turns"]
    assert set(turns) == {"sha256"}
    full = bridge.AUDIT_DIR / f"{turns['sha256']}.turns.jsonl.gz"
    assert hashlib.sha256(full.read_bytes()).hexdigest() == turns["sha256"]
    with gzip.open(full, "rt") as source:
        assert [json.loads(line) for line in source] == json.loads(json.dumps(records))
    assert not output.with_suffix(".turns.jsonl.gz").exists()
    assert not saved["artifacts"]["traces"]["sampled"]


def test_large_critical_dump_is_sampled_deterministically_per_run(tmp_path, monkeypatch):
    import tools.learning.bridge as bridge

    monkeypatch.setattr(bridge, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(bridge, "MAX_REVIEW_BYTES", 1)
    records = [
        {"run": run, "path": "text", "trace_id": str(index), "flags": {"wrong_move": True}}
        for run in ("stub", "degraded")
        for index in range(220)
    ]
    report = {"label": "SIMULATED", "smokes": {}}
    write_report(tmp_path / "first", report, records)
    first = deepcopy(report["artifacts"]["traces"])
    with gzip.open(first["path"], "rt") as source:
        rows = [json.loads(line) for line in source]
    assert first["sampled"] and first["full_rows"] == 440
    assert first["rows"] == 400
    assert all(sum(row["run"] == run for row in rows) == 200 for run in ("stub", "degraded"))
    full = bridge.AUDIT_DIR / f"{first['full_sha256']}.traces.jsonl.gz"
    assert hashlib.sha256(full.read_bytes()).hexdigest() == first["full_sha256"]
    write_report(tmp_path / "second", report, list(reversed(records)))
    assert report["artifacts"]["traces"]["sha256"] == first["sha256"]
