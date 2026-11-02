"""The frozen extraction evaluation set and its measurement tool."""

import hashlib
import json
from pathlib import Path

from tools.response import evaluate_extraction, make_extraction_cases

ROOT = Path(__file__).parents[3]
REQUESTS = {"none", "skip", "faster", "slower", "hint", "answer", "stop", "change_goal"}
ASSISTANCE = {"none", "hint", "example", "answer_shown", "unknown"}


def load():
    cases = evaluate_extraction.load_jsonl(evaluate_extraction.CASES_PATH)
    contexts = json.loads(evaluate_extraction.CONTEXTS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(evaluate_extraction.MANIFEST_PATH.read_text(encoding="utf-8"))
    return cases, contexts, manifest


def test_committed_files_match_the_builder_and_the_frozen_test_hash():
    contexts = make_extraction_cases.build_contexts()
    outputs = make_extraction_cases.render(make_extraction_cases.build_cases(contexts), contexts)
    for path, text in outputs.items():
        assert path.read_text(encoding="utf-8") == text, path
    cases, _, manifest = load()
    test_text = "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n"
        for case in cases if case["split"] == "test"
    )
    assert hashlib.sha256(test_text.encode("utf-8")).hexdigest() == manifest["test_sha256"]


def test_eval_set_shape_labels_and_provenance():
    cases, contexts, manifest = load()
    assert len(cases) >= 200
    assert manifest["synthetic"] is True
    assert manifest["provenance"]["reviewer_status"] == "pending"
    splits = {}
    for case in cases:
        gold = case["gold"]
        assert case["split"] in {"dev", "test"}
        assert gold["explicit_request"] in REQUESTS
        assert gold["assistance"] in ASSISTANCE
        assert gold["correctness"] in {None, "correct", "incorrect"}
        if gold["explicit_request"] not in {"skip", "change_goal"}:
            assert gold["requested_concept_id"] is None
        for item in gold["misconception_hypotheses"]:
            assert item["misconception_id"] in {f"M0{index}" for index in range(1, 9)}
            for span in item["evidence_spans"]:
                assert 0 <= span["start"] < span["end"] <= len(case["message"])
        assert case["provenance"]["synthetic"] is True
        assert case["provenance"]["reviewer_status"] == "pending_xueming_review"
        assert case["context"]["context_id"] in contexts
        splits.setdefault(case["category"], set()).add(case["split"])
    assert all(found == {"dev", "test"} for found in splits.values())
    assert len(splits) == 20


def test_score_reports_denominators_and_the_grade_guard_on_the_live_extractor():
    cases, contexts, _ = load()
    predictions = evaluate_extraction.run_extractor(cases, contexts)
    results = evaluate_extraction.score(cases, predictions)
    for split in ("test", "dev"):
        metrics = results[split]["metrics"]
        assert metrics["grade_guard"]["n"] == metrics["grade_guard"]["d"] == results[split]["n"]
        assert metrics["schema_valid"]["n"] == results[split]["n"]
        assert metrics["requested_concept_id"]["d"] < results[split]["n"]
        assert set(results[split]["confusion"]) == {
            "explicit_request", "assistance", "answer_presence", "misconception",
        }
    report = evaluate_extraction.render_report(
        results, manifest=json.loads(
            evaluate_extraction.MANIFEST_PATH.read_text(encoding="utf-8")
        ), after_meta={"commit": "test", "extractor_version": "x"},
    )
    assert "## test split" in report and "gold \\ predicted" in report
