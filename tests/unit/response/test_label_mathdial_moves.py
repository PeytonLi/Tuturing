import json
from pathlib import Path

import pytest
import yaml

from tests.unit.response.fake_openai import FakeOpenAIServer, Reply
from tools.response import label_mathdial_moves as labeler
from tools.response.label_mathdial_moves import (
    ChatClient,
    LabelStore,
    SpendTracker,
    act_mapping_agreement,
    fit_tokens,
    label_agreement,
    label_one,
    label_turns,
    load_api_key,
    make_estimator,
    parse_label,
    request_cost,
    select_within_budget,
)

ROOT = Path(__file__).resolve().parents[3]
PRICES = {"model": "fake", "input_per_m": 1.0, "cached_input_per_m": 0.1, "output_per_m": 10.0}


def _turn(index, split="train", act="focus"):
    return {
        "example_id": f"mathdial:train:{index:04d}:t02",
        "split": split,
        "source_act": act,
        "teacher_text": f"Look at the number of boxes in step {index}.",
        "context_last_student": "I multiplied 3 by 4 and got 12.",
    }


def _client(server, **overrides):
    options = {"temperature": 0, "reasoning_effort": "minimal", "sleep": lambda _: None}
    options.update(overrides)
    return ChatClient(server.base_url, "sk-test", "fake", **options)


def test_parse_label_accepts_the_strict_format_and_rejects_the_rest():
    assert parse_label("LABEL: HINT\nREASON: points at the next step") == (
        "HINT", "points at the next step"
    )
    assert parse_label("**LABEL:** REFLECT  \n**REASON:** asks why.") == ("REFLECT", "asks why.")
    assert parse_label("LABEL: EXPLAIN\nREASON: not a move") == (None, None)
    assert parse_label("I think it is a hint.") == (None, None)
    assert parse_label(None) == (None, None)


def test_malformed_reply_is_retried_once_then_excluded():
    with FakeOpenAIServer(["no label here", "still nothing"]) as server:
        result, usages, outputs = label_one(_client(server), _turn(1), "system")
    assert result == {"label": "EXCLUDE", "reason": "malformed output after one retry",
                      "status": "malformed"}
    assert len(usages) == 2 and len(outputs) == 2
    retry_body = server.requests[1]["body"]
    assert retry_body["messages"][-1]["content"].startswith("Your reply did not follow")
    with FakeOpenAIServer(["??", "LABEL: PROBE\nREASON: asks a quantity"]) as server:
        result, _, _ = label_one(_client(server), _turn(2), "system")
    assert result["label"] == "PROBE" and result["status"] == "labeled_after_retry"


def test_client_adapts_parameters_the_model_rejects():
    rejection = Reply(
        status=400,
        raw_body=json.dumps({"error": {"message": (
            "Unsupported value: 'temperature' does not support 0 with this model. "
            "Only the default (1) value is supported."
        )}}).encode(),
    )
    effort = Reply(
        status=400,
        raw_body=json.dumps({"error": {"message": (
            "Unsupported value: 'reasoning_effort' does not support 'minimal' with this model. "
            "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'."
        )}}).encode(),
    )
    with FakeOpenAIServer([rejection, effort, "LABEL: HINT\nREASON: cue"]) as server:
        client = _client(server)
        content, usage = client.complete([{"role": "user", "content": "x"}])
    assert content.startswith("LABEL: HINT")
    bodies = [request["body"] for request in server.requests]
    assert "temperature" in bodies[0] and "temperature" not in bodies[1]
    assert bodies[1]["reasoning_effort"] == "minimal"
    assert bodies[2]["reasoning_effort"] == "none"
    assert "max_completion_tokens" in bodies[2]
    assert len(client.adjustments) == 2
    assert all("sk-test" not in note for note in client.adjustments)
    with FakeOpenAIServer([rejection, "LABEL: HINT\nREASON: cue"]) as server:
        client = _client(server, temperature=None)
        content, _ = client.complete([{"role": "user", "content": "x"}])
    assert content.startswith("LABEL: HINT") and client.adjustments == []


def test_client_retries_transient_errors_and_reports_hard_failures():
    with FakeOpenAIServer([Reply(status=429), "LABEL: PROBE\nREASON: ok"]) as server:
        content, _ = _client(server).complete([{"role": "user", "content": "x"}])
    assert content.startswith("LABEL: PROBE")
    with FakeOpenAIServer([Reply(status=401)]) as server:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            _client(server).complete([{"role": "user", "content": "x"}])


def test_label_turns_is_resumable_and_stops_at_the_cap(tmp_path):
    turns = [_turn(index) for index in range(4)]
    tracker = SpendTracker(tmp_path / "ledger.jsonl", cap_usd=1.0, prices_by_model={"fake": PRICES})
    store = LabelStore(tmp_path / "labels.jsonl")
    usage = {"prompt_tokens": 1000, "completion_tokens": 10}
    replies = [Reply("LABEL: HINT\nREASON: cue", usage=usage) for _ in range(2)]
    with FakeOpenAIServer(replies) as server:
        outcome = label_turns(
            turns[:2], _client(server), store, tracker, stage="full",
            estimate_per_turn=lambda turn: 0.001, system_prompt="system",
            prompt_version="test_prompt", concurrency=2, log=lambda _: None,
        )
    assert outcome["labeled"] == 2 and outcome["errors"] == 0
    assert set(store.records) == {turns[0]["example_id"], turns[1]["example_id"]}
    assert tracker.requests == 2
    assert tracker.spent == pytest.approx(2 * request_cost(
        {"prompt_tokens": 1000, "completion_tokens": 10}, PRICES
    ))
    assert request_cost({"prompt_tokens": 1000, "cached_tokens": 500, "completion_tokens": 0},
                        PRICES) == pytest.approx((500 * 1.0 + 500 * 0.1) / 1e6)
    ledger_lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(ledger_lines) == 2 and "sk-test" not in "".join(ledger_lines)

    resumed_tracker = SpendTracker(tmp_path / "ledger.jsonl", 1.0, {"fake": PRICES})
    assert resumed_tracker.spent == pytest.approx(tracker.spent)
    resumed_store = LabelStore(tmp_path / "labels.jsonl")
    with FakeOpenAIServer(["LABEL: PROBE\nREASON: asks"]) as server:
        outcome = label_turns(
            turns[:3], _client(server), resumed_store, resumed_tracker, stage="full",
            estimate_per_turn=lambda turn: 0.001, system_prompt="system",
            prompt_version="test_prompt", concurrency=2, log=lambda _: None,
        )
    assert outcome["already_done"] == 2 and outcome["labeled"] == 1
    assert len(server.requests) == 1

    capped = SpendTracker(
        tmp_path / "capped.jsonl", cap_usd=0.0005, prices_by_model={"fake": PRICES}
    )
    with FakeOpenAIServer(["LABEL: PROBE\nREASON: asks"]) as server:
        outcome = label_turns(
            [turns[3]], _client(server), LabelStore(tmp_path / "capped_labels.jsonl"), capped,
            stage="full", estimate_per_turn=lambda turn: 0.001, system_prompt="system",
            prompt_version="test_prompt", concurrency=1, log=lambda _: None,
        )
    assert outcome["labeled"] == 0 and outcome["skipped_for_cap"] == 1
    assert server.requests == []


def test_agreement_fit_and_budget_selection_helpers():
    calibration = {
        "a": {"draft": "HINT", "source_act": "focus"},
        "b": {"draft": "PROBE", "source_act": "focus"},
        "c": {"draft": "EXCLUDE", "source_act": "generic"},
    }
    move_map = {"mappings": {"focus": {"move": "HINT"}, "generic": {"exclude": True}}}
    act = act_mapping_agreement(calibration, move_map)
    assert act["overall"] == {"agree": 2, "n": 3, "rate": 0.6667}
    assert act["per_act"]["focus"] == {"agree": 1, "n": 2, "rate": 0.5}
    records = {"a": {"label": "HINT"}, "b": {"label": "HINT"}}
    agreement = label_agreement(records, calibration)
    assert agreement["overall"] == {"agree": 1, "n": 2, "rate": 0.5}
    assert agreement["draft_vs_model"] == {"HINT": {"HINT": 1}, "PROBE": {"HINT": 1}}

    fit = fit_tokens({
        "x": {"words": 10, "usage": {"prompt_tokens": 620, "completion_tokens": 40}},
        "y": {"words": 110, "usage": {"prompt_tokens": 780, "completion_tokens": 60}},
    })
    assert fit["per_word"] == pytest.approx(1.6) and fit["intercept"] == pytest.approx(604.0)
    assert fit["cached_fraction"] == 0.0
    estimator = make_estimator(fit, PRICES, safety=1.0)
    turn = {"context_last_student": "one two three", "teacher_text": "four five"}
    assert estimator(turn) == pytest.approx((604 + 1.6 * 5) * 1.0 / 1e6 + 50 * 10.0 / 1e6)

    turns = [
        *[_turn(index, "train", "focus") for index in range(4)],
        *[_turn(10 + index, "train", "probing") for index in range(4)],
        _turn(20, "validation"), _turn(21, "test"),
    ]
    selected, planned = select_within_budget(turns, lambda turn: 0.1, budget_usd=0.65)
    assert selected[0]["split"] == "test"
    assert len(selected) == 6 and planned == pytest.approx(0.6)
    assert [turn["split"] for turn in selected[1:]] == ["train"] * 4 + ["validation"]
    assert {turn["source_act"] for turn in selected[1:5]} == {"focus", "probing"}


def test_few_shot_examples_are_held_out_of_the_agreement(tmp_path):
    calibration = {
        f"id{index}": {
            "draft": "HINT" if index % 2 else "PROBE", "source_act": "focus",
            "teacher_text": f"turn {index}", "reason": "why", "ambiguous": "no",
        }
        for index in range(8)
    }
    prompt = tmp_path / "prompt_v9.txt"
    prompt.write_text("Rules.\n\n{examples}\n\nReply.", encoding="utf-8")
    version, text, ids = labeler.load_prompt(
        {"prompt_file": str(prompt), "few_shot": {"per_label": 2, "seed": 1}}, calibration
    )
    assert version == "prompt_v9-fs2-s1" and len(ids) == 4
    assert "{examples}" not in text and '-> HINT (why)' in text and text.endswith("Reply.")
    plain_version, plain_text, plain_ids = labeler.load_prompt({"prompt_file": str(prompt)})
    assert plain_version == "prompt_v9" and plain_ids == [] and "{examples}" not in plain_text


def test_wilson_interval_and_repeat_consistency():
    low, high = labeler.wilson_interval(146, 182)
    assert low == pytest.approx(0.7376, abs=1e-3) and high == pytest.approx(0.8538, abs=1e-3)
    assert labeler.wilson_interval(0, 0) is None
    first = {"a": {"label": "HINT"}, "b": {"label": "PROBE"}, "c": {"label": "HINT"}}
    second = {"a": {"label": "HINT"}, "b": {"label": "HINT"}}
    assert labeler.repeat_consistency(first, second, ["a", "b", "c"]) == {
        "rows": 2, "same_label": 1, "rate": 0.5,
    }


def test_key_file_parsing_never_echoes_the_value(tmp_path):
    env = tmp_path / "llm.env"
    env.write_text("# keys\nexport OTHER=1\nOPENAI_API_KEY=\"sk-secret-value\"\n", encoding="utf-8")
    assert load_api_key(env, "OPENAI_API_KEY") == "sk-secret-value"
    with pytest.raises(KeyError):
        load_api_key(env, "MISSING")


def test_pilot_full_and_report_run_against_the_fake_server(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/response/mathdial_moves_v1.yaml").read_text())
    config["run_dir"] = str(tmp_path / "run")
    config["labels_path"] = str(tmp_path / "labels.jsonl")
    config["manifest_path"] = str(tmp_path / "manifest.json")
    config["calibration_csv"] = str(tmp_path / "calibration.csv")
    config["prompt_file"] = str(ROOT / config["prompt_file"])
    config["few_shot"] = {}
    config["concurrency"] = 2
    config["candidates"] = [PRICES]
    monkeypatch.setattr(labeler, "ROOT", Path("/"))
    splits = ("train", "train", "validation", "test")
    turns = [_turn(index, split) for index, split in enumerate(splits)]
    (tmp_path / "calibration.csv").write_text(
        "example_id,source_act,model_draft_label\n"
        f"{turns[0]['example_id']},focus,HINT\n{turns[1]['example_id']},focus,PROBE\n",
        encoding="utf-8",
    )
    with FakeOpenAIServer(["LABEL: HINT\nREASON: cue", "LABEL: HINT\nREASON: cue"]) as server:
        summary = labeler.pilot(config, "fake", turns, "sk-test", base_url=server.base_url,
                                log=lambda _: None)
    assert summary["agreement"]["overall"] == {"agree": 1, "n": 2, "rate": 0.5}
    assert summary["meets_threshold"] is False
    assert summary["cost_usd"] > 0
    replies = ["LABEL: PROBE\nREASON: asks", "LABEL: EXCLUDE\nREASON: praise",
               "garbage", "LABEL: REFLECT\nREASON: why", "LABEL: HINT\nREASON: cue"]
    with FakeOpenAIServer(replies) as server:
        full_summary = labeler.full(config, "fake", turns, "sk-test", base_url=server.base_url,
                                    log=lambda _: None)
    assert full_summary["outcome"]["labeled"] == 4
    assert full_summary["selection"]["rule"] == "all pending turns"
    with FakeOpenAIServer([]) as server:
        again = labeler.full(config, "fake", turns, "sk-test", base_url=server.base_url,
                             log=lambda _: None)
    assert again["outcome"]["already_done"] == 4 and len(again["invocations"]) == 2
    universe = {"teacher_turns_total": 4, "skipped": {}, "labelable": 4, "by_split_and_act": {}}
    source = {"source_id": "mathdial", "revision": "rev-test"}
    config["act_move_map"] = str(ROOT / "configs/response/mathdial_move_map_v1.yaml")
    manifest = labeler.report(config, "fake", turns, universe, source, log=lambda _: None)
    assert manifest["labeled"] == 4 and manifest["unlabeled"]["count"] == 0
    assert manifest["counts_per_label"] == {"EXCLUDE": 1, "HINT": 1, "PROBE": 1, "REFLECT": 1}
    assert manifest["status_counts"] == {"labeled": 3, "labeled_after_retry": 1}
    assert manifest["model_selection"]["chosen_model"] == "fake"
    repeat = manifest["calibration"]["repeat_consistency_pilot_vs_full"]
    assert repeat["rows"] == 2 and 0 <= repeat["same_label"] <= 2
    pilot_entry = next(iter(manifest["model_selection"]["pilots"].values()))
    assert pilot_entry["agreement_wilson_95"] == labeler.wilson_interval(1, 2)
    assert "sk-test" not in json.dumps(manifest)
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert set(rows[0]) == {"example_id", "split", "source_act", "label", "status", "model",
                            "prompt_version"}
    assert all("boxes" not in json.dumps(row) for row in rows)


def test_committed_config_is_consistent():
    config = yaml.safe_load((ROOT / "configs/response/mathdial_moves_v1.yaml").read_text())
    assert config["key_name"] == "OPENAI_API_KEY"
    assert config["spend_cap_usd"] == 5.0 and config["agreement_threshold"] == 0.85
    prices = [candidate["input_per_m"] for candidate in config["candidates"]]
    assert prices == sorted(prices)
    for key in ("source_config", "act_move_map", "calibration_csv", "prompt_file"):
        assert (ROOT / config[key]).exists(), key
