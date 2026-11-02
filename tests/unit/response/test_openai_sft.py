"""Offline tests for OpenAI fine-tuning (option B) and fine-tuned model evaluation entries."""

import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from packages.response.prompt import compile_prompt
from tools.response import benchmark as bench
from tools.response import openai_sft as osft
from tools.response.fireworks_sft import SFTError, write_jsonl

ROOT = Path(__file__).parents[3]
PLAN_DIR = Path(__file__).parent / "fixtures" / "plans"
SECRET = "sk-test-secret-value-1234567890"


def plan(name="practice"):
    return json.loads((PLAN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def example(name="practice", reply="Here is your problem. Give it a try."):
    return {"example_id": f"t:{name}", "plan": plan(name), "prompt": compile_prompt(plan(name)),
            "completion": [{"role": "assistant", "content": reply}]}


def words(text):
    return text.split()


# ---------------------------------------------------------------------------
# Format, tokens and cost


def test_record_puts_weight_only_on_the_reply():
    record = osft.to_openai_record(example())
    messages = record["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert "weight" not in messages[0] and "weight" not in messages[1]
    assert messages[2]["weight"] == 1
    assert [messages[0]["content"], messages[1]["content"]] == [
        m["content"] for m in compile_prompt(plan())]
    assert osft.record_errors(record) == []


@pytest.mark.parametrize("record, message", [
    ({"messages": [{"role": "user", "content": "q", "weight": 0},
                   {"role": "assistant", "content": "a"}]}, "unexpected keys ['weight']"),
    ({"messages": [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "a", "weight": 2}]}, "weight must be 0 or 1"),
    ({"messages": [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "a", "weight": 0}]}, "no trained assistant"),
    ({"messages": [{"role": "assistant", "content": "a"},
                   {"role": "user", "content": "q"}]}, "last message"),
    ({"messages": [{"role": "user", "content": "q"}, {"role": "system", "content": "s"},
                   {"role": "assistant", "content": "a"}]}, "must come first"),
    ({"messages": [{"role": "tool", "content": "x"},
                   {"role": "assistant", "content": "a"}]}, "not system, user or assistant"),
    ({"messages": [{"role": "user", "content": " "},
                   {"role": "assistant", "content": "a"}]}, "non-empty string"),
    ({"messages": [{"role": "assistant", "content": "a"}], "tools": []}, "top-level keys"),
    ({"messages": []}, "non-empty list"),
])
def test_record_errors_catch_format_problems(record, message):
    assert any(message in error for error in osft.record_errors(record))


def test_token_count_and_file_check(tmp_path):
    record = {"messages": [{"role": "system", "content": "a b"},
                           {"role": "user", "content": "c d e"},
                           {"role": "assistant", "content": "f g", "weight": 1}]}
    # 3 per example + per message (3 + role word + content words)
    assert osft.count_tokens(record, words) == (3 + (3 + 1 + 2) + (3 + 1 + 3) + (3 + 1 + 2), 2)
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [record] * 12)
    report = osft.check_file(path, words, max_example_tokens=100)
    assert report["error_count"] == 0 and report["rows"] == 12
    assert report["total_tokens"] == 12 * 22 and report["trained_tokens"] == 24
    short = osft.check_file(path, words, max_example_tokens=20, min_examples=20)
    assert any("exceeds 20" in e for e in short["errors"])
    assert any("at least 20" in e for e in short["errors"])
    path.write_text(path.read_text() + "not json\n", encoding="utf-8")
    assert any("invalid JSON" in e for e in osft.check_file(path, words)["errors"])


def test_heuristic_encoder_and_cost_estimate():
    assert len(osft.heuristic_encode("abcdefg")) == 3
    estimate = osft.estimate_cost(200_000, 40_000, 3, {"training_usd_per_mtok": 5.0})
    assert estimate["billed_training_tokens_expected"] == 600_000
    assert estimate["expected_usd"] == pytest.approx(3.0)
    assert estimate["upper_bound_usd"] == pytest.approx(3.6)
    assert estimate["reserve_usd"] == pytest.approx(5.4)


def test_multipart_form_has_purpose_and_file():
    body, content_type = osft.multipart_form(
        {"purpose": "fine-tune"}, {"file": ("t.jsonl", b"{}\n", "application/jsonl")}, "B")
    assert content_type == "multipart/form-data; boundary=B"
    assert b'name="purpose"\r\n\r\nfine-tune\r\n' in body
    assert b'name="file"; filename="t.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n{}\n' in body
    assert body.endswith(b"--B--\r\n")


def test_client_redacts_the_key_and_returns_raw_bytes():
    seen = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b"step,train_loss\n1,1.5\n"

    def urlopen(request, timeout=None):
        seen.append(request)
        if request.full_url.endswith("/bad"):
            raise urllib.error.HTTPError(
                request.full_url, 401, "no", {},
                __import__("io").BytesIO(f'{{"error": "bad key {SECRET}"}}'.encode()))
        return Response()

    client = osft.OpenAIClient(SECRET, urlopen=urlopen)
    status, payload = client.request("GET", "/v1/bad")
    assert status == 401 and SECRET not in json.dumps(payload)
    status, content = client.request("GET", "/v1/files/f/content", raw=True)
    assert content.startswith(b"step,train_loss")
    assert seen[1].get_header("Authorization") == f"Bearer {SECRET}"


# ---------------------------------------------------------------------------
# Files, job and artifacts against an API double


class FakeOpenAI:
    def __init__(self, final_status="succeeded", trained_tokens=586_527, events_pages=2):
        self.calls = []
        self.final_status = final_status
        self.trained_tokens = trained_tokens
        self.polls = 0
        self.events_pages = events_pages
        self.files = {}

    def request(self, method, path, body=None, query=None, *, data=None,
                content_type="application/json", raw=False):
        self.calls.append((method, path, body, query))
        if method == "POST" and path == "/v1/files":
            assert content_type.startswith("multipart/form-data")
            assert b'name="purpose"\r\n\r\nfine-tune' in data
            file_id = f"file-{len(self.files) + 1}"
            self.files[file_id] = data
            return 200, {"id": file_id, "filename": "x.jsonl", "status": "processed"}
        if method == "GET" and path.startswith("/v1/files/") and path.endswith("/content"):
            return 200, b"step,train_loss,valid_loss\n1,2.0,2.1\n"
        if method == "GET" and path.startswith("/v1/files/"):
            return (200, {"id": path.rsplit("/", 1)[1]}) if path.rsplit("/", 1)[1] in self.files \
                else (404, {"error": "no such file"})
        if method == "POST" and path == "/v1/fine_tuning/jobs":
            return 200, {"id": "ftjob-1", "status": "validating_files", **body}
        if method == "GET" and path == "/v1/fine_tuning/jobs/ftjob-1":
            self.polls += 1
            done = self.polls >= 3
            job = {"id": "ftjob-1", "model": "gpt-4.1-mini-2025-04-14",
                   "status": self.final_status if done else "running",
                   "hyperparameters": {"n_epochs": 3, "batch_size": 1,
                                       "learning_rate_multiplier": 2.0}}
            if done and self.final_status == "succeeded":
                job.update(fine_tuned_model="ft:gpt-4.1-mini-2025-04-14:org:tut-sft-full-v1:abc",
                           trained_tokens=self.trained_tokens, result_files=["file-r1"])
            return 200, job
        if path == "/v1/fine_tuning/jobs/ftjob-1/events":
            page = 1 if not (query or {}).get("after") else 2
            return 200, {"data": [{"id": f"ev-{page}", "message": "step"}],
                         "has_more": page < self.events_pages}
        if path == "/v1/fine_tuning/jobs/ftjob-1/checkpoints":
            return 200, {"data": [
                {"step_number": 870, "fine_tuned_model_checkpoint": "ft:x:ckpt-step-870",
                 "metrics": {"full_valid_loss": 0.9}},
                {"step_number": 435, "fine_tuned_model_checkpoint": "ft:x:ckpt-step-435",
                 "metrics": {"full_valid_loss": 1.1}},
            ]}
        if method == "POST" and path.endswith("/cancel"):
            return 200, {"status": "cancelled"}
        raise AssertionError(f"unexpected {method} {path}")


def openai_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/response/sft_openai_full_v1.yaml").read_text())
    cfg["openai"]["run_dir"] = str(tmp_path / "runs")
    return cfg


def prepared(tmp_path, api, cap=25.0):
    cfg = openai_config(tmp_path)
    run_dir = Path(cfg["openai"]["run_dir"])
    files = {}
    for key, rows in (("train", 12), ("validation", 10)):
        path = run_dir / "data" / f"{key}.jsonl"
        write_jsonl(path, [osft.to_openai_record(example())] * rows)
        files[key] = osft.check_file(path, words)
    estimate = osft.estimate_cost(195_509, 41_730, 3, cfg["openai"]["pricing"])
    (run_dir / "check.json").write_text(json.dumps({**files, "estimate": estimate}))
    ledger = bench.Ledger(run_dir / "ledger.jsonl")
    clock = {"now": 0.0}
    run = osft.OpenAIRun(api, cfg, ledger, cap, out=lambda *_a: None,
                         sleep=lambda s: clock.update(now=clock["now"] + s),
                         clock=lambda: clock["now"])
    return run, run_dir, estimate


def test_upload_reuses_identical_files_and_refuses_unchecked_ones(tmp_path):
    api = FakeOpenAI()
    run, run_dir, _ = prepared(tmp_path, api)
    first = run.upload("train", run_dir / "data" / "train.jsonl")
    again = run.upload("train", run_dir / "data" / "train.jsonl")
    assert first["id"] == again["id"] == "file-1" and len(api.files) == 1
    (run_dir / "data" / "validation.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SFTError, match="changed since the check"):
        run.upload("validation", run_dir / "data" / "validation.jsonl")


def test_job_body_dry_run_budget_and_settlement_from_trained_tokens(tmp_path):
    api = FakeOpenAI()
    run, run_dir, estimate = prepared(tmp_path, api)
    for key in ("train", "validation"):
        run.upload(key, run_dir / "data" / f"{key}.jsonl")
    dry = run.create_job(estimate)
    body = dry["body"]
    assert body["model"] == "gpt-4.1-mini-2025-04-14" and body["suffix"] == "tut-sft-full-v1"
    assert body["method"] == {"type": "supervised",
                              "supervised": {"hyperparameters": {"n_epochs": 3}}}
    assert body["validation_file"] == "file-2" and body["seed"] == 17
    assert not any(c[1] == "/v1/fine_tuning/jobs" for c in api.calls)
    small, small_dir, _ = prepared(tmp_path / "small", FakeOpenAI(), cap=1.0)
    small.upload("train", small_dir / "data" / "train.jsonl")
    with pytest.raises(SFTError, match="passes the cap"):
        small.create_job(estimate, confirm=True)
    run.create_job(estimate, confirm=True)
    assert run.ledger.committed() == pytest.approx(estimate["reserve_usd"])
    final = run.poll(timeout_s=1000, poll_s=60)
    assert final["status"] == "succeeded"
    assert [t["status"] for t in run.state["job"]["transitions"]] == ["running", "succeeded"]
    assert run.state["job"]["settled"]["final_usd"] == pytest.approx(586_527 * 5 / 1e6, abs=1e-4)
    assert run.ledger.spent() == pytest.approx(586_527 * 5 / 1e6, abs=1e-4)


def test_failed_job_without_trained_tokens_keeps_the_reserve(tmp_path):
    api = FakeOpenAI(final_status="failed")
    run, run_dir, estimate = prepared(tmp_path, api)
    run.upload("train", run_dir / "data" / "train.jsonl")
    run.create_job(estimate, confirm=True)
    run.poll(timeout_s=1000, poll_s=60)
    assert run.state["job"]["settled"]["final_usd"] == pytest.approx(estimate["reserve_usd"])


def test_fetch_writes_models_with_epoch_checkpoints_and_select_survives(tmp_path):
    api = FakeOpenAI()
    run, run_dir, estimate = prepared(tmp_path, api)
    run.upload("train", run_dir / "data" / "train.jsonl")
    run.create_job(estimate, confirm=True)
    run.poll(timeout_s=1000, poll_s=60)
    models = run.fetch()
    assert models["final"].startswith("ft:gpt-4.1-mini")
    assert (models["epoch_2"], models["epoch_3"]) == ("ft:x:ckpt-step-435", "ft:x:ckpt-step-870")
    assert run.state["artifacts"]["events"] == 2
    assert (run_dir / "artifacts" / "file-r1.csv").read_bytes().startswith(b"step,")
    selected = run.select("epoch_2", "lowest math and disclosure failures on validation")
    assert selected["selected"] == "ft:x:ckpt-step-435"
    with pytest.raises(SFTError, match="has no epoch_9"):
        run.select("epoch_9", "x")
    assert run.fetch()["selected"] == "ft:x:ckpt-step-435"  # a refetch keeps the choice


# ---------------------------------------------------------------------------
# Fine-tuned entries in the benchmark harness


def test_fine_tuned_entries_resolve_from_models_json(tmp_path):
    models_file = tmp_path / "models.json"
    entry = {"id": "ft", "provider": "openai_compatible", "model": "{fine_tuned}",
             "api_key_env": "OPENAI_API_KEY", "decoding": {"temperature": 0, "max_tokens": 384},
             "fine_tuned": {"models_file": str(models_file), "key": "final"},
             "pricing": {"basis": "tokens", "input_usd_per_mtok": 0.8,
                         "output_usd_per_mtok": 3.2}}
    config = {"models": [entry], "keys_file": "unused"}
    ctx = bench.RunContext(config=config, keys={"OPENAI_API_KEY": SECRET}, cap=25.0,
                           ledger=bench.Ledger(tmp_path / "ledger.jsonl"), run_dir=tmp_path)
    ok, reason = bench.readiness(entry, ctx)
    assert not ok and "no final model" in reason
    with pytest.raises(bench.BenchmarkError, match="not available yet"):
        bench.resolve_model(entry, ctx)
    models_file.write_text(json.dumps({"final": "ft:gpt-4.1-mini:org:a:1"}))
    assert bench.readiness(entry, ctx) == (True, "ready")
    assert bench.resolve_model(entry, ctx) == "ft:gpt-4.1-mini:org:a:1"
    first = bench.decoding_id(entry)
    models_file.write_text(json.dumps({"final": "ft:gpt-4.1-mini:org:a:2"}))
    assert bench.decoding_id(entry) != first  # a new model never reuses old records


def test_committed_openai_configs():
    supported = {"gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14"}
    for name, rows_source in (("sft_openai_full_v1.yaml", "mix_v1"),
                              ("sft_openai_smoke_v1.yaml", "original_v1")):
        text = (ROOT / "configs/response" / name).read_text()
        cfg = yaml.safe_load(text)
        oa = cfg["openai"]
        assert oa["base_model"] in supported
        assert 1 <= len(oa["suffix"]) <= 64
        assert oa["hyperparameters"]["n_epochs"] <= 3  # PRD 13.5
        assert rows_source in cfg["data"]["train"]
        assert oa["pricing"]["training_usd_per_mtok"] > 0 and oa["run_dir"].endswith(
            ("/full", "/smoke"))
        assert "sk-" not in text
    evaluation = bench.load_config(ROOT / "configs/response/sft_openai_eval_v1.yaml")
    assert not any("probability_cases" in spec["path"] for spec in evaluation["case_sets"])
    full_run_dir = yaml.safe_load(
        (ROOT / "configs/response/sft_openai_full_v1.yaml").read_text())["openai"]["run_dir"]
    assert evaluation["run_dir"] == full_run_dir  # one ledger for training and evaluation
    for entry in evaluation["models"]:
        if entry.get("fine_tuned"):
            assert entry["decoding"]["token_param"] == "max_completion_tokens"
            assert "reasoning_effort" not in (entry["decoding"].get("extra") or {})
    test = bench.load_config(ROOT / "configs/response/sft_openai_test_v1.yaml")
    assert [spec["path"] for spec in test["case_sets"]] == [
        "evaluation/response/probability_cases_v1.jsonl"]
    assert [entry["fine_tuned"]["key"] for entry in test["models"]] == ["selected"]
    assert test["min_subsample_cases"] == 55
