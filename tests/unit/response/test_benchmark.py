"""Offline tests for the response benchmark harness, deployment lifecycle and report."""

import copy
import io
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

from tools.response import benchmark as bench
from tools.response.benchmark_report import build_report
from tools.response.fireworks_deployment import DeploymentError, DeploymentRun

ROOT = Path(__file__).parents[3]
PLAN_DIR = Path(__file__).parent / "fixtures" / "plans"
SECRET = "sk-test-secret-value-1234567890"


def plans():
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PLAN_DIR.glob("*.json"))
    }


class FakeHTTPResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeChat:
    """OpenAI-compatible chat endpoint returning fixed text per request."""

    def __init__(self, text="Here is a hint: List the two outcomes. What is your next step?",
                 usage=None, fail_after=None):
        self.text = text
        self.usage = usage or {
            "prompt_tokens": 100, "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 10},
        }
        self.requests = []
        self.fail_after = fail_after

    def __call__(self, request, timeout=None):
        self.requests.append((request, timeout))
        if self.fail_after is not None and len(self.requests) > self.fail_after:
            raise RuntimeError("boom")
        if request.full_url.endswith("/models"):
            return FakeHTTPResponse({"data": []})
        return FakeHTTPResponse({
            "model": "fake", "choices": [{"message": {"content": self.text},
                                          "finish_reason": "stop"}],
            "usage": self.usage,
        })


def make_config(tmp_path, cases_dir, **entry_overrides):
    entry = {
        "id": "paid-model", "label": "Paid", "provider": "openai_compatible",
        "base_url": "https://example.invalid/v1", "model": "fake-model",
        "api_key_env": "FAKE_API_KEY", "concurrency": 2,
        "decoding": {"temperature": None, "max_tokens": 512,
                     "token_param": "max_completion_tokens",
                     "extra": {"reasoning_effort": "low"}},
        "pricing": {"basis": "tokens", "input_usd_per_mtok": 2.0,
                    "cached_input_usd_per_mtok": 0.1, "output_usd_per_mtok": 10.0},
    }
    entry.update(entry_overrides)
    return {
        "benchmark_id": "test", "run_dir": str(tmp_path / "runs"),
        "report_dir": str(tmp_path / "report"), "keys_file": str(tmp_path / "llm.env"),
        "spend_cap_env": "TUTURING_BENCH_SPEND_CAP_USD", "min_subsample_cases": 2,
        "estimate_safety_factor": 1.0,
        "case_sets": [{"name": "fixtures", "format": "plan_dir", "path": str(cases_dir)}],
        "review": {"cases": 3, "seed": 1},
        "models": [{"id": "fallback", "label": "Fallback", "provider": "fallback"}, entry],
    }


def write_keys(tmp_path, cap="5"):
    lines = [f"FAKE_API_KEY={SECRET}"]
    if cap is not None:
        lines.append(f"TUTURING_BENCH_SPEND_CAP_USD={cap}")
    (tmp_path / "llm.env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_ctx(config, chat=None, keys=None):
    keys = keys if keys is not None else bench.load_keys(config["keys_file"])
    run_dir = Path(config["run_dir"])
    return bench.RunContext(
        config=config, keys=keys, cap=bench.spend_cap(keys, config),
        ledger=bench.Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments"),
        run_dir=run_dir, urlopen=chat, sleep=lambda _s: None, out=lambda *_a: None,
    )


def test_load_keys_parses_lines_and_redact_hides_values(tmp_path):
    path = tmp_path / "llm.env"
    path.write_text(
        "# comment\nexport A_KEY='abc-def-ghi-jkl'\nB_KEY=\"x\"\nTUTURING_BENCH_SPEND_CAP_USD=25\n"
        "broken line\n", encoding="utf-8")
    keys = bench.load_keys(path)
    assert keys == {"A_KEY": "abc-def-ghi-jkl", "B_KEY": "x", "TUTURING_BENCH_SPEND_CAP_USD": "25"}
    assert bench.spend_cap(keys, {}) == 25.0
    assert bench.redact("token abc-def-ghi-jkl here", keys) == "token <redacted> here"
    assert bench.load_keys(tmp_path / "missing.env") == {}


def test_readiness_skips_missing_key_and_missing_cap(tmp_path):
    config = make_config(tmp_path, PLAN_DIR)
    entry = config["models"][1]
    ctx = make_ctx(config, keys={})
    ok, reason = bench.readiness(entry, ctx)
    assert not ok and "FAKE_API_KEY is not set" in reason
    ctx = make_ctx(config, keys={"FAKE_API_KEY": SECRET})
    ok, reason = bench.readiness(entry, ctx)
    assert not ok and "no TUTURING_BENCH_SPEND_CAP_USD" in reason
    write_keys(tmp_path)
    ctx = make_ctx(config)
    assert bench.readiness(entry, ctx) == (True, "ready")
    assert bench.readiness(config["models"][0], ctx)[0]


def test_run_one_sends_expected_request_and_records_cost_without_secrets(tmp_path):
    config = make_config(tmp_path, PLAN_DIR)
    write_keys(tmp_path)
    chat = FakeChat()
    ctx = make_ctx(config, chat)
    entry = config["models"][1]
    case = bench.Case("fixtures", "hint", plans()["hint"], "test")
    record = bench.run_one(entry, case, ctx, model="fake-model", tag="pilot")
    request, timeout = chat.requests[0]
    body = json.loads(request.data)
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert body["max_completion_tokens"] == 512 and "max_tokens" not in body
    assert body["reasoning_effort"] == "low" and "temperature" not in body
    assert body["messages"] == bench.compile_prompt(case.plan)
    assert record["checks"]["action_faithful"] and record["delivery"] == "model"
    assert record["usage"] == {
        "input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 40,
        "reasoning_tokens": 10, "visible_output_tokens": 30,
    }
    assert record["cost_usd"] == pytest.approx((100 * 2.0 + 40 * 10.0) / 1e6)
    assert SECRET not in json.dumps(record)


def test_execute_refuses_paid_call_that_would_pass_the_cap(tmp_path):
    config = make_config(tmp_path, PLAN_DIR)
    write_keys(tmp_path, cap="0.000001")
    chat = FakeChat()
    ctx = make_ctx(config, chat)
    entry = config["models"][1]
    cases = [bench.Case("fixtures", "hint", plans()["hint"], "test")]
    records, stopped = bench.execute(entry, cases, ctx, model="fake-model", tag="full")
    assert records == [] and stopped and "would pass the cap" in stopped[0]
    assert chat.requests == []
    with pytest.raises(bench.BenchmarkError, match="without a spend cap"):
        bench.execute(entry, cases, make_ctx(config, chat, keys={"FAKE_API_KEY": SECRET}),
                      model="fake-model", tag="full")


def test_run_entry_requires_pilot_then_subsamples_to_budget_and_resumes(tmp_path):
    cap = 0.004
    config = make_config(tmp_path, PLAN_DIR, concurrency=1,
                         decoding={"temperature": 0, "max_tokens": 64})
    write_keys(tmp_path, cap=str(cap))
    chat = FakeChat()
    ctx = make_ctx(config, chat)
    entry = config["models"][1]
    cases, _ = bench.load_all_cases(config)
    assert len(cases) == 8
    with pytest.raises(bench.BenchmarkError, match="run a pilot first"):
        bench.run_entry(entry, cases, ctx)
    pilot = bench.run_entry(entry, cases, ctx, pilot=2)
    assert len(pilot) == 2 and len(chat.requests) == 2
    per_case = (100 * 2.0 + 40 * 10.0) / 1e6
    assert ctx.ledger.spent() == pytest.approx(2 * per_case)
    full = bench.run_entry(entry, cases, ctx)
    estimate = [e for e in ctx.ledger.events() if e["event"] == "estimate"][-1]
    expected_selection = int((cap - 2 * per_case) // per_case)
    assert estimate["pending_cases"] == 6 and estimate["selected_cases"] == expected_selection
    assert 1 <= len(full) <= expected_selection, "worst-case reserve may stop earlier"
    assert ctx.ledger.spent() <= cap
    done_before = len(bench.load_results(ctx.run_dir, entry["id"]))
    assert done_before == 2 + len(full)
    # The remaining budget fits fewer than min_subsample_cases, so a resume must refuse
    # before any call rather than spend the last cents on an unrepresentative sample.
    with pytest.raises(bench.BenchmarkError, match="minimal subsample"):
        bench.run_entry(entry, cases, ctx, skip_pilot=True)
    done_after = bench.load_results(ctx.run_dir, entry["id"])
    assert len(done_after) == done_before
    assert ctx.ledger.spent() <= cap and len(chat.requests) == len(done_after)
    assert all(r["prompt_sha256"] for r in done_after.values())
    assert len(bench.pending_cases(entry, cases, ctx)) == 8 - len(done_after)
    stale = copy.deepcopy(entry)
    stale["decoding"] = {**entry["decoding"], "max_tokens": 1}
    assert len(bench.pending_cases(stale, cases, ctx)) == 8, "changed decoding invalidates"


def test_fit_to_budget_and_stratified_order():
    cases = [bench.Case("s", f"c{i}", {"teaching_action": {"move": m}}, "t")
             for i, m in enumerate(["HINT", "PROBE"] * 5)]
    assert bench.fit_to_budget(cases, 0.0, 1.0, min_cases=2) == cases
    assert len(bench.fit_to_budget(cases, 1.0, 4.5, min_cases=2)) == 4
    assert bench.fit_to_budget(cases, 1.0, 1.5, min_cases=2) is None
    ordered = bench.stratified_order(cases)
    assert [c.move for c in ordered[:4]] == ["HINT", "PROBE", "HINT", "PROBE"]
    assert {c.case_id for c in ordered} == {c.case_id for c in cases}


def test_usage_thinking_and_diagnostic_helpers():
    assert bench.normalize_usage({"prompt_tokens": 5, "completion_tokens": 7})[
        "visible_output_tokens"] == 7
    assert bench.strip_thinking("<think>secret</think>Answer") == ("Answer", True)
    assert bench.strip_thinking("plain") == ("plain", False)
    assert bench.delivery_for([]) == "model"
    assert bench.delivery_for(["length_valid"]) == "repair"
    assert bench.delivery_for(["math_supported"]) == "fallback"
    assert bench.delivery_for([], error=True) == "error"
    plan = copy.deepcopy(plans()["probe"])
    plan["response_context"]["last_student_message"] = "I think it is 37 out of 99"
    plan["constraints"]["forbidden_answer_fingerprints"] = ["37"]
    unsupported, learner = bench.number_diagnostics(plan, "You said 37 and 99, but what about 41?")
    assert unsupported == ["37", "99", "41"] and learner == ["37", "99"]
    disclosed, in_context = bench.disclosure_diagnostics(plan, "Is 37 right?")
    assert disclosed == ["37"] and in_context is True
    assert bench.disclosure_diagnostics(plan, "Think again.") == ([], None)
    bare = copy.deepcopy(plans()["hint"])
    bare["content"]["permitted_hint"] = None
    assert not bench.action_check_applicable(bare)
    assert bench.action_check_applicable(plans()["hint"])


def test_call_model_retries_scaling_up_and_reports_errors_without_key(monkeypatch):
    attempts = []

    def flaky(request, timeout=None):
        attempts.append(request)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                request.full_url, 503, "scaling", {},
                io.BytesIO(json.dumps({"error": {"code": "DEPLOYMENT_SCALING_UP"}}).encode()),
            )
        return FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    entry = {"base_url": "https://x.invalid/v1", "api_key_env": "K",
             "retry": {"max_attempts": 1, "scaling_attempts": 5, "base_delay_s": 0},
             "decoding": {"temperature": 0, "max_tokens": 10}}
    result = bench.call_model(entry, "m", [{"role": "user", "content": "hi"}], {"K": SECRET},
                              urlopen=flaky, sleep=lambda _s: None)
    assert result["text"] == "ok" and result["attempts"] == 3

    def forbidden(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {},
                                     io.BytesIO(f"bad key {SECRET}".encode()))

    with pytest.raises(bench.CallError) as excinfo:
        bench.call_model(entry, "m", [], {"K": SECRET}, urlopen=forbidden, sleep=lambda _s: None)
    assert SECRET not in str(excinfo.value) and excinfo.value.status == 401


class FakeFireworks:
    """Control-plane double: create -> READY after two polls -> delete -> DELETED."""

    def __init__(self, create_status=200, polls_until_ready=2):
        self.calls = []
        self.create_status = create_status
        self.polls = 0
        self.polls_until_ready = polls_until_ready
        self.deleted = False
        self.name = None

    def account_id(self):
        return "acct"

    def request(self, method, path, body=None, query=None):
        self.calls.append((method, path, body, query))
        if path == "/v1/accounts":
            return 200, {"accounts": [{"name": "accounts/acct"}]}
        if method == "POST" and path.endswith("/deployments"):
            if self.create_status != 200:
                return self.create_status, {"code": 9, "message": "payment method is required"}
            self.name = f"accounts/acct/deployments/{query['deploymentId']}"
            return 200, {"name": self.name, "state": "CREATING", **body}
        if method == "DELETE":
            self.deleted = True
            return 200, {}
        if method == "GET" and self.name and path == f"/v1/{self.name}":
            if self.deleted:
                return 404, {"code": 5, "message": "not found"}
            self.polls += 1
            state = "READY" if self.polls >= self.polls_until_ready else "CREATING"
            return 200, {"name": self.name, "state": state, "acceleratorType": "NVIDIA_H100_80GB"}
        if method == "GET" and path.endswith("/deployments"):
            return 200, {"deployments": []}
        raise AssertionError(f"unexpected call {method} {path}")


def deployment_entry():
    return {
        "id": "qwen-dep", "provider": "openai_compatible", "base_url": "https://fw.invalid/v1",
        "model": "{deployment}", "api_key_env": "FW", "concurrency": 1,
        "decoding": {"temperature": 0, "max_tokens": 64}, "pricing": {"basis": "gpu_time"},
        "deployment": {
            "base_model": "accounts/fireworks/models/qwen3p8-27b",
            "deployment_shape": "accounts/fireworks/deploymentShapes/sva2n3mh",
            "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 1,
            "usd_per_gpu_hour": 8.0, "max_minutes": 30, "delete_buffer_s": 0,
            "ready_timeout_s": 100, "warmup_timeout_s": 100, "scale_to_zero_window": "300s",
        },
    }


def test_deployment_lifecycle_records_state_cost_and_verified_deletion(tmp_path):
    clock = {"now": 1000.0}
    ledger = bench.Ledger(tmp_path / "ledger.jsonl", clock=lambda: clock["now"])
    api = FakeFireworks()
    run = DeploymentRun(api, deployment_entry(), tmp_path / "dep.json", ledger,
                        out=lambda *_a: None, sleep=lambda s: clock.update(now=clock["now"] + s),
                        clock=lambda: clock["now"])
    run.create()
    create_call = next(c for c in api.calls if c[0] == "POST")
    assert create_call[2]["deploymentShape"].endswith("sva2n3mh")
    assert create_call[2]["minReplicaCount"] == 0 and create_call[2]["maxReplicaCount"] == 1
    assert create_call[2]["autoscalingPolicy"]["scaleToZeroWindow"] == "300s"
    assert ledger.open_deployment_cost() == 0.0  # ledger has no deployments dir here
    run.wait_ready(timeout_s=100, poll_s=10)
    assert run.state["state"] == "READY"
    clock["now"] += 900  # 15 minutes of use
    state = run.delete_and_verify(attempts=3, poll_s=1)
    assert state["deleted_verified"] is True and state["verification_get"]["http"] == 404
    assert state["estimated_gpu_usd"] == pytest.approx((900 + 10) / 3600 * 8.0, abs=0.01)
    events = ledger.events()
    assert [e["event"] for e in events] == ["deployment_create", "gpu"]
    assert ledger.spent() == pytest.approx(state["estimated_gpu_usd"])
    saved = json.loads((tmp_path / "dep.json").read_text())
    assert saved["name"] == api.name and saved["deleted_verified"]
    with pytest.raises(DeploymentError, match="payment method"):
        DeploymentRun(FakeFireworks(create_status=400), deployment_entry(),
                      tmp_path / "dep2.json", ledger, out=lambda *_a: None).create()
    failed = json.loads((tmp_path / "dep2.json").read_text())
    assert failed["deleted_verified"] is True and failed["state"] == "NOT_CREATED"


def test_run_with_deployment_always_deletes_even_when_a_call_crashes(tmp_path, monkeypatch):
    import tools.response.fireworks_deployment as fwd

    config = make_config(tmp_path, PLAN_DIR)
    config["models"][1] = deployment_entry()
    (tmp_path / "llm.env").write_text(
        f"FW={SECRET}\nTUTURING_BENCH_SPEND_CAP_USD=25\n", encoding="utf-8")
    api = FakeFireworks(polls_until_ready=1)
    monkeypatch.setattr(fwd, "FireworksAPI", lambda *_a, **_k: api)
    chat = FakeChat(text="Here is a hint: List the two outcomes. What next?", fail_after=2)
    ctx = make_ctx(config, chat)
    cases, _ = bench.load_all_cases(config)
    with pytest.raises(RuntimeError, match="boom"):
        bench.run_with_deployment(config["models"][1], cases, ctx, pilot=1)
    assert api.deleted
    state = json.loads((ctx.run_dir / "deployments" / "qwen-dep.json").read_text())
    assert state["deleted_verified"] is True
    assert any(e["event"] == "gpu" for e in ctx.ledger.events())
    body = json.loads(chat.requests[1][0].data)
    assert body["model"] == api.name
    assert bench.readiness(config["models"][1], ctx)[1].startswith("skipped: no active")


def test_report_writes_tables_and_blinded_review(tmp_path):
    config = make_config(tmp_path, PLAN_DIR)
    write_keys(tmp_path)
    chat = FakeChat(text="A fair coin is flipped once. What is the probability of heads? Try it.")
    ctx = make_ctx(config, chat)
    cases, _ = bench.load_all_cases(config)
    bench.run_entry(config["models"][0], cases, ctx)
    bench.run_entry(config["models"][1], cases, ctx, pilot=8)
    written = build_report(config, ctx)
    report_dir = Path(config["report_dir"])
    names = {path.name for path in written}
    assert {"results_by_model.csv", "results_by_move.csv", "results_by_case_set.csv",
            "results_by_case.csv", "report.md", "review_sheet_blinded.csv"} <= names
    rows = list(__import__("csv").DictReader(
        (report_dir / "results_by_model.csv").open(encoding="utf-8")))
    assert {row["model_id"] for row in rows} == {"fallback", "paid-model"}
    fallback_row = next(row for row in rows if row["model_id"] == "fallback")
    assert fallback_row["n"] == "8" and fallback_row["all_applicable_pass"] == "8"
    sheet = (report_dir / "review" / "review_sheet_blinded.csv").read_text(encoding="utf-8")
    assert "paid-model" not in sheet and "Fallback" not in sheet and "R1" in sheet
    key = (report_dir / "review" / "review_key_do_not_open_before_rating.csv").read_text()
    assert "paid-model" in key and "fallback" in key
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "## Limitations" in report and "| fixtures | 8 |" in report
    assert SECRET not in report and SECRET not in key


def test_committed_config_is_consistent():
    config = yaml.safe_load((ROOT / "configs/response/benchmark_v1.yaml").read_text())
    ids = [entry["id"] for entry in config["models"]]
    assert len(ids) == len(set(ids))
    for entry in config["models"]:
        if entry["provider"] == "fallback":
            continue
        basis = bench.cost_basis(entry)
        assert basis in {"tokens", "gpu_time", "free"}, entry["id"]
        if basis != "free":
            assert entry.get("api_key_env"), entry["id"]
            assert not entry["api_key_env"].startswith("sk-")
        if basis == "tokens":
            assert entry["pricing"]["input_usd_per_mtok"] > 0
            assert entry["pricing"].get("source"), entry["id"]
        if basis == "gpu_time":
            assert entry["deployment"]["usd_per_gpu_hour"] > 0
            assert entry["deployment"]["min_replica_count"] == 0
        assert bench.max_output_tokens(entry) >= 384
    text = (ROOT / "configs/response/benchmark_v1.yaml").read_text()
    assert "sk-" not in text and "fw_" not in text
