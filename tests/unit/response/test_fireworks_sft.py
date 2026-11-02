"""Offline tests for the Fireworks SFT tooling, LoRA addon loading and adapter evaluation."""

import json
import math
import re
from pathlib import Path

import pytest
import yaml

from packages.response.prompt import compile_prompt
from tools.response import benchmark as bench
from tools.response import fireworks_sft as fsft
from tools.response.fireworks_deployment import DeploymentError, DeploymentRun
from tools.response.train_sft import check_config

ROOT = Path(__file__).parents[3]
PLAN_DIR = Path(__file__).parent / "fixtures" / "plans"
SECRET = "fw-test-secret-value-1234567890"


def plan(name="practice"):
    return json.loads((PLAN_DIR / f"{name}.json").read_text(encoding="utf-8"))


def example(name="practice", reply="Here is your problem. Give it a try.", **overrides):
    record = {
        "example_id": f"t:{name}",
        "plan": plan(name),
        "prompt": compile_prompt(plan(name)),
        "completion": [{"role": "assistant", "content": reply}],
    }
    record.update(overrides)
    return record


class TinyTokenizer:
    """Chat-template double: one token per character, ``<|im_end|>`` is one token."""

    eos_token = "<|im_end|>"
    eos_token_id = 999
    im_end_token_id = 999

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize,
                            enable_thinking=None):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
        ids = []
        for part in re.split(r"(<\|im_end\|>)", text):
            if part == "<|im_end|>":
                ids.append(self.im_end_token_id)
            elif part:
                ids.extend(ord(char) for char in part)
        return {"input_ids": ids}


# ---------------------------------------------------------------------------
# Conversion, tokens and cost


def test_record_trains_only_the_reply_and_matches_the_compiled_prompt():
    record = fsft.to_fireworks_record(example())
    messages = record["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert [m["weight"] for m in messages] == [0, 0, 1]
    compiled = compile_prompt(plan())
    assert messages[0]["content"] == compiled[0]["content"]
    assert messages[1]["content"] == compiled[1]["content"]
    assert messages[2]["content"] == "Here is your problem. Give it a try."
    assert set(record) == {"messages"}


@pytest.mark.parametrize("change, message", [
    ({"prompt": compile_prompt(plan())[:1]}, "prompt roles"),
    ({"prompt": [{"role": "system", "content": "other"}, compile_prompt(plan())[1]]},
     "differs from compile_prompt"),
    ({"completion": []}, "one assistant message"),
    ({"completion": [{"role": "assistant", "content": "  "}]}, "empty tutor reply"),
    ({"completion": [{"role": "assistant", "content": "<think>x</think> Try it."}]},
     "thinking marker"),
])
def test_record_rejects_inconsistent_examples(change, message):
    with pytest.raises(ValueError, match=message):
        fsft.to_fireworks_record(example(**change))


def test_convert_and_export_write_hashable_jsonl(tmp_path):
    source = tmp_path / "validation.jsonl"
    rows, digest = fsft.write_jsonl(source, [example("practice"), example("hint")])
    assert rows == 2
    converted = fsft.convert_split(source, tmp_path / "out" / "validation.jsonl")
    assert converted["rows"] == 2 and converted["source_sha256"] == digest
    lines = (tmp_path / "out" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["messages"][2]["weight"] == 1
    exported = fsft.export_cases(source, tmp_path / "cases.jsonl")
    assert exported["sha256"] == digest  # byte-identical copy of the split
    cases, _ = bench.load_case_set({"name": "v", "path": str(tmp_path / "cases.jsonl")})
    assert [case.case_id for case in cases] == ["t:practice", "t:hint"]


def test_token_stats_steps_and_cost_estimate():
    stats = fsft.token_stats(TinyTokenizer(), [example(), example("hint", reply="Try it.")], 10)
    assert stats["examples"] == 2
    assert stats["total_tokens"] == stats["prompt_tokens"] + stats["target_tokens"]
    # Reply characters plus the end-of-turn token are the trained tokens.
    assert stats["target_tokens"] == len("Here is your problem. Give it a try.") + 1 + 7 + 1
    assert stats["over_max_sequence_tokens"] == 2
    assert fsft.optimizer_steps(261, 16, 3) == {
        "per_epoch": 17, "total": 51, "per_epoch_full_batches_only": 16}
    estimate = fsft.estimate_training_cost(
        {"total_tokens": 100_000}, {"total_tokens": 10_000}, {"epochs": 3},
        {"usd_per_mtok": 3.0, "estimate_safety_factor": 1.5})
    assert estimate["expected_usd"] == pytest.approx(0.9)
    assert estimate["upper_bound_usd"] == pytest.approx((300_000 + 40_000) * 3 / 1e6)
    assert estimate["reserve_usd"] == pytest.approx(estimate["upper_bound_usd"] * 1.5, abs=1e-4)


def test_multipart_and_money_helpers():
    body, content_type = fsft.multipart_file("file", "train.jsonl", b'{"a": 1}\n', "XYZ")
    assert content_type == "multipart/form-data; boundary=XYZ"
    assert body.startswith(b"--XYZ\r\nContent-Disposition: form-data; name=\"file\"; "
                           b"filename=\"train.jsonl\"\r\n")
    assert body.endswith(b'{"a": 1}\n\r\n--XYZ--\r\n')
    assert fsft.money_usd({"currencyCode": "USD", "units": "1", "nanos": 250000000}) == 1.25
    assert fsft.money_usd({}) is None and fsft.money_usd(None) is None


def test_config_cap_can_only_lower_the_keys_file_cap():
    keys = {"TUTURING_BENCH_SPEND_CAP_USD": "25"}
    assert bench.spend_cap(keys, {"spend_cap_usd": 20}) == 20
    assert bench.spend_cap(keys, {"spend_cap_usd": 60}) == 25
    assert bench.spend_cap({}, {"spend_cap_usd": 20}) is None


# ---------------------------------------------------------------------------
# Datasets and the job, against a control-plane double


class FakeControlPlane:
    """Datasets, one SFT job, deployments and deployed models."""

    def __init__(self, upload_status=200, job_states=("JOB_STATE_RUNNING",
                                                      "JOB_STATE_COMPLETED"),
                 estimated_usd=0.5):
        self.calls = []
        self.datasets = {}
        self.upload_status = upload_status
        self.job = None
        self.job_states = list(job_states)
        self.estimated_usd = estimated_usd
        self.deployment = None
        self.deployed_model = None
        self.deployed_polls = 0
        self.deleted = False

    def account_id(self):
        return "acct"

    def request(self, method, path, body=None, query=None, *, data=None,
                content_type="application/json"):
        self.calls.append((method, path, body, query, content_type))
        if method == "GET" and "/datasets/" in path:
            name = path.removeprefix("/v1/")
            if name not in self.datasets:
                return 404, {"code": 5}
            return 200, self.datasets[name]
        if method == "POST" and path == "/v1/accounts/acct/datasets":
            name = f"accounts/acct/datasets/{body['datasetId']}"
            self.datasets[name] = {"name": name, "state": "UPLOADING",
                                   "exampleCount": body["dataset"]["exampleCount"]}
            return 200, self.datasets[name]
        if method == "POST" and path.endswith(":upload"):
            assert content_type.startswith("multipart/form-data; boundary=")
            if self.upload_status != 200:
                return self.upload_status, {"raw": "dataset uploads are temporarily disabled"}
            name = path.removeprefix("/v1/").removesuffix(":upload")
            self.datasets[name].update(state="READY", estimatedTokenCount="1234")
            return 200, {"id": name}
        if method == "DELETE" and "/datasets/" in path:
            self.datasets.pop(path.removeprefix("/v1/"), None)
            return 200, {}
        if method == "POST" and path.endswith("/supervisedFineTuningJobs"):
            name = f"accounts/acct/supervisedFineTuningJobs/{query['supervisedFineTuningJobId']}"
            self.job = {"name": name, "state": "JOB_STATE_CREATING", **body}
            return 200, dict(self.job)
        if self.job and path == f"/v1/{self.job['name']}":
            if method == "DELETE":
                self.job["state"] = "JOB_STATE_CANCELLED"
                return 200, {}
            if self.job_states and self.job["state"] not in fsft.TERMINAL_STATES:
                self.job["state"] = self.job_states.pop(0)
            units = int(self.estimated_usd)
            self.job["estimatedCost"] = {
                "currencyCode": "USD", "units": str(units),
                "nanos": round((self.estimated_usd - units) * 1e9)}
            self.job["metricsFileSignedUrl"] = "https://storage.invalid/metrics?sig=1"
            return 200, dict(self.job)
        if method == "POST" and path.endswith("/deployments"):
            self.deployment = f"accounts/acct/deployments/{query['deploymentId']}"
            return 200, {"name": self.deployment, "state": "CREATING", **body}
        if method == "POST" and path.endswith("/deployedModels"):
            assert query == {"replaceMergedAddon": "true"}
            self.deployed_model = "accounts/acct/deployedModels/dm-1"
            return 200, {"name": self.deployed_model, "state": "DEPLOYING", **body}
        if self.deployed_model and path == f"/v1/{self.deployed_model}":
            if self.deleted:
                return 404, {"code": 5}
            self.deployed_polls += 1
            state = "DEPLOYED" if self.deployed_polls >= 2 else "DEPLOYING"
            return 200, {"name": self.deployed_model, "state": state}
        if method == "DELETE" and self.deployment and path == f"/v1/{self.deployment}":
            self.deleted = True
            return 200, {}
        if method == "GET" and self.deployment and path == f"/v1/{self.deployment}":
            if self.deleted:
                return 404, {"code": 5}
            return 200, {"name": self.deployment, "state": "READY"}
        if method == "GET" and path.endswith("/deployments"):
            return 200, {"deployments": []}
        raise AssertionError(f"unexpected call {method} {path}")


def sft_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/response/sft_smoke_qwen38_fireworks.yaml").read_text())
    cfg["fireworks"]["run_dir"] = str(tmp_path / "runs")
    return cfg


def make_run(tmp_path, api, cap=20.0, urlopen=None):
    run_dir = tmp_path / "runs"
    ledger = bench.Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments")
    clock = {"now": 1000.0}
    return fsft.SFTRun(api, sft_config(tmp_path), ledger, cap, out=lambda *_a: None,
                       sleep=lambda s: clock.update(now=clock["now"] + s),
                       clock=lambda: clock["now"], urlopen=urlopen)


def write_dataset(tmp_path, name, rows=3):
    path = tmp_path / f"{name}.jsonl"
    fsft.write_jsonl(path, [fsft.to_fireworks_record(example()) for _ in range(rows)])
    return path


ESTIMATE = {"expected_usd": 0.8, "upper_bound_usd": 1.0, "reserve_usd": 1.5}


def test_upload_creates_datasets_and_job_body_uses_the_prd_settings(tmp_path):
    api = FakeControlPlane()
    run = make_run(tmp_path, api)
    record = run.ensure_dataset("train", write_dataset(tmp_path, "train"))
    assert record["state"] == "READY" and record["rows"] == 3 and record["upload_http"] == 200
    run.ensure_dataset("validation", write_dataset(tmp_path, "validation", rows=2))
    again = run.ensure_dataset("train", write_dataset(tmp_path, "train"))  # idempotent
    assert again["name"] == "accounts/acct/datasets/tut-sft-smoke-v1-train"
    assert sum(1 for c in api.calls if c[1].endswith(":upload")) == 2
    body = run.job_body()
    assert body["baseModel"] == "accounts/fireworks/models/qwen3p8-27b"
    assert body["evaluationDataset"].endswith("tut-sft-smoke-v1-validation")
    assert (body["loraRank"], body["loraAlpha"], body["learningRate"]) == (16, 32, 0.0001)
    assert (body["epochs"], body["batchSizeSamples"], body["maxContextLength"]) == (3, 16, 4096)
    assert body["evalAutoCarveout"] is False and "gradientAccumulationSteps" not in body


def test_refused_upload_deletes_the_empty_dataset(tmp_path):
    api = FakeControlPlane(upload_status=403)
    run = make_run(tmp_path, api)
    with pytest.raises(fsft.SFTError, match="temporarily disabled"):
        run.ensure_dataset("train", write_dataset(tmp_path, "train"))
    assert api.datasets == {}
    recorded = run.state["datasets"]["train"]
    assert recorded["deleted_after_failed_upload"]["http"] == 200
    assert recorded["upload_http"] is None


def prepared_run(tmp_path, api, cap=20.0, urlopen=None):
    run = make_run(tmp_path, api, cap=cap, urlopen=urlopen)
    run.ensure_dataset("train", write_dataset(tmp_path, "train"))
    run.ensure_dataset("validation", write_dataset(tmp_path, "validation"))
    return run


def test_job_dry_run_budget_refusal_create_poll_and_settle(tmp_path):
    api = FakeControlPlane()
    run = prepared_run(tmp_path, api)
    dry = run.create_job(ESTIMATE)
    assert dry["dry_run"] and api.job is None and run.ledger.events() == []
    with pytest.raises(fsft.SFTError, match="passes the cap"):
        prepared_run(tmp_path / "small", FakeControlPlane(), cap=1.0).create_job(
            ESTIMATE, confirm=True)
    created = run.create_job(ESTIMATE, confirm=True)
    assert created["name"].endswith("/tut-sft-smoke-v1")
    assert run.ledger.committed() == pytest.approx(1.5)  # reserve held while it runs
    final = run.poll(timeout_s=100, poll_s=1)
    assert final["state"] == "JOB_STATE_COMPLETED"
    states = [t["state"] for t in run.state["job"]["transitions"]]
    assert states == ["JOB_STATE_RUNNING", "JOB_STATE_COMPLETED"]
    settled = run.state["job"]["settled"]
    assert settled["final_usd"] == pytest.approx(0.8)  # max(Fireworks 0.50, tokens 0.80)
    assert run.ledger.spent() == pytest.approx(0.8)
    assert [e["event"] for e in run.ledger.events()] == [
        "estimate", "sft_job_reserve", "sft_job_settle"]


def test_poll_cancels_a_job_whose_estimate_passes_the_reserve(tmp_path):
    api = FakeControlPlane(job_states=["JOB_STATE_RUNNING"] * 3, estimated_usd=0.0)
    run = prepared_run(tmp_path, api)
    run.create_job(ESTIMATE, confirm=True)
    api.estimated_usd = 2.25  # above the $1.50 reserve
    with pytest.raises(fsft.SFTError, match="cancelled"):
        run.poll(timeout_s=100, poll_s=1)
    assert run.state["job"]["cancel"]["http"] == 200


def test_signed_artifacts_are_fetched_without_the_api_key(tmp_path):
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def read(self):
            return b'{"step": 1, "loss": 1.5}\n'

    def urlopen(request, timeout=None):
        seen.append(request)
        return Response()

    api = FakeControlPlane()
    run = prepared_run(tmp_path, api, urlopen=urlopen)
    run.create_job(ESTIMATE, confirm=True)
    _, payload = run.get_job()
    fetched = run.fetch_artifacts(payload)
    assert fetched["metricsFileSignedUrl"]["bytes"] == 25
    assert fetched["renderSamplesSignedUrl"] is None
    assert len(seen) == 1 and seen[0].get_header("Authorization") is None
    assert (tmp_path / "runs" / "artifacts" / "metrics").read_bytes().startswith(b'{"step"')


# ---------------------------------------------------------------------------
# Hot-reload deployment and adapter evaluation


def deployment_entry():
    return {
        "id": "base", "provider": "openai_compatible", "base_url": "https://fw.invalid/v1",
        "model": "{deployment}", "api_key_env": "FW", "concurrency": 1,
        "decoding": {"temperature": 0, "max_tokens": 64}, "pricing": {"basis": "gpu_time"},
        "deployment": {
            "base_model": "accounts/fireworks/models/qwen3p8-27b",
            "deployment_shape": "accounts/fireworks/deploymentShapes/sva2n3mh",
            "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 1,
            "usd_per_gpu_hour": 8.0, "max_minutes": 30, "delete_buffer_s": 0,
            "ready_timeout_s": 100, "warmup_timeout_s": 100,
            "enable_hot_reload_latest_addon": True, "purpose": "sft-eval",
        },
    }


def adapter_entry():
    return {
        "id": "sft", "provider": "openai_compatible", "base_url": "https://fw.invalid/v1",
        "model": "{adapter}#{deployment}", "api_key_env": "FW", "concurrency": 1,
        "decoding": {"temperature": 0, "max_tokens": 64}, "pricing": {"basis": "gpu_time"},
        "adapter": {"model": "accounts/acct/models/lora", "deployment_entry": "base",
                    "model_candidates": ["{adapter}#{deployment}", "{deployment}"],
                    "warmup_rounds": 2, "warmup_poll_s": 0, "load_poll_s": 0},
    }


def test_hot_reload_body_and_addon_load(tmp_path):
    ledger = bench.Ledger(tmp_path / "ledger.jsonl")
    api = FakeControlPlane()
    run = DeploymentRun(api, deployment_entry(), tmp_path / "dep.json", ledger,
                        out=lambda *_a: None, sleep=lambda _s: None)
    body = run.create_body()
    assert body["enableHotReloadLatestAddon"] is True and "enableAddons" not in body
    assert body["annotations"]["custom/purpose"] == "sft-eval"
    run.create()
    loaded = run.load_addon("accounts/acct/models/lora", timeout_s=100, poll_s=1)
    assert loaded["state"] == "DEPLOYED"
    run.annotate_addon(served_as="x#y")
    addon = run.state["addons"][0]
    assert addon["deployed_model"] == "accounts/acct/deployedModels/dm-1"
    assert addon["served_as"] == "x#y"
    state = run.delete_and_verify(attempts=2, poll_s=1)
    assert state["deleted_verified"] and state["addons"][0]["after_delete"]["http"] == 404


class RoutingChat:
    """Chat endpoint whose reply depends on the routed model string."""

    def __init__(self, adapter_text, fail_adapter_routes=()):
        self.adapter_text = adapter_text
        self.fail = set(fail_adapter_routes)
        self.models = []

    def __call__(self, request, timeout=None):
        import urllib.error

        body = json.loads(request.data)
        self.models.append(body["model"])
        if body["model"] in self.fail:
            raise urllib.error.HTTPError(request.full_url, 404, "not found", {}, None)
        text = (self.adapter_text if "#" in body["model"]
                else "A fair coin is flipped once. What is the probability of heads? Try it.")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def read(self):
                return json.dumps({"model": body["model"], "choices": [
                    {"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()

        return Response()


def adapter_context(tmp_path, chat):
    config = {
        "benchmark_id": "t", "run_dir": str(tmp_path / "runs"), "keys_file": "unused",
        "spend_cap_env": "CAP", "min_subsample_cases": 1, "estimate_safety_factor": 1.0,
        "case_sets": [{"name": "fixtures", "format": "plan_dir", "path": str(PLAN_DIR)}],
        "models": [deployment_entry(), adapter_entry()],
    }
    keys = {"FW": SECRET, "CAP": "20"}
    run_dir = Path(config["run_dir"])
    ctx = bench.RunContext(
        config=config, keys=keys, cap=bench.spend_cap(keys, config),
        ledger=bench.Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments"),
        run_dir=run_dir, urlopen=chat, sleep=lambda _s: None, out=lambda *_a: None,
    )
    return config, ctx


def test_adapter_runs_after_the_base_on_the_same_deployment(tmp_path, monkeypatch):
    import tools.response.fireworks_deployment as fwd

    api = FakeControlPlane()
    monkeypatch.setattr(fwd, "FireworksAPI", lambda *_a, **_k: api)
    chat = RoutingChat("Let's practice. A fair coin is flipped once. Find P(heads). Try it.")
    config, ctx = adapter_context(tmp_path, chat)
    cases, _ = bench.load_all_cases(config)
    assert bench.readiness(config["models"][1], ctx, managing=True) == (True, "ready")
    assert bench.readiness(config["models"][1], ctx)[1].startswith("skipped: adapter")
    records = bench.run_with_deployment(config["models"][0], cases, ctx, pilot=1,
                                        adapters=[config["models"][1]])
    assert len(records) == 2 * len(cases)
    adapter_model = f"accounts/acct/models/lora#{api.deployment}"
    first_adapter_call = chat.models.index(adapter_model)
    assert all(model == api.deployment for model in chat.models[:first_adapter_call])
    assert api.deleted
    state = json.loads((ctx.run_dir / "deployments" / "base.json").read_text())
    assert state["deleted_verified"] and state["addons"][0]["served_as"] == adapter_model
    sft = bench.load_results(ctx.run_dir, "sft")
    assert len(sft) == len(cases) and all(r["model"] == adapter_model for r in sft.values())
    assert bench.decoding_id(config["models"][1]) != bench.decoding_id(config["models"][0])


def test_adapter_identical_to_base_stops_and_still_deletes(tmp_path, monkeypatch):
    import tools.response.fireworks_deployment as fwd

    api = FakeControlPlane()
    monkeypatch.setattr(fwd, "FireworksAPI", lambda *_a, **_k: api)
    same = "A fair coin is flipped once. What is the probability of heads? Try it."
    chat = RoutingChat(same, fail_adapter_routes=())
    config, ctx = adapter_context(tmp_path, chat)
    cases, _ = bench.load_all_cases(config)
    with pytest.raises(bench.BenchmarkError, match="does not appear to be applied"):
        bench.run_with_deployment(config["models"][0], cases, ctx, pilot=1,
                                  adapters=[config["models"][1]])
    assert api.deleted


def test_unservable_adapter_routes_raise_and_delete(tmp_path, monkeypatch):
    import tools.response.fireworks_deployment as fwd

    api = FakeControlPlane()
    monkeypatch.setattr(fwd, "FireworksAPI", lambda *_a, **_k: api)
    chat = RoutingChat("x")
    config, ctx = adapter_context(tmp_path, chat)
    cases, _ = bench.load_all_cases(config)
    original = bench._ping

    def ping(entry, model, ctx_):
        if entry["id"] == "sft":
            raise bench.CallError("HTTP 404: no route", status=404)
        return original(entry, model, ctx_)

    monkeypatch.setattr(bench, "_ping", ping)
    with pytest.raises(bench.BenchmarkError, match="no candidate model string served"):
        bench.run_with_deployment(config["models"][0], cases, ctx, pilot=1,
                                  adapters=[config["models"][1]])
    assert api.deleted


def test_load_addon_failure_raises(tmp_path):
    class Refusing(FakeControlPlane):
        def request(self, method, path, *args, **kwargs):
            if method == "POST" and path.endswith("/deployedModels"):
                return 400, {"message": "addons not supported"}
            return super().request(method, path, *args, **kwargs)

    api = Refusing()
    run = DeploymentRun(api, deployment_entry(), tmp_path / "dep.json",
                        bench.Ledger(tmp_path / "ledger.jsonl"), out=lambda *_a: None,
                        sleep=lambda _s: None)
    run.create()
    with pytest.raises(DeploymentError, match="LoRA load failed"):
        run.load_addon("accounts/acct/models/lora")
    assert run.state["addons"][0]["error"] == {"message": "addons not supported"}


# ---------------------------------------------------------------------------
# Committed configs


@pytest.mark.parametrize("name, rows", [
    ("sft_smoke_qwen38_fireworks.yaml", 261), ("sft_full_qwen38_fireworks.yaml", 435)])
def test_committed_sft_configs_match_the_prd_values(name, rows):
    text = (ROOT / "configs/response" / name).read_text()
    cfg = yaml.safe_load(text)
    assert check_config(cfg) == []
    job = cfg["fireworks"]["job"]
    assert job["lora_rank"] == cfg["lora_rank"] == 16 and job["lora_alpha"] == cfg["lora_alpha"]
    assert job["learning_rate"] == cfg["learning_rate"]
    assert job["epochs"] <= cfg["epochs_max"]
    assert job["batch_size_samples"] == (
        cfg["micro_batch_size"] * cfg["gradient_accumulation_steps"])
    assert job["max_context_length"] == cfg["max_sequence_tokens"]
    steps = fsft.optimizer_steps(rows, job["batch_size_samples"], job["epochs"])["total"]
    assert job["learning_rate_warmup_steps"] == math.ceil(cfg["warmup_ratio"] * steps)
    assert cfg["base_model_revision"] == cfg["tokenizer_revision"]
    assert re.fullmatch(r"[0-9a-f]{40}", cfg["base_model_revision"])
    assert "fw_" not in text and "sk-" not in text


def test_committed_eval_config_pairs_the_adapter_with_a_hot_reload_base():
    config = bench.load_config(ROOT / "configs/response/sft_smoke_eval_v1.yaml")
    entries = {entry["id"]: entry for entry in config["models"]}
    for entry in entries.values():
        adapter = entry.get("adapter")
        if adapter:
            base = entries[adapter["deployment_entry"]]
            assert base["deployment"]["enable_hot_reload_latest_addon"] is True
            assert base["deployment"]["min_replica_count"] == 0
            assert entry["decoding"] == base["decoding"]
    assert config["spend_cap_usd"] == 20
    assert not any("probability_cases" in spec["path"] for spec in config["case_sets"])
