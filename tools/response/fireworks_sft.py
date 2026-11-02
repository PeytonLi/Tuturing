"""Fireworks managed LoRA SFT for the response model: data, job, artifacts and spend.

Converts SFTExample JSONL (``data/sft/original_v1``) into Fireworks chat JSONL with the loss
on the tutor reply only, counts tokens with the pinned tokenizer, uploads the datasets,
launches one ``supervisedFineTuningJob``, polls it, and downloads its per-step metrics and
rendered loss masks. Every paid step is checked against the spend ledger in the run
directory first. Config: ``configs/response/sft_smoke_qwen38_fireworks.yaml``; runbook:
``docs/runbooks/sft_fireworks.md``.

    uv run python -m tools.response.fireworks_sft convert
    uv run --extra response-train python -m tools.response.fireworks_sft tokens
    uv run python -m tools.response.fireworks_sft upload
    uv run python -m tools.response.fireworks_sft train            # dry run: body + estimate
    uv run python -m tools.response.fireworks_sft train --confirm  # paid
    uv run python -m tools.response.fireworks_sft poll
    uv run python -m tools.response.fireworks_sft fetch
    uv run python -m tools.response.fireworks_sft status

Dataset format (https://docs.fireworks.ai/fine-tuning/fine-tuning-models): one JSON object
per line with a ``messages`` array; ``weight: 0`` on a message excludes it from the loss and
any other number trains it. Keys come only from the keys file and are never printed.
"""

import argparse
import hashlib
import json
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

from packages.response.prompt import compile_prompt

DEFAULT_CONFIG = "configs/response/sft_smoke_qwen38_fireworks.yaml"
TERMINAL_STATES = frozenset({
    "JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    "JOB_STATE_EARLY_STOPPED", "JOB_STATE_DELETED", "JOB_STATE_ARCHIVED",
})
_JOB_FIELDS = (
    "name", "displayName", "state", "status", "createTime", "updateTime", "completedTime",
    "baseModel", "dataset", "evaluationDataset", "evalAutoCarveout", "outputModel", "epochs",
    "learningRate", "learningRateWarmupSteps", "lrScheduler", "maxContextLength", "loraRank",
    "loraAlpha", "batchSize", "batchSizeSamples", "gradientAccumulationSteps",
    "optimizerWeightDecay", "isTurbo", "purpose", "estimatedCost", "jobProgress",
    "rendererHuggingFaceRepoId", "thinkingTraceHistoryMode", "tuneVisionEncoder",
    "stateTimestamps", "createdBy",
)
_JOB_BODY_KEYS = {
    "epochs": "epochs",
    "batch_size_samples": "batchSizeSamples",
    "learning_rate": "learningRate",
    "learning_rate_warmup_steps": "learningRateWarmupSteps",
    "lr_scheduler": "lrScheduler",
    "lora_rank": "loraRank",
    "lora_alpha": "loraAlpha",
    "max_context_length": "maxContextLength",
    "optimizer_weight_decay": "optimizerWeightDecay",
    "eval_auto_carveout": "evalAutoCarveout",
    "is_turbo": "isTurbo",
}


class SFTError(RuntimeError):
    """A precondition, budget or API problem that stops the SFT workflow."""


def utc_now():
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    """Write records the way build_original_sft does; return (rows, sha256)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(records), sha256_bytes(data)


# ---------------------------------------------------------------------------
# Data conversion


def to_fireworks_record(example):
    """One SFTExample as a Fireworks chat record; only the tutor reply carries loss.

    The stored prompt must equal ``compile_prompt(plan)``, which is what the benchmark
    harness and the response service send at inference time.
    """
    example_id = example.get("example_id", "<unknown>")
    prompt = example.get("prompt") or []
    completion = example.get("completion") or []
    roles = [message.get("role") for message in prompt]
    if roles != ["system", "user"]:
        raise ValueError(f"{example_id}: prompt roles {roles}, expected system then user")
    if len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError(f"{example_id}: completion must be one assistant message")
    if compile_prompt(example["plan"]) != prompt:
        raise ValueError(f"{example_id}: stored prompt differs from compile_prompt(plan)")
    reply = completion[0].get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError(f"{example_id}: empty tutor reply")
    if "<think>" in reply or "</think>" in reply:
        raise ValueError(f"{example_id}: tutor reply contains a thinking marker")
    for message in prompt:
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{example_id}: empty {message.get('role')} message")
    return {"messages": [
        {"role": "system", "content": prompt[0]["content"], "weight": 0},
        {"role": "user", "content": prompt[1]["content"], "weight": 0},
        {"role": "assistant", "content": reply, "weight": 1},
    ]}


def convert_split(in_path, out_path):
    examples = read_jsonl(in_path)
    records = [to_fireworks_record(example) for example in examples]
    rows, digest = write_jsonl(out_path, records)
    return {
        "source": str(in_path),
        "source_sha256": sha256_bytes(Path(in_path).read_bytes()),
        "path": str(out_path),
        "rows": rows,
        "sha256": digest,
    }


def export_cases(in_path, out_path):
    """Copy an SFT split as a benchmark case set (same format as probability_cases_v1)."""
    examples = read_jsonl(in_path)
    for example in examples:
        if not isinstance(example.get("plan"), dict) or not example.get("example_id"):
            raise ValueError(f"{in_path}: every record needs example_id and plan")
    rows, digest = write_jsonl(out_path, examples)
    return {"source": str(in_path), "path": str(out_path), "rows": rows, "sha256": digest}


# ---------------------------------------------------------------------------
# Tokens and cost


def token_stats(tokenizer, examples, max_tokens):
    """Token counts with the loss mask train_sft.py uses (prompt masked, reply trained)."""
    from tools.response.train_sft import build_loss_mask

    totals, targets = [], []
    for example in examples:
        result = build_loss_mask(tokenizer, example)
        totals.append(len(result["input_ids"]))
        targets.append(sum(result["loss_mask"]))
    count = len(totals)
    return {
        "examples": count,
        "total_tokens": sum(totals),
        "target_tokens": sum(targets),
        "prompt_tokens": sum(totals) - sum(targets),
        "mean_total_tokens": round(sum(totals) / count, 1) if count else 0.0,
        "max_total_tokens": max(totals, default=0),
        "min_target_tokens": min(targets, default=0),
        "max_target_tokens": max(targets, default=0),
        "over_max_sequence_tokens": sum(1 for total in totals if total > max_tokens),
        "max_sequence_tokens": max_tokens,
    }


def optimizer_steps(rows, batch_size, epochs):
    per_epoch = -(-rows // batch_size)
    return {"per_epoch": per_epoch, "total": per_epoch * epochs,
            "per_epoch_full_batches_only": rows // batch_size}


def estimate_training_cost(train_stats, eval_stats, job, pricing):
    """Upper-bound USD for one managed LoRA SFT job, from token counts and list prices.

    Fireworks: "Training tokens can be estimated with number of tokens in training dataset
    * number of epochs." Whether evaluation tokens are billed is not documented, so the
    bound adds one evaluation pass per epoch plus one, then applies the safety factor.
    """
    epochs = int(job["epochs"])
    price = float(pricing["usd_per_mtok"])
    train_tokens = int(train_stats["total_tokens"]) * epochs
    eval_tokens = int(eval_stats["total_tokens"]) * (epochs + 1) if eval_stats else 0
    expected = train_tokens * price / 1_000_000
    bound = (train_tokens + eval_tokens) * price / 1_000_000
    safety = float(pricing.get("estimate_safety_factor", 1.5))
    return {
        "billed_training_tokens_expected": train_tokens,
        "evaluation_tokens_upper_bound": eval_tokens,
        "usd_per_mtok": price,
        "expected_usd": round(expected, 4),
        "upper_bound_usd": round(bound, 4),
        "reserve_usd": round(bound * safety, 4),
        "safety_factor": safety,
    }


# ---------------------------------------------------------------------------
# Fireworks control plane


def multipart_file(field, filename, data, boundary=None):
    """Encode one file part as multipart/form-data; return (body, content_type)."""
    boundary = boundary or f"tuturing-{uuid.uuid4().hex}"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


def money_usd(money):
    """Convert a google.type.Money-like dict to float USD; None when absent."""
    if not isinstance(money, dict) or not money:
        return None
    return float(money.get("units") or 0) + float(money.get("nanos") or 0) / 1e9


def summarize_job(payload):
    return {key: payload[key] for key in _JOB_FIELDS if key in payload}


class SFTRun:
    """Datasets, one SFT job and its artifacts, persisted to ``<run_dir>/sft_job.json``."""

    def __init__(self, api, cfg, ledger, cap, *, out=print, sleep=time.sleep, clock=time.time,
                 urlopen=None):
        self.api = api
        self.cfg = cfg
        self.fw = cfg["fireworks"]
        self.ledger = ledger
        self.cap = cap
        self.out = out
        self.sleep = sleep
        self.clock = clock
        self.urlopen = urlopen or urllib.request.urlopen
        self.run_dir = Path(self.fw["run_dir"])
        self.state_path = self.run_dir / "sft_job.json"
        self.state = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def account(self):
        account = self.state.get("account_id") or self.fw.get("account_id")
        if not account:
            account = self.api.account_id()
        self.state["account_id"] = account
        return account

    # -- datasets -----------------------------------------------------------------

    def ensure_dataset(self, key, path, timeout_s=600, poll_s=5):
        """Create and upload one dataset unless an identical upload is already recorded."""
        account = self.account()
        dataset_id = self.fw["datasets"][key]
        data = Path(path).read_bytes()
        digest = sha256_bytes(data)
        rows = sum(1 for line in data.decode("utf-8").splitlines() if line.strip())
        recorded = self.state.setdefault("datasets", {}).get(key)
        name = f"accounts/{account}/datasets/{dataset_id}"
        status, payload = self.api.request("GET", f"/v1/{name}")
        created = False
        if status == 200:
            if not recorded or recorded.get("sha256") != digest:
                raise SFTError(
                    f"dataset {name} exists but this run did not upload {path} to it; "
                    "choose a new dataset id in the config"
                )
            if payload.get("state") == "READY":
                recorded.update(self._dataset_summary(payload))
                self.save()
                return recorded
        elif status != 404:
            raise SFTError(f"GET {name} returned HTTP {status}: {json.dumps(payload)[:300]}")
        else:
            body = {"datasetId": dataset_id, "dataset": {
                "displayName": f"tuturing {self.cfg.get('run_id', 'sft')} {key}"[:63],
                "exampleCount": str(rows),
                "userUploaded": {},
                "format": "CHAT",
            }}
            status, payload = self.api.request("POST", f"/v1/accounts/{account}/datasets", body)
            if status != 200:
                raise SFTError(f"dataset create: HTTP {status}: {json.dumps(payload)[:200]}")
            recorded = {"name": name, "path": str(path), "sha256": digest, "rows": rows,
                        "bytes": len(data), "created_at": utc_now()}
            self.state["datasets"][key] = recorded
            self.save()
            created = True
        if recorded.get("upload_http") != 200:
            body_bytes, content_type = multipart_file("file", Path(path).name, data)
            status, payload = self.api.request(
                "POST", f"/v1/{name}:upload", data=body_bytes, content_type=content_type)
            recorded.update({"upload_http": status, "upload_at": utc_now()})
            if status != 200:
                recorded["upload_error"] = payload
                if created:
                    # Leave nothing behind: an empty dataset object only blocks a later rerun.
                    delete_status, _ = self.api.request("DELETE", f"/v1/{name}")
                    recorded["deleted_after_failed_upload"] = {"http": delete_status,
                                                               "at": utc_now()}
                    recorded["upload_http"] = None
                self.save()
                raise SFTError(f"dataset upload: HTTP {status}: {json.dumps(payload)[:200]}")
            self.save()
        deadline = self.clock() + timeout_s
        while True:
            status, payload = self.api.request("GET", f"/v1/{name}")
            if status == 200 and payload.get("state") == "READY":
                recorded.update(self._dataset_summary(payload))
                self.save()
                if int(payload.get("exampleCount") or rows) != rows:
                    raise SFTError(f"{name} reports {payload.get('exampleCount')} examples, "
                                   f"expected {rows}")
                self.out(f"dataset {name} READY ({rows} rows, "
                         f"~{payload.get('estimatedTokenCount')} tokens by Fireworks)")
                return recorded
            if self.clock() > deadline:
                raise SFTError(f"dataset {name} not READY after {timeout_s}s")
            self.sleep(poll_s)

    @staticmethod
    def _dataset_summary(payload):
        return {
            "state": payload.get("state"),
            "example_count": payload.get("exampleCount"),
            "estimated_token_count": payload.get("estimatedTokenCount"),
            "average_turn_count": payload.get("averageTurnCount"),
            "format": payload.get("format"),
            "status": payload.get("status"),
        }

    # -- job ----------------------------------------------------------------------

    def job_body(self):
        job = self.fw["job"]
        datasets = self.state.get("datasets") or {}
        if "train" not in datasets or "validation" not in datasets:
            raise SFTError("upload the train and validation datasets first")
        body = {
            "displayName": job.get("display_name", "tuturing response sft")[:63],
            "baseModel": self.fw["base_model"],
            "dataset": datasets["train"]["name"],
            "evaluationDataset": datasets["validation"]["name"],
            "outputModel": job["output_model"],
        }
        for key, field in _JOB_BODY_KEYS.items():
            if key in job and job[key] is not None:
                body[field] = job[key]
        return body

    def create_job(self, estimate, *, confirm=False):
        if self.state.get("job", {}).get("name"):
            raise SFTError(f"job {self.state['job']['name']} already recorded; use poll")
        if self.cap is None:
            raise SFTError("no spend cap: the keys file must set the cap variable")
        account = self.account()
        job = self.fw["job"]
        body = self.job_body()
        if not body["outputModel"].startswith("accounts/"):
            body["outputModel"] = f"accounts/{account}/models/{body['outputModel']}"
        reserve = float(estimate["reserve_usd"])
        committed = self.ledger.committed()
        decision = {
            "event": "estimate", "model_id": "sft-train", "expected_usd": estimate["expected_usd"],
            "upper_bound_usd": estimate["upper_bound_usd"], "reserve_usd": reserve,
            "committed_usd": round(committed, 4), "cap_usd": self.cap,
        }
        if committed + reserve > self.cap + 1e-9:
            raise SFTError(f"committed ${committed:.2f} + reserve ${reserve:.2f} passes the "
                           f"cap ${self.cap:.2f}; not training")
        if not confirm:
            return {"dry_run": True, "body": body, "decision": decision}
        self.ledger.append(decision)
        job_id = job["id"]
        self.state["job"] = {"job_id": job_id, "request": body, "estimate": estimate,
                             "create_requested_at": utc_now(), "transitions": []}
        self.save()
        status, payload = self.api.request(
            "POST", f"/v1/accounts/{account}/supervisedFineTuningJobs", body,
            {"supervisedFineTuningJobId": job_id},
        )
        self.state["job"]["create_http"] = status
        if status != 200 or "name" not in payload:
            self.state["job"]["error"] = payload
            self.save()
            raise SFTError(f"job create failed (HTTP {status}): {json.dumps(payload)[:500]}")
        self.state["job"].update({"name": payload["name"], "created": summarize_job(payload)})
        self.save()
        self.ledger.append({
            "event": "sft_job_reserve", "model_id": "sft-train", "job": payload["name"],
            "actual_usd": reserve,
            "basis": "upper bound held until the job settles (training tokens at list price)",
        })
        self.out(f"created {payload['name']} (state {payload.get('state')})")
        return payload

    def get_job(self):
        name = self.state.get("job", {}).get("name")
        if not name:
            raise SFTError("no job recorded")
        return self.api.request("GET", f"/v1/{name}")

    def poll(self, timeout_s=10800, poll_s=60):
        """Poll until a terminal state, logging transitions; settle the ledger at the end."""
        record = self.state["job"]
        deadline = self.clock() + timeout_s
        reserve = float(record["estimate"]["reserve_usd"])
        while True:
            status, payload = self.get_job()
            if status != 200:
                self.out(f"GET job returned HTTP {status}; retrying")
                if self.clock() > deadline:
                    raise SFTError(f"job not finished after {timeout_s}s")
                self.sleep(poll_s)
                continue
            state = payload.get("state")
            transitions = record.setdefault("transitions", [])
            if not transitions or transitions[-1]["state"] != state:
                transitions.append({"state": state, "seen_at": utc_now(),
                                    "status": payload.get("status")})
                self.out(f"{utc_now()} {state} {json.dumps(payload.get('status'))}")
            record["last"] = summarize_job(payload)
            self.save()
            estimated = money_usd(payload.get("estimatedCost"))
            if estimated is not None and estimated > reserve and state not in TERMINAL_STATES:
                self.cancel(f"Fireworks estimatedCost ${estimated:.2f} exceeds the reserve "
                            f"${reserve:.2f}")
                raise SFTError("job cancelled: estimated cost above the reserve")
            if state in TERMINAL_STATES:
                record["finished_at"] = utc_now()
                self.save()
                self.settle(payload)
                return payload
            if self.clock() > deadline:
                raise SFTError(f"job still {state} after {timeout_s}s; poll again later")
            self.sleep(poll_s)

    def cancel(self, reason):
        name = self.state["job"]["name"]
        status, payload = self.api.request("DELETE", f"/v1/{name}")
        self.state["job"]["cancel"] = {"at": utc_now(), "http": status, "reason": reason}
        self.save()
        self.out(f"cancel requested for {name}: HTTP {status} ({reason})")

    def settle(self, payload):
        record = self.state["job"]
        if record.get("settled"):
            return record["settled"]
        reserve = float(record["estimate"]["reserve_usd"])
        fireworks_usd = money_usd(payload.get("estimatedCost"))
        token_usd = float(record["estimate"]["expected_usd"])
        completed = payload.get("state") == "JOB_STATE_COMPLETED"
        candidates = [value for value in (fireworks_usd, token_usd if completed else None)
                      if value is not None]
        final = max(candidates) if candidates else reserve
        settled = {"final_usd": round(final, 4), "fireworks_estimated_usd": fireworks_usd,
                   "token_estimate_usd": token_usd, "reserve_usd": reserve,
                   "state": payload.get("state"), "at": utc_now()}
        self.ledger.append({
            "event": "sft_job_settle", "model_id": "sft-train", "job": record["name"],
            "actual_usd": round(final - reserve, 4), "final_usd": round(final, 4),
            "fireworks_estimated_usd": fireworks_usd, "token_estimate_usd": token_usd,
            "basis": "replaces the reserve with max(Fireworks estimatedCost, token estimate)",
        })
        record["settled"] = settled
        self.save()
        return settled

    # -- artifacts ----------------------------------------------------------------

    def fetch_artifacts(self, payload=None):
        """Download the metrics file, rendered samples and trainer logs (signed URLs).

        Signed URLs point at object storage; they are fetched without the API key.
        """
        if payload is None:
            status, payload = self.get_job()
            if status != 200:
                raise SFTError(f"GET job returned HTTP {status}")
        out_dir = self.run_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        fetched = {}
        for field, filename in (("metricsFileSignedUrl", "metrics"),
                                ("renderSamplesSignedUrl", "render_samples"),
                                ("trainerLogsSignedUrl", "trainer_logs")):
            url = payload.get(field)
            if not url:
                fetched[field] = None
                continue
            request = urllib.request.Request(url, method="GET")
            with self.urlopen(request, timeout=120) as response:
                data = response.read()
            path = out_dir / filename
            path.write_bytes(data)
            fetched[field] = {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}
        self.state.setdefault("artifacts", {}).update(fetched)
        self.state["artifacts"]["fetched_at"] = utc_now()
        self.save()
        return fetched

    def output_model(self):
        name = self.state.get("job", {}).get("request", {}).get("outputModel")
        if not name:
            raise SFTError("no output model recorded")
        status, payload = self.api.request("GET", f"/v1/{name}")
        summary = {
            "http": status, "name": payload.get("name"), "state": payload.get("state"),
            "kind": payload.get("kind"), "peftDetails": payload.get("peftDetails"),
            "fineTuningJob": payload.get("fineTuningJob"), "createTime": payload.get("createTime"),
        }
        self.state["output_model"] = summary
        self.save()
        return summary


# ---------------------------------------------------------------------------
# CLI


def _paths(cfg):
    fw = cfg["fireworks"]
    run_dir = Path(fw["run_dir"])
    return {
        "train_in": Path(cfg["data"]["train"]),
        "validation_in": Path(cfg["data"]["validation"]),
        "train_out": run_dir / "data" / "train.jsonl",
        "validation_out": run_dir / "data" / "validation.jsonl",
        "cases_in": Path(cfg["evaluation"].get("validation_cases_source")
                         or cfg["data"]["validation"]),
        "cases_out": Path(cfg["evaluation"]["validation_cases"]),
        "tokens": run_dir / "tokens.json",
    }


def _context(cfg, keys_file=None):
    from tools.response.benchmark import Ledger, load_keys, spend_cap
    from tools.response.fireworks_deployment import FireworksAPI

    fw = cfg["fireworks"]
    keys = load_keys(keys_file or fw.get("keys_file", "~/.config/tuturing/llm.env"))
    api_key = keys.get(fw.get("api_key_env", "FIREWORKS_API_KEY"))
    if not api_key:
        raise SystemExit("the Fireworks key is not set in the keys file")
    run_dir = Path(fw["run_dir"])
    ledger = Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments")
    cap = spend_cap(keys, {"spend_cap_env": fw.get("spend_cap_env",
                                                    "TUTURING_BENCH_SPEND_CAP_USD"),
                           "spend_cap_usd": fw.get("spend_cap_usd")})
    return SFTRun(FireworksAPI(api_key), cfg, ledger, cap)


def command_convert(cfg):
    paths = _paths(cfg)
    result = {
        "train": convert_split(paths["train_in"], paths["train_out"]),
        "validation": convert_split(paths["validation_in"], paths["validation_out"]),
        "validation_cases": export_cases(paths["cases_in"], paths["cases_out"]),
    }
    print(json.dumps(result, indent=2))
    return 0


def command_tokens(cfg):
    from transformers import AutoTokenizer

    paths = _paths(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["tokenizer_revision"])
    stats = {
        "tokenizer": {"repo": cfg["base_model"], "revision": cfg["tokenizer_revision"],
                      "class": type(tokenizer).__name__},
        "train": token_stats(tokenizer, read_jsonl(paths["train_in"]),
                             int(cfg["max_sequence_tokens"])),
        "validation": token_stats(tokenizer, read_jsonl(paths["validation_in"]),
                                  int(cfg["max_sequence_tokens"])),
    }
    job = cfg["fireworks"]["job"]
    stats["optimizer_steps"] = optimizer_steps(
        stats["train"]["examples"], int(job["batch_size_samples"]), int(job["epochs"]))
    stats["estimate"] = estimate_training_cost(
        stats["train"], stats["validation"], job, cfg["fireworks"]["pricing"])
    paths["tokens"].parent.mkdir(parents=True, exist_ok=True)
    paths["tokens"].write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


def _load_estimate(cfg):
    path = _paths(cfg)["tokens"]
    if not path.exists():
        raise SystemExit("run the tokens command first; the estimate comes from token counts")
    return json.loads(path.read_text(encoding="utf-8"))["estimate"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--keys-file")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "tokens", "upload", "poll", "fetch", "status"):
        sub.add_parser(name)
    train = sub.add_parser("train")
    train.add_argument("--confirm", action="store_true", help="create the paid job")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "convert":
        return command_convert(cfg)
    if args.command == "tokens":
        return command_tokens(cfg)
    run = _context(cfg, args.keys_file)
    paths = _paths(cfg)
    if args.command == "upload":
        for key in ("train", "validation"):
            run.ensure_dataset(key, paths[f"{key}_out"])
        print(json.dumps(run.state["datasets"], indent=2))
        return 0
    if args.command == "train":
        result = run.create_job(_load_estimate(cfg), confirm=args.confirm)
        print(json.dumps(result if args.confirm is False else summarize_job(result), indent=2))
        return 0
    if args.command == "poll":
        fw = cfg["fireworks"]
        payload = run.poll(timeout_s=float(fw.get("max_wait_minutes", 180)) * 60,
                           poll_s=float(fw.get("poll_interval_s", 60)))
        print(json.dumps(summarize_job(payload), indent=2))
        print(json.dumps(run.state["job"].get("settled"), indent=2))
        return 0 if payload.get("state") == "JOB_STATE_COMPLETED" else 1
    if args.command == "fetch":
        print(json.dumps(run.fetch_artifacts(), indent=2))
        print(json.dumps(run.output_model(), indent=2))
        return 0
    status, payload = run.get_job()
    print(json.dumps({"http": status, **summarize_job(payload)}, indent=2))
    print(f"ledger committed ${run.ledger.committed():.4f} of cap {run.cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
