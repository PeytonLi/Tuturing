"""OpenAI supervised fine-tuning of the response model (option B): data, job, artifacts, spend.

Converts SFTExample JSONL into OpenAI chat fine-tuning JSONL (system and user prompt from
``compile_prompt``, the tutor reply as the only trained turn), validates the files against
OpenAI's format and limits, counts tokens and estimates the cost, uploads the files, then
creates, polls and settles one fine-tuning job and downloads its events, checkpoints and
result files. Every paid step is checked against the run directory's spend ledger first.
Configs: ``configs/response/sft_openai_{smoke,full}_v1.yaml``; runbook:
``docs/runbooks/sft_openai.md``.

    uv run python -m tools.response.openai_sft --config <config> convert
    uv run --with tiktoken python -m tools.response.openai_sft --config <config> check
    uv run python -m tools.response.openai_sft --config <config> upload        # needs approval
    uv run python -m tools.response.openai_sft --config <config> train         # dry run
    uv run python -m tools.response.openai_sft --config <config> train --confirm
    uv run python -m tools.response.openai_sft --config <config> poll
    uv run python -m tools.response.openai_sft --config <config> fetch
    uv run python -m tools.response.openai_sft --config <config> select --key epoch_2 --reason ...

Format and limits (developers.openai.com, checked 2026-11-02): chat-format JSONL, one
example per line; the optional ``weight`` key (0 or 1) is set on assistant messages only;
at least 10 examples; 65,536 tokens per example for the gpt-4.1 family (longer examples are
truncated from the end). Training is billed per token: tokens in the training file times
epochs. Keys come only from the keys file and are never printed.
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import yaml

from tools.response.fireworks_sft import (
    SFTError,
    chat_turns,
    convert_split,
    export_cases,
    sha256_bytes,
    utc_now,
)

DEFAULT_CONFIG = "configs/response/sft_openai_full_v1.yaml"
API_BASE = "https://api.openai.com"
USER_AGENT = "tuturing-response-sft/1.0"
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
MIN_EXAMPLES = 10
EXAMPLE_TOKEN_LIMIT = 65_536
TOKENS_PER_MESSAGE = 3   # chat formatting overhead per message (OpenAI cookbook count)
TOKENS_PER_EXAMPLE = 3   # reply priming
_JOB_FIELDS = (
    "id", "status", "model", "created_at", "finished_at", "estimated_finish",
    "fine_tuned_model", "training_file", "validation_file", "trained_tokens", "seed",
    "hyperparameters", "method", "result_files", "error", "suffix", "user_provided_suffix",
    "metadata", "organization_id",
)


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


# ---------------------------------------------------------------------------
# Format, tokens and cost


def to_openai_record(example):
    """One SFTExample as an OpenAI chat fine-tuning example; only the reply is trained."""
    system, user, reply = chat_turns(example)
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply, "weight": 1},
    ]}


def record_errors(record):
    """Problems that would make OpenAI reject or mis-train one example."""
    if not isinstance(record, dict):
        return ["the example is not a JSON object"]
    errors = []
    if set(record) != {"messages"}:
        errors.append(f"unexpected top-level keys {sorted(set(record) - {'messages'})}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return errors + ["messages must be a non-empty list"]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message {index} is not an object")
            continue
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            errors.append(f"message {index}: role {role!r} is not system, user or assistant")
        allowed = {"role", "content"} | ({"weight"} if role == "assistant" else set())
        unexpected = sorted(set(message) - allowed)
        if unexpected:
            errors.append(f"message {index}: unexpected keys {unexpected}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"message {index}: content must be a non-empty string")
        if role == "assistant" and "weight" in message and message["weight"] not in (0, 1):
            errors.append(f"message {index}: weight must be 0 or 1")
        if role == "system" and index != 0:
            errors.append(f"message {index}: a system message must come first")
    trained = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
               and m.get("weight", 1) == 1]
    if not trained:
        errors.append("no trained assistant message")
    if isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
        errors.append("the last message must be the assistant reply")
    return errors


def heuristic_encode(text):
    """Conservative stand-in when tiktoken is absent: one token per three characters."""
    return [0] * math.ceil(len(text) / 3)


def load_encoder(name):
    try:
        import tiktoken
    except ImportError:
        return heuristic_encode, "heuristic ceil(chars / 3); tiktoken not installed"
    return tiktoken.get_encoding(name).encode, f"tiktoken {tiktoken.__version__} {name}"


def count_tokens(record, encode):
    """(total, trained) tokens of one example, counted the way OpenAI's cookbook does."""
    total, trained = TOKENS_PER_EXAMPLE, 0
    for message in record["messages"]:
        content_tokens = len(encode(message["content"]))
        total += TOKENS_PER_MESSAGE + len(encode(message["role"])) + content_tokens
        if message["role"] == "assistant" and message.get("weight", 1) == 1:
            trained += content_tokens
    return total, trained


def check_file(path, encode, *, max_example_tokens=EXAMPLE_TOKEN_LIMIT,
               min_examples=MIN_EXAMPLES):
    """Validate one OpenAI fine-tuning file and count its tokens."""
    data = Path(path).read_bytes()
    errors, totals, trained = [], [], []
    rows = 0
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"line {number}: blank line")
            continue
        rows += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: invalid JSON ({error.msg})")
            continue
        problems = record_errors(record)
        if problems:
            errors.extend(f"line {number}: {problem}" for problem in problems)
            continue
        total, target = count_tokens(record, encode)
        totals.append(total)
        trained.append(target)
        if total > max_example_tokens:
            errors.append(f"line {number}: {total} tokens exceeds {max_example_tokens}")
    if rows < min_examples:
        errors.append(f"{rows} examples; OpenAI needs at least {min_examples}")
    return {
        "path": str(path), "rows": rows, "bytes": len(data), "sha256": sha256_bytes(data),
        "error_count": len(errors), "errors": errors[:50],
        "total_tokens": sum(totals), "trained_tokens": sum(trained),
        "max_example_tokens": max(totals, default=0),
        "max_example_token_limit": max_example_tokens,
    }


def estimate_cost(train_tokens, validation_tokens, n_epochs, pricing):
    """Training USD: training-file tokens x epochs at the per-token training price.

    Validation tokens are not documented as billed; the upper bound adds them per epoch.
    """
    price = float(pricing["training_usd_per_mtok"])
    billed = int(train_tokens) * int(n_epochs)
    expected = billed * price / 1_000_000
    upper = (billed + int(validation_tokens) * int(n_epochs)) * price / 1_000_000
    safety = float(pricing.get("estimate_safety_factor", 1.5))
    return {
        "billed_training_tokens_expected": billed,
        "training_usd_per_mtok": price,
        "expected_usd": round(expected, 4),
        "upper_bound_usd": round(upper, 4),
        "reserve_usd": round(upper * safety, 4),
        "safety_factor": safety,
    }


# ---------------------------------------------------------------------------
# HTTP


def multipart_form(fields, files, boundary=None):
    """Encode text fields and (filename, bytes, type) files; return (body, content type)."""
    boundary = boundary or f"tuturing-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    for name, (filename, data, content_type) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
            + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAIClient:
    """Minimal OpenAI REST client. Never logs the key; error bodies are redacted."""

    def __init__(self, api_key, *, base=API_BASE, urlopen=None, timeout=120):
        self._key = api_key
        self.base = base.rstrip("/")
        self.urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout

    def _clean(self, text):
        return text.replace(self._key, "<redacted>") if self._key else text

    def request(self, method, path, body=None, query=None, *, data=None,
                content_type="application/json", raw=False):
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        if data is None and body is not None:
            data = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                status = getattr(response, "status", 200)
        except urllib.error.HTTPError as error:
            text = self._clean(error.read().decode("utf-8", errors="replace"))
            try:
                return error.code, json.loads(text)
            except json.JSONDecodeError:
                return error.code, {"raw": text[:1000]}
        if raw:
            return status, payload
        text = payload.decode("utf-8")
        return status, json.loads(text) if text.strip() else {}


# ---------------------------------------------------------------------------
# Files, job and artifacts


def summarize_job(payload):
    return {key: payload[key] for key in _JOB_FIELDS if key in payload}


class OpenAIRun:
    """Uploaded files, one fine-tuning job and its artifacts, in ``<run_dir>/sft_job.json``."""

    def __init__(self, client, cfg, ledger, cap, *, out=print, sleep=time.sleep, clock=time.time):
        self.client = client
        self.cfg = cfg
        self.oa = cfg["openai"]
        self.ledger = ledger
        self.cap = cap
        self.out = out
        self.sleep = sleep
        self.clock = clock
        self.run_dir = Path(self.oa["run_dir"])
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

    def checked(self):
        path = self.run_dir / "check.json"
        if not path.exists():
            raise SFTError("run the check command first")
        return json.loads(path.read_text(encoding="utf-8"))

    def upload(self, key, path):
        """Upload one checked file with purpose fine-tune; reuse an identical earlier upload."""
        data = Path(path).read_bytes()
        digest = sha256_bytes(data)
        check = self.checked()[key]
        if check["sha256"] != digest or check["error_count"]:
            raise SFTError(f"{path} changed since the check or failed it; run check again")
        files = self.state.setdefault("files", {})
        recorded = files.get(key)
        if recorded and recorded.get("sha256") == digest and recorded.get("id"):
            status, payload = self.client.request("GET", f"/v1/files/{recorded['id']}")
            if status == 200:
                return recorded
        body, content_type = multipart_form(
            {"purpose": "fine-tune"},
            {"file": (f"tuturing-{self.cfg['run_id']}-{key}.jsonl", data, "application/jsonl")},
        )
        status, payload = self.client.request(
            "POST", "/v1/files", data=body, content_type=content_type)
        if status != 200 or "id" not in payload:
            files[key] = {"sha256": digest, "upload_http": status, "error": payload}
            self.save()
            raise SFTError(f"file upload: HTTP {status}: {json.dumps(payload)[:300]}")
        files[key] = {"id": payload["id"], "sha256": digest, "rows": check["rows"],
                      "bytes": len(data), "filename": payload.get("filename"),
                      "status": payload.get("status"), "uploaded_at": utc_now()}
        self.save()
        self.out(f"uploaded {key}: {payload['id']} ({len(data)} bytes)")
        return files[key]

    def job_body(self):
        files = self.state.get("files") or {}
        if not files.get("train", {}).get("id"):
            raise SFTError("upload the training file first")
        body = {
            "model": self.oa["base_model"],
            "training_file": files["train"]["id"],
            "suffix": self.oa["suffix"],
            "method": {"type": "supervised", "supervised": {
                "hyperparameters": dict(self.oa.get("hyperparameters") or {})}},
        }
        if files.get("validation", {}).get("id"):
            body["validation_file"] = files["validation"]["id"]
        if self.oa.get("seed") is not None:
            body["seed"] = int(self.oa["seed"])
        if self.oa.get("metadata"):
            body["metadata"] = {str(k): str(v) for k, v in self.oa["metadata"].items()}
        return body

    def create_job(self, estimate, *, confirm=False):
        if self.state.get("job", {}).get("id"):
            raise SFTError(f"job {self.state['job']['id']} already recorded; use poll")
        if self.cap is None:
            raise SFTError("no spend cap: the keys file must set the cap variable")
        body = self.job_body()
        reserve = float(estimate["reserve_usd"])
        committed = self.ledger.committed()
        decision = {
            "event": "estimate", "model_id": "sft-train", "base_model": body["model"],
            "expected_usd": estimate["expected_usd"],
            "upper_bound_usd": estimate["upper_bound_usd"],
            "reserve_usd": reserve, "committed_usd": round(committed, 4), "cap_usd": self.cap,
        }
        if committed + reserve > self.cap + 1e-9:
            raise SFTError(f"committed ${committed:.2f} + reserve ${reserve:.2f} passes the "
                           f"cap ${self.cap:.2f}; not training")
        if not confirm:
            return {"dry_run": True, "body": body, "decision": decision}
        self.ledger.append(decision)
        self.state["job"] = {"request": body, "estimate": estimate,
                             "create_requested_at": utc_now(), "transitions": []}
        self.save()
        status, payload = self.client.request("POST", "/v1/fine_tuning/jobs", body)
        self.state["job"]["create_http"] = status
        if status != 200 or "id" not in payload:
            self.state["job"]["error"] = payload
            self.save()
            raise SFTError(f"job create: HTTP {status}: {json.dumps(payload)[:500]}")
        self.state["job"].update({"id": payload["id"], "created": summarize_job(payload)})
        self.save()
        self.ledger.append({
            "event": "sft_job_reserve", "model_id": "sft-train", "job": payload["id"],
            "actual_usd": reserve,
            "basis": "upper bound held until the job settles (training tokens at list price)",
        })
        self.out(f"created {payload['id']} ({payload.get('status')})")
        return payload

    def get_job(self):
        job_id = self.state.get("job", {}).get("id")
        if not job_id:
            raise SFTError("no job recorded")
        return self.client.request("GET", f"/v1/fine_tuning/jobs/{job_id}")

    def poll(self, timeout_s=14400, poll_s=60):
        record = self.state["job"]
        deadline = self.clock() + timeout_s
        while True:
            status, payload = self.get_job()
            if status != 200:
                self.out(f"GET job returned HTTP {status}; retrying")
            else:
                state = payload.get("status")
                transitions = record.setdefault("transitions", [])
                if not transitions or transitions[-1]["status"] != state:
                    transitions.append({"status": state, "seen_at": utc_now()})
                    self.out(f"{utc_now()} {state}")
                record["last"] = summarize_job(payload)
                self.save()
                if state in TERMINAL_STATES:
                    record["finished_seen_at"] = utc_now()
                    self.save()
                    self.settle(payload)
                    return payload
            if self.clock() > deadline:
                raise SFTError(f"job not finished after {timeout_s:.0f}s; poll again later")
            self.sleep(poll_s)

    def cancel(self, reason):
        job_id = self.state["job"]["id"]
        status, _ = self.client.request("POST", f"/v1/fine_tuning/jobs/{job_id}/cancel", {})
        self.state["job"]["cancel"] = {"at": utc_now(), "http": status, "reason": reason}
        self.save()
        return status

    def settle(self, payload):
        """Replace the reserve with trained_tokens at the training price, when OpenAI reports it."""
        record = self.state["job"]
        if record.get("settled"):
            return record["settled"]
        reserve = float(record["estimate"]["reserve_usd"])
        price = float(record["estimate"]["training_usd_per_mtok"])
        trained = payload.get("trained_tokens")
        final = trained * price / 1_000_000 if isinstance(trained, int) else reserve
        settled = {"final_usd": round(final, 4), "trained_tokens": trained,
                   "reserve_usd": reserve, "status": payload.get("status"), "at": utc_now(),
                   "basis": "trained_tokens x training price" if isinstance(trained, int)
                   else "trained_tokens absent: the reserve stays as the charge"}
        self.ledger.append({
            "event": "sft_job_settle", "model_id": "sft-train", "job": record["id"],
            "actual_usd": round(final - reserve, 4), "final_usd": round(final, 4),
            "trained_tokens": trained, "basis": settled["basis"],
        })
        record["settled"] = settled
        self.save()
        return settled

    def fetch(self):
        """Save events, checkpoints and result files; write models.json for the evaluation."""
        status, job = self.get_job()
        if status != 200:
            raise SFTError(f"GET job returned HTTP {status}")
        job_id = job["id"]
        out_dir = self.run_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        events, after = [], None
        for _ in range(50):
            query = {"limit": "100", **({"after": after} if after else {})}
            status, page = self.client.request(
                "GET", f"/v1/fine_tuning/jobs/{job_id}/events", query=query)
            if status != 200:
                break
            events.extend(page.get("data") or [])
            if not page.get("has_more") or not page.get("data"):
                break
            after = page["data"][-1]["id"]
        (out_dir / "events.json").write_text(json.dumps(events, indent=1) + "\n",
                                             encoding="utf-8")
        status, page = self.client.request(
            "GET", f"/v1/fine_tuning/jobs/{job_id}/checkpoints", query={"limit": "10"})
        checkpoints = sorted(page.get("data") or [], key=lambda c: c.get("step_number", 0)) \
            if status == 200 else []
        (out_dir / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1) + "\n",
                                                  encoding="utf-8")
        result_files = []
        for file_id in job.get("result_files") or []:
            status, content = self.client.request(
                "GET", f"/v1/files/{file_id}/content", raw=True)
            if status == 200:
                path = out_dir / f"{file_id}.csv"
                path.write_bytes(content)
                result_files.append({"id": file_id, "path": str(path), "bytes": len(content)})
        models = {"final": job.get("fine_tuned_model"), "job": job_id,
                  "base_model": job.get("model")}
        n_epochs = int((job.get("hyperparameters") or {}).get("n_epochs") or len(checkpoints))
        first_epoch = n_epochs - len(checkpoints) + 1
        for offset, checkpoint in enumerate(checkpoints):
            models[f"epoch_{first_epoch + offset}"] = checkpoint.get("fine_tuned_model_checkpoint")
        models_path = self.run_dir / "models.json"
        existing = json.loads(models_path.read_text()) if models_path.exists() else {}
        if existing.get("selected"):
            models["selected"] = existing["selected"]
            models["selection"] = existing.get("selection")
        models_path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
        self.state["artifacts"] = {"events": len(events), "checkpoints": len(checkpoints),
                                   "result_files": result_files, "fetched_at": utc_now()}
        self.state["models"] = models
        self.save()
        return models

    def select(self, key, reason):
        models_path = self.run_dir / "models.json"
        models = json.loads(models_path.read_text(encoding="utf-8"))
        if not models.get(key):
            raise SFTError(f"models.json has no {key}; choose from "
                           f"{sorted(k for k, v in models.items() if str(v).startswith('ft:'))}")
        models["selected"] = models[key]
        models["selection"] = {"key": key, "reason": reason, "at": utc_now()}
        models_path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
        return models


# ---------------------------------------------------------------------------
# CLI


def _context(cfg, keys_file=None):
    from tools.response.benchmark import Ledger, load_keys, spend_cap

    oa = cfg["openai"]
    keys = load_keys(keys_file or oa.get("keys_file", "~/.config/tuturing/llm.env"))
    api_key = keys.get(oa.get("api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise SystemExit("the OpenAI key is not set in the keys file")
    run_dir = Path(oa["run_dir"])
    ledger = Ledger(run_dir / "ledger.jsonl", deployments_dir=run_dir / "deployments")
    cap = spend_cap(keys, {"spend_cap_env": oa.get("spend_cap_env",
                                                    "TUTURING_BENCH_SPEND_CAP_USD"),
                           "spend_cap_usd": oa.get("spend_cap_usd")})
    return OpenAIRun(OpenAIClient(api_key), cfg, ledger, cap)


def _paths(cfg):
    run_dir = Path(cfg["openai"]["run_dir"])
    return {
        "train_in": Path(cfg["data"]["train"]),
        "validation_in": Path(cfg["data"]["validation"]),
        "train": run_dir / "data" / "train.jsonl",
        "validation": run_dir / "data" / "validation.jsonl",
        "check": run_dir / "check.json",
    }


def command_convert(cfg):
    paths = _paths(cfg)
    result = {key: convert_split(paths[f"{key}_in"], paths[key], to_record=to_openai_record)
              for key in ("train", "validation")}
    evaluation = cfg.get("evaluation") or {}
    if evaluation.get("validation_cases"):
        result["validation_cases"] = export_cases(
            evaluation.get("validation_cases_source") or cfg["data"]["validation"],
            evaluation["validation_cases"])
    print(json.dumps(result, indent=2))
    return 0


def command_check(cfg):
    oa = cfg["openai"]
    paths = _paths(cfg)
    encode, method = load_encoder(oa.get("tokenizer", "o200k_base"))
    limits = oa.get("limits") or {}
    result = {"token_counting": method}
    for key in ("train", "validation"):
        result[key] = check_file(
            paths[key], encode,
            max_example_tokens=int(limits.get("max_example_tokens", EXAMPLE_TOKEN_LIMIT)),
            min_examples=int(limits.get("min_examples", MIN_EXAMPLES)))
    n_epochs = (oa.get("hyperparameters") or {}).get("n_epochs")
    if not isinstance(n_epochs, int):
        raise SystemExit("set hyperparameters.n_epochs to an integer so the cost is bounded")
    result["estimate"] = estimate_cost(result["train"]["total_tokens"],
                                       result["validation"]["total_tokens"], n_epochs,
                                       oa["pricing"])
    for name, alternative in (oa.get("alternatives") or {}).items():
        result.setdefault("alternatives", {})[name] = estimate_cost(
            result["train"]["total_tokens"], result["validation"]["total_tokens"], n_epochs,
            alternative)
    paths["check"].parent.mkdir(parents=True, exist_ok=True)
    paths["check"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not (result["train"]["error_count"] or result["validation"]["error_count"]) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--keys-file")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("convert", "check", "upload", "poll", "fetch", "status"):
        sub.add_parser(name)
    train = sub.add_parser("train")
    train.add_argument("--confirm", action="store_true", help="create the paid job")
    select = sub.add_parser("select")
    select.add_argument("--key", required=True, help="final, epoch_1, epoch_2, ...")
    select.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "convert":
        return command_convert(cfg)
    if args.command == "check":
        return command_check(cfg)
    run = _context(cfg, args.keys_file)
    paths = _paths(cfg)
    if args.command == "upload":
        for key in ("train", "validation"):
            run.upload(key, paths[key])
        print(json.dumps(run.state["files"], indent=2))
        return 0
    if args.command == "train":
        result = run.create_job(run.checked()["estimate"], confirm=args.confirm)
        print(json.dumps(result if not args.confirm else summarize_job(result), indent=2))
        return 0
    if args.command == "poll":
        oa = cfg["openai"]
        payload = run.poll(timeout_s=float(oa.get("max_wait_minutes", 240)) * 60,
                           poll_s=float(oa.get("poll_interval_s", 60)))
        print(json.dumps(summarize_job(payload), indent=2))
        print(json.dumps(run.state["job"].get("settled"), indent=2))
        return 0 if payload.get("status") == "succeeded" else 1
    if args.command == "fetch":
        print(json.dumps(run.fetch(), indent=2))
        return 0
    if args.command == "select":
        print(json.dumps(run.select(args.key, args.reason), indent=2))
        return 0
    status, payload = run.get_job()
    print(json.dumps({"http": status, **summarize_job(payload)}, indent=2))
    print(f"ledger committed ${run.ledger.committed():.4f} of cap {run.cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
