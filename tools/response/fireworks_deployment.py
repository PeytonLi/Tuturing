"""Fireworks on-demand deployment lifecycle for the response benchmark.

Creates one deployment from a validated deployment shape with scale-to-zero autoscaling,
waits until it serves, and deletes it with a verifying GET. Each step is written to a state
file under the benchmark run directory, and the GPU time goes into the spend ledger as an
estimate (wall time from the create request to verified deletion, times the list price).

    uv run python -m tools.response.fireworks_deployment validate --entry qwen3.8-27b-fw-h100
    uv run python -m tools.response.fireworks_deployment status --entry qwen3.8-27b-fw-h100
    uv run python -m tools.response.fireworks_deployment delete --entry qwen3.8-27b-fw-h100
    uv run python -m tools.response.fireworks_deployment list

``benchmark run --manage-deployment`` uses the same classes and always deletes in a
``finally`` block. API reference: https://docs.fireworks.ai/api-reference/create-deployment
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

API_BASE = "https://api.fireworks.ai"
USER_AGENT = "tuturing-response-benchmark/1.0"
_SUMMARY_FIELDS = (
    "name", "displayName", "state", "status", "createTime", "updateTime", "deleteTime",
    "purgeTime", "baseModel", "deploymentShape", "acceleratorType", "acceleratorCount",
    "precision", "region", "placement", "minReplicaCount", "maxReplicaCount",
    "desiredReplicaCount", "replicaCount", "replicaStats", "autoscalingPolicy",
)


class DeploymentError(RuntimeError):
    pass


def _iso(epoch):
    return datetime.fromtimestamp(epoch, UTC).replace(microsecond=0).isoformat()


def summarize(payload):
    return {key: payload[key] for key in _SUMMARY_FIELDS if key in payload}


class FireworksAPI:
    """Minimal JSON client for the Fireworks control plane. Never logs the key."""

    def __init__(self, api_key, *, base=API_BASE, urlopen=None, timeout=60):
        self._key = api_key
        self.base = base.rstrip("/")
        self.urlopen = urlopen or urllib.request.urlopen
        self.timeout = timeout

    def _clean(self, text):
        return text.replace(self._key, "<redacted>") if self._key else text

    def request(self, method, path, body=None, query=None):
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        })
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
                return getattr(response, "status", 200), json.loads(text) if text.strip() else {}
        except urllib.error.HTTPError as error:
            text = self._clean(error.read().decode("utf-8", errors="replace"))
            try:
                return error.code, json.loads(text)
            except json.JSONDecodeError:
                return error.code, {"raw": text[:1000]}

    def account_id(self):
        status, payload = self.request("GET", "/v1/accounts")
        accounts = payload.get("accounts") or []
        if status != 200 or not accounts:
            raise DeploymentError(f"could not list Fireworks accounts (HTTP {status})")
        if len(accounts) > 1:
            raise DeploymentError("several Fireworks accounts; set deployment.account_id")
        return accounts[0]["name"].split("/", 1)[1]


class DeploymentRun:
    """One deployment's lifecycle, persisted to ``state_path``."""

    def __init__(self, api, entry, state_path, ledger, *, out=print, sleep=time.sleep,
                 clock=time.time):
        self.api = api
        self.entry = entry
        self.spec = entry["deployment"]
        self.state_path = Path(state_path)
        self.ledger = ledger
        self.out = out
        self.sleep = sleep
        self.clock = clock
        self.state = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    @property
    def usd_per_hour(self):
        return float(self.spec["usd_per_gpu_hour"]) * int(self.spec.get("accelerator_count", 1))

    @property
    def create_requested_epoch(self):
        return self.state["create_requested_epoch"]

    def _save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def account(self):
        return self.spec.get("account_id") or self.api.account_id()

    def create_body(self):
        spec = self.spec
        body = {
            "displayName": spec.get("display_name", "tuturing response benchmark")[:63],
            "description": spec.get("description", "Tuturing response benchmark; delete after"),
            "baseModel": spec["base_model"],
            "deploymentShape": spec["deployment_shape"],
            "minReplicaCount": int(spec.get("min_replica_count", 0)),
            "maxReplicaCount": int(spec.get("max_replica_count", 1)),
            "autoscalingPolicy": {
                "scaleToZeroWindow": spec.get("scale_to_zero_window", "300s"),
                "scaleDownWindow": spec.get("scale_down_window", "300s"),
                "scaleUpWindow": spec.get("scale_up_window", "30s"),
            },
            "annotations": {
                "custom/project": "tuturing",
                "custom/purpose": "response-benchmark",
            },
        }
        return body

    def validate(self):
        """Dry run: the server returns the deployment it would create. Free."""
        account = self.account()
        status, payload = self.api.request(
            "POST", f"/v1/accounts/{account}/deployments", self.create_body(),
            {"validateOnly": "true", "deploymentId": self._new_id()},
        )
        return status, payload

    def _new_id(self):
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{self.spec.get('id_prefix', 'tut-bench')}-{stamp}"

    def create(self):
        if self.state.get("name") and not self.state.get("deleted_verified"):
            raise DeploymentError(
                f"state file already tracks live deployment {self.state['name']}; delete it first"
            )
        account = self.account()
        deployment_id = self._new_id()
        self.state = {
            "entry_id": self.entry["id"],
            "deployment_id": deployment_id,
            "account_id": account,
            "base_model": self.spec["base_model"],
            "deployment_shape": self.spec["deployment_shape"],
            "accelerator_type": self.spec.get("accelerator_type"),
            "accelerator_count": int(self.spec.get("accelerator_count", 1)),
            "usd_per_gpu_hour": float(self.spec["usd_per_gpu_hour"]),
            "usd_per_hour": self.usd_per_hour,
            "price_source": self.spec.get("price_source"),
            "create_requested_epoch": self.clock(),
            "create_requested_at": _iso(self.clock()),
            "deleted_verified": False,
            "events": [],
        }
        self._save()
        self.out(f"creating deployment {deployment_id} ({self.spec['deployment_shape']})")
        status, payload = self.api.request(
            "POST", f"/v1/accounts/{account}/deployments", self.create_body(),
            {"deploymentId": deployment_id},
        )
        self.state["events"].append({"at": _iso(self.clock()), "op": "create", "http": status})
        if status != 200 or "name" not in payload:
            self.state.update({"deleted_verified": True, "error": payload, "state": "NOT_CREATED"})
            self._save()
            raise DeploymentError(f"create failed (HTTP {status}): {json.dumps(payload)[:500]}")
        self.state.update({"name": payload["name"], "state": payload.get("state"),
                           "create_response": summarize(payload)})
        self._save()
        self.ledger.append({
            "event": "deployment_create", "model_id": self.entry["id"],
            "deployment": payload["name"], "usd_per_hour": self.usd_per_hour,
        })
        return payload

    def get(self):
        status, payload = self.api.request("GET", f"/v1/{self.state['name']}")
        return status, payload

    def wait_ready(self, timeout_s=2400, poll_s=15):
        deadline = self.clock() + timeout_s
        while True:
            status, payload = self.get()
            state = payload.get("state")
            self.state.update({"state": state, "last_get": summarize(payload)})
            self._save()
            if status == 200 and state == "READY":
                self.out(f"deployment {self.state['name']} is READY")
                return payload
            if state in {"FAILED", "DELETED", "DELETING"}:
                raise DeploymentError(f"deployment entered state {state}: {payload.get('status')}")
            if self.clock() > deadline:
                raise DeploymentError(f"deployment not READY after {timeout_s:.0f}s ({state})")
            self.sleep(poll_s)

    def warm_up(self, call, timeout_s=2400, poll_s=20):
        """Send a tiny request until a replica answers (scale-from-zero returns 503)."""
        deadline = self.clock() + timeout_s
        attempts = 0
        while True:
            attempts += 1
            try:
                call()
            except Exception as exc:  # CallError from the benchmark client
                if self.clock() > deadline:
                    raise DeploymentError(f"no replica served after {timeout_s:.0f}s: {exc}")
                if attempts == 1 or attempts % 6 == 0:
                    self.out(f"waiting for a replica ({str(exc)[:120]})")
                self.sleep(poll_s)
                continue
            self.state["first_served_at"] = _iso(self.clock())
            self._save()
            self.out(f"deployment serving after {attempts} warm-up attempt(s)")
            return

    def delete_and_verify(self, attempts=40, poll_s=15):
        """Delete, then confirm with GET (state DELETED or 404). Records GPU cost."""
        name = self.state.get("name")
        if not name or self.state.get("deleted_verified"):
            return self.state
        status, payload = self.api.request("DELETE", f"/v1/{name}")
        self.state["events"].append({"at": _iso(self.clock()), "op": "delete", "http": status})
        if status not in (200, 404):
            status, payload = self.api.request("DELETE", f"/v1/{name}", query={
                "ignoreChecks": "true"})
            self.state["events"].append({
                "at": _iso(self.clock()), "op": "delete_ignore_checks", "http": status})
        self._save()
        verified = None
        for _ in range(attempts):
            get_status, get_payload = self.get()
            state = get_payload.get("state")
            if get_status == 404 or state == "DELETED":
                verified = {"http": get_status, **summarize(get_payload)}
                break
            self.sleep(poll_s)
        end = self.clock()
        hours = max(end - self.state["create_requested_epoch"], 0) / 3600
        cost = round(hours * self.usd_per_hour, 4)
        self.state.update({
            "delete_requested_http": status,
            "verification_get": verified,
            "deleted_verified": verified is not None,
            "ended_epoch": end,
            "ended_at": _iso(end),
            "billable_hours_upper_bound": round(hours, 4),
            "estimated_gpu_usd": cost,
        })
        if verified is None:
            self.state["state"] = "DELETE_UNVERIFIED"
            self._save()
            self.out(f"WARNING: could not verify deletion of {name}; check it now")
            return self.state
        self.state["state"] = verified.get("state") or "NOT_FOUND"
        try:
            list_status, listing = self.api.request(
                "GET", f"/v1/accounts/{self.state['account_id']}/deployments",
                query={"showDeleted": "true", "pageSize": "50"})
            matches = [d for d in listing.get("deployments", []) if d.get("name") == name]
            self.state["verification_list"] = {
                "http": list_status,
                "entries_for_name": [summarize(d) for d in matches],
            }
        except Exception as exc:  # listing is supplementary evidence only
            self.state["verification_list"] = {"error": str(exc)[:200]}
        self._save()
        self.ledger.append({
            "event": "gpu",
            "model_id": self.entry["id"],
            "deployment": name,
            "accelerator": f"{self.state['accelerator_count']}x {self.state['accelerator_type']}",
            "start": self.state["create_requested_at"],
            "end": self.state["ended_at"],
            "hours": round(hours, 4),
            "usd_per_hour": self.usd_per_hour,
            "actual_usd": cost,
            "basis": "upper bound: create request to verified deletion at list price",
        })
        self.out(f"deleted {name}; verified state {self.state['state']}; "
                 f"GPU estimate ${cost:.2f} for {hours * 60:.1f} min")
        return self.state


def _context(args):
    from tools.response.benchmark import (
        deployment_state_path,
        entry_by_id,
        load_config,
        make_context,
    )

    config = load_config(args.config)
    ctx = make_context(args, config)
    entry = entry_by_id(ctx.config, args.entry) if getattr(args, "entry", None) else None
    api_key = ctx.keys.get((entry or {}).get("api_key_env") or "FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is not set in the keys file")
    api = FireworksAPI(api_key)
    run = None
    if entry is not None:
        run = DeploymentRun(api, entry, deployment_state_path(ctx, entry), ctx.ledger)
    return ctx, api, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/response/benchmark_v1.yaml")
    parser.add_argument("--keys-file")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "status", "delete"):
        sub.add_parser(name).add_argument("--entry", required=True)
    sub.add_parser("list")
    args = parser.parse_args(argv)
    ctx, api, run = _context(args)
    if args.command == "validate":
        status, payload = run.validate()
        print(json.dumps({"http": status, **summarize(payload)}, indent=2, sort_keys=True))
        return 0 if status == 200 else 1
    if args.command == "status":
        if not run.state.get("name"):
            print("no deployment recorded for this entry")
            return 0
        status, payload = run.get()
        print(json.dumps({"http": status, **summarize(payload)}, indent=2, sort_keys=True))
        return 0
    if args.command == "delete":
        state = run.delete_and_verify()
        print(json.dumps({k: state.get(k) for k in (
            "name", "state", "deleted_verified", "estimated_gpu_usd")}, indent=2))
        return 0 if state.get("deleted_verified", True) else 1
    account = api.account_id()
    status, payload = api.request("GET", f"/v1/accounts/{account}/deployments",
                                  query={"pageSize": "50"})
    rows = [summarize(item) for item in payload.get("deployments", [])]
    print(json.dumps({"http": status, "deployments": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
