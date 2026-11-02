"""Paired SIMULATED structured-to-text integration evaluation (PRD G8)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from apps.response_service.config import load_config
from packages.learning.episode import HORIZON, load_scenarios, run_episode
from packages.simulator.randomness import EventRandomBank
from packages.simulator.scenarios import ROOT
from packages.simulator.student import DEFAULT_SIMULATOR_CONFIG
from packages.simulator.text_path import (
    CRITICAL_FLAGS,
    CallBudget,
    CallLimitExceeded,
    DegradedRealizer,
    ServiceRealizer,
    TextPath,
    llm_verbalizer,
)
from packages.simulator.text_renderer import RENDERER_VERSION
from tools.learning.evaluate import bootstrap_mean_interval, policy_from_name

STUB_CONFIG = "configs/response/serve_stub.yaml"
MODEL_CONFIG = "configs/response/serve_openai_compatible.yaml"
FIDELITY_RULE = (
    "PRD line 895: content_fidelity=0 for unsupported or contradictory delivered content. "
    "Here any failed independent A check or item-identity check sets content_realized=False; "
    "otherwise the simulator's configured stub_realized coefficient (at most 1) and "
    "seeded failure process apply. Extraction disagreement is separate from content fidelity."
)
FAILURE_TYPES = sorted(CRITICAL_FLAGS | {"fallback_used", "draft_validator_failure"})


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def _estimate(values, seed):
    return {
        "mean": sum(values) / len(values),
        "ci_95": bootstrap_mean_interval(values, seed),
    }


def g8_gate(report: dict, records: list[dict], reviews: list[dict] | None = None) -> dict:
    flags = {(row["path"], row["trace_id"]): row for row in records if any(row["flags"].values())}
    reviews_by_key = {}
    for row in reviews or []:
        key = (row["path"], row["trace_id"])
        if key not in flags or key in reviews_by_key:
            raise ValueError("review must uniquely match a flagged trace in this run")
        if type(row.get("critical_failure")) is not bool or row.get("reviewed") is not True:
            raise ValueError("review requires reviewed=true and boolean critical_failure")
        if row.get("review_hash") != flags[key]["review_hash"] or not row.get("reviewed_by"):
            raise ValueError("review must bind to review_hash and name a human reviewer")
        reviews_by_key[key] = row
    complete = reviews is not None and set(reviews_by_key) == set(flags)
    critical = sum(any(row["flags"].get(name, False) for name in CRITICAL_FLAGS) for row in records)
    reviewed_critical = sum(row["critical_failure"] for row in reviews_by_key.values())
    outcome_ok = report["paired_text_minus_structured"]["mean"] >= -0.02
    eligible = report["split"] == "test" and report["episodes"] >= 300
    real_text = (
        report["reply"] == "openai"
        and report["paths"]["text"]["model_delivered_episodes"] == report["episodes"]
    )
    passed = eligible and real_text and complete and not reviewed_critical and outcome_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "eligible_test_budget": eligible,
        "real_text_run": real_text,
        "outcome_threshold_met": outcome_ok,
        "paired_uncertainty_published": True,
        "review_complete": complete,
        "reviewed_traces": len(reviews_by_key),
        "flagged_traces": len(flags),
        "automated_critical_candidates": critical,
        "human_critical_failures": reviewed_critical,
        "rule": "test pairs>=300 AND real replies AND all flagged traces reviewed with no "
        "critical failures AND text-minus-structured mean>=-0.02; "
        "publish paired bootstrap 95% CI (CI crossing -0.02 is reported, not an extra PRD gate)",
    }


def run_bridge(
    *,
    episodes: int = 300,
    split: str = "validation",
    seed: int = 20250115,
    policy: str = "rules",
    reply: str = "stub",
    response_config: str = MODEL_CONFIG,
    simulator_config: str = DEFAULT_SIMULATOR_CONFIG,
    degradation_rate: float = 0.2,
    max_calls: int = 0,
    dry_run: bool = False,
    unseal_test: bool = False,
    verbalizer_config: str | None = None,
    reviews: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    if split == "test" and not unseal_test:
        raise ValueError("test split is sealed; pass --unseal-test")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if reply not in {"stub", "degraded", "openai"}:
        raise ValueError("reply must be stub, degraded, or openai")
    budget = CallBudget(max_calls)
    scenarios = [row for row in load_scenarios() if row["split"] == split]
    if not scenarios:
        raise ValueError(f"no scenarios for split {split}")
    strategist, strategist_name = policy_from_name(
        policy if policy == "rules" or policy.startswith("checkpoint:") else f"checkpoint:{policy}"
    )
    if reply == "openai" and not dry_run and max_calls == 0:
        raise ValueError("real replies require an explicit positive --max-calls")
    if verbalizer_config and not dry_run and max_calls == 0:
        raise ValueError("LLM verbalizer requires an explicit positive --max-calls")
    response_spec = load_config(response_config) if reply == "openai" else load_config(STUB_CONFIG)
    if reply == "openai" and response_spec.responder["kind"] == "template_stub":
        raise ValueError("openai reply requires a model responder, not template_stub")
    max_repairs = response_spec.max_repairs if reply == "openai" else 0
    max_turns = sum(
        min(HORIZON, int(scenarios[index % len(scenarios)]["teaching_horizon"]))
        for index in range(episodes)
    )
    structured = TextPath(ServiceRealizer(STUB_CONFIG, budget=budget), structured=True)
    text = None
    if not dry_run:
        realizer = ServiceRealizer(
            response_config if reply == "openai" else STUB_CONFIG, budget=budget
        )
        if reply == "degraded":
            realizer = DegradedRealizer(realizer, degradation_rate)
        text = TextPath(
            realizer,
            verbalizer=llm_verbalizer(verbalizer_config, budget) if verbalizer_config else None,
        )
    results = {"structured": [], "text": []}
    pairs = []
    started = datetime.now(UTC).isoformat()
    for index in range(episodes):
        scenario = scenarios[index % len(scenarios)]
        # Both arms share scenario, episode seed, frozen policy pair and the same event bank.
        bank = EventRandomBank()
        options = dict(
            episode_seed=seed + index,
            bank=bank,
            strategist=strategist,
            strategist_name=strategist_name,
            run_id="text-bridge-v1",
            horizon=min(HORIZON, int(scenario["teaching_horizon"])),
            simulator_config=simulator_config,
        )
        reference = run_episode(scenario, text_path=structured, **options)
        results["structured"].append(reference)
        if text is not None:
            realized = run_episode(scenario, text_path=text, **options)
            results["text"].append(realized)
            pairs.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "seed": seed + index,
                    "structured_gain": reference.gain,
                    "text_gain": realized.gain,
                    "difference": realized.gain - reference.gain,
                    "structured_turns": reference.turns,
                    "text_turns": realized.turns,
                }
            )
    records = [
        {"path": name, **row}
        for name, path in (("structured", structured), ("text", text))
        if path is not None
        for row in path.records
    ]
    nominal = len(structured.records)
    report = {
        "label": "SIMULATED",
        "version": "text_bridge_v1",
        "started_at": started,
        "split": split,
        "episodes": episodes,
        "seed": seed,
        "policy": policy,
        "navigator": "rules",
        "reply": reply,
        "dry_run": dry_run,
        "content_fidelity_rule": FIDELITY_RULE,
        "renderer_version": RENDERER_VERSION,
        "bridge_source_sha256": _hash(__file__),
        "text_path_source_sha256": _hash(ROOT / "packages/simulator/text_path.py"),
        "renderer_source_sha256": _hash(ROOT / "packages/simulator/text_renderer.py"),
        "simulator_config": simulator_config,
        "simulator_config_sha256": _hash(simulator_config),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_diff_sha256": hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest(),
        "dependency_lock_sha256": _hash(ROOT / "uv.lock"),
        "scenario_bank_sha256": hashlib.sha256(
            json.dumps(scenarios, sort_keys=True).encode()
        ).hexdigest(),
        "checkpoint_sha256": _hash(policy.removeprefix("checkpoint:"))
        if policy != "rules"
        else None,
        "response_config_sha256": _hash(response_spec.path),
        "verbalizer_config_sha256": _hash(verbalizer_config) if verbalizer_config else None,
        "call_estimate": {
            "structured_rehearsal_turns": nominal,
            "planned_primary_calls_at_stub_trajectory": nominal if reply == "openai" else 0,
            "planned_verbalizer_calls_at_stub_trajectory": nominal if verbalizer_config else 0,
            "maximum_turns_per_path": max_turns,
            "maximum_primary_calls": max_turns if reply == "openai" else 0,
            "maximum_repair_calls": max_turns * max_repairs,
            "maximum_verbalizer_calls": max_turns if verbalizer_config else 0,
            "actual_model_calls": budget.calls,
            "max_calls_budget": max_calls,
            "note": "Nominal count follows stub decisions; text may change decisions and length. "
            "Upper bounds include every turn up to each scenario horizon, including repairs.",
        },
        "paths": {},
        "pairs": pairs,
    }
    if text is not None:
        report["text_realizer"] = text.realizer.description
    for name, items in results.items():
        if not items:
            continue
        path_records = [row for row in records if row["path"] == name]
        counts = Counter(
            flag for row in path_records for flag, active in row["flags"].items() if active
        )
        report["paths"][name] = {
            "gain": _estimate([row.gain for row in items], seed),
            "turns": len(path_records),
            "episodes": len(items),
            "model_delivered_turns": sum(
                row["response"]["delivery_mode"] in {"model", "repaired"} for row in path_records
            ),
            "model_delivered_episodes": len(
                {
                    row["plan"]["session_id"]
                    for row in path_records
                    if row["response"]["delivery_mode"] in {"model", "repaired"}
                }
            ),
            "failure_counts": {flag: counts[flag] for flag in FAILURE_TYPES},
            "critical_turns": sum(
                any(row["flags"].get(flag) for flag in CRITICAL_FLAGS) for row in path_records
            ),
        }
    if not dry_run:
        report["paired_text_minus_structured"] = _estimate(
            [row["difference"] for row in pairs], seed + 1
        )
        report["g8"] = g8_gate(report, records, reviews)
    return report, records


def markdown(report: dict) -> str:
    reports = report.get("smokes", {report.get("reply", "run"): report})
    lines = [
        "# SIMULATED text bridge",
        "",
        "These are not measured human learning outcomes.",
        "",
        "| Run | Pairs | Structured gain (95% CI) | Text gain (95% CI) | "
        "Text minus structured (paired 95% CI) | Outcome >= -0.02 | G8 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    def fmt(estimate):
        low, high = estimate["ci_95"]
        return f"{estimate['mean']:.6f} [{low:.6f}, {high:.6f}]"

    for name, row in reports.items():
        if row["dry_run"]:
            lines += [f"Dry run ({name}), no model calls: " + json.dumps(row["call_estimate"])]
            continue
        lines.append(
            f"| {name} | {row['episodes']} | {fmt(row['paths']['structured']['gain'])} | "
            f"{fmt(row['paths']['text']['gain'])} | {fmt(row['paired_text_minus_structured'])} | "
            f"{row['g8']['outcome_threshold_met']} | {row['g8']['status']} |"
        )
    lines += [
        "",
        FIDELITY_RULE,
        "",
        "## Flag counts (turn denominator)",
        "",
        "| Run | Path | Turns | " + " | ".join(FAILURE_TYPES) + " |",
        "|---|---|---:|" + "---:|" * len(FAILURE_TYPES),
    ]
    for name, row in reports.items():
        for path, values in row["paths"].items():
            counts = " | ".join(str(values["failure_counts"][key]) for key in FAILURE_TYPES)
            lines.append(f"| {name} | {path} | {values['turns']} | {counts} |")
        lines += ["", f"{name} calls: `{json.dumps(row['call_estimate'], sort_keys=True)}`", ""]
        if not row["dry_run"]:
            lines += [f"{name} G8 detail: `{json.dumps(row['g8'], sort_keys=True)}`", ""]
    lines += [
        "Validation and stub/fake runs are engineering checks, not G8 clearance. "
        "Fallback and rejected drafts are flagged for review but do not by themselves "
        "mean the delivered content is critical. Human reviews are absent unless supplied.",
        "Bootstrap resamples paired episode differences (1,000 resamples); repeated seeds within "
        "the same scenario are not new scenario families. Intervals are conditional on this "
        "scenario bank, simulator and policy, not intervals for human effects.",
        "Item identity uses A's public item_fidelity check (validator_v2); "
        "these checks are heuristics, not a semantic/math oracle. Review remains mandatory.",
    ]
    return "\n".join(lines) + "\n"


def write_report(output: str | Path, report: dict, records: list[dict]) -> None:
    prefix = Path(output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix, rows in (
        ("turns", records),
        ("traces", [row for row in records if any(row["flags"].values())]),
    ):
        path = prefix.with_suffix(f".{suffix}.jsonl.gz")
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_bytes(gzip.compress(content.encode(), mtime=0))
        artifacts[suffix] = {"path": str(path), "sha256": _hash(path), "rows": len(rows)}
    report["artifacts"] = artifacts
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    prefix.with_suffix(".md").write_text(markdown(report))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=20250115)
    parser.add_argument("--policy", default="rules", help="rules, checkpoint:PATH, or PATH")
    parser.add_argument("--reply", choices=("stub", "degraded", "openai"), default="stub")
    parser.add_argument("--response-config", default=MODEL_CONFIG)
    parser.add_argument("--simulator-config", default=DEFAULT_SIMULATOR_CONFIG)
    parser.add_argument("--degradation-rate", type=float, default=0.2)
    parser.add_argument("--verbalizer-config")
    parser.add_argument("--max-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unseal-test", action="store_true")
    parser.add_argument(
        "--review-file",
        help="JSON array: path, trace_id, review_hash, reviewed_by, reviewed, critical_failure",
    )
    parser.add_argument("--output", default="docs/results/bridge_validation_v1")
    parser.add_argument(
        "--smoke-suite", action="store_true", help="stub + degraded, no model calls"
    )
    args = parser.parse_args(argv)
    if args.split == "test" and not args.unseal_test:
        parser.error("test split is sealed; pass --unseal-test")
    if args.smoke_suite and (args.reply == "openai" or args.verbalizer_config or args.review_file):
        parser.error("smoke suite only permits local stub/fake paths without reviews")
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in {"output", "smoke_suite", "review_file"}
    }
    options["reviews"] = (
        json.loads(Path(args.review_file).read_text()) if args.review_file else None
    )
    if args.split == "test":
        print("WARNING: explicitly unsealing test; all results are SIMULATED")
    try:
        if args.smoke_suite:
            report = {"label": "SIMULATED", "smokes": {}}
            records = []
            for reply in ("stub", "degraded"):
                result, rows = run_bridge(**{**options, "reply": reply})
                report["smokes"][reply] = result
                records.extend({"run": reply, **row} for row in rows)
        else:
            report, records = run_bridge(**options)
        write_report(args.output, report, records)
    except (ValueError, CallLimitExceeded) as error:
        # Never publish a silently truncated paired estimate when the safety cap trips.
        prefix = Path(args.output)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.with_suffix(".aborted.json").write_text(
            json.dumps(
                {
                    "label": "SIMULATED",
                    "status": "ABORTED",
                    "reason": str(error),
                }
            )
            + "\n"
        )
        parser.error(str(error))
    print(markdown(report))


if __name__ == "__main__":
    main()
