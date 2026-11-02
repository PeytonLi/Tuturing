from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path

import yaml

from packages.learning.episode import load_scenarios, simulate_episodes
from packages.policies.network import NetworkStrategist
from tools.learning.train_ppo import episode_reward

ROOT = Path(__file__).resolve().parents[2]


def bootstrap_mean_interval(values: list[float], seed: int, resamples: int = 1000):
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples)
    )
    return [means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))]]


def summarize(results, seed: int) -> dict:
    gains = [result.gain for result in results]
    transitions = [item for result in results for item in result.strategist_transitions]
    disclosure = sum(
        item["reward_components"]["disclosure_penalty"] < 0 for item in transitions
    ) / max(1, len(transitions))
    return {
        "label": "simulated",
        "episodes": len(results),
        "mean_gain": sum(gains) / max(1, len(gains)),
        "mean_gain_ci_95": bootstrap_mean_interval(gains, seed),
        "disclosure_rate": disclosure,
        "dropout_rate": sum(result.dropout for result in results) / max(1, len(results)),
        "mean_turns": sum(result.turns for result in results) / max(1, len(results)),
        "mean_total_reward": sum(episode_reward(result) for result in results)
        / max(1, len(results)),
    }


def policy_from_name(name: str):
    if name in {"rules", "random", "always_hint"}:
        return None, name
    if name.startswith("checkpoint:"):
        checkpoint = name.split(":", 1)[1]
        if not checkpoint:
            raise ValueError("checkpoint policy requires a path")
        return NetworkStrategist(checkpoint, greedy=True), "network"
    raise ValueError(f"unknown policy {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/smoke.yaml")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--policies", default="rules,random,always_hint")
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument("--unseal-test", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 20250115))
    if args.split == "test" and not args.unseal_test:
        parser.error("test split is sealed; pass --unseal-test to run it")
    if args.split == "test":
        warnings.warn("Unsealing the test split; results are simulated.", stacklevel=1)
        print("WARNING: unsealing test split; all results are simulated.")
    scenarios = [row for row in load_scenarios() if row["split"] == args.split]
    if not scenarios:
        raise ValueError(f"no scenarios for split {args.split}")
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    report = {
        "label": "simulated",
        "split": args.split,
        "seed": seed,
        "episodes_per_scenario": args.episodes_per_scenario,
        "policies": {},
    }
    for policy_index, name in enumerate(policies):
        strategist, strategist_name = policy_from_name(name)
        results = simulate_episodes(
            episodes=len(scenarios) * args.episodes_per_scenario,
            split=args.split,
            scenarios=scenarios,
            seed=seed + policy_index * 100_000,
            strategist_name=strategist_name,
            strategist=strategist,
            run_id=f"evaluation-{args.split}-{policy_index}",
        )
        policy_report = {
            "label": "simulated",
            "overall": summarize(results, seed + policy_index),
            "profiles": {},
        }
        for profile_index, profile in enumerate(sorted({row.profile_id for row in results})):
            group = [row for row in results if row.profile_id == profile]
            policy_report["profiles"][profile] = summarize(
                group, seed + policy_index * 100 + profile_index
            )
        report["policies"][name] = policy_report
    name = args.name or f"{args.split}_seed{seed}"
    output_dir = ROOT / "experiments/learning/evals" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Simulated learning evaluation",
        "",
        f"Split: **{args.split}**. All results are **simulated**.",
        "",
        "| Policy | Profile | Mean gain (95% bootstrap CI) | Disclosure | Dropout | "
        "Mean turns | Mean reward |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy, values in report["policies"].items():
        groups = [("overall", values["overall"]), *values["profiles"].items()]
        for profile, summary in groups:
            low, high = summary["mean_gain_ci_95"]
            lines.append(
                f"| {policy} | {profile} | {summary['mean_gain']:.4f} "
                f"({low:.4f}, {high:.4f}) | {summary['disclosure_rate']:.3f} "
                f"| {summary['dropout_rate']:.3f} | {summary['mean_turns']:.2f} "
                f"| {summary['mean_total_reward']:.4f} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"reports: {output_dir}")


if __name__ == "__main__":
    main()
