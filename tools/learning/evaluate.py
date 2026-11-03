from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from packages.learning.episode import HORIZON, load_scenarios, run_episode, simulate_episodes
from packages.policies.llm import LLMStrategist
from packages.policies.network import NetworkStrategist
from packages.policies.rules import action_label
from packages.simulator.student import DEFAULT_SIMULATOR_CONFIG
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
    move_counts = {}
    pace_counts = {}
    for result in results:
        for move, count in result.move_histogram.items():
            move_counts[move] = move_counts.get(move, 0) + count
    for transition in transitions:
        pace = action_label(transition["action_id"])[2]
        pace_counts[pace] = pace_counts.get(pace, 0) + 1
    total_moves = sum(move_counts.values())
    total_turns = sum(pace_counts.values())
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
        "move_frequency": {
            move: move_counts[move] / max(1, total_moves) for move in sorted(move_counts)
        },
        "pace_frequency": {
            pace: pace_counts[pace] / max(1, total_turns) for pace in sorted(pace_counts)
        },
        "mean_total_reward": sum(episode_reward(result) for result in results)
        / max(1, len(results)),
    }


def load_llm_aliases() -> dict:
    aliases = yaml.safe_load(
        (ROOT / "configs/learning/llm_strategists.yaml").read_text(encoding="utf-8")
    )
    if not isinstance(aliases, dict):
        raise ValueError("LLM strategist config must be a mapping of aliases")
    return aliases


def policy_from_name(
    name: str,
    *,
    llm_aliases: dict | None = None,
    max_calls: int = 2000,
    llm_base_url_override: str | None = None,
    dry_run: bool = False,
):
    if name in {"rules", "rules_fastready", "random", "always_hint", "always_worked_example"}:
        return None, name
    if name.startswith("checkpoint:"):
        checkpoint = name.split(":", 1)[1]
        if not checkpoint:
            raise ValueError("checkpoint policy requires a path")
        return NetworkStrategist(checkpoint, greedy=True), "network"
    if name.startswith("llm:"):
        alias = name.split(":", 1)[1]
        if not alias:
            raise ValueError("LLM policy requires an alias")
        aliases = llm_aliases if llm_aliases is not None else load_llm_aliases()
        if alias not in aliases:
            raise ValueError(f"unknown LLM strategist alias {alias!r}")
        config = aliases[alias]
        model_env = config.get("model_env")
        model = config.get("model")
        if model_env:
            model = os.environ.get(model_env)
            if model is None:
                if not dry_run:
                    raise RuntimeError(f"LLM policy requires environment variable {model_env}")
                model = f"<{model_env} not set>"
        api_key_env = config["api_key_env"]
        api_key = os.environ.get(api_key_env, "")
        if not api_key and not dry_run:
            raise RuntimeError(f"LLM policy requires environment variable {api_key_env}")
        return (
            LLMStrategist(
                base_url=llm_base_url_override or config["base_url"],
                api_key=api_key,
                model=model,
                max_tokens_field=config["max_tokens_field"],
                max_tokens=int(config["max_tokens"]),
                temperature=config.get("temperature"),
                max_calls=max_calls,
                prompt_path=config["prompt"],
            ),
            "llm",
        )
    raise ValueError(f"unknown policy {name!r}")


def policy_seed(seed: int, policy_index: int, paired: bool) -> int:
    return seed if paired else seed + policy_index * 100_000


def simulate_policy(
    *,
    episodes: int,
    split: str,
    scenarios: list[dict],
    seed: int,
    strategist_name: str,
    strategist,
    run_id: str,
    simulator_config: str,
    workers: int = 1,
):
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return simulate_episodes(
            episodes=episodes,
            split=split,
            scenarios=scenarios,
            seed=seed,
            strategist_name=strategist_name,
            strategist=strategist,
            run_id=run_id,
            simulator_config=simulator_config,
        )
    if episodes < 1:
        raise ValueError("episodes must be positive")
    candidates = [scenario for scenario in scenarios if scenario["split"] == split]
    if not candidates:
        raise ValueError(f"no scenarios available for split {split}")

    def run(index: int):
        scenario = candidates[index % len(candidates)]
        return run_episode(
            scenario,
            run_id=run_id,
            episode_seed=seed + index,
            strategist_name=strategist_name,
            strategist=strategist,
            horizon=min(int(scenario["teaching_horizon"]), HORIZON),
            simulator_config=simulator_config,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, range(episodes)))


class _PromptCaptured(Exception):
    pass


def first_prompt_for(
    strategist: LLMStrategist,
    *,
    scenario: dict,
    seed: int,
    simulator_config: str,
) -> str:
    captured = {}

    class PromptCapture:
        def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
            captured["prompt"] = strategist.build_prompt(context, mask)
            raise _PromptCaptured

    try:
        run_episode(
            scenario,
            run_id="evaluation-dry-run",
            episode_seed=seed,
            strategist_name="llm",
            strategist=PromptCapture(),
            horizon=min(int(scenario["teaching_horizon"]), HORIZON),
            simulator_config=simulator_config,
        )
    except _PromptCaptured:
        pass
    if "prompt" not in captured:
        raise RuntimeError("could not capture the first LLM prompt")
    return captured["prompt"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/smoke.yaml")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--policies", default="rules,random,always_hint")
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--paired-seeds", action="store_true")
    parser.add_argument("--simulator-config")
    parser.add_argument("--name")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-base-url-override")
    parser.add_argument("--unseal-test", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_calls < 0:
        parser.error("--max-calls must be non-negative")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    simulator_config = (
        args.simulator_config
        if args.simulator_config is not None
        else config.get("simulator_config", DEFAULT_SIMULATOR_CONFIG)
    )
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
    llm_aliases = (
        load_llm_aliases() if any(name.startswith("llm:") for name in policies) else None
    )
    selected_policies = [
        (
            policy_index,
            name,
            *policy_from_name(
                name,
                llm_aliases=llm_aliases,
                max_calls=args.max_calls,
                llm_base_url_override=args.llm_base_url_override,
                dry_run=args.dry_run,
            ),
        )
        for policy_index, name in enumerate(policies)
    ]
    if args.dry_run:
        llm_policies = [
            (policy_index, name, strategist)
            for policy_index, name, strategist, _ in selected_policies
            if isinstance(strategist, LLMStrategist)
        ]
        if not llm_policies:
            parser.error("--dry-run requires at least one llm:<alias> policy")
        estimated_calls = len(scenarios) * args.episodes_per_scenario * 16
        for policy_index, name, strategist in llm_policies:
            prompt = first_prompt_for(
                strategist,
                scenario=scenarios[0],
                seed=policy_seed(seed, policy_index, args.paired_seeds),
                simulator_config=simulator_config,
            )
            print(f"=== {name} ({strategist.model}) ===")
            print(prompt, end="" if prompt.endswith("\n") else "\n")
            print(f"Estimated calls: {estimated_calls}")
        return
    first_policy_gains: list[float] | None = None
    report = {
        "label": "simulated",
        "split": args.split,
        "seed": seed,
        "episodes_per_scenario": args.episodes_per_scenario,
        "simulator_config": simulator_config,
        "paired_seeds": args.paired_seeds,
        "workers": args.workers,
        "policies": {},
    }
    for policy_index, name, strategist, strategist_name in selected_policies:
        results = simulate_policy(
            episodes=len(scenarios) * args.episodes_per_scenario,
            split=args.split,
            scenarios=scenarios,
            seed=policy_seed(seed, policy_index, args.paired_seeds),
            strategist_name=strategist_name,
            strategist=strategist,
            run_id=f"evaluation-{args.split}-{policy_index}",
            simulator_config=simulator_config,
            workers=args.workers,
        )
        policy_report = {
            "label": "simulated",
            "overall": summarize(results, seed + policy_index),
            "profiles": {},
        }
        if isinstance(strategist, LLMStrategist):
            policy_report["llm_usage"] = strategist.usage_summary()
        if args.paired_seeds and first_policy_gains is not None:
            gain_differences = [
                result.gain - first_gain
                for result, first_gain in zip(results, first_policy_gains, strict=True)
            ]
            policy_report["paired_gain_difference_vs_first"] = {
                "mean": sum(gain_differences) / len(gain_differences),
                "ci_95": bootstrap_mean_interval(gain_differences, seed + policy_index),
            }
        elif first_policy_gains is None:
            first_policy_gains = [result.gain for result in results]
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
        "Mean turns | Mean reward | Pace frequency |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for policy, values in report["policies"].items():
        groups = [("overall", values["overall"]), *values["profiles"].items()]
        for profile, summary in groups:
            low, high = summary["mean_gain_ci_95"]
            lines.append(
                f"| {policy} | {profile} | {summary['mean_gain']:.4f} "
                f"({low:.4f}, {high:.4f}) | {summary['disclosure_rate']:.3f} "
                f"| {summary['dropout_rate']:.3f} | {summary['mean_turns']:.2f} "
                f"| {summary['mean_total_reward']:.4f} | "
                + ", ".join(
                    f"{pace}={frequency:.3f}"
                    for pace, frequency in summary["pace_frequency"].items()
                )
                + " |"
            )
    llm_policies = [
        (policy, values["llm_usage"])
        for policy, values in report["policies"].items()
        if "llm_usage" in values
    ]
    if llm_policies:
        lines.extend(
            [
                "",
                "## LLM usage",
                "",
                "| Policy | Model | Calls | Cache hits | Fallback rate | Prompt tokens | "
                "Completion tokens |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy, usage in llm_policies:
            lines.append(
                f"| {policy} | {usage['model']} | {usage['calls']} | "
                f"{usage['cache_hits']} | {usage['fallback_rate']:.4f} | "
                f"{usage['prompt_tokens']} | {usage['completion_tokens']} |"
            )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"reports: {output_dir}")


if __name__ == "__main__":
    main()
