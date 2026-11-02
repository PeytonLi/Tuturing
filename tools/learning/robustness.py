"""Paired SIMULATED validation checks, never training or checkpoint selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.learning.actions import MOVES
from packages.policies import rules
from tools.learning.evaluate import (
    ROOT,
    bootstrap_mean_interval,
    load_scenarios,
    policy_from_name,
    simulate_episodes,
    summarize,
)

DEFAULT_CONFIGS = (
    "configs/learning/simulator_proto_v2.yaml",
    "configs/learning/simulator_proto_alt.yaml",
)


def load_policies(names: list[str]) -> tuple[dict, list[str]]:
    policies = {}
    skipped = []
    for name in dict.fromkeys(["rules", *names]):
        try:
            policies[name] = policy_from_name(name)
        except ValueError:
            if name != "always_worked_example":
                raise
            policy_class = getattr(rules, "AlwaysWorkedExampleStrategist", None)
            if policy_class is None:
                skipped.append(name)
            else:
                policies[name] = (policy_class(), name)
    return policies, skipped


def paired_summary(results, reference, seed: int) -> dict:
    if [row.episode_id for row in results] != [row.episode_id for row in reference]:
        raise ValueError("paired evaluation requires matching episode IDs and order")
    summary = summarize(results, seed)
    differences = [
        row.gain - baseline.gain for row, baseline in zip(results, reference, strict=True)
    ]
    summary.update(
        paired_difference_vs_rules=sum(differences) / len(differences),
        paired_difference_vs_rules_ci_95=bootstrap_mean_interval(differences, seed),
        disclosure_count=sum(
            transition["reward_components"]["disclosure_penalty"] < 0
            for row in results
            for transition in row.strategist_transitions
        ),
    )
    summary["move_frequency"] = {
        move: summary["move_frequency"].get(move, 0.0) for move in MOVES
    }
    return summary


def evaluate_robustness(
    policies: dict, simulator_configs: list[str], *, episodes: int, seed: int
) -> dict:
    scenarios = [row for row in load_scenarios() if row["split"] == "validation"]
    report = {
        "label": "SIMULATED",
        "purpose": "robustness only; never training or checkpoint selection",
        "split": "validation",
        "seed": seed,
        "episodes": episodes,
        "bootstrap_resamples": 1000,
        "paired_comparison": "policy minus rules, matched scenario and episode seed",
        "disclosure_count_definition": "turns with a negative disclosure_penalty",
        "episode_schedule": [],
        "simulator_configs": {},
    }
    for simulator_config in dict.fromkeys(simulator_configs):
        config_report = {"policies": {}}
        reference = None
        for name, (strategist, strategist_name) in policies.items():
            results = simulate_episodes(
                episodes=episodes,
                split="validation",
                scenarios=scenarios,
                seed=seed,
                strategist_name=strategist_name,
                strategist=strategist,
                run_id="robustness-validation",
                simulator_config=simulator_config,
            )
            schedule = [
                {
                    "episode_id": row.episode_id,
                    "scenario_id": row.scenario_id,
                    "profile_id": row.profile_id,
                    "seed": seed + index,
                }
                for index, row in enumerate(results)
            ]
            if not report["episode_schedule"]:
                report["episode_schedule"] = schedule
            elif report["episode_schedule"] != schedule:
                raise ValueError("episode schedule changed across policies or configs")
            if reference is None:
                if name != "rules":
                    raise ValueError("rules must be evaluated first as the paired reference")
                reference = results
            config_report["policies"][name] = {
                "overall": paired_summary(results, reference, seed),
                "profiles": {
                    profile: paired_summary(
                        [row for row in results if row.profile_id == profile],
                        [row for row in reference if row.profile_id == profile],
                        seed,
                    )
                    for profile in sorted({row.profile_id for row in results})
                },
            }
        report["simulator_configs"][simulator_config] = config_report
    return report


def format_interval(mean: float, interval: list[float]) -> str:
    return f"{mean:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def markdown_report(report: dict) -> str:
    lines = [
        "# SIMULATED alternate-simulator robustness",
        "",
        f"Validation only; {report['episodes']} episodes per policy/config, seed {report['seed']}. "
        "Robustness only, never used for training or checkpoint selection.",
        "All numbers are SIMULATED, not human learning evidence.",
        "",
        "The same ordered scenario/episode-seed schedule is reused across every policy and config. "
        "95% percentile bootstrap CIs use 1,000 episode resamples; differences resample paired "
        "policy-minus-rules gains. These CIs do not capture simulator-model uncertainty.",
        "Disclosure count is the number of turns with a negative disclosure penalty. "
        "Rules is always included as the reference, even when omitted from --policies.",
        "The alternate config halves per-move action effects (ell stays fixed), raises guess/slip "
        "by 0.05, and lowers configured initial mastery ranges by 0.10 (floor 0.05). "
        "The novice's hard-coded unrelated-concept initialization is unchanged.",
        "",
        "| Config | Policy | Mean gain [95% CI] | Paired difference vs rules [95% CI] | "
        "Dropout | Disclosures | Mean turns |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for config, values in report["simulator_configs"].items():
        for policy, groups in values["policies"].items():
            summary = groups["overall"]
            gain = format_interval(summary["mean_gain"], summary["mean_gain_ci_95"])
            difference = format_interval(
                summary["paired_difference_vs_rules"],
                summary["paired_difference_vs_rules_ci_95"],
            )
            lines.append(
                f"| {Path(config).stem} | {policy} | {gain} | {difference} | "
                f"{summary['dropout_rate']:.1%} | {summary['disclosure_count']} | "
                f"{summary['mean_turns']:.2f} |"
            )
    lines.extend([
        "", "## Per-profile gain (SIMULATED)", "",
        "| Config | Policy | Profile | Episodes | Mean gain [95% CI] |",
        "|---|---|---|---:|---:|",
    ])
    for config, values in report["simulator_configs"].items():
        for policy, groups in values["policies"].items():
            for profile, summary in groups["profiles"].items():
                gain = format_interval(summary["mean_gain"], summary["mean_gain_ci_95"])
                lines.append(
                    f"| {Path(config).stem} | {policy} | {profile} | "
                    f"{summary['episodes']} | {gain} |"
                )
    lines.extend([
        "", "## Move frequencies (SIMULATED)", "",
        "| Config | Policy | " + " | ".join(MOVES) + " |",
        "|---|---|" + "---:|" * len(MOVES),
    ])
    for config, values in report["simulator_configs"].items():
        for policy, groups in values["policies"].items():
            frequencies = groups["overall"]["move_frequency"]
            cells = " | ".join(f"{frequencies[move]:.1%}" for move in MOVES)
            lines.append(f"| {Path(config).stem} | {policy} | {cells} |")
    if report["skipped_policies"]:
        lines.extend(["", "Skipped unavailable policies: " + ", ".join(report["skipped_policies"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default="rules,random,always_hint,always_worked_example")
    parser.add_argument(
        "--checkpoint", action="append", default=[], help="repeat for each checkpoint"
    )
    parser.add_argument("--simulator-config", action="append", help="repeat for each YAML config")
    parser.add_argument("--split", choices=("validation",), default="validation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20250115)
    parser.add_argument(
        "--output-prefix", default="evaluation/learning/results/robustness_validation"
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    names = [name.strip() for name in args.policies.split(",") if name.strip()]
    names.extend(f"checkpoint:{path}" for path in args.checkpoint)
    try:
        policies, skipped = load_policies(names)
    except ValueError as error:
        parser.error(str(error))
    report = evaluate_robustness(
        policies, args.simulator_config or list(DEFAULT_CONFIGS),
        episodes=args.episodes, seed=args.seed,
    )
    report["skipped_policies"] = skipped
    markdown = markdown_report(report)
    prefix = ROOT / args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(f"{prefix}.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"reports: {prefix}.json and {prefix}.md")


if __name__ == "__main__":
    main()
