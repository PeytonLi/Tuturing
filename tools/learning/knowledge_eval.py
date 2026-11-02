"""Run policies on the hidden-knowledge test set and score how they adapt (simulated only)."""

import argparse
import json
from pathlib import Path

from packages.learning.episode import load_scenarios
from tools.learning.evaluate import (
    LLMStrategist,
    load_llm_aliases,
    policy_from_name,
    simulate_policy,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = "data/scenarios/knowledge_v1.yaml"
SIM_CONFIG = "configs/learning/simulator_knowledge_v1.yaml"
TEACHING = ("HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE")


def episode_metrics(result) -> dict:
    diags = result.private_diagnostics
    first_active = set(diags[0]["private_active_misconceptions"]) if diags else set()
    final_active = set(diags[-1]["private_active_misconceptions"]) if diags else set()
    fixed_turn = next(
        (d["turn"] for d in diags if not d["private_active_misconceptions"]), None
    )
    moves = result.move_histogram
    total = max(1, sum(moves.values()))
    return {
        "scenario_id": result.scenario_id,
        "profile_id": result.profile_id,
        "turns": result.turns,
        "gain": result.gain,
        "post_score": result.post_score,
        "dropout": result.dropout,
        "had_misconception": bool(first_active),
        "misconception_fixed": bool(first_active) and not final_active,
        "turn_fixed": fixed_turn if first_active else None,
        "teaching_share": sum(moves.get(m, 0) for m in TEACHING) / total,
        "probe_share": moves.get("PROBE", 0) / total,
        "reviewed_prerequisite": "REVIEW_PREREQUISITE"
        in json.dumps(result.navigator_transitions, default=str),
        "moves": moves,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--episodes-per-scenario", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--llm-base-url-override")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scenarios = load_scenarios(SCENARIOS)
    aliases = load_llm_aliases() if args.policy.startswith("llm:") else None
    strategist, strategist_name = policy_from_name(
        args.policy,
        llm_aliases=aliases,
        max_calls=args.max_calls,
        llm_base_url_override=args.llm_base_url_override,
        dry_run=False,
    )
    results = simulate_policy(
        episodes=len(scenarios) * args.episodes_per_scenario,
        split="validation",
        scenarios=scenarios,
        seed=args.seed,
        strategist_name=strategist_name,
        strategist=strategist,
        run_id="knowledge-eval",
        simulator_config=SIM_CONFIG,
        workers=args.workers,
    )
    report = {
        "label": "simulated",
        "policy": args.policy,
        "seed": args.seed,
        "episodes": [episode_metrics(r) for r in results],
    }
    if isinstance(strategist, LLMStrategist):
        report["llm_usage"] = strategist.usage_summary()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report.get("llm_usage", {}), default=str))


if __name__ == "__main__":
    main()
