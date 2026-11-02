from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from packages.learning.episode import ROOT, simulate_episodes, write_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/smoke.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strategist", choices=("rules", "random", "always_hint"), default="rules")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    episodes = int(args.episodes if args.episodes is not None else config.get("episodes", 100))
    seed = int(args.seed if args.seed is not None else config.get("seed", 20250115))
    split = str(config.get("split", "train"))
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_id = f"{config_path.stem}_{args.strategist}_seed{seed}_{episodes}"
    output_dir = (
        Path(args.output_dir) if args.output_dir else ROOT / "experiments/learning/runs" / run_id
    )
    results = simulate_episodes(
        episodes=episodes,
        split=split,
        seed=seed,
        strategist_name=args.strategist,
        run_id=run_id,
    )
    manifest = {
        "run_id": run_id,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": config_hash,
        "seed": seed,
        "episodes_requested": episodes,
        "split": split,
        "strategist": args.strategist,
        "navigator": "rules",
        "policy_versions": {
            "navigator": "rules_navigator_v1",
            "strategist": args.strategist,
        },
        "simulator_config": config.get(
            "simulator_config", "configs/learning/simulator_proto_v1.yaml"
        ),
        "reward_config": config.get("reward_config", "configs/learning/reward_v1.yaml"),
        "graph_version": config.get("graph_version", "graph_proto_v1"),
        "content_version": config.get("content_version", "content_proto_v1"),
        "feature_version": "features_v1",
        "response_artifact": "response_stub_v1",
        "horizon": min(int(config.get("horizon", 16)), 16),
    }
    summary = write_run(results, output_dir, manifest=manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run directory: {output_dir}")


if __name__ == "__main__":
    main()
