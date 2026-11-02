"""Behavior-clone an LLM Strategist: label train-split simulator turns with the LLM, then BC."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from packages.learning.episode import HORIZON, load_scenarios, run_episode
from packages.policies.llm import CallLimitExceeded
from packages.policies.network import (
    ACTION_SPEC_VERSION,
    FEATURE_SPEC_VERSION,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from tools.learning.evaluate import load_llm_aliases, policy_from_name
from tools.learning.train_bc import action_agreement, returns_for_result, train_epoch

ROOT = Path(__file__).resolve().parents[2]
INPUT_USD_PER_M = 2.0
OUTPUT_USD_PER_M = 10.0


class _Recorder:
    def __init__(self, teacher, local: threading.local) -> None:
        self.teacher = teacher
        self.local = local
        self.calls: list[tuple[int, bool]] = []

    def act(self, observation, mask, context):
        self.local.fell_back = False
        action_id = self.teacher.act(observation, mask, context)
        self.calls.append((action_id, bool(self.local.fell_back)))
        return action_id


def collect(config: dict, seed: int, episodes: int) -> tuple[list[dict], dict]:
    teacher, name = policy_from_name(
        config["teacher"], llm_aliases=load_llm_aliases(), max_calls=int(config["max_calls"])
    )
    local = threading.local()
    original_fallback = teacher._fallback_action

    def flagged_fallback(*args, **kwargs):
        local.fell_back = True
        return original_fallback(*args, **kwargs)

    teacher._fallback_action = flagged_fallback
    scenarios = [
        s for s in load_scenarios(config["scenarios"]) if s["split"] == config["split"]
    ]
    gamma = float(config["gamma"])

    def run(index: int):
        scenario = scenarios[index % len(scenarios)]
        recorder = _Recorder(teacher, local)
        try:
            result = run_episode(
                scenario,
                run_id=f"{config['run_name']}-seed{seed}",
                episode_seed=seed + index,
                strategist_name=name,
                strategist=recorder,
                horizon=min(int(scenario["teaching_horizon"]), HORIZON),
                simulator_config=config["simulator_config"],
            )
        except CallLimitExceeded:
            return index, None, "call_limit"
        transitions = result.strategist_transitions
        if len(transitions) != len(recorder.calls) or any(
            item["action_id"] != action for item, (action, _) in zip(
                transitions, recorder.calls, strict=True
            )
        ):
            return index, None, "misaligned"
        targets = returns_for_result(result, gamma)
        rows = [
            {
                "episode_index": index,
                "episode_id": result.episode_id,
                "scenario_id": result.scenario_id,
                "profile_id": result.profile_id,
                "observation": item["observation"],
                "action_mask": item["action_mask"],
                "action_id": item["action_id"],
                "value_target": target,
                "fell_back": fell_back,
            }
            for item, target, (_, fell_back) in zip(
                transitions, targets, recorder.calls, strict=True
            )
        ]
        return index, rows, "ok"

    started = time.time()
    rows: list[dict] = []
    status = {"ok": 0, "call_limit": 0, "misaligned": 0}
    with ThreadPoolExecutor(max_workers=int(config["workers"])) as executor:
        for _, episode_rows, outcome in executor.map(run, range(episodes)):
            status[outcome] += 1
            if episode_rows:
                rows.extend(episode_rows)
    usage = teacher.usage_summary()
    usage["cost_usd"] = round(
        usage["prompt_tokens"] * INPUT_USD_PER_M / 1e6
        + usage["completion_tokens"] * OUTPUT_USD_PER_M / 1e6,
        3,
    )
    usage["wall_seconds"] = round(time.time() - started, 1)
    usage["episodes"] = status
    return rows, usage


def train(config: dict, seed: int, rows: list[dict]) -> dict:
    kept = [row for row in rows if not row["fell_back"]]
    episode_ids = sorted({row["episode_id"] for row in kept})
    holdout_count = min(
        len(episode_ids) - 1, max(1, round(len(episode_ids) * config["holdout_fraction"]))
    )
    order = list(range(len(episode_ids)))
    random.Random(seed).shuffle(order)
    holdout_ids = {episode_ids[index] for index in order[:holdout_count]}
    train_samples, holdout_samples = [], []
    for row in kept:
        destination = holdout_samples if row["episode_id"] in holdout_ids else train_samples
        destination.append(
            (row["observation"], row["action_mask"], row["action_id"], row["value_target"])
        )
    if not train_samples:
        raise ValueError("distillation split has no transitions")
    torch.manual_seed(seed)
    model = StrategistActorCritic(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = random.Random(seed)
    epoch_losses = [
        train_epoch(
            model,
            optimizer,
            [r[0] for r in train_samples],
            [r[1] for r in train_samples],
            [r[2] for r in train_samples],
            [r[3] for r in train_samples],
            int(config["batch_size"]),
            rng,
        )
        for _ in range(int(config["epochs"]))
    ]
    run_dir = ROOT / "experiments/learning/runs" / f"{config['run_name']}_seed{seed}"
    save_checkpoint(
        model,
        run_dir / "bc.pt",
        {
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "action_spec_version": ACTION_SPEC_VERSION,
            "seed": seed,
            "init": "scratch",
            "policy": "behavior_cloning",
            "teacher": config["teacher"],
            "simulator_config": config["simulator_config"],
        },
    )
    return {
        "seed": seed,
        "labeled_turns": len(rows),
        "fallback_turns_excluded": len(rows) - len(kept),
        "train_transitions": len(train_samples),
        "holdout_transitions": len(holdout_samples),
        "train_action_agreement": action_agreement(model, train_samples),
        "holdout_action_agreement": action_agreement(model, holdout_samples),
        "epoch_losses": epoch_losses,
        "checkpoint": str(run_dir / "bc.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/bc_gpt_distill.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--labels", help="reuse a labels.jsonl instead of calling the LLM")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    episodes = int(args.episodes if args.episodes is not None else config["episodes"])
    run_dir = ROOT / "experiments/learning/runs" / f"{config['run_name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    usage = None
    if args.labels:
        lines = Path(args.labels).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines]
    else:
        rows, usage = collect(config, seed, episodes)
        with (run_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        (run_dir / "teacher_usage.json").write_text(
            json.dumps(usage, indent=2) + "\n", encoding="utf-8"
        )
    metrics = train(config, seed, rows)
    metrics["teacher_usage"] = usage
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "epoch_losses"}, indent=2))


if __name__ == "__main__":
    main()
