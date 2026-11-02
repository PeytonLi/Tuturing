from __future__ import annotations

import argparse
import copy
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from packages.learning.episode import load_scenarios, simulate_episodes
from packages.policies.network import (
    ACTION_SPEC_VERSION,
    FEATURE_SPEC_VERSION,
    NetworkStrategist,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from packages.policies.rules import RandomStrategist, RulesStrategist
from packages.simulator.student import DEFAULT_SIMULATOR_CONFIG

ROOT = Path(__file__).resolve().parents[2]


class DaggerStrategist:
    def __init__(
        self,
        behavior_policy: object,
        teacher_prob: float,
        seed: int,
        *,
        record_label_inputs: bool = False,
    ) -> None:
        self.behavior_policy = behavior_policy
        self.teacher_prob = teacher_prob
        self.rng = random.Random(seed)
        self.teacher = RulesStrategist()
        self.labels: list[int] = []
        self.actions: list[int] = []
        self.label_inputs: list[tuple[list[float], list[bool], dict]] = []
        self.record_label_inputs = record_label_inputs

    def act(self, observation: list[float], mask: list[bool], context: dict) -> int:
        label = self.teacher.act(observation, mask, context)
        self.labels.append(label)
        if self.record_label_inputs:
            self.label_inputs.append(
                (list(observation), list(mask), copy.deepcopy(context))
            )
        action = (
            label
            if self.rng.random() < self.teacher_prob
            else self.behavior_policy.act(observation, mask, context)
        )
        self.actions.append(action)
        return action


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _normalize_dagger_mixes(config: dict) -> list[dict] | None:
    if "dagger_mixes" not in config:
        return None
    raw_mixes = config["dagger_mixes"]
    if not isinstance(raw_mixes, list) or not raw_mixes:
        raise ValueError("dagger_mixes must be a non-empty list")

    mixes = []
    for index, raw_mix in enumerate(raw_mixes):
        if not isinstance(raw_mix, dict):
            raise ValueError(f"dagger_mixes[{index}] must be a mapping")
        behavior = raw_mix.get("behavior")
        if not isinstance(behavior, str):
            raise ValueError(f"dagger_mixes[{index}].behavior must be a string")
        if behavior in {"rules", "random"}:
            normalized_behavior = behavior
        elif behavior.startswith("checkpoint:") and behavior.partition(":")[2]:
            checkpoint = Path(behavior.partition(":")[2])
            if not checkpoint.is_absolute():
                checkpoint = ROOT / checkpoint
            normalized_behavior = f"checkpoint:{checkpoint.resolve()}"
        else:
            raise ValueError(f"unsupported DAgger behavior: {behavior!r}")
        try:
            episodes = int(raw_mix["episodes"])
            teacher_prob = float(raw_mix["teacher_prob"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"dagger_mixes[{index}] requires episodes and teacher_prob"
            ) from error
        if episodes < 1:
            raise ValueError(f"dagger_mixes[{index}].episodes must be positive")
        if not 0.0 <= teacher_prob <= 1.0:
            raise ValueError(
                f"dagger_mixes[{index}].teacher_prob must be between 0 and 1"
            )
        mixes.append(
            {
                "behavior": normalized_behavior,
                "episodes": episodes,
                "teacher_prob": teacher_prob,
            }
        )
    return mixes


def _behavior_policy(behavior: str, episode_seed: int, checkpoint_policies: dict):
    if behavior == "rules":
        return RulesStrategist()
    if behavior == "random":
        return RandomStrategist(episode_seed)
    checkpoint_path = behavior.partition(":")[2]
    if checkpoint_path not in checkpoint_policies:
        checkpoint_policies[checkpoint_path] = NetworkStrategist(
            checkpoint_path, greedy=True
        )
    return checkpoint_policies[checkpoint_path]


def _collect_dagger_episodes(
    mixes: list[dict],
    *,
    seed: int,
    run_name: str,
    simulator_config: str,
) -> list[dict]:
    train_scenarios = [
        scenario for scenario in load_scenarios() if scenario["split"] == "train"
    ]
    if not train_scenarios:
        raise ValueError("no scenarios available for split train")
    checkpoint_policies = {}
    collected = []
    for mix_index, mix in enumerate(mixes):
        for episode_index in range(mix["episodes"]):
            episode_seed = seed + 1_000_000 * mix_index + episode_index
            behavior_policy = _behavior_policy(
                mix["behavior"], episode_seed, checkpoint_policies
            )
            strategist = DaggerStrategist(
                behavior_policy,
                mix["teacher_prob"],
                seed=episode_seed + mix_index,
            )
            scenario = train_scenarios[episode_index % len(train_scenarios)]
            result = simulate_episodes(
                episodes=1,
                split="train",
                seed=episode_seed,
                strategist=strategist,
                scenarios=[scenario],
                run_id=f"{run_name}-seed{seed}-mix{mix_index}-episode{episode_index}",
                simulator_config=simulator_config,
            )[0]
            assert len(strategist.labels) == len(result.strategist_transitions)
            collected.append(
                {
                    "mix_index": mix_index,
                    "result": result,
                    "labels": strategist.labels,
                }
            )
    return collected


def _run_command() -> str:
    return shlex.join(["python", "-m", "tools.learning.train_bc", *sys.argv[1:]])


def discounted_episode_returns(transitions: list[dict], gamma: float) -> list[float]:
    returns = [0.0] * len(transitions)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        running = float(transitions[index]["reward"]) + gamma * running
        returns[index] = running
    return returns


def returns_for_result(result, gamma: float) -> list[float]:
    transitions = [dict(item) for item in result.strategist_transitions]
    if transitions and not transitions[-1]["terminated"] and any(
        item["terminated"] and item["duration"] == 0 for item in result.navigator_transitions
    ):
        transitions[-1]["reward"] += sum(result.terminal_components.values())
    return discounted_episode_returns(transitions, gamma)


def _batch_losses(model, observations, masks, actions, value_targets):
    logits, values = model(torch.as_tensor(observations, dtype=torch.float32))
    logits = model.mask_logits(logits, torch.as_tensor(masks, dtype=torch.bool))
    targets = torch.as_tensor(actions, dtype=torch.long)
    value_targets = torch.as_tensor(value_targets, dtype=torch.float32)
    policy_loss = torch.nn.functional.cross_entropy(logits, targets)
    value_loss = torch.nn.functional.mse_loss(values, value_targets)
    return policy_loss, value_loss


def train_epoch(model, optimizer, observations, masks, actions, value_targets, batch_size, rng):
    order = list(range(len(actions)))
    rng.shuffle(order)
    total = 0.0
    count = 0
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        batch = (
            [observations[i] for i in indexes],
            [masks[i] for i in indexes],
            [actions[i] for i in indexes],
            [value_targets[i] for i in indexes],
        )
        policy_loss, value_loss = _batch_losses(model, *batch)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(indexes)
        count += len(indexes)
    return total / max(1, count)


def action_agreement(model, samples) -> float:
    if not samples:
        return 0.0
    correct = 0
    with torch.no_grad():
        for observation, mask, action, _ in samples:
            predicted, _, _ = model.act(observation, mask, greedy=True)
            correct += int(predicted == action)
    return correct / len(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/bc.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    episodes = int(args.episodes if args.episodes is not None else config["episodes"])
    simulator_config = config.get("simulator_config", DEFAULT_SIMULATOR_CONFIG)
    dagger_mixes = _normalize_dagger_mixes(config)
    torch.manual_seed(seed)
    train_samples = []
    holdout_samples = []
    gamma = float(config["gamma"])
    dagger_mix_metrics = None
    if dagger_mixes is None:
        results = simulate_episodes(
            episodes=episodes,
            split="train",
            seed=seed,
            strategist_name="rules",
            run_id=f"{config['run_name']}-seed{seed}",
            simulator_config=simulator_config,
        )
        holdout_count = min(
            episodes - 1, max(1, round(episodes * config["holdout_fraction"]))
        )
        order = list(range(episodes))
        random.Random(seed).shuffle(order)
        holdout_ids = {results[index].episode_id for index in order[:holdout_count]}
        for result in results:
            transitions = result.strategist_transitions
            targets = returns_for_result(result, gamma)
            destination = (
                holdout_samples if result.episode_id in holdout_ids else train_samples
            )
            destination.extend(
                (item["observation"], item["action_mask"], item["action_id"], target)
                for item, target in zip(transitions, targets, strict=True)
            )
        episodes_run = episodes
    else:
        collected = _collect_dagger_episodes(
            dagger_mixes,
            seed=seed,
            run_name=config["run_name"],
            simulator_config=simulator_config,
        )
        episodes_run = len(collected)
        holdout_count = min(
            episodes_run - 1,
            max(1, round(episodes_run * config["holdout_fraction"])),
        )
        order = list(range(episodes_run))
        random.Random(seed).shuffle(order)
        holdout_indexes = set(order[:holdout_count])
        per_mix_train_samples = [[] for _ in dagger_mixes]
        per_mix_holdout_samples = [[] for _ in dagger_mixes]
        per_mix_transition_counts = [0 for _ in dagger_mixes]
        for episode_index, row in enumerate(collected):
            result = row["result"]
            transitions = result.strategist_transitions
            labels = row["labels"]
            targets = returns_for_result(result, gamma)
            assert len(labels) == len(transitions)
            destination = (
                holdout_samples if episode_index in holdout_indexes else train_samples
            )
            mix_destination = (
                per_mix_holdout_samples[row["mix_index"]]
                if episode_index in holdout_indexes
                else per_mix_train_samples[row["mix_index"]]
            )
            samples = [
                (item["observation"], item["action_mask"], label, target)
                for item, label, target in zip(
                    transitions, labels, targets, strict=True
                )
            ]
            destination.extend(samples)
            mix_destination.extend(samples)
            per_mix_transition_counts[row["mix_index"]] += len(transitions)
        dagger_mix_metrics = {
            "train_transition_counts": [
                len(samples) for samples in per_mix_train_samples
            ],
            "holdout_transition_counts": [
                len(samples) for samples in per_mix_holdout_samples
            ],
            "mixes": None,
        }
    if not train_samples:
        raise ValueError("BC training split has no transitions")
    model = StrategistActorCritic(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = random.Random(seed)
    epoch_losses = []
    for _ in range(int(config["epochs"])):
        epoch_losses.append(
            train_epoch(
                model,
                optimizer,
                [row[0] for row in train_samples],
                [row[1] for row in train_samples],
                [row[2] for row in train_samples],
                [row[3] for row in train_samples],
                int(config["batch_size"]),
                rng,
            )
        )
    run_dir = ROOT / "experiments/learning/runs" / f"{config['run_name']}_seed{seed}"
    metadata = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "seed": seed,
        "init": "scratch",
        "policy": "behavior_cloning",
        "simulator_config": simulator_config,
    }
    save_checkpoint(model, run_dir / "bc.pt", metadata)
    resolved_config = dict(config)
    resolved_config.update(
        {
            "seed": seed,
            "simulator_config": simulator_config,
            "split": "train",
        }
    )
    if dagger_mixes is not None:
        resolved_config["dagger_mixes"] = dagger_mixes
        resolved_config["effective_episodes"] = episodes_run
    else:
        resolved_config["episodes"] = episodes
    metrics = {
        "seed": seed,
        "episodes": episodes_run,
        "train_transitions": len(train_samples),
        "holdout_transitions": len(holdout_samples),
        "train_action_agreement": action_agreement(model, train_samples),
        "holdout_action_agreement": action_agreement(model, holdout_samples),
        "epoch_losses": epoch_losses,
        "checkpoint": str(run_dir / "bc.pt"),
        "simulator_config": simulator_config,
        "command": _run_command(),
        "git_sha": _git_sha(),
        "resolved_config": resolved_config,
    }
    if dagger_mix_metrics is not None:
        dagger_mix_metrics["mixes"] = [
            {
                **mix,
                "mix_index": index,
                "transition_count": per_mix_transition_counts[index],
                "train_transition_count": dagger_mix_metrics[
                    "train_transition_counts"
                ][index],
                "holdout_transition_count": dagger_mix_metrics[
                    "holdout_transition_counts"
                ][index],
                "holdout_action_agreement": (
                    action_agreement(model, per_mix_holdout_samples[index])
                    if per_mix_holdout_samples[index]
                    else None
                ),
            }
            for index, mix in enumerate(dagger_mixes)
        ]
        metrics["dagger_mix_metrics"] = dagger_mix_metrics
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
