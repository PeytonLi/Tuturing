from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from packages.learning.episode import simulate_episodes
from packages.policies.network import (
    ACTION_SPEC_VERSION,
    FEATURE_SPEC_VERSION,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)

ROOT = Path(__file__).resolve().parents[2]


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
    torch.manual_seed(seed)
    results = simulate_episodes(
        episodes=episodes,
        split="train",
        seed=seed,
        strategist_name="rules",
        run_id=f"{config['run_name']}-seed{seed}",
    )
    holdout_count = min(episodes - 1, max(1, round(episodes * config["holdout_fraction"])))
    order = list(range(episodes))
    random.Random(seed).shuffle(order)
    holdout_ids = {results[index].episode_id for index in order[:holdout_count]}
    train_samples = []
    holdout_samples = []
    gamma = float(config["gamma"])
    for result in results:
        transitions = result.strategist_transitions
        targets = returns_for_result(result, gamma)
        destination = holdout_samples if result.episode_id in holdout_ids else train_samples
        destination.extend(
            (item["observation"], item["action_mask"], item["action_id"], target)
            for item, target in zip(transitions, targets, strict=True)
        )
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
    }
    save_checkpoint(model, run_dir / "bc.pt", metadata)
    metrics = {
        "seed": seed,
        "episodes": episodes,
        "train_transitions": len(train_samples),
        "holdout_transitions": len(holdout_samples),
        "train_action_agreement": action_agreement(model, train_samples),
        "holdout_action_agreement": action_agreement(model, holdout_samples),
        "epoch_losses": epoch_losses,
        "checkpoint": str(run_dir / "bc.pt"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
