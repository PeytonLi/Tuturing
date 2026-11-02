from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import yaml

from packages.learning.episode import simulate_episodes
from packages.policies.network import (
    ACTION_SPEC_VERSION,
    FEATURE_SPEC_VERSION,
    NetworkStrategist,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from packages.policies.returns import duration_aware_gae

ROOT = Path(__file__).resolve().parents[2]
METRIC_FIELDS = (
    "update",
    "sessions",
    "mean_episode_reward",
    "mean_gain",
    "dropout_rate",
    "disclosure_rate",
    "entropy",
    "approx_kl",
    "clip_frac",
    "value_loss",
)


def episode_reward(result) -> float:
    reward = sum(float(item["reward"]) for item in result.strategist_transitions)
    if result.strategist_transitions and not result.strategist_transitions[-1]["terminated"]:
        if any(
            item["terminated"] and item["duration"] == 0 for item in result.navigator_transitions
        ):
            reward += sum(result.terminal_components.values())
    return reward


def _ppo_update(model, optimizer, samples, config, rng):
    gamma = float(config["gamma"])
    lam = float(config["lam"])
    observations = []
    masks = []
    actions = []
    actor_eligible = []
    old_log_probs = []
    old_values = []
    next_values = []
    rewards = []
    terminated = []
    truncated = []
    continuation = []
    episode_stats = []
    for result in samples:
        transitions = result.strategist_transitions
        if not transitions:
            continue
        episode_rewards = [float(item["reward"]) for item in transitions]
        episode_terminated = [bool(item["terminated"]) for item in transitions]
        if not episode_terminated[-1] and any(
            item["terminated"] and item["duration"] == 0 for item in result.navigator_transitions
        ):
            episode_rewards[-1] += sum(result.terminal_components.values())
            episode_terminated[-1] = True
        observations.extend(item["observation"] for item in transitions)
        masks.extend(item["action_mask"] for item in transitions)
        actions.extend(item["action_id"] for item in transitions)
        actor_eligible.extend(bool(item["actor_eligible"]) for item in transitions)
        rewards.extend(episode_rewards)
        terminated.extend(episode_terminated)
        truncated.extend([False] * (len(transitions) - 1) + [not episode_terminated[-1]])
        continuation.extend([True] * (len(transitions) - 1) + [False])
        episode_stats.append(result)
    if not observations:
        raise ValueError("PPO collection produced no strategist transitions")
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    mask_tensor = torch.as_tensor(masks, dtype=torch.bool)
    action_tensor = torch.as_tensor(actions, dtype=torch.long)
    with torch.no_grad():
        logits, value_tensor = model(obs_tensor)
        distribution = torch.distributions.Categorical(
            logits=model.mask_logits(logits, mask_tensor)
        )
        old_log_probs = distribution.log_prob(action_tensor).tolist()
        old_values = value_tensor.tolist()
        following_obs = torch.as_tensor(
            [
                item["next_observation"]
                for result in episode_stats
                for item in result.strategist_transitions
            ],
            dtype=torch.float32,
        )
        _, following_values = model(following_obs)
        next_values = following_values.tolist()
    advantages = []
    returns = []
    offset = 0
    for result in episode_stats:
        length = len(result.strategist_transitions)
        output = duration_aware_gae(
            rewards[offset : offset + length],
            old_values[offset : offset + length],
            [1] * length,
            next_values[offset : offset + length],
            terminated[offset : offset + length],
            truncated[offset : offset + length],
            continuation[offset : offset + length],
            gamma=gamma,
            lam=lam,
        )
        advantages.extend(output.advantages)
        returns.extend(output.returns)
        offset += length
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32)
    advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
        advantage_tensor.std(unbiased=False) + 1e-8
    )
    old_log_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    actor_tensor = torch.as_tensor(actor_eligible, dtype=torch.bool)
    order = list(range(len(actions)))
    rng.shuffle(order)
    records = []
    for _ in range(int(config["epochs"])):
        for start in range(0, len(order), int(config["minibatch_size"])):
            indexes = order[start : start + int(config["minibatch_size"])]
            log_prob, entropy, values = model.evaluate_actions(
                obs_tensor[indexes], mask_tensor[indexes], action_tensor[indexes]
            )
            ratio = torch.exp(log_prob - old_log_tensor[indexes])
            adv = advantage_tensor[indexes]
            clip = float(config["clip"])
            actor_rows = actor_tensor[indexes]
            if actor_rows.any():
                actor_ratio = ratio[actor_rows]
                actor_adv = adv[actor_rows]
                actor_log = log_prob[actor_rows]
                actor_old_log = old_log_tensor[indexes][actor_rows]
                actor_entropy = entropy[actor_rows]
                policy_loss = -torch.minimum(
                    actor_ratio * actor_adv,
                    torch.clamp(actor_ratio, 1 - clip, 1 + clip) * actor_adv,
                ).mean()
                entropy_mean = actor_entropy.mean()
                approx_kl = (actor_old_log - actor_log).mean()
                clip_frac = ((actor_ratio - 1).abs() > clip).float().mean()
            else:
                policy_loss = torch.zeros((), dtype=torch.float32)
                entropy_mean = torch.zeros((), dtype=torch.float32)
                approx_kl = torch.zeros((), dtype=torch.float32)
                clip_frac = torch.zeros((), dtype=torch.float32)
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor[indexes])
            loss = (
                policy_loss
                + float(config["value_coef"]) * value_loss
                - float(config["entropy_coef"]) * entropy_mean
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            records.append(
                (
                    float(entropy_mean.detach()),
                    float(approx_kl.detach()),
                    float(clip_frac.detach()),
                    float(value_loss.detach()),
                )
            )
    means = [sum(row[i] for row in records) / len(records) for i in range(4)]
    disclosure_count = sum(
        int(item["reward_components"]["disclosure_penalty"] < 0)
        for result in episode_stats
        for item in result.strategist_transitions
    )
    turn_count = sum(len(result.strategist_transitions) for result in episode_stats)
    return {
        "mean_episode_reward": sum(episode_reward(item) for item in episode_stats)
        / len(episode_stats),
        "mean_gain": sum(item.gain for item in episode_stats) / len(episode_stats),
        "dropout_rate": sum(item.dropout for item in episode_stats) / len(episode_stats),
        "disclosure_rate": disclosure_count / max(1, turn_count),
        "entropy": means[0],
        "approx_kl": means[1],
        "clip_frac": means[2],
        "value_loss": means[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/ppo_strategist.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init")
    parser.add_argument("--sessions", type=int)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    sessions = int(args.sessions if args.sessions is not None else config["sessions"])
    torch.manual_seed(seed)
    if args.init:
        model = NetworkStrategist(args.init).model
        init = str(Path(args.init))
    else:
        model = StrategistActorCritic(seed=seed)
        init = "scratch"
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    policy = NetworkStrategist.from_model(model, greedy=False)
    rng = random.Random(seed)
    run_name = f"{config['run_name']}_seed{seed}_{sessions}"
    run_dir = ROOT / "experiments/learning/runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    updates = (sessions + int(config["episodes_per_update"]) - 1) // int(
        config["episodes_per_update"]
    )
    completed = 0
    next_checkpoint = 500
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for update in range(1, updates + 1):
            count = min(
                int(config["episodes_per_update"]),
                sessions - completed,
                next_checkpoint - completed,
            )
            torch.manual_seed(seed + completed)
            results = simulate_episodes(
                episodes=count,
                split="train",
                seed=seed + completed,
                strategist=policy,
                strategist_name="network",
                run_id=f"{run_name}-update{update}",
            )
            metrics = _ppo_update(model, optimizer, results, config, rng)
            completed += count
            row = {"update": update, "sessions": completed, **metrics}
            writer.writerow(row)
            handle.flush()
            metadata = {
                "feature_spec_version": FEATURE_SPEC_VERSION,
                "action_spec_version": ACTION_SPEC_VERSION,
                "seed": seed,
                "init": init,
                "sessions": completed,
                "policy": "ppo_strategist",
            }
            if completed % 500 == 0 or completed == sessions:
                save_checkpoint(model, run_dir / f"ppo_{completed}.pt", metadata)
                save_checkpoint(model, run_dir / "ppo.pt", metadata)
                next_checkpoint += 500
            print(json.dumps(row, sort_keys=True))
    print(f"checkpoint: {run_dir / 'ppo.pt'}")


if __name__ == "__main__":
    main()
