from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from packages.learning.episode import HORIZON, run_episode
from packages.policies.network import (
    ACTION_SPEC_VERSION,
    FEATURE_SPEC_VERSION,
    StrategistActorCritic,
    save_checkpoint,
    torch,
)
from packages.policies.rules import (
    KnownAwareRulesStrategist,
    RulesFastReadyStrategist,
    RulesStrategist,
)
from packages.simulator.scenarios import load_scenarios
from packages.simulator.student import DEFAULT_SIMULATOR_CONFIG

ROOT = Path(__file__).resolve().parents[2]
TEACHERS = {
    "rules": RulesStrategist,
    "rules_fastready": RulesFastReadyStrategist,
    "rules_known_aware": KnownAwareRulesStrategist,
}
DAGGER_SEED_STRIDE = 1_000_003


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


class LabelingStrategist:
    """Acts with `actor` (or the teacher when actor is None) and records the teacher's label."""

    def __init__(self, teacher, actor=None) -> None:
        self.teacher = teacher
        self.actor = actor
        self.labels: dict[int, tuple[list[float], int]] = {}
        self.policy_version = getattr(actor or teacher, "policy_version", "policy_v1")

    def act(self, observation, mask, context) -> int:
        label = int(self.teacher.act(observation, mask, context))
        self.labels[id(observation)] = (observation, label)
        if self.actor is None:
            return label
        action, _, _ = self.actor.act(observation, mask, greedy=True)
        return int(action)


_WORKER: dict = {}


def _init_worker(job: dict) -> None:
    torch.set_num_threads(1)
    _WORKER.clear()
    _WORKER.update(job)
    _WORKER["scenarios"] = [
        scenario
        for scenario in load_scenarios(
            include_pace_scenarios=job["include_pace_scenarios"],
            include_strong_scenarios=job["include_strong_scenarios"],
        )
        if scenario["split"] == "train"
    ]
    model = None
    if job.get("state_dict") is not None:
        model = StrategistActorCritic(hidden=tuple(job["hidden"]))
        model.load_state_dict(job["state_dict"])
        model.eval()
    _WORKER["model"] = model


def _rollout(index: int) -> dict:
    job = _WORKER
    scenarios = job["scenarios"]
    scenario = scenarios[index % len(scenarios)]
    strategist = LabelingStrategist(TEACHERS[job["teacher"]](), job["model"])
    with torch.no_grad():
        result = run_episode(
            scenario,
            run_id=job["run_id"],
            episode_seed=job["seed"] + index,
            strategist=strategist,
            horizon=min(int(scenario["teaching_horizon"]), HORIZON),
            simulator_config=job["simulator_config"],
        )
    targets = returns_for_result(result, job["gamma"])
    rows = []
    for item, target in zip(result.strategist_transitions, targets, strict=True):
        if not item["actor_eligible"]:
            continue
        _, label = strategist.labels[id(item["observation"])]
        rows.append((item["observation"], item["action_mask"], label, target, item["action_id"]))
    return {"episode_id": result.episode_id, "index": index, "rows": rows}


def collect(job: dict, episodes: int, workers: int, cache: bool) -> list[dict]:
    cache_path = None
    if cache and job.get("state_dict") is None:
        key = json.dumps(
            {k: v for k, v in job.items() if k not in {"state_dict", "hidden"}}, sort_keys=True
        )
        digest = hashlib.sha1(f"{key}|{episodes}".encode()).hexdigest()[:16]
        cache_path = ROOT / "experiments/learning/runs/_bc_cache" / f"{digest}.pkl"
        if cache_path.is_file():
            return pickle.loads(cache_path.read_bytes())
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(job,)) as pool:
        episodes_out = list(pool.map(_rollout, range(episodes), chunksize=8))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(episodes_out))
    return episodes_out


def to_tensors(rows: list[tuple]) -> tuple:
    return (
        torch.as_tensor([row[0] for row in rows], dtype=torch.float32),
        torch.as_tensor([row[1] for row in rows], dtype=torch.bool),
        torch.as_tensor([row[2] for row in rows], dtype=torch.long),
        torch.as_tensor([row[3] for row in rows], dtype=torch.float32),
    )


def agreement(model, data) -> float:
    observations, masks, actions, _ = data
    if len(actions) == 0:
        return 0.0
    with torch.no_grad():
        logits, _ = model(observations)
        predicted = model.mask_logits(logits, masks).argmax(dim=-1)
    return float((predicted == actions).float().mean())


def fit(config: dict, seed: int, train, holdout) -> tuple[StrategistActorCritic, dict]:
    hidden = tuple(config.get("hidden", (128, 128)))
    model = StrategistActorCritic(seed=seed, hidden=hidden)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    value_weight = float(config.get("value_loss_weight", 1.0))
    batch_size = int(config["batch_size"])
    patience = int(config.get("patience", 0))
    generator = torch.Generator().manual_seed(seed)
    observations, masks, actions, value_targets = train
    best = (-1.0, None, 0)
    history = []
    stale = 0
    for epoch in range(int(config["epochs"])):
        model.train()
        order = torch.randperm(len(actions), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            logits, values = model(observations[index])
            logits = model.mask_logits(logits, masks[index])
            policy_loss = torch.nn.functional.cross_entropy(logits, actions[index])
            value_loss = torch.nn.functional.mse_loss(values, value_targets[index])
            loss = policy_loss + value_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        model.eval()
        score = agreement(model, holdout)
        history.append({"epoch": epoch, "loss": total / len(order), "holdout_agreement": score})
        if score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()), epoch)
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                break
    if patience and best[1] is not None:
        model.load_state_dict(best[1])
    return model, {"best_epoch": best[2], "epochs_run": len(history), "history": history}


def split_rows(episodes_out: list[dict], holdout_indexes: set[int]) -> tuple[list, list]:
    train, holdout = [], []
    for episode in episodes_out:
        (holdout if episode["index"] in holdout_indexes else train).extend(episode["rows"])
    return train, holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learning/bc.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config["seed"])
    episodes = int(args.episodes if args.episodes is not None else config["episodes"])
    run_name = args.run_name or config["run_name"]
    simulator_config = config.get("simulator_config", DEFAULT_SIMULATOR_CONFIG)
    teacher = config.get("teacher", "rules")
    torch.manual_seed(seed)
    job = {
        "teacher": teacher,
        "seed": seed,
        "run_id": f"{config['run_name']}-seed{seed}",
        "simulator_config": simulator_config,
        "gamma": float(config["gamma"]),
        "include_pace_scenarios": bool(config.get("include_pace_scenarios", False)),
        "include_strong_scenarios": bool(config.get("include_strong_scenarios", False)),
        "hidden": list(config.get("hidden", (128, 128))),
        "state_dict": None,
    }
    teacher_episodes = collect(job, episodes, args.workers, cache=args.workers > 1)
    holdout_count = min(episodes - 1, max(1, round(episodes * config["holdout_fraction"])))
    order = list(range(episodes))
    random.Random(seed).shuffle(order)
    holdout_indexes = set(order[:holdout_count])
    train_rows, holdout_rows = split_rows(teacher_episodes, holdout_indexes)
    if not train_rows:
        raise ValueError("BC training split has no transitions")
    holdout = to_tensors(holdout_rows)
    model, fit_info = fit(config, seed, to_tensors(train_rows), holdout)
    rounds = [{"round": 0, "train_transitions": len(train_rows),
               "holdout_action_agreement": agreement(model, holdout), **fit_info}]
    for round_index in range(1, int(config.get("dagger_rounds", 0)) + 1):
        dagger_job = dict(job, state_dict=model.state_dict(),
                          seed=seed + DAGGER_SEED_STRIDE * round_index)
        visited = collect(dagger_job, int(config.get("dagger_episodes", episodes)),
                          args.workers, cache=False)
        rows = [row for episode in visited for row in episode["rows"]]
        on_policy = sum(row[2] == row[4] for row in rows) / max(1, len(rows))
        train_rows.extend(rows)
        model, fit_info = fit(config, seed, to_tensors(train_rows), holdout)
        rounds.append({"round": round_index, "on_policy_agreement_before": on_policy,
                       "added_transitions": len(rows), "train_transitions": len(train_rows),
                       "holdout_action_agreement": agreement(model, holdout), **fit_info})
        print(json.dumps({k: v for k, v in rounds[-1].items() if k != "history"}))
    run_dir = ROOT / "experiments/learning/runs" / f"{run_name}_seed{seed}"
    metadata = {
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "seed": seed,
        "init": "scratch",
        "policy": "behavior_cloning",
        "teacher": teacher,
        "hidden": list(model.hidden),
        "dagger_rounds": len(rounds) - 1,
        "simulator_config": simulator_config,
    }
    save_checkpoint(model, run_dir / "bc.pt", metadata)
    metrics = {
        "seed": seed,
        "episodes": episodes,
        "teacher": teacher,
        "config": str(args.config),
        "train_transitions": len(train_rows),
        "holdout_transitions": len(holdout_rows),
        "train_action_agreement": agreement(model, to_tensors(train_rows)),
        "holdout_action_agreement": agreement(model, holdout),
        "rounds": rounds,
        "checkpoint": str(run_dir / "bc.pt"),
        "simulator_config": simulator_config,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {k: v for k, v in metrics.items() if k != "rounds"}
    summary["rounds"] = [{k: v for k, v in r.items() if k != "history"} for r in rounds]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
