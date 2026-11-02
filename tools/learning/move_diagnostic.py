"""Expected one-step value of each Strategist move under a simulator config.

All numbers are SIMULATED: they come from the prototype simulator's assumed dynamics,
not from human learners.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
from pathlib import Path

import yaml

from packages.learning.episode import load_graph
from packages.simulator.randomness import EventRandomBank
from packages.simulator.reward import expected_score
from packages.simulator.scenarios import load_scenarios
from packages.simulator.student import (
    DEFAULT_SIMULATOR_CONFIG,
    PrivateStudentState,
    apply_teaching_effect,
    sample_student,
)

ROOT = Path(__file__).resolve().parents[2]
TEACHING_MOVES = (
    "PROBE",
    "CHECK_MASTERY",
    "PRACTICE",
    "HINT",
    "EXPLAIN_CONCEPT",
    "WORKED_EXAMPLE",
    "CHALLENGE",
    "REFLECT",
)


def _deterministic_config(config: dict) -> dict:
    """Same dynamics without random misconception removal; removal is added in expectation."""
    result = copy.deepcopy(config)
    result["misconceptions"]["deactivate_probability"] = {"surfaced": 0.0, "not_surfaced": 0.0}
    result["content_fidelity"]["failure_probability"] = 0.0
    return result


def _deactivation_probability(config: dict, student: PrivateStudentState, misconception: str):
    probabilities = config["misconceptions"]["deactivate_probability"]
    surfaced = misconception in student.surfaced_misconceptions
    return float(probabilities["surfaced"] if surfaced else probabilities["not_surfaced"])


def move_value(
    student: PrivateStudentState,
    *,
    move: str,
    concept: str,
    goal: str,
    split: str,
    scenario: dict,
    graph: dict,
    config: dict,
    config_path: str,
    last_answer_incorrect: bool,
) -> tuple[float, float]:
    """Return (expected mastery change on concept, expected change in post-assessment score)."""
    before = expected_score(student, goal, split, config_path=config_path)
    trial = copy.deepcopy(student)
    if last_answer_incorrect:
        trial.last_answer_correct = False
        trial.last_answer_item_id = "diagnostic-item"
        trial.previous_step = "PRACTICE on " + concept
    mastery_before = trial.mastery[concept]
    apply_teaching_effect(
        trial,
        concept_id=concept,
        move=move,
        difficulty="medium",
        turn=0,
        scenario=scenario,
        episode_seed=0,
        graph=graph,
        bank=EventRandomBank(),
        config=_deterministic_config(config),
        content_realized=True,
        item_id="diagnostic-item",
    )
    mastery_delta = trial.mastery[concept] - mastery_before
    after = expected_score(trial, goal, split, config_path=config_path)
    if move in {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"} and student.active_misconceptions:
        for misconception in sorted(student.active_misconceptions):
            probability = _deactivation_probability(config, student, misconception)
            cleared = copy.deepcopy(trial)
            cleared.active_misconceptions.discard(misconception)
            cleared_score = expected_score(cleared, goal, split, config_path=config_path)
            after += probability * (cleared_score - after)
    return mastery_delta, after - before


def consecutive_curve(
    student: PrivateStudentState,
    *,
    move: str,
    concept: str,
    goal: str,
    split: str,
    scenario: dict,
    graph: dict,
    config: dict,
    config_path: str,
    repeats: int,
) -> list[float]:
    """Expected post-score gain after 1..repeats consecutive uses of one move."""
    trial = copy.deepcopy(student)
    survive = {misconception: 1.0 for misconception in trial.active_misconceptions}
    start = expected_score(student, goal, split, config_path=config_path)
    curve = []
    deterministic = _deterministic_config(config)
    for turn in range(repeats):
        if move in {"EXPLAIN_CONCEPT", "WORKED_EXAMPLE"}:
            for misconception in survive:
                removal = _deactivation_probability(config, trial, misconception)
                survive[misconception] *= 1 - removal
        apply_teaching_effect(
            trial,
            concept_id=concept,
            move=move,
            difficulty="medium",
            turn=turn,
            scenario=scenario,
            episode_seed=0,
            graph=graph,
            bank=EventRandomBank(),
            config=deterministic,
            content_realized=True,
            item_id="diagnostic-item",
        )
        active = copy.deepcopy(trial)
        cleared = copy.deepcopy(trial)
        cleared.active_misconceptions = set()
        score_active = expected_score(active, goal, split, config_path=config_path)
        score_cleared = expected_score(cleared, goal, split, config_path=config_path)
        keep = min(survive.values()) if survive else 1.0
        curve.append(keep * score_active + (1 - keep) * score_cleared - start)
    return curve


class _CachedYaml:
    """The simulator re-reads YAML on every call; cache it so the sweep stays fast."""

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def safe_load(text: str):
        return yaml.safe_load(text)


def main() -> None:
    import packages.simulator.reward as reward_module
    import packages.simulator.student as student_module

    reward_module.yaml = _CachedYaml
    student_module.yaml = _CachedYaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator-config", default=DEFAULT_SIMULATOR_CONFIG)
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--students-per-scenario", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.simulator_config).read_text(encoding="utf-8"))
    graph = load_graph()
    scenarios = [row for row in load_scenarios() if row["split"] == args.split]
    sums: dict[tuple[str, str, str], list[float]] = {}
    curves: dict[tuple[str, str], list[list[float]]] = {}
    for scenario in scenarios:
        profile = scenario["profile_id"]
        concept = scenario["start_concept_id"]
        goal = scenario["goal_concept_id"]
        for seed in range(args.students_per_scenario):
            student = sample_student(scenario, 900_000 + seed, config_path=args.simulator_config)
            for move in TEACHING_MOVES:
                for state in ("fresh", "after_incorrect", "surfaced"):
                    trial = copy.deepcopy(student)
                    if state == "surfaced":
                        trial.surfaced_misconceptions = set(trial.active_misconceptions)
                    mastery_delta, score_delta = move_value(
                        trial,
                        move=move,
                        concept=concept,
                        goal=goal,
                        split=scenario["split"],
                        scenario=scenario,
                        graph=graph,
                        config=config,
                        config_path=args.simulator_config,
                        last_answer_incorrect=state != "fresh",
                    )
                    bucket = sums.setdefault((profile, move, state), [0.0, 0.0, 0.0])
                    bucket[0] += mastery_delta
                    bucket[1] += score_delta
                    bucket[2] += 1
            for move in ("WORKED_EXAMPLE", "EXPLAIN_CONCEPT", "PRACTICE", "HINT"):
                curves.setdefault((profile, move), []).append(
                    consecutive_curve(
                        student,
                        move=move,
                        concept=concept,
                        goal=goal,
                        split=scenario["split"],
                        scenario=scenario,
                        graph=graph,
                        config=config,
                        config_path=args.simulator_config,
                        repeats=args.repeats,
                    )
                )
    report = {
        "label": "simulated",
        "simulator_config": args.simulator_config,
        "split": args.split,
        "students_per_scenario": args.students_per_scenario,
        "one_step": [
            {
                "profile": profile,
                "move": move,
                "state": state,
                "mean_mastery_delta": values[0] / values[2],
                "mean_post_score_delta": values[1] / values[2],
                "students": int(values[2]),
            }
            for (profile, move, state), values in sorted(sums.items())
        ],
        "consecutive_post_score_gain": {
            f"{profile}/{move}": [
                sum(curve[index] for curve in rows) / len(rows) for index in range(args.repeats)
            ]
            for (profile, move), rows in sorted(curves.items())
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        output = ROOT / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    profiles = sorted({row["profile"] for row in report["one_step"]})
    for state in ("fresh", "after_incorrect", "surfaced"):
        print(f"\nExpected one-step post-score change x1000 (simulated), state={state}")
        print("| Move | " + " | ".join(profiles) + " |")
        print("|---|" + "---:|" * len(profiles))
        for move in TEACHING_MOVES:
            cells = [
                next(
                    row["mean_post_score_delta"]
                    for row in report["one_step"]
                    if row["profile"] == profile and row["move"] == move and row["state"] == state
                )
                * 1000
                for profile in profiles
            ]
            print(f"| {move} | " + " | ".join(f"{cell:.2f}" for cell in cells) + " |")
    print("\nCumulative expected post-score gain after k consecutive uses (simulated)")
    for key, values in report["consecutive_post_score_gain"].items():
        print(key, " ".join(f"{value:.4f}" for value in values))


if __name__ == "__main__":
    main()
