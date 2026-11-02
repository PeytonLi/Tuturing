from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

CONCEPTS = [
    ("C01", "Outcomes events and sample spaces", []),
    ("C02", "Probability bounds and equally likely outcomes", ["C01"]),
    ("C03", "Complements", ["C02"]),
    ("C04", "Unions intersections and set notation", ["C01"]),
    ("C05", "Addition rule", ["C02", "C04"]),
    ("C06", "Conditional probability", ["C02", "C04"]),
    ("C07", "Independence", ["C06"]),
    ("C08", "Multiplication rule and probability trees", ["C06"]),
    ("C09", "Total probability", ["C05", "C08"]),
    ("C10", "Bayes theorem", ["C06", "C09"]),
    ("C11", "Counting with order", ["C01"]),
    ("C12", "Counting without order", ["C11"]),
    ("C13", "Bernoulli trials", ["C02", "C03", "C07"]),
    ("C14", "Binomial probability", ["C12", "C13"]),
    ("C15", "Expected value of a discrete variable", ["C02", "C05"]),
]
ENABLED = {"C01", "C02", "C03", "C06", "C08"}
FULL_EDGES = [
    (prerequisite, concept_id)
    for concept_id, _, prerequisites in CONCEPTS
    for prerequisite in prerequisites
]
PROTO_EDGES = [("C01", "C02"), ("C02", "C03"), ("C02", "C06"), ("C06", "C08")]
MISCONCEPTIONS = [
    "Treating unequal outcomes as equally likely",
    "Confusing an event with a single outcome",
    "Assigning probability above one",
    "Subtracting the wrong complement",
    "Reversing conditional probabilities",
    "Using the wrong conditional denominator",
    "Assuming independence without evidence",
    "Multiplying along incompatible tree paths",
]


def write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    records = []
    for index, (concept_id, name, prerequisites) in enumerate(CONCEPTS):
        concept_key = concept_id.lower()
        records.append(
            {
                "concept_id": concept_id,
                "index": index,
                "name": name,
                "enabled": concept_id in ENABLED,
                "measurable_objective": (
                    f"Use evidence and probability rules to reason about {name.lower()}."
                ),
                "prerequisites": prerequisites,
                "diagnostic_template_ids": [f"{concept_key}_assess_01"]
                if concept_id in ENABLED
                else [],
                "practice_template_ids": [
                    f"{concept_key}_teach_{variant:02d}" for variant in range(1, 5)
                ]
                if concept_id in ENABLED
                else [],
                "allowed_misconception_ids": ["M05", "M06"]
                if concept_id in {"C06", "C08"}
                else ["M01", "M02", "M03", "M04"],
                "content_provenance": "original prototype content; authored for this project",
            }
        )
    graphs = {
        "graph_full_v1": [
            {"source_concept_id": source, "dependent_concept_id": target, "status": "proposed"}
            for source, target in FULL_EDGES
        ],
        "graph_proto_v1": [
            {"source_concept_id": source, "dependent_concept_id": target, "status": "proposed"}
            for source, target in PROTO_EDGES
        ],
    }
    write_yaml(
        ROOT / "data/original_content/concepts.yaml",
        {
            "schema_version": "concepts_v1",
            "concepts": records,
            "graphs": graphs,
        },
    )
    slots = [
        {
            "misconception_id": f"M{index:02d}",
            "index": index - 1,
            "active": index <= 8,
            "description": MISCONCEPTIONS[index - 1]
            if index <= 8
            else f"Inactive reserved misconception slot {index:02d}.",
        }
        for index in range(1, 26)
    ]
    write_yaml(
        ROOT / "data/original_content/misconceptions.yaml",
        {
            "schema_version": "misconceptions_v1",
            "misconceptions": slots,
        },
    )
    allowlists = {
        split: [
            *(
                f"{concept_id.lower()}_teach_{variant:02d}"
                for concept_id in sorted(ENABLED)
                for variant, family_split in enumerate(("train", "train", "validation", "test"), 1)
                if family_split == split
            ),
            *(
                f"{concept_id.lower()}_assess_{variant:02d}"
                for concept_id in sorted(ENABLED)
                for variant, family_split in enumerate(("train", "validation", "test"), 1)
                if family_split == split
            ),
        ]
        for split in ("train", "validation", "test")
    }
    groups = [
        ("prerequisite_gap", "novice_missing_prereqs", 8, 2, 2),
        ("misconception", "confident_misconception", 4, 1, 1),
        ("knowledgeable", "knowledgeable_concise", 4, 1, 1),
        ("mixed", "mixed", 4, 1, 1),
    ]
    scenarios = []
    for group_name, profile, train_count, validation_count, test_count in groups:
        counts = {"train": train_count, "validation": validation_count, "test": test_count}
        for split, count in counts.items():
            for _ in range(count):
                number = len(scenarios) + 1
                goal = "C06" if number in {5, 10, 15, 20, 25, 30} else "C08"
                start = (
                    "C01"
                    if group_name == "prerequisite_gap"
                    else ("C02" if goal == "C06" else "C06")
                )
                actual_profile = (
                    ("novice_missing_prereqs", "knowledgeable_concise", "confident_misconception")[
                        number % 3
                    ]
                    if profile == "mixed"
                    else profile
                )
                history = []
                if group_name == "mixed":
                    request = "faster" if number % 2 == 0 else "skip"
                    history = [
                        {
                            "schema_version": "1.0",
                            "event_id": f"scenario-event-{number:02d}",
                            "session_id": f"scenario-session-{number:02d}",
                            "trace_id": f"scenario-trace-{number:02d}",
                            "source_response_id": f"scenario-response-{number:02d}",
                            "problem_instance_id": None,
                            "observed_at": "2026-11-02T00:00:00Z",
                            "normalized_answer": None,
                            "correctness": "not_applicable",
                            "grade_source": "none",
                            "confidence": 1.0,
                            "assistance": "none",
                            "explicit_request": request,
                            "requested_concept_id": goal if request == "skip" else None,
                            "misconception_hypotheses": [],
                            "raw_message_ref": f"raw-{number:02d}",
                            "extractor_version": "scenario_fixture_v1",
                            "extensions": {},
                        }
                    ]
                scenarios.append(
                    {
                        "scenario_id": f"proto-scenario-{number:02d}",
                        "scenario_family_id": f"proto-family-{group_name}-{number:02d}",
                        "split": split,
                        "goal_concept_id": goal,
                        "start_concept_id": start,
                        "profile_id": actual_profile,
                        "simulator_config_id": "simulator_proto_v1",
                        "private_initial_state_ref": f"private-init-{number:02d}",
                        "public_history": history,
                        "template_family_allowlist": allowlists[split],
                        "teaching_horizon": 16,
                        "provenance": {
                            "author": "Person B",
                            "reviewed": False,
                            "revision": "proto_v1",
                        },
                    }
                )
    write_yaml(
        ROOT / "data/scenarios/proto_v1.yaml",
        {"schema_version": "scenarios_v1", "scenarios": scenarios},
    )


if __name__ == "__main__":
    main()
