from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from packages.learning.content.templates.families import FAMILIES

ROOT = Path(__file__).resolve().parents[2]
ENABLED = {"C01", "C02", "C03", "C06", "C08"}
EXPECTED_PROTO = {
    ("C01", "C02"),
    ("C02", "C03"),
    ("C02", "C06"),
    ("C06", "C08"),
}
EXPECTED_GROUPS = {
    "train": Counter({"prerequisite_gap": 8, "misconception": 4, "knowledgeable": 4, "mixed": 4}),
    "validation": Counter(
        {"prerequisite_gap": 2, "misconception": 1, "knowledgeable": 1, "mixed": 1}
    ),
    "test": Counter({"prerequisite_gap": 2, "misconception": 1, "knowledgeable": 1, "mixed": 1}),
}


def _acyclic(edges: list[dict], concept_ids: set[str]) -> bool:
    outgoing = {concept_id: [] for concept_id in concept_ids}
    for edge in edges:
        outgoing[edge["source_concept_id"]].append(edge["dependent_concept_id"])
    active: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in active:
            return False
        if node in visited:
            return True
        active.add(node)
        if not all(visit(neighbor) for neighbor in outgoing[node]):
            return False
        active.remove(node)
        visited.add(node)
        return True

    return all(visit(node) for node in concept_ids)


def validate_content(config_path: str = "configs/learning/prototype.yaml") -> dict:
    with (ROOT / config_path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if (
        config.get("graph_version") != "graph_proto_v1"
        or set(config.get("enabled_concepts", [])) != ENABLED
    ):
        raise ValueError(
            "prototype config must select graph_proto_v1 and the five frozen enabled concepts"
        )
    catalogue = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    concepts = catalogue["concepts"]
    concept_ids = {record["concept_id"] for record in concepts}
    indices = [record["index"] for record in concepts]
    if (
        len(concepts) != 15
        or len(concept_ids) != 15
        or len(set(indices)) != 15
        or sorted(indices) != list(range(15))
    ):
        raise ValueError("concept IDs and stable indices must be unique and complete")
    if {record["concept_id"] for record in concepts if record["enabled"]} != ENABLED:
        raise ValueError("enabled concepts must be C01, C02, C03, C06, and C08")
    by_id = {record["concept_id"]: record for record in concepts}
    misconception_records = yaml.safe_load(
        (ROOT / "data/original_content/misconceptions.yaml").read_text(encoding="utf-8")
    )["misconceptions"]
    misconception_ids = {record["misconception_id"] for record in misconception_records}
    if len(misconception_records) != 25 or len(misconception_ids) != 25:
        raise ValueError("misconception catalogue must have 25 unique slots")
    descriptions = {
        record["misconception_id"]: record["description"] for record in misconception_records
    }
    if "Reversing conditional probabilities" not in descriptions["M05"]:
        raise ValueError("M05 must represent reversing conditional probabilities")
    if "conditional denominator" not in descriptions["M06"].lower():
        raise ValueError("M06 must represent the wrong conditional denominator")
    if any(record["active"] != (record["index"] < 8) for record in misconception_records):
        raise ValueError("only misconception slots M01 through M08 may be active")
    family_ids = set(FAMILIES)
    if len(FAMILIES) != 50:
        raise ValueError(f"expected 50 template families; found {len(FAMILIES)}")
    for record in concepts:
        if any(prerequisite not in concept_ids for prerequisite in record["prerequisites"]):
            raise ValueError(f"{record['concept_id']} references an unknown prerequisite")
        if any(
            misconception_id not in misconception_ids
            for misconception_id in record["allowed_misconception_ids"]
        ):
            raise ValueError(f"{record['concept_id']} references an unknown misconception")
        if any(
            family_id not in family_ids
            for family_id in record["diagnostic_template_ids"] + record["practice_template_ids"]
        ):
            raise ValueError(f"{record['concept_id']} references an unknown template family")
    for graph in catalogue["graphs"].values():
        edges = graph
        if any(
            edge["source_concept_id"] not in concept_ids
            or edge["dependent_concept_id"] not in concept_ids
            for edge in edges
        ):
            raise ValueError("graph edge references an unknown concept")
        if not _acyclic(edges, concept_ids):
            raise ValueError("concept graph contains a cycle")
        if any(edge["status"] != "proposed" for edge in edges):
            raise ValueError("all prototype graph edges must remain proposed")
    proto_edges = {
        (edge["source_concept_id"], edge["dependent_concept_id"])
        for edge in catalogue["graphs"]["graph_proto_v1"]
    }
    if proto_edges != EXPECTED_PROTO:
        raise ValueError("prototype graph edges do not match the frozen four-edge graph")
    if len({family.template_family_id for family in FAMILIES.values()}) != len(FAMILIES):
        raise ValueError("template family IDs are not unique")
    for concept_id in ENABLED:
        record = by_id[concept_id]
        if not record["diagnostic_template_ids"] or not record["practice_template_ids"]:
            raise ValueError(f"{concept_id} lacks diagnostic or practice families")
        for family_id in record["diagnostic_template_ids"] + record["practice_template_ids"]:
            if family_id not in FAMILIES:
                raise ValueError(f"{concept_id} references missing template {family_id}")
        teaching = [
            family
            for family in FAMILIES.values()
            if family.concept_id == concept_id and family.role == "teaching"
        ]
        assessment = [
            family
            for family in FAMILIES.values()
            if family.concept_id == concept_id and family.role == "assessment"
        ]
        if len(teaching) != 4 or len(assessment) != 6:
            raise ValueError(f"{concept_id} must have four teaching and six assessment families")
        if sorted(family.split for family in teaching) != ["test", "train", "train", "validation"]:
            raise ValueError(f"{concept_id} teaching split allocation is invalid")
        if sorted(family.split for family in assessment) != [
            "test",
            "test",
            "train",
            "train",
            "validation",
            "validation",
        ]:
            raise ValueError(f"{concept_id} assessment split allocation is invalid")
        if any(family.pool == "teaching" for family in assessment):
            raise ValueError("assessment family leaked into teaching pool")
        expected_splits = Counter(
            {
                ("teaching", "train"): 2,
                ("teaching", "validation"): 1,
                ("teaching", "test"): 1,
                ("assessment", "train"): 2,
                ("assessment", "validation"): 2,
                ("assessment", "test"): 2,
            }
        )
        actual_splits = Counter(
            (family.role, family.split)
            for family in FAMILIES.values()
            if family.concept_id == concept_id
        )
        if actual_splits != expected_splits:
            raise ValueError(f"{concept_id} family split/role distribution is invalid")
    for family in FAMILIES.values():
        if family.split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split on {family.template_family_id}")
        expected_pool = (
            "teaching"
            if family.role == "teaching"
            else "train-reward"
            if family.split == "train"
            else family.split
        )
        if family.pool != expected_pool:
            raise ValueError(f"invalid family pool on {family.template_family_id}")
        for seed in range(200):
            instance = family.sample(seed)
            parameters = family.sampler(seed)
            answer = instance["answer"]
            if not isinstance(answer, Fraction):
                raise ValueError(f"{family.template_family_id} solver did not return Fraction")
            if parameters["denominator"] <= 0 or not 0 <= answer <= 1:
                raise ValueError(f"{family.template_family_id} has invalid probability parameters")
            for misconception_id in family.error_generators:
                if family.generate_error(misconception_id, instance) == answer:
                    raise ValueError(
                        f"{family.template_family_id}/{misconception_id} error equals the answer"
                    )
            example = family.worked_example(seed, answer)
            if family.answer_normalizer(example["answer"], 1e-9) == answer:
                raise ValueError(
                    f"{family.template_family_id} worked example repeats the instance answer"
                )
    scenarios = yaml.safe_load((ROOT / "data/scenarios/proto_v1.yaml").read_text(encoding="utf-8"))[
        "scenarios"
    ]
    if len(scenarios) != 30 or [
        sum(item["split"] == split for item in scenarios)
        for split in ("train", "validation", "test")
    ] != [20, 5, 5]:
        raise ValueError("scenario split allocation must be 20/5/5")
    learning_schema = yaml.safe_load(
        (ROOT / "contracts/learning_v1.schema.json").read_text(encoding="utf-8")
    )
    scenario_validator = Draft202012Validator(
        {
            "$schema": learning_schema["$schema"],
            "$defs": learning_schema["$defs"],
            "$ref": "#/$defs/Scenario",
        }
    )
    group_counts = {split: Counter() for split in ("train", "validation", "test")}
    group_totals = Counter()
    for scenario in scenarios:
        scenario_validator.validate(scenario)
        group = scenario["scenario_family_id"].split("-")[2]
        group_counts[scenario["split"]][group] += 1
        group_totals[group] += 1
        allowed = set(scenario["template_family_allowlist"])
        if not allowed <= family_ids:
            raise ValueError(f"scenario {scenario['scenario_id']} has unknown template families")
        if any(FAMILIES[family_id].split != scenario["split"] for family_id in allowed):
            raise ValueError(f"scenario {scenario['scenario_id']} crosses content splits")
        if any(FAMILIES[family_id].role not in {"teaching", "assessment"} for family_id in allowed):
            raise ValueError(f"scenario {scenario['scenario_id']} has an invalid teaching pool")
        for concept_id in ENABLED:
            assessment_families = {
                family_id
                for family_id in allowed
                if FAMILIES[family_id].concept_id == concept_id
                and FAMILIES[family_id].role == "assessment"
                and FAMILIES[family_id].split == scenario["split"]
            }
            if len(assessment_families) < 2:
                raise ValueError(
                    f"scenario {scenario['scenario_id']} must allow at least two "
                    f"{scenario['split']} assessment families for {concept_id}"
                )
        if group == "mixed":
            requests = {event["explicit_request"] for event in scenario["public_history"]}
            if not requests & {"faster", "skip"}:
                raise ValueError(f"mixed scenario {scenario['scenario_id']} lacks an early request")
    if group_totals != Counter(
        {"prerequisite_gap": 12, "misconception": 6, "knowledgeable": 6, "mixed": 6}
    ):
        raise ValueError(f"unexpected overall scenario family mix: {group_totals}")
    if group_counts != EXPECTED_GROUPS:
        raise ValueError(f"scenario groups are not proportionally distributed: {group_counts}")
    return {
        "concepts": len(concepts),
        "families": len(FAMILIES),
        "sampled_instances": len(FAMILIES) * 200,
        "scenarios": len(scenarios),
        "scenario_splits": dict(Counter(scenario["split"] for scenario in scenarios)),
        "scenario_groups": dict(group_totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = validate_content(args.config)
    print(
        f"content validation passed: {report['concepts']} concepts, {report['families']} families, "
        f"{report['sampled_instances']} sampled instances, {report['scenarios']} scenarios"
    )
    print(f"scenario splits: {report['scenario_splits']}; groups: {report['scenario_groups']}")


if __name__ == "__main__":
    main()
