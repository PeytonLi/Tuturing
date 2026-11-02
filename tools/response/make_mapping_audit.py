"""Produce a deterministic, stratified sample for independent move review."""

import argparse
import csv
import random
from pathlib import Path

from tools.response.prepare_sft import (
    build_examples,
    load_config,
    load_move_map,
    load_source,
    validate_raw_counts,
)

AUDIT_COLUMNS = (
    "example_id", "qid", "source_act", "context_last_student", "teacher_text",
    "draft_move", "label_person_a", "label_person_b", "notes",
)


def sample_audit_rows(candidates, move_map, seed=20261102, per_act=50):
    mapping = move_map["mappings"]
    by_act = {}
    for candidate in candidates:
        act = candidate["source_act"]
        if act:
            by_act.setdefault(act, []).append(candidate)
    rows = []
    for act in sorted(by_act):
        eligible = sorted(by_act[act], key=lambda item: item["example_id"])
        if len(eligible) < per_act:
            raise ValueError(
                f"need {per_act} audit turns for {act}; found {len(eligible)}"
            )
        sampled = random.Random(f"{seed}:{act}").sample(eligible, per_act)
        mapping_entry = mapping.get(act) or {}
        draft_move = (
            f"EXCLUDED_{mapping_entry.get('reason', 'excluded_act')}"
            if mapping_entry.get("exclude")
            else mapping_entry.get("move", "UNMAPPED")
        )
        for item in sampled:
            rows.append({
                "example_id": item["example_id"],
                "qid": item["qid"],
                "source_act": act,
                "context_last_student": item["context_last_student"],
                "teacher_text": item["teacher_text"],
                "draft_move": draft_move,
                "label_person_a": "",
                "label_person_b": "",
                "notes": "",
            })
    return sorted(rows, key=lambda item: (item["source_act"], item["example_id"]))


def write_audit(rows, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_audit(config_path, out_path, seed):
    config = load_config(config_path)
    move_map = load_move_map(config["move_map"])
    rows_by_official, _, _ = load_source(config)
    validate_raw_counts(rows_by_official, config["expected_counts"])
    _, _, candidates, assignments = build_examples(rows_by_official, config, move_map)
    train_qids = {qid for qid, split in assignments.items() if split == "train"}
    train_candidates = [
        item for item in candidates
        if item["qid"] in train_qids and item["teacher_text"] and item["source_act"]
    ]
    rows = sample_audit_rows(train_candidates, move_map, seed=seed, per_act=50)
    write_audit(rows, out_path)
    attribution_path = Path("evaluation/response/ATTRIBUTION.md")
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_path.write_text(
        "# MathDial attribution\n\n"
        f"{config['attribution']}\n\n"
        f"Source revision: `{config['revision']}`. Licensed under "
        f"[CC BY 4.0]({config['license']['url']}). This derived review sample "
        "contains 200 stratified teacher turns; speaker and teacher-act prefixes "
        "were stripped, and no future dialogue turns were included in context.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} audit rows to {out_path}")
    print("rows per act: " + repr({
        act: sum(row["source_act"] == act for row in rows)
        for act in sorted({row["source_act"] for row in rows})
    }))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/response/mathdial_v1.yaml")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out", default="evaluation/response/mathdial_mapping_audit_200.csv")
    args = parser.parse_args(argv)
    make_audit(args.config, args.out, args.seed)


if __name__ == "__main__":
    main()
