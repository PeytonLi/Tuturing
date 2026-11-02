"""Prepare provenance-tracked, turn-level MathDial SFT examples."""

import argparse
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from packages.response.hashing import canonical_json
from packages.response.prompt import compile_prompt
from packages.response.validate import run_checks

GENERATOR_VERSION = "tools.response.prepare_sft v1.1"
USER_AGENT = "Tutoring-response-sft-prep/1.0"
MOVE_ORDER = (
    "PROBE", "HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE",
    "PRACTICE", "CHALLENGE", "CHECK_MASTERY", "REFLECT",
)
_TEACHER_ACT = re.compile(r"^Teacher:\s*\((\w+)\)\s*")
_SPEAKER_PREFIX = re.compile(r"^([^:\n]{1,40}):\s*")


class RawCountMismatch(ValueError):
    """Raised if the pinned source revision differs from expected counts."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_move_map(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _download_if_missing(path, url):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def load_source(config):
    """Download missing pinned assets, then hash and parse all source files."""
    raw_dir = Path(config["raw_dir"])
    revision = config["revision"]
    url_template = config["url_template"]
    names = [*config["files"], config.get("download_readme", "README.md")]
    file_info = []
    rows_by_official = {}
    urls = []
    for name in names:
        url = url_template.format(revision=revision, file=name)
        urls.append(url)
        path = raw_dir / name
        _download_if_missing(path, url)
        payload = path.read_bytes()
        rows = None
        if name.endswith(".jsonl"):
            rows = parse_jsonl(payload.decode("utf-8"))
            rows_by_official[name.removesuffix(".jsonl")] = rows
        file_info.append({
            "name": name,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "rows": len(rows) if rows is not None else None,
        })
    return rows_by_official, file_info, urls


def parse_jsonl(text):
    rows = []
    for line_number, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append((line_number, json.loads(line)))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number + 1}") from error
    return rows


def _teacher_turn_text(turn):
    after_speaker = turn[len("Teacher:"):].strip() if turn.startswith("Teacher:") else turn
    act_match = _TEACHER_ACT.match(turn)
    if act_match:
        after_speaker = turn[act_match.end():]
    return after_speaker.strip()


def _teacher_act(turn):
    match = _TEACHER_ACT.match(turn)
    return match.group(1).lower() if match else None


def _teacher_turns(row):
    conversation = row.get("conversation") or ""
    return conversation.split("|EOM|")


def compute_raw_counts(rows_by_official):
    train_rows = rows_by_official.get("train", [])
    test_rows = rows_by_official.get("test", [])
    all_rows = [*train_rows, *test_rows]
    qids_train = {int(row["qid"]) for _, row in train_rows}
    qids_test = {int(row["qid"]) for _, row in test_rows}
    acts = Counter()
    empty_teacher_turns = 0
    teacher_turns = 0
    for _, row in all_rows:
        for turn in _teacher_turns(row):
            if not turn.startswith("Teacher:"):
                continue
            teacher_turns += 1
            act = _teacher_act(turn)
            if act:
                acts[act] += 1
            if not _teacher_turn_text(turn):
                empty_teacher_turns += 1
    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "dialogues": len(all_rows),
        "unique_qids": len({int(row["qid"]) for _, row in all_rows}),
        "teacher_turns": teacher_turns,
        "test_qids_in_train": len(qids_train & qids_test),
        "teacher_acts": dict(sorted(acts.items())),
        "empty_teacher_turns": empty_teacher_turns,
    }


def validate_raw_counts(rows_by_official, expected_counts):
    actual = compute_raw_counts(rows_by_official)
    differences = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in expected_counts.items()
        if actual.get(key) != expected
    }
    if differences:
        raise RawCountMismatch(
            "raw count mismatch (expected vs actual): "
            + canonical_json(differences)
        )
    return actual


def assign_qid_splits(rows_by_official, split_salt):
    qids = {
        int(row["qid"])
        for rows in rows_by_official.values()
        for _, row in rows
    }
    assignment = {}
    for qid in qids:
        digest = hashlib.sha256(f"{split_salt}:{qid}".encode()).hexdigest()
        bucket = int(digest, 16) % 10
        assignment[qid] = "train" if bucket <= 7 else ("validation" if bucket == 8 else "test")
    return assignment


def parse_turn(turn, student_name=""):
    if turn.startswith("Teacher:"):
        return {
            "role": "tutor",
            "act": _teacher_act(turn),
            "text": _teacher_turn_text(turn),
        }
    text = turn
    match = _SPEAKER_PREFIX.match(turn)
    if match and match.group(1).strip().casefold() in {
        "student", student_name.strip().casefold()
    }:
        text = turn[match.end():].strip()
    return {"role": "student", "act": None, "text": text.strip()}


def _normalized(text):
    return " ".join(str(text).casefold().split())


def protected_rejection(plan, prompt, teacher_text, ground_truth):
    """Return a rejection reason for a reveal-forbidden example, or None.

    A protected prompt must not contain the full reference solution, even when
    an earlier tutor turn pasted it into the dialogue, and a protected target
    must not state the forbidden final answer (PRD 13.2 steps 2 and 6).
    """
    if plan["constraints"]["reveal_allowed"]:
        return None
    solution = _normalized(ground_truth or "")
    prompt_text = _normalized("\n".join(message["content"] for message in prompt))
    if solution and solution in prompt_text:
        return "reference_solution_in_protected_context"
    if not run_checks(plan, teacher_text, plan["teaching_action"])["disclosure_allowed"]:
        return "target_states_protected_answer"
    return None


def _action_id(move):
    difficulty_index = 1  # medium
    pace_index = 1  # normal
    return MOVE_ORDER.index(move) * 9 + difficulty_index * 3 + pace_index


def _teaching_plan(dialogue_id, example_id, row, move, recent_dialogue, last_student, config):
    reveal = move == "EXPLAIN_CONCEPT"
    content = {
        "concept_name": "Arithmetic word problem",
        "goal_name": "Arithmetic word problem",
        "problem_stem": row.get("question") or "",
        "approved_facts": [],
        "permitted_hint": None,
        "worked_example": None,
        "reveal_solution": row.get("ground_truth") if reveal else None,
        "previous_step": None,
    }
    ground_truth = row.get("ground_truth") or ""
    final_answer = next(
        (line.strip() for line in reversed(ground_truth.splitlines()) if line.strip()),
        "",
    )
    plan = {
        "schema_version": "1.0",
        "plan_id": f"mathdial_plan:{example_id}",
        "session_id": dialogue_id,
        "trace_id": "mathdial_v1",
        "based_on_state_version": 0,
        "option_id": "mathdial_option",
        "navigation_action": {
            "action_id": -1,
            "operation": "STAY",
            "target_concept_id": "EXT_MATHDIAL",
        },
        "teaching_action": {
            "move": move,
            "difficulty": "medium",
            "pace": "normal",
            "action_id": _action_id(move),
        },
        "actor_eligible": False,
        "concept_id": "EXT_MATHDIAL",
        "goal_concept_id": "EXT_MATHDIAL",
        "problem_instance_id": f"mathdial:q{row['qid']}",
        "template_family_id": f"mathdial:q{row['qid']}",
        "assessment_mode": "none",
        "assistance": {
            "PROBE": "none",
            "HINT": "hint",
            "EXPLAIN_CONCEPT": "answer_shown",
        }[move],
        "content_bundle_id": "mathdial_v1",
        "content": content,
        "constraints": {
            "reveal_allowed": reveal,
            "max_words": config["max_words"],
            "max_questions": 1,
            "must_request_student_attempt": False,
            "forbidden_answer_fingerprints": (
                [] if reveal else ([final_answer] if final_answer else [])
            ),
        },
        "response_context": {
            "last_student_message": last_student,
            "recent_dialogue": recent_dialogue,
            "qualified_misconception_summary": None,
        },
        "versions": {
            "graph": "external_none",
            "content": "mathdial_v1",
            "state_estimator": "none",
            "navigator": "none",
            "strategist": "none",
            "mask": "none",
            "feature": "none",
        },
        "expires_at": "2099-01-01T00:00:00Z",
    }
    return plan


def build_examples(rows_by_official, config, move_map):
    """Build accepted teacher-turn examples and rejection records from inline rows."""
    mapping = move_map["mappings"]
    assignments = assign_qid_splits(rows_by_official, config.get("split_salt", "mathdial"))
    accepted = []
    rejected = []
    audit_candidates = []
    for official_split, rows in rows_by_official.items():
        for row_index, row in rows:
            qid = int(row["qid"])
            split = assignments[qid]
            dialogue_id = f"mathdial:{official_split}:{row_index:04d}"
            turns = _teacher_turns(row)
            student_profile = row.get("student_profile") or ""
            student_name = student_profile.split()[0] if student_profile.split() else ""
            parsed_turns = [parse_turn(turn, student_name) for turn in turns]
            context_prefix = []
            if config.get("include_student_solution_as_opening_turn"):
                opening = row.get("student_incorrect_solution")
                if isinstance(opening, str) and opening.strip():
                    context_prefix.append({"role": "student", "text": opening.strip()})
            for turn_index, (raw_turn, parsed) in enumerate(zip(turns, parsed_turns)):
                if parsed["role"] != "tutor":
                    continue
                teacher_text = parsed["text"]
                act = parsed["act"]
                example_id = f"{dialogue_id}:t{turn_index:02d}"
                prior = [*context_prefix, *parsed_turns[:turn_index]]
                recent_dialogue = [
                    {"role": item["role"], "text": item["text"]}
                    for item in prior
                    if item["text"]
                ]
                last_student = next(
                    (
                        item["text"]
                        for item in reversed(recent_dialogue)
                        if item["role"] == "student"
                    ),
                    "",
                )
                mapping_entry = mapping.get(act) if act else None
                candidate = {
                    "example_id": example_id,
                    "qid": qid,
                    "source_act": act or "",
                    "context_last_student": last_student,
                    "teacher_text": teacher_text,
                    "official_split": official_split,
                    "row_index": row_index,
                    "turn_index": turn_index,
                    "row": row,
                    "split": split,
                }
                if teacher_text:
                    audit_candidates.append(candidate)
                rejection = None
                if not teacher_text:
                    rejection = "empty_text"
                elif not act:
                    rejection = "missing_act"
                elif not mapping_entry:
                    rejection = "unmapped_act"
                elif mapping_entry.get("exclude"):
                    rejection = mapping_entry.get("reason", "excluded_act")
                elif len(teacher_text.split()) > config["max_words"]:
                    rejection = "over_max_words"
                if rejection:
                    rejected.append({
                        "example_id": example_id,
                        "reason": rejection,
                        "source_act": act,
                        "split": split,
                    })
                    continue
                move = mapping_entry["move"]
                plan = _teaching_plan(
                    dialogue_id, example_id, row, move, recent_dialogue,
                    last_student, config,
                )
                prompt = compile_prompt(plan)
                rejection = protected_rejection(
                    plan, prompt, teacher_text, row.get("ground_truth")
                )
                if rejection:
                    rejected.append({
                        "example_id": example_id,
                        "reason": rejection,
                        "source_act": act,
                        "split": split,
                    })
                    continue
                accepted.append({
                    "example_id": example_id,
                    "dialogue_id": dialogue_id,
                    "original_problem_id": f"mathdial:q{qid}",
                    "template_family_id": f"mathdial:q{qid}",
                    "split": split,
                    "source_id": "mathdial",
                    "source_revision": config["revision"],
                    "license_ref": "CC-BY-4.0",
                    "synthetic": False,
                    "reviewer_status": "pending",
                    "action_label_source": "mapped",
                    "plan": plan,
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": teacher_text}],
                    "checks": {
                        "math_correct": None,
                        "faithful": None,
                        "no_early_answer": None,
                    },
                    "extensions": {
                        "mapping_status": move_map.get("status", "draft_unaudited"),
                        "checks_status": "not_computed",
                        "mathdial": {
                            "qid": qid,
                            "scenario": row.get("scenario"),
                            "official_split": official_split,
                            "row_index": row_index,
                            "turn_index": turn_index,
                            "source_act": act,
                        },
                    },
                })
    accepted.sort(key=lambda example: example["example_id"])
    return accepted, rejected, audit_candidates, assignments


def _write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in sorted(records, key=lambda item: item["example_id"]):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _source_id_summary(ids):
    ordered = sorted(ids)
    return {"count": len(ordered), "sha256_sorted_ids": sha256_bytes("\n".join(ordered).encode())}


def _split_statistics(accepted, assignments):
    examples_by_split = Counter(example["split"] for example in accepted)
    qids_by_split = defaultdict(set)
    dialogues_by_split = defaultdict(set)
    moves_by_split = Counter()
    for qid, split in assignments.items():
        qids_by_split[split].add(qid)
    for example in accepted:
        dialogues_by_split[example["split"]].add(example["dialogue_id"])
        moves_by_split[(example["split"], example["plan"]["teaching_action"]["move"])] += 1
    counts = {
        split: {
            "dialogues": len(dialogues_by_split[split]),
            "qids": len(qids_by_split[split]),
            "examples": examples_by_split[split],
        }
        for split in ("train", "validation", "test")
    }
    by_move = {
        split: dict(sorted(
            (move, count)
            for (which_split, move), count in moves_by_split.items()
            if which_split == split
        ))
        for split in ("train", "validation", "test")
    }
    return counts, by_move


def prepare(config_path):
    config = load_config(config_path)
    move_map = load_move_map(config["move_map"])
    rows_by_official, downloads, download_urls = load_source(config)
    raw_counts = validate_raw_counts(rows_by_official, config["expected_counts"])
    accepted, rejected, _, assignments = build_examples(rows_by_official, config, move_map)

    out_dir = Path(config["out_dir"])
    split_records = {
        split: [item for item in accepted if item["split"] == split]
        for split in ("train", "validation", "test")
    }
    output_files = []
    for split, records in split_records.items():
        name = f"{split}.jsonl"
        path = out_dir / name
        _write_jsonl(path, records)
        output_files.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(records),
        })

    counts_per_split, counts_per_split_and_move = _split_statistics(accepted, assignments)
    all_rows = [
        row for rows in rows_by_official.values() for _, row in rows
    ]
    conversations = [row.get("conversation", "") for row in all_rows]
    duplicate_count = len(conversations) - len(set(conversations))
    dialogue_ids = [
        f"mathdial:{official}:{row_index:04d}"
        for official, rows in rows_by_official.items()
        for row_index, _ in rows
    ]
    problem_ids = sorted({f"mathdial:q{int(row['qid'])}" for row in all_rows})
    official_split_crosstab = {}
    for official, rows in rows_by_official.items():
        counts = Counter(
            assignments[int(row["qid"])] for _, row in rows
        )
        official_split_crosstab[official] = dict(sorted(counts.items()))
    qids = {
        split: sorted(int(qid) for qid, assigned in assignments.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    mapping_path = Path(config["move_map"])
    config_path_obj = Path(config_path)
    manifest = {
        "source_id": config["source_id"],
        "publisher_url": "https://huggingface.co/datasets/eth-nlped/mathdial",
        "revision": config["revision"],
        "download_urls": download_urls,
        "download_date": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "files": downloads,
        "license": {
            "name": config["license"]["name"],
            "url": config["license"]["url"],
            "local_snapshot_file": config.get("download_readme", "README.md"),
            "local_snapshot_sha256": next(
                item["sha256"] for item in downloads
                if item["name"] == config.get("download_readme", "README.md")
            ),
        },
        "access_restrictions": "public, no gating",
        "permitted_purposes": [
            "research prototype SFT and response evaluation consistent with CC-BY-4.0",
        ],
        "raw_counts": raw_counts,
        "accepted_count": len(accepted),
        "rejected_count_by_reason": dict(sorted(Counter(
            item["reason"] for item in rejected
        ).items())),
        "source_ids": {
            "problem": _source_id_summary(problem_ids),
            "dialogue": _source_id_summary(dialogue_ids),
        },
        "deduplication": {
            "method": "exact conversation text match",
            "duplicates_found": duplicate_count,
        },
        "split_rule": (
            "For each pooled qid, int(sha256(f'mathdial:{qid}').hexdigest(), 16) % 10; "
            "0-7 train, 8 validation, 9 test. All turns of a qid stay together; "
            "official split is provenance only."
        ),
        "split_assignment_sha256": sha256_bytes(canonical_json({
            str(qid): assignments[qid] for qid in sorted(assignments)
        }).encode("utf-8")),
        "split_qids": qids,
        "counts_per_split": counts_per_split,
        "counts_per_split_and_move": counts_per_split_and_move,
        "counts_per_move": dict(sorted(Counter(
            item["plan"]["teaching_action"]["move"] for item in accepted
        ).items())),
        "official_split_crosstab": official_split_crosstab,
        "attribution": config["attribution"],
        "responsible_owner": "Person A",
        "mapping": {
            "path": config["move_map"],
            "sha256": sha256_file(mapping_path),
            "status": move_map.get("status", "draft_unaudited"),
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path_obj)},
        "generator": GENERATOR_VERSION,
        "output_files": output_files,
        "lineage": [
            "Downloaded pinned MathDial train/test JSONL and README license snapshot if absent.",
            "Hashed and counted raw source files; validated the pinned raw count assertions.",
            "Pooled dialogues and split qids deterministically before expanding turns.",
            "Parsed preceding dialogue context; stripped known speaker and teacher-act prefixes.",
            "Mapped probing/focus/telling acts; excluded generic, malformed and overlength turns.",
            "Rejected reveal-forbidden examples whose prompt already holds the full reference "
            "solution or whose target states the forbidden final answer.",
            "Compiled deterministic prompts and wrote turn-level SFT records.",
        ],
        "notes": [
            "SFT checks are null and not computed; mapped labels are draft_unaudited.",
            "The disclosure check (packages.response.validate) runs only as an admission "
            "filter on reveal-forbidden targets; its result is not stored in checks.",
            "External MathDial concepts use navigation action_id -1 because they have no "
            "catalogue index.",
            "MathDial HINT and EXPLAIN_CONCEPT plans lack the corresponding PRD 9.3 "
            "approved hint/facts.",
            "Only EXPLAIN_CONCEPT plans include ground_truth, as reveal_solution.",
        ],
    }
    manifest_path = Path(config["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"raw_counts: {json.dumps(raw_counts, sort_keys=True)}")
    accepted_per_split = {
        split: len(rows) for split, rows in split_records.items()
    }
    print(f"accepted per split: {json.dumps(accepted_per_split, sort_keys=True)}")
    print(f"rejected by reason: {json.dumps(manifest['rejected_count_by_reason'], sort_keys=True)}")
    print(f"mapped move per split: {json.dumps(counts_per_split_and_move, sort_keys=True)}")
    print(f"dialogues/qids per split: {json.dumps(counts_per_split, sort_keys=True)}")
    print(f"manifest: {manifest_path}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/response/mathdial_v1.yaml")
    args = parser.parse_args(argv)
    prepare(args.config)


if __name__ == "__main__":
    main()
