import json
from collections import Counter

import pytest

from tools.response.make_mapping_audit import AUDIT_COLUMNS, sample_audit_rows
from tools.response.prepare_sft import (
    RawCountMismatch,
    assign_qid_splits,
    build_examples,
    compute_raw_counts,
    load_move_map,
    load_turn_labels,
    validate_raw_counts,
)


def inline_input(include_opening=True):
    config = {
        "revision": "rev-test",
        "max_words": 150,
        "split_salt": "mathdial",
        "include_student_solution_as_opening_turn": include_opening,
    }
    mapping = {
        "status": "draft_unaudited",
        "mappings": {
            "probing": {"move": "PROBE"},
            "focus": {"move": "HINT"},
            "telling": {"move": "EXPLAIN_CONCEPT"},
            "generic": {"exclude": True, "reason": "generic_act"},
        },
    }
    shared = {
        "scenario": "arithmetic",
        "question": "How many apples remain?",
        "ground_truth": "Private ground truth: 17",
        "student_incorrect_solution": "I subtracted the wrong amounts.",
        "student_profile": "Ada learner profile must not leak",
        "teacher_described_confusion": "TEACHER_CONFUSION_PRIVATE",
        "self-correctness": "SELF_CORRECTNESS_PRIVATE",
        "self-typical-confusion": "hidden_confusion",
        "self-typical-interactions": "hidden_interaction",
    }
    rows_by_official = {
        "train": [
            (0, {
                **shared,
                "qid": 7,
                "conversation": (
                    "Teacher: (probing) Ask the first question.|EOM|"
                    "Ada: Let me try.|EOM|"
                    "Teacher: (generic) A general thought.|EOM|"
                    "Teacher: (probing) Let's continue from the earlier turn.|EOM|"
                    "Student: FUTURE_STUDENT_PRIVATE|EOM|"
                    "Teacher: (probing) FUTURE_TEACHER_PRIVATE"
                ),
            }),
            (1, {
                **shared,
                "qid": 9,
                "conversation": (
                    "Teacher: (focus) Notice what is being counted.|EOM|"
                    "Ada: I am unsure.|EOM|"
                    "Teacher: (probing) "
                ),
            }),
        ],
        "test": [
            (0, {
                **shared,
                "qid": 7,
                "conversation": (
                    "Teacher: (telling) Here is the explanation.|EOM|"
                    "Student: I will think about that."
                ),
            }),
        ],
    }
    return config, mapping, rows_by_official


def test_prepare_uses_prior_turns_strips_prefixes_and_excludes_generic_and_empty():
    config, mapping, rows = inline_input()
    accepted, rejected, _, assignments = build_examples(rows, config, mapping)
    assert len(accepted) == 5
    assert {item["reason"] for item in rejected} == {"generic_act", "empty_text"}
    assert Counter(item["reason"] for item in rejected) == {
        "generic_act": 1, "empty_text": 1
    }
    first_probe = next(
        item for item in accepted
        if item["example_id"] == "mathdial:train:0000:t00"
    )
    assert first_probe["completion"][0]["content"] == "Ask the first question."
    assert "Teacher:" not in first_probe["completion"][0]["content"]
    late_probe = next(
        item for item in accepted
        if item["example_id"] == "mathdial:train:0000:t03"
    )
    serialized = json.dumps({"plan": late_probe["plan"], "prompt": late_probe["prompt"]})
    assert "FUTURE_STUDENT_PRIVATE" not in serialized
    assert "FUTURE_TEACHER_PRIVATE" not in serialized
    assert "Let's continue from the earlier turn." not in serialized
    assert any(
        turn["text"] == "Let me try."
        for turn in late_probe["plan"]["response_context"]["recent_dialogue"]
    )
    assert assignments[7] == next(
        item["split"] for item in accepted
        if item["example_id"] == "mathdial:train:0000:t00"
    )
    assert assignments[7] == next(
        item["split"] for item in accepted
        if item["example_id"] == "mathdial:test:0000:t00"
    )
    assert {assignments[qid] for qid in assignments} <= {
        "train", "validation", "test"
    }


def test_ground_truth_is_reveal_only_and_private_labels_never_enter_prompt():
    config, mapping, rows = inline_input()
    accepted, _, _, _ = build_examples(rows, config, mapping)
    for example in accepted:
        plan = example["plan"]
        prompt_text = "\n".join(item["content"] for item in example["prompt"])
        serialized = json.dumps({"plan": plan, "prompt": example["prompt"]})
        assert "TEACHER_CONFUSION_PRIVATE" not in serialized
        assert "SELF_CORRECTNESS_PRIVATE" not in serialized
        assert "Ada learner profile must not leak" not in serialized
        if plan["teaching_action"]["move"] == "EXPLAIN_CONCEPT":
            assert plan["content"]["reveal_solution"] == "Private ground truth: 17"
            assert "Private ground truth: 17" in prompt_text
        else:
            assert plan["content"]["reveal_solution"] is None
            assert "Private ground truth: 17" not in prompt_text
    focus = next(item for item in accepted if item["plan"]["teaching_action"]["move"] == "HINT")
    assert focus["plan"]["response_context"]["recent_dialogue"][0]["text"] == (
        "I subtracted the wrong amounts."
    )


def test_opening_student_solution_turn_can_be_disabled():
    config, mapping, rows = inline_input(include_opening=True)
    accepted_open, _, _, _ = build_examples(rows, config, mapping)
    config["include_student_solution_as_opening_turn"] = False
    accepted_closed, _, _, _ = build_examples(rows, config, mapping)
    focus_open = next(
        item
        for item in accepted_open
        if item["plan"]["teaching_action"]["move"] == "HINT"
    )
    focus_closed = next(
        item
        for item in accepted_closed
        if item["plan"]["teaching_action"]["move"] == "HINT"
    )
    assert focus_open["plan"]["response_context"]["recent_dialogue"][0]["text"] == (
        "I subtracted the wrong amounts."
    )
    assert focus_closed["plan"]["response_context"]["recent_dialogue"] == []


def test_per_turn_labels_replace_the_act_mapping(tmp_path):
    config, _, rows = inline_input()
    labeled = {
        "mathdial:train:0000:t00": "REFLECT",
        "mathdial:train:0000:t02": "PROBE",
        "mathdial:train:0000:t03": "EXCLUDE",
        "mathdial:train:0001:t00": "PRACTICE",
    }
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("".join(
        json.dumps({"example_id": example_id, "label": label, "model": "labeler-x",
                    "prompt_version": "prompt-y", "status": "labeled"}) + "\n"
        for example_id, label in labeled.items()
    ), encoding="utf-8")
    map_path = tmp_path / "map.yaml"
    map_path.write_text(f"status: model_labeled\nlabels_file: {labels_path}\n", encoding="utf-8")
    move_map = load_move_map(map_path)
    assert move_map["kind"] == "per_turn"
    accepted, rejected, _, _ = build_examples(rows, config, move_map)
    assert {item["example_id"]: item["plan"]["teaching_action"]["move"] for item in accepted} == {
        "mathdial:train:0000:t00": "REFLECT",
        "mathdial:train:0000:t02": "PROBE",
        "mathdial:train:0001:t00": "PRACTICE",
    }
    assert Counter(item["reason"] for item in rejected) == {
        "excluded_by_label": 1, "unlabeled_turn": 2, "empty_text": 1,
    }
    reflect = next(item for item in accepted if item["example_id"] == "mathdial:train:0000:t00")
    assert reflect["action_label_source"] == "mapped"
    assert reflect["extensions"]["mapping_status"] == "model_labeled"
    assert reflect["extensions"]["move_label"] == {
        "source": "model", "model": "labeler-x", "prompt_version": "prompt-y", "status": "labeled",
    }
    assert reflect["plan"]["constraints"]["reveal_allowed"] is False
    assert reflect["plan"]["content"]["reveal_solution"] is None
    assert reflect["plan"]["assistance"] == "none"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"example_id": "x", "label": "TELLING"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        load_turn_labels(bad)
    act_map = tmp_path / "act.yaml"
    act_map.write_text("status: draft\nmappings:\n  probing:\n    move: PROBE\n", encoding="utf-8")
    assert load_move_map(act_map)["kind"] == "per_act"


def test_targets_using_an_unseen_profile_name_are_flagged_without_storing_it():
    config, mapping, rows = inline_input()
    rows["train"].append((2, {
        **rows["train"][0][1],
        "qid": 11,
        "conversation": (
            "Teacher: (probing) Hi Ada, walk me through your answer.|EOM|"
            "Ada: I added them.|EOM|"
            "Teacher: (focus) Look at what is being taken away."
        ),
    }))
    accepted, _, _, _ = build_examples(rows, config, mapping)
    flags = {item["example_id"]: item["extensions"]["target_uses_unseen_name"] for item in accepted}
    assert flags["mathdial:train:0002:t00"] is True
    assert flags["mathdial:train:0002:t02"] is False
    assert sum(flags.values()) == 1
    for item in accepted:
        assert "Ada" not in json.dumps(item["extensions"])
        assert "Ada learner profile must not leak" not in json.dumps(item)


def test_raw_count_mismatch_reports_expected_and_actual():
    _, _, rows = inline_input()
    actual = compute_raw_counts(rows)
    assert actual["dialogues"] == 3
    assert actual["unique_qids"] == 2
    with pytest.raises(RawCountMismatch, match='"train_rows":\\{"actual":2,"expected":99\\}'):
        validate_raw_counts(rows, {"train_rows": 99})


def test_all_qid_rows_share_one_regrouped_split():
    _, _, rows = inline_input()
    assignment = assign_qid_splits(rows, "mathdial")
    assert len(assignment) == 2
    for official_rows in rows.values():
        for _, row in official_rows:
            assert assignment[int(row["qid"])] in {"train", "validation", "test"}


def test_mapping_audit_sampler_is_deterministic_stratified_and_column_exact():
    _, mapping, _ = inline_input()
    candidates = []
    for act in ("probing", "focus", "telling", "generic"):
        for index in range(5):
            candidates.append({
                "example_id": f"mathdial:train:{act}:{index}",
                "qid": index,
                "source_act": act,
                "context_last_student": f"prior {index}",
                "teacher_text": f"{act} quote {index}",
            })
    first = sample_audit_rows(candidates, mapping, seed=17, per_act=3)
    second = sample_audit_rows(candidates, mapping, seed=17, per_act=3)
    assert first == second
    assert all(set(row) == set(AUDIT_COLUMNS) for row in first)
    assert Counter(row["source_act"] for row in first) == {
        "probing": 3, "focus": 3, "telling": 3, "generic": 3
    }
    assert first == sorted(first, key=lambda row: (row["source_act"], row["example_id"]))
    assert {row["draft_move"] for row in first if row["source_act"] == "generic"} == {
        "EXCLUDED_generic_act"
    }
