import json
import random
from pathlib import Path

import yaml

from tools.response import build_sft_mix as mixer
from tools.response.prepare_sft import _teaching_plan

ROOT = Path(__file__).resolve().parents[3]


def _example(example_id, move, source, concept="C01"):
    return {
        "example_id": example_id,
        "source_id": source,
        "plan": {"teaching_action": {"move": move}, "concept_id": concept},
        "completion": [{"role": "assistant", "content": f"reply {example_id}"}],
    }


def _original(count):
    moves = ("PROBE", "HINT", "PRACTICE")
    return [_example(f"original_v1:o{index}", moves[index % 3], "original_v1",
                     concept=("C01", "C02")[index % 2]) for index in range(count)]


def _mathdial(count):
    moves = ("PROBE", "HINT", "EXPLAIN_CONCEPT", "REFLECT")
    return [_example(f"mathdial:m{index}", moves[index % 4], "mathdial", concept="EXT_MATHDIAL")
            for index in range(count)]


def test_sample_by_move_covers_rare_moves_first():
    pool = [*_mathdial(40), _example("mathdial:rare", "PRACTICE", "mathdial")]
    chosen = mixer.sample_by_move(pool, 5, random.Random(1))
    assert len(chosen) == 5
    assert {mixer.move_of(item) for item in chosen} == {
        "EXPLAIN_CONCEPT", "HINT", "PRACTICE", "PROBE", "REFLECT"
    }
    assert len({item["example_id"] for item in chosen}) == 5


def test_split_mix_follows_the_original_count_and_reports_repeats():
    config = {"shares": {"original": 0.6, "mathdial": 0.4}, "original_repeats": 1}
    records, stats = mixer.build_split(_original(30), _mathdial(100), config, random.Random(3))
    assert stats["original_exposures"] == 30 and stats["mathdial_sampled"] == 20
    assert stats["mix"] == {"original": 0.6, "mathdial": 0.4}
    assert stats["moves"]["mathdial"] == {"EXPLAIN_CONCEPT": 5, "HINT": 5, "PROBE": 5, "REFLECT": 5}
    assert stats["repeated_exposure"]["original_examples_repeated"] == 0
    assert len(records) == 50 and len({item["example_id"] for item in records}) == 50

    capped_records, capped = mixer.build_split(
        _original(30), _mathdial(8), config, random.Random(3)
    )
    assert capped["mathdial_target"] == 20 and capped["mathdial_sampled"] == 8
    assert capped["mix"]["mathdial"] == round(8 / 38, 4)

    config["original_repeats"] = 2
    repeated_records, repeated = mixer.build_split(
        _original(30), _mathdial(100), config, random.Random(3)
    )
    assert repeated["original_exposures"] == 60 and repeated["mathdial_sampled"] == 40
    assert repeated["repeated_exposure"] == {
        "original_examples_repeated": 30, "times": 2, "mathdial_examples_repeated": 0
    }
    assert len(repeated_records) == 100


def test_build_writes_split_files_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(mixer, "ROOT", tmp_path)
    for name, examples in (("original_v1", _original(12)), ("mathdial_v1", _mathdial(30))):
        for split in mixer.SPLITS:
            mixer.write_jsonl(tmp_path / "data/sft" / name / f"{split}.jsonl", examples)
        manifest = {"action_label_source": "synthetic" if name == "original_v1" else "mapped",
                    "mapping": {"status": "model_labeled"}}
        path = tmp_path / "data/manifests/response" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "dataset_id": "mix_test",
        "out_dir": "data/sft/mix_test",
        "manifest_path": "data/manifests/response/mix_test.json",
        "seed": 5,
        "shares": {"original": 0.6, "mathdial": 0.4},
        "original_repeats": 1,
        "sources": {
            "original": {"dir": "data/sft/original_v1",
                         "manifest": "data/manifests/response/original_v1.json"},
            "mathdial": {"dir": "data/sft/mathdial_v1",
                         "manifest": "data/manifests/response/mathdial_v1.json"},
        },
    }
    config_path = tmp_path / "configs/response/sft_mix_test.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = mixer.build("configs/response/sft_mix_test.yaml")
    for split in mixer.SPLITS:
        rows = mixer.read_jsonl(tmp_path / "data/sft/mix_test" / f"{split}.jsonl")
        assert len(rows) == 20
        assert sum(row["source_id"] == "mathdial" for row in rows) == 8
    assert manifest["totals"] == {
        "examples": 60, "original_exposures": 36, "mathdial_sampled": 24,
        "mathdial_admitted": 90, "mathdial_available": 90,
    }
    assert manifest["sources"]["mathdial"]["action_label_source"] == "mapped"
    assert manifest["sources"]["mathdial"]["mapping"] == {"status": "model_labeled"}
    assert len(manifest["output_files"]) == 3
    again = mixer.build("configs/response/sft_mix_test.yaml")
    assert [item["sha256"] for item in again["output_files"]] == [
        item["sha256"] for item in manifest["output_files"]
    ]


def _mathdial_example(example_id, move, text):
    row = {
        "qid": 1,
        "question": "Tom has 12 apples and eats 5. How many apples are left?",
        "ground_truth": "12 - 5 = 7\n7",
    }
    plan = _teaching_plan("mathdial:train:0001", example_id, row, move, [], "", {"max_words": 150})
    return {"example_id": example_id, "source_id": "mathdial", "plan": plan,
            "completion": [{"role": "assistant", "content": text}]}


def test_mathdial_admission_rules_follow_the_committed_config():
    rules = yaml.safe_load((ROOT / "configs/response/sft_mix_v1.yaml").read_text())[
        "mathdial_admission"
    ]
    examples = [
        _mathdial_example("keep_hint", "HINT", "Look at how many apples Tom eats."),
        _mathdial_example(
            "keep_practice", "PRACTICE",
            "Try it again: Tom has 12 apples and eats 5. How many apples are left?",
        ),
        _mathdial_example("unsupported", "HINT", "He had 12, so 12 x 3 = 36 apples in all."),
        _mathdial_example("discloses", "HINT", "So 7 apples are left."),
        _mathdial_example("two_questions", "PROBE", "How many did he eat? How many are left?"),
        _mathdial_example(
            "new_item", "PRACTICE",
            "Now try this one: Sam has 3 pears and gives away 2. How many pears remain?",
        ),
        _mathdial_example("worked", "WORKED_EXAMPLE", "If Ann has 4 plums and eats 1, 3 remain."),
    ]
    admitted, reasons, by_move = mixer.admit_mathdial(examples, rules)
    assert [item["example_id"] for item in admitted] == ["keep_hint", "keep_practice"]
    assert reasons == {
        "failed_check:math_supported": 1,
        "failed_check:disclosure_allowed": 1,
        "failed_check:question_count_valid": 1,
        "new_item_under_presenting_move": 1,
        "excluded_move:WORKED_EXAMPLE": 1,
    }
    assert by_move["PRACTICE"] == {"new_item_under_presenting_move": 1}
    assert mixer.admit_mathdial(examples, {})[0] == examples


def test_committed_config_is_consistent():
    config = yaml.safe_load((ROOT / "configs/response/sft_mix_v1.yaml").read_text())
    assert config["shares"] == {"original": 0.6, "mathdial": 0.4}
    assert config["out_dir"].startswith("data/sft/")
    for source in config["sources"].values():
        assert (ROOT / source["manifest"]).exists()
