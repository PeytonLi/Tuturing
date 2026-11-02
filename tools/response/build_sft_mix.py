"""Combine original_v1 and mathdial_v1 SFT examples into the mix_v1 sets.

PRD 13.2 proposes about 60 percent original probability examples and 40
percent permitted general tutoring examples, capped by the data that passes
review, sampled by action and concept coverage rather than source size, with
any repeated sampling reported as repeated exposure. Every original example
of a split is used; MathDial examples are drawn from the same split (they keep
their own qid-level family split) in a per-move round robin until the mix
reaches the configured share. Protected-assessment MathDial examples never
reach this tool: prepare_sft already rejects them (see the leak review).

    python -m tools.response.build_sft_mix --config configs/response/sft_mix_v1.yaml
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TOOL_VERSION = "tools.response.build_sft_mix v1"
SPLITS = ("train", "validation", "test")
DEFAULT_CONFIG = "configs/response/sft_mix_v1.yaml"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def move_of(example):
    return example["plan"]["teaching_action"]["move"]


def concept_of(example):
    return example["plan"].get("concept_id")


def sample_by_move(examples, count, rng):
    """Draw ``count`` examples in a per-move round robin so rare moves are covered first."""
    by_move = defaultdict(list)
    for example in sorted(examples, key=lambda item: item["example_id"]):
        by_move[move_of(example)].append(example)
    for bucket in by_move.values():
        rng.shuffle(bucket)
    moves = sorted(by_move)
    chosen = []
    while len(chosen) < count and any(by_move[move] for move in moves):
        for move in moves:
            if len(chosen) >= count:
                break
            if by_move[move]:
                chosen.append(by_move[move].pop())
    return chosen


def mathdial_target(original_count, original_share, mathdial_share):
    """How many MathDial examples give the configured share next to the original ones."""
    if original_count == 0 or mathdial_share <= 0:
        return 0
    return round(original_count * mathdial_share / original_share)


def build_split(original, mathdial, config, rng):
    """Return (records, stats) for one split."""
    repeats = int(config.get("original_repeats", 1))
    shares = config["shares"]
    kept_original = sorted(original, key=lambda item: item["example_id"])
    exposures = [example for _ in range(repeats) for example in kept_original]
    target = min(
        mathdial_target(len(exposures), shares["original"], shares["mathdial"]),
        len(mathdial),
    )
    chosen_mathdial = sample_by_move(mathdial, target, rng)
    records = [*exposures, *chosen_mathdial]
    rng.shuffle(records)
    total = len(records)
    stats = {
        "total": total,
        "original_examples": len(kept_original),
        "original_exposures": len(exposures),
        "original_repeats": repeats,
        "mathdial_available": len(mathdial),
        "mathdial_target": mathdial_target(len(exposures), shares["original"], shares["mathdial"]),
        "mathdial_sampled": len(chosen_mathdial),
        "mix": {
            "original": round(len(exposures) / total, 4) if total else None,
            "mathdial": round(len(chosen_mathdial) / total, 4) if total else None,
        },
        "moves": {
            "original": dict(sorted(Counter(move_of(item) for item in kept_original).items())),
            "mathdial": dict(sorted(Counter(move_of(item) for item in chosen_mathdial).items())),
            "mathdial_available": dict(sorted(Counter(move_of(item) for item in mathdial).items())),
        },
        "concepts": {
            "original": dict(sorted(Counter(concept_of(item) for item in kept_original).items())),
        },
        "repeated_exposure": {
            "original_examples_repeated": len(kept_original) if repeats > 1 else 0,
            "times": repeats,
            "mathdial_examples_repeated": 0,
        },
    }
    return records, stats


def build(config_path):
    config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rng = random.Random(int(config.get("seed", 20261102)))
    sources = config["sources"]
    out_dir = ROOT / config["out_dir"]
    per_split = {}
    output_files = []
    for split in SPLITS:
        original = read_jsonl(ROOT / sources["original"]["dir"] / f"{split}.jsonl")
        mathdial = read_jsonl(ROOT / sources["mathdial"]["dir"] / f"{split}.jsonl")
        records, stats = build_split(original, mathdial, config, rng)
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        per_split[split] = stats
        output_files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                             "rows": len(records)})
    source_info = {}
    for name, source in sources.items():
        manifest_path = ROOT / source["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_info[name] = {
            "dir": source["dir"],
            "manifest": {"path": source["manifest"], "sha256": sha256_file(manifest_path)},
            "split_files": {
                split: {
                    "path": f"{source['dir']}/{split}.jsonl",
                    "sha256": sha256_file(ROOT / source["dir"] / f"{split}.jsonl"),
                }
                for split in SPLITS
            },
            "action_label_source": manifest.get("action_label_source")
            or ("mapped" if name == "mathdial" else None),
            "mapping": manifest.get("mapping"),
            "reviewer_status": manifest.get("reviewer_status", "pending"),
        }
    mix_manifest = {
        "dataset_id": config["dataset_id"],
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "tool_version": TOOL_VERSION,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
        "seed": int(config.get("seed", 20261102)),
        "shares_target": config["shares"],
        "rule": (
            "Every original example of a split is used (repeated original_repeats times); "
            "MathDial examples of the same split are drawn in a per-move round robin until "
            "the mathdial share is reached or the split's MathDial pool is exhausted. MathDial "
            "keeps its own qid-level split; protected-assessment MathDial examples were "
            "already rejected by prepare_sft."
        ),
        "sources": source_info,
        "counts_per_split": per_split,
        "totals": {
            "examples": sum(stats["total"] for stats in per_split.values()),
            "original_exposures": sum(stats["original_exposures"] for stats in per_split.values()),
            "mathdial_sampled": sum(stats["mathdial_sampled"] for stats in per_split.values()),
        },
        "repeated_exposure": {
            split: stats["repeated_exposure"] for split, stats in per_split.items()
        },
        "output_files": output_files,
        "notes": [
            "Original examples are synthetic and pending review; MathDial labels are model-made "
            "(action_label_source mapped), never human labels.",
            "The mix is capped by what passes: the MathDial count follows the original count.",
        ],
    }
    manifest_path = ROOT / config["manifest_path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(mix_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return mix_manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    manifest = build(args.config)
    print(json.dumps({
        split: {
            "total": stats["total"],
            "original": stats["original_exposures"],
            "mathdial": stats["mathdial_sampled"],
            "mix": stats["mix"],
        }
        for split, stats in manifest["counts_per_split"].items()
    }, sort_keys=True))
    print(f"manifest: {manifest['config']['path']} -> {manifest['output_files'][0]['path']}")


if __name__ == "__main__":
    main()
