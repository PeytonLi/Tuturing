"""Build the committed deterministic eight-action response smoke suite."""

import argparse
import json
from pathlib import Path

from packages.response.fallbacks import render_fallback
from packages.response.prompt import compile_prompt

PLAN_DIR = Path("tests/unit/response/fixtures/plans")
OUTPUT_PATH = Path("evaluation/response/smoke_cases_v1.jsonl")


def build_smoke_cases(plan_dir=PLAN_DIR):
    cases = []
    for path in sorted(Path(plan_dir).glob("*.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem
        cases.append({
            "example_id": f"smoke_{name}_v1",
            "dialogue_id": f"smoke_dialogue_{name}_v1",
            "original_problem_id": f"smoke_problem_{name}_v1",
            "template_family_id": f"smoke_family_{name}_v1",
            "split": "test",
            "source_id": "original_fixture",
            "source_revision": "fixtures_v1",
            "license_ref": "original",
            "synthetic": True,
            "reviewer_status": "pending",
            "action_label_source": "synthetic",
            "plan": plan,
            "prompt": compile_prompt(plan),
            "completion": [{"role": "assistant", "content": render_fallback(plan)}],
            "checks": {
                "math_correct": None,
                "faithful": None,
                "no_early_answer": None,
            },
        })
    return sorted(cases, key=lambda item: item["example_id"])


def write_smoke_cases(output_path=OUTPUT_PATH, plan_dir=PLAN_DIR):
    cases = build_smoke_cases(plan_dir)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases to {path}")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    args = parser.parse_args(argv)
    write_smoke_cases(args.out)


if __name__ == "__main__":
    main()
