"""Build the original_v1 SFT examples from authored records and Person B's content.

Each authored record (data/response/original_v1/records/*.jsonl) names one of
Person B's teaching families and a sampler seed, a teaching action, a
navigation action, the learner's message, the recent dialogue and the tutor's
reply. The build resolves the item with B's sampler and solver, builds the
TeachingPlan with B's plan builder, applies the Person A normalizations listed
in instructions_v1.md, renders the placeholders, and admits an example only
when every check passes: learning_v1 schema and plan rules, B's response stub,
B's grader on the learner's answer, the six response checks, solver agreement,
the SFTExample schema, the split invariants and duplicate detection.

    python -m tools.response.build_original_sft
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from packages.learning.actions import encode_navigator, encode_strategist
from packages.learning.content.templates.families import FAMILIES
from packages.learning.episode import _build_plan, _problem_stem
from packages.learning.grader import grade_answer
from packages.learning.plan_rules import validate_plan_rules
from packages.response.hashing import canonical_json
from packages.response.numbers import extract_numbers
from packages.response.prompt import PROMPT_VERSION, compile_prompt
from packages.response.validate import VALIDATOR_VERSION, run_checks
from packages.simulator.response_adapter import realize_plan

ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "tools.response.build_original_sft v1"
SOURCE_ID = "original_v1"
INSTRUCTIONS_VERSION = "instructions_v1"
GENERATING_MODEL = {
    "name": "Claude Fable 5.1",
    "model_id": "claude-fable-5-1",
    "runtime": "Claude Code agent acting for Person A",
}
MOVES = (
    "PROBE", "HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE",
    "PRACTICE", "CHALLENGE", "CHECK_MASTERY", "REFLECT",
)
SPLITS = ("train", "validation", "test")
CHECK_NAMES = (
    "schema_valid", "action_faithful", "math_supported", "disclosure_allowed",
    "length_valid", "question_count_valid",
)
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")
PHRASE_FRACTION = re.compile(r"\b(\d+)\s+(?:out of|in|over|of)\s+(\d+)\b")
WORKED_EXAMPLE = re.compile(
    r"has (\d+) favorable sections out of (\d+)\. The answer is (\d+)/(\d+)\."
)
WORD_FRACTIONS = {
    Fraction(1, 2): ("half", "halves", "fifty percent", "fifty-fifty"),
    Fraction(1, 4): ("quarter", "a fourth", "one fourth"),
    Fraction(3, 4): ("three quarters", "three-quarters", "three fourths"),
    Fraction(1, 3): ("a third", "one third"),
    Fraction(2, 3): ("two thirds", "two-thirds"),
    Fraction(1, 5): ("a fifth", "one fifth"),
    Fraction(1, 10): ("a tenth", "one tenth"),
}
NEAR_DUPLICATE_THRESHOLD = 0.85
DEFAULT_RECORDS_DIR = "data/response/original_v1"
DEFAULT_OUT_DIR = "data/sft/original_v1"
DEFAULT_MANIFEST = "data/manifests/response/original_v1.json"
DEFAULT_REJECTIONS = "evaluation/response/original_v1_rejection_report.json"
DEFAULT_CASES = "evaluation/response/probability_cases_v1.jsonl"
DEFAULT_REVIEW = "evaluation/response/original_v1_review_sample.md"
DEFAULT_MATHDIAL_DIR = "data/sft/mathdial_v1"


class RecordError(ValueError):
    """Raised when an authored record cannot be built into a valid example."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(records_dir):
    files = sorted((Path(records_dir) / "records").glob("*.jsonl"))
    records = []
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_source"] = f"{path.name}:{line_number}"
            records.append(record)
    return records, files


def load_concept_names():
    document = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    return {item["concept_id"]: item["name"] for item in document["concepts"]}


def _schema_validator(path, definition):
    schema = json.loads((ROOT / path).read_text(encoding="utf-8"))
    for key in ("oneOf", "anyOf", "allOf", "$ref"):
        schema.pop(key, None)
    schema["$ref"] = f"#/$defs/{definition}"
    return Draft202012Validator(schema, format_checker=FormatChecker())


def exact_decimal(fraction):
    """Return the exact decimal text of a terminating fraction, or None."""
    denominator = fraction.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        return None
    text = format(Decimal(fraction.numerator) / Decimal(fraction.denominator), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def answer_value(spec, family, instance):
    kind = spec["kind"]
    answer = instance["answer"]
    if kind == "correct":
        return answer
    if kind == "arithmetic":
        value = answer + Fraction(1, int(spec.get("k", 3)))
        return value if value != answer else answer + Fraction(1, 3)
    if kind in family.error_generators:
        return family.generate_error(kind, instance)
    raise RecordError(f"unknown answer kind {kind}")


def format_answer(value, fmt, instance):
    if fmt == "fraction":
        return f"{value.numerator}/{value.denominator}"
    if fmt == "unreduced":
        if value != instance["answer"]:
            raise RecordError("unreduced format is only defined for the correct answer")
        return f"{instance['numerator']}/{instance['denominator']}"
    if fmt == "decimal":
        text = exact_decimal(value)
        if text is None:
            raise RecordError(f"{value} has no terminating decimal form")
        return text
    if fmt == "percent":
        text = exact_decimal(value * 100)
        if text is None:
            raise RecordError(f"{value} has no exact percent form")
        return f"{text}%"
    raise RecordError(f"unknown answer format {fmt}")


def render(text, context, where):
    def substitute(match):
        key = match.group(1)
        if context.get(key) is None:
            raise RecordError(f"placeholder {{{key}}} is not available in {where}")
        return str(context[key])

    return PLACEHOLDER.sub(substitute, text)


def phrase_discloses(text, answer):
    lowered = text.casefold()
    for match in PHRASE_FRACTION.finditer(lowered):
        numerator, denominator = int(match.group(1)), int(match.group(2))
        if denominator and Fraction(numerator, denominator) == answer:
            return True
    return any(
        value == answer and any(word in lowered for word in words)
        for value, words in WORD_FRACTIONS.items()
    )


def normalize_for_dedup(text):
    text = text.casefold()
    text = re.sub(r"\d+(?:[.,/]\d+)?%?", "#", text)
    text = re.sub(r"[^a-z#\s]+", " ", text)
    return " ".join(text.split())


def shingles(normalized, size=3):
    words = normalized.split()
    if len(words) <= size:
        return {tuple(words)}
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


def jaccard(left, right):
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def answer_text(fraction):
    return f"{fraction.numerator}/{fraction.denominator}"


def build_plan(record, family, instance, split_manifest, names):
    split = family.split
    scenario = {
        "scenario_id": split_manifest["plan_envelope_scenarios"][split][0],
        "split": split,
    }
    group = record["group"]
    turn = int(record["turn"])
    move = record["move"]
    operation, target = record["nav"]
    if operation == "END_SESSION" or not isinstance(target, str):
        raise RecordError("a teaching plan needs a navigation operation with a target concept")
    reveal = bool(record.get("reveal", False))
    if reveal and move != "EXPLAIN_CONCEPT":
        raise RecordError("reveal is only authored with EXPLAIN_CONCEPT, as B's collector does")
    hint_level = int(record.get("hint_level", 0))
    active_item = {
        "family": family,
        "instance": instance,
        "item_id": f"{group}-item-{instance['seed']}",
        "hint_level": hint_level,
    }
    plan = _build_plan(
        scenario=scenario,
        episode_id=f"{SOURCE_ID}-{group}",
        turn=turn,
        current=family.concept_id,
        goal=record["goal"],
        navigation_action_id=encode_navigator(operation, int(target[1:]) - 1),
        teaching_action_id=encode_strategist(move, record["difficulty"], record["pace"]),
        active_item=active_item,
        beliefs={},
        hint_level=hint_level,
        previous_step=record.get("previous_step"),
        assessment_independent=bool(record.get("independent", False)),
        actor_eligible=not reveal,
        forced_answer=reveal,
    )
    plan["content"]["concept_name"] = names[family.concept_id]
    plan["content"]["goal_name"] = names[record["goal"]]
    if not plan["constraints"]["reveal_allowed"]:
        plan["constraints"]["forbidden_answer_fingerprints"] = [answer_text(instance["answer"])]
    return plan


def placeholder_context(record, family, instance, plan, learner_answer, prior_answers):
    content = plan["content"]
    facts = content["approved_facts"]
    context = {
        "stem": content["problem_stem"] or _problem_stem(family, instance),
        "hint": content["permitted_hint"]
        or family.hint_ladder[min(int(record.get("hint_level", 0)), 2)],
        "hint1": family.hint_ladder[0],
        "hint2": family.hint_ladder[1],
        "hint3": family.hint_ladder[2],
        "example": content["worked_example"],
        "previous_step": content["previous_step"],
        "solution": content["reveal_solution"],
        "n": instance["numerator"],
        "d": instance["denominator"],
        "kind": family.item_kind,
        "concept": content["concept_name"],
        "goal": content["goal_name"],
        "facts": " ".join(facts),
        "fact1": facts[0] if facts else None,
        "fact2": facts[1] if len(facts) > 1 else None,
        "answer": learner_answer,
    }
    context.update(prior_answers)
    return context


def solver_problems(plan, family, instance, reply):
    """Return reasons why the plan or reply disagrees with B's solver."""
    problems = []
    answer = instance["answer"]
    expected = answer_text(answer)
    parameters = {"numerator": instance["numerator"], "denominator": instance["denominator"]}
    if family.solver(parameters) != answer:
        problems.append("instance answer differs from the family solver")
    content = plan["content"]
    constraints = plan["constraints"]
    if content["reveal_solution"] is not None and content["reveal_solution"] != expected:
        problems.append("reveal_solution differs from the solver answer")
    fingerprints = constraints["forbidden_answer_fingerprints"]
    if not constraints["reveal_allowed"] and fingerprints != [expected]:
        problems.append("forbidden fingerprint differs from the solver answer")
    example = content["worked_example"]
    if example is not None:
        match = WORKED_EXAMPLE.search(example)
        if not match:
            problems.append("worked example has an unexpected form")
        else:
            favorable, total, numerator, denominator = (int(group) for group in match.groups())
            if Fraction(favorable, total) != Fraction(numerator, denominator):
                problems.append("worked example answer disagrees with its own counts")
            if not constraints["reveal_allowed"] and Fraction(numerator, denominator) == answer:
                problems.append("worked example answer equals the protected item answer")
    allowed = set()
    for value in (
        content["problem_stem"], *content["approved_facts"], content["permitted_hint"],
        content["worked_example"], content["previous_step"],
        content["reveal_solution"] if constraints["reveal_allowed"] else None,
    ):
        if isinstance(value, str):
            allowed.update(token.value for token in extract_numbers(value))
    for token in extract_numbers(reply):
        if token.kind != "int" and token.value not in allowed:
            problems.append(f"reply number {token.raw.strip()} is not in the approved content")
        elif token.kind == "int" and token.value not in allowed and not 0 <= token.value <= 10:
            problems.append(f"reply integer {token.raw.strip()} is not in the approved content")
    return problems


def build_example(record, split_manifest, names, validators):
    """Return (example, meta) for one record or raise RecordError."""
    family = FAMILIES.get(record["family"])
    if family is None:
        raise RecordError(f"unknown family {record['family']}")
    if family.role != "teaching":
        raise RecordError("assessment families are reserved and never SFT targets")
    expected_split = split_manifest["teaching_families"][family.template_family_id]["sft_split"]
    if expected_split != family.split:
        raise RecordError("split manifest disagrees with B's family split; refreeze first")
    instance = family.sample(int(record["seed"]))
    if instance["answer"] >= 1:
        raise RecordError("instance answer is not below one; choose another seed")
    attack_group = record.get("attack_group")
    if attack_group is not None:
        groups = split_manifest["adversarial_attack_groups"]["groups"]
        if attack_group not in groups:
            raise RecordError(f"unknown attack group {attack_group}")
        if groups[attack_group]["split"] != family.split:
            raise RecordError("attack group is pinned to a different split than the family")
    hidden_solution = "{solution}" in record["reply"] and not record.get("reveal")
    if "{answer}" in record["reply"] or hidden_solution:
        raise RecordError("the reply must not echo the learner's answer or a hidden solution")

    prior_answers = {}
    for name, spec in (record.get("prior_answers") or {}).items():
        prior_answers[name] = format_answer(
            answer_value(spec, family, instance), spec.get("format", "fraction"), instance
        )
    learner_answer = None
    answer_spec = record.get("answer")
    learner_type = record["learner_type"]
    if answer_spec is not None:
        value = answer_value(answer_spec, family, instance)
        learner_answer = format_answer(value, answer_spec.get("format", "fraction"), instance)
        grade = grade_answer(learner_answer, instance, family)
        expected_grade = "correct" if answer_spec["kind"] == "correct" else "incorrect"
        if grade != expected_grade:
            raise RecordError(
                f"B's grader says {grade} for {learner_answer}, expected {expected_grade}"
            )
        type_says_correct = (
            learner_type.endswith("correct") and not learner_type.startswith("incorrect")
        )
        if type_says_correct != (answer_spec["kind"] == "correct"):
            raise RecordError(f"learner_type {learner_type} disagrees with answer kind")
        if learner_type.startswith("incorrect_M") and answer_spec["kind"] != learner_type[10:]:
            raise RecordError(f"learner_type {learner_type} names a different misconception")
        if answer_spec["kind"].startswith("M") and record.get("misconception_summary") is None:
            raise RecordError("a misconception answer needs a qualified misconception summary")
    elif learner_type.startswith("incorrect") or learner_type.endswith("correct"):
        raise RecordError(f"learner_type {learner_type} needs an answer spec")

    plan = build_plan(record, family, instance, split_manifest, names)
    context = placeholder_context(record, family, instance, plan, learner_answer, prior_answers)
    message = render(record["message"], context, "message")
    dialogue = [
        {"role": turn["role"], "text": render(turn["text"], context, "dialogue")}
        for turn in record.get("dialogue") or []
    ]
    if message.strip():
        if not dialogue or dialogue[-1] != {"role": "student", "text": message}:
            dialogue.append({"role": "student", "text": message})
    elif learner_type != "first_turn":
        raise RecordError("only first_turn records may have an empty learner message")
    plan["response_context"] = {
        "last_student_message": message,
        "recent_dialogue": dialogue,
        "qualified_misconception_summary": record.get("misconception_summary"),
    }
    reply = render(record["reply"], context, "reply")

    schema_errors = [error.message for error in validators["plan"].iter_errors(plan)]
    schema_errors.extend(validate_plan_rules(plan))
    if schema_errors:
        raise RecordError("plan invalid: " + "; ".join(schema_errors[:3]))
    try:
        realized = realize_plan(plan, request_id=f"{SOURCE_ID}-{record['id']}-request")
    except Exception as error:  # B's stub raises ResponseValidationError subclasses
        raise RecordError(f"B's response stub rejected the plan: {error}") from error
    if not all(realized["checks"].values()):
        raise RecordError("B's response stub failed its own checks on the plan")

    checks = run_checks(plan, reply, plan["teaching_action"])
    failed = [name for name in CHECK_NAMES if not checks[name]]
    problems = solver_problems(plan, family, instance, reply)
    phrase = (
        not plan["constraints"]["reveal_allowed"] and phrase_discloses(reply, instance["answer"])
    )
    if failed or problems or phrase:
        detail = []
        if failed:
            detail.append("failed checks: " + ", ".join(failed))
        detail.extend(problems)
        if phrase:
            detail.append("reply states the protected answer in words")
        raise RecordError("; ".join(detail))
    example = {
        "example_id": f"{SOURCE_ID}:{record['id']}",
        "dialogue_id": f"{SOURCE_ID}:{record['group']}",
        "original_problem_id": f"{family.template_family_id}:seed{instance['seed']}",
        "template_family_id": family.template_family_id,
        "split": family.split,
        "source_id": SOURCE_ID,
        "source_revision": INSTRUCTIONS_VERSION,
        "license_ref": "original",
        "synthetic": True,
        "reviewer_status": "pending",
        "action_label_source": "synthetic",
        "plan": plan,
        "prompt": compile_prompt(plan),
        "completion": [{"role": "assistant", "content": reply}],
        "checks": {"math_correct": True, "faithful": True, "no_early_answer": True},
    }
    example_errors = [error.message for error in validators["example"].iter_errors(example)]
    if example_errors:
        raise RecordError("SFTExample invalid: " + "; ".join(example_errors[:3]))
    meta = {
        "example_id": example["example_id"],
        "record_id": record["id"],
        "source": record["_source"],
        "split": family.split,
        "concept_id": family.concept_id,
        "family": family.template_family_id,
        "seed": instance["seed"],
        "move": record["move"],
        "difficulty": record["difficulty"],
        "pace": record["pace"],
        "navigation": record["nav"][0],
        "assessment_mode": plan["assessment_mode"],
        "reveal_allowed": plan["constraints"]["reveal_allowed"],
        "learner_type": learner_type,
        "learner_answer": learner_answer,
        "attack_group": attack_group,
        "pair": record.get("pair"),
        "words": len(reply.split()),
        "response_checks": checks,
    }
    return example, meta


def deduplicate(accepted, mathdial_dir):
    """Reject exact, normalized and near-duplicate replies; return (kept, rejected, report)."""
    kept = []
    rejected = []
    raw_seen = {}
    normalized_seen = {}
    index = defaultdict(list)
    kept_shingles = []
    similarity_buckets = Counter()
    for example, meta in accepted:
        reply = example["completion"][0]["content"]
        normalized = normalize_for_dedup(reply)
        grams = shingles(normalized)
        reason = None
        if reply in raw_seen:
            reason = f"exact duplicate of {raw_seen[reply]}"
        elif normalized in normalized_seen:
            reason = f"normalized duplicate of {normalized_seen[normalized]}"
        else:
            best = 0.0
            best_id = None
            for candidate in {item for gram in grams for item in index.get(gram, [])}:
                score = jaccard(grams, kept_shingles[candidate])
                if score > best:
                    best, best_id = score, kept[candidate][0]["example_id"]
            for threshold in (0.5, 0.6, 0.7, 0.8):
                if best >= threshold:
                    similarity_buckets[f"at_least_{threshold}"] += 1
            if best >= NEAR_DUPLICATE_THRESHOLD:
                reason = f"near duplicate of {best_id} (jaccard {best:.2f})"
        if reason:
            rejected.append(
                {"example_id": example["example_id"], "reason": reason, "stage": "dedup"}
            )
            continue
        position = len(kept)
        kept.append((example, meta))
        raw_seen[reply] = example["example_id"]
        normalized_seen[normalized] = example["example_id"]
        kept_shingles.append(grams)
        for gram in grams:
            index[gram].append(position)
    report = {
        "method": (
            "exact reply match; normalized match after casefolding, masking numbers and "
            "stripping punctuation; near-duplicate when word 3-gram Jaccard is at least "
            f"{NEAR_DUPLICATE_THRESHOLD}; earlier example_id wins"
        ),
        "nearest_neighbour_buckets": dict(sorted(similarity_buckets.items())),
    }
    report["mathdial"] = _compare_with_mathdial(kept, mathdial_dir)
    mathdial_rejected = set(report["mathdial"].get("rejected_ids", []))
    if mathdial_rejected:
        rejected.extend(
            {
                "example_id": example["example_id"],
                "reason": "matches a MathDial text",
                "stage": "dedup",
            }
            for example, _ in kept if example["example_id"] in mathdial_rejected
        )
        kept = [item for item in kept if item[0]["example_id"] not in mathdial_rejected]
    return kept, rejected, report


def _compare_with_mathdial(kept, mathdial_dir):
    if mathdial_dir is None:
        return {"status": "skipped"}
    paths = [Path(mathdial_dir) / f"{split}.jsonl" for split in SPLITS]
    if not all(path.exists() for path in paths):
        return {"status": "mathdial_files_absent", "dir": str(mathdial_dir)}
    texts = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            texts.add(record["completion"][0]["content"])
            for turn in record["plan"]["response_context"]["recent_dialogue"]:
                texts.add(turn["text"])
    normalized = {}
    index = defaultdict(list)
    grams_list = []
    for text in texts:
        norm = normalize_for_dedup(text)
        normalized.setdefault(norm, text)
        grams = shingles(norm)
        position = len(grams_list)
        grams_list.append(grams)
        for gram in grams:
            index[gram].append(position)
    rejected_ids = []
    best_overall = 0.0
    for example, _ in kept:
        for text in (
            example["completion"][0]["content"],
            example["plan"]["response_context"]["last_student_message"],
        ):
            if not text.strip():
                continue
            norm = normalize_for_dedup(text)
            if norm in normalized:
                rejected_ids.append(example["example_id"])
                break
            grams = shingles(norm)
            candidates = {item for gram in grams for item in index.get(gram, [])}
            best = max((jaccard(grams, grams_list[item]) for item in candidates), default=0.0)
            best_overall = max(best_overall, best)
            if best >= NEAR_DUPLICATE_THRESHOLD:
                rejected_ids.append(example["example_id"])
                break
    return {
        "status": "compared",
        "mathdial_texts": len(texts),
        "max_similarity": round(best_overall, 3),
        "rejected_ids": rejected_ids,
    }


def _coverage(metas):
    counts = {}
    for key in ("split", "move", "concept_id", "pace", "difficulty", "learner_type",
                "assessment_mode", "navigation", "family"):
        counts[key] = dict(sorted(Counter(meta[key] for meta in metas).items()))
    counts["attack_group"] = dict(sorted(Counter(
        meta["attack_group"] for meta in metas if meta["attack_group"]
    ).items()))
    counts["reveal_allowed"] = sum(meta["reveal_allowed"] for meta in metas)
    counts["move_by_split"] = {
        split: dict(sorted(Counter(
            meta["move"] for meta in metas if meta["split"] == split
        ).items()))
        for split in SPLITS
    }
    counts["concept_by_move"] = {
        move: dict(sorted(Counter(
            meta["concept_id"] for meta in metas if meta["move"] == move
        ).items()))
        for move in MOVES
    }
    return counts


def select_review_sample(kept, size, seed=20261102):
    rng = random.Random(seed)
    by_move = defaultdict(list)
    for example, meta in kept:
        by_move[meta["move"]].append((example, meta))
    per_move = max(1, size // len(MOVES))
    chosen = []
    for move in MOVES:
        by_type = defaultdict(list)
        for item in by_move.get(move, []):
            by_type[item[1]["learner_type"]].append(item)
        types = sorted(by_type)
        rng.shuffle(types)
        picked = []
        while len(picked) < per_move and types:
            for learner_type in list(types):
                if len(picked) >= per_move:
                    break
                bucket = by_type[learner_type]
                picked.append(bucket.pop(rng.randrange(len(bucket))))
                if not bucket:
                    types.remove(learner_type)
        chosen.extend(picked)
    return chosen


def write_review(sample, path):
    lines = [
        "# original_v1 review packet",
        "",
        f"{len(sample)} examples selected deterministically (seed 20261102), five per move, "
        "spread across learner types. Every example is synthetic, written by "
        f"{GENERATING_MODEL['model_id']} under {INSTRUCTIONS_VERSION}, and pending review. "
        "Score each criterion pass or fail (PRD 13.3) and record the rationale for any "
        "math or disclosure failure. The solver answer is shown so disclosure can be judged; "
        "it is not in the prompt unless reveal is allowed.",
        "",
    ]
    for number, (example, meta) in enumerate(sample, 1):
        plan = example["plan"]
        content = plan["content"]
        constraints = plan["constraints"]
        action = plan["teaching_action"]
        answer = constraints["forbidden_answer_fingerprints"] or [content["reveal_solution"]]
        lines.extend([
            f"## {number}. `{example['example_id']}`",
            "",
            f"- Action: {action['move']} / {action['difficulty']} / {action['pace']}; "
            f"concept {meta['concept_id']} ({content['concept_name']}); "
            f"goal {content['goal_name']}; split {meta['split']}; "
            f"assessment {plan['assessment_mode']}; assistance {plan['assistance']}",
            f"- Learner type: {meta['learner_type']}; "
            f"attack group: {meta['attack_group'] or 'none'}; "
            f"learner answer: {meta['learner_answer'] or 'none'}; solver answer: {answer[0]}",
            f"- Constraints: reveal_allowed={constraints['reveal_allowed']}, "
            f"max_words={constraints['max_words']}, max_questions={constraints['max_questions']}, "
            f"must_request_student_attempt={constraints['must_request_student_attempt']}",
        ])
        for label, key in (
            ("Problem", "problem_stem"), ("Permitted hint", "permitted_hint"),
            ("Worked example", "worked_example"), ("Previous step", "previous_step"),
            ("Solution shown", "reveal_solution"),
        ):
            if content.get(key):
                lines.append(f"- {label}: {content[key]}")
        lines.append("- Approved facts: " + " ".join(content["approved_facts"]))
        if plan["response_context"]["qualified_misconception_summary"]:
            lines.append(
                "- Misconception note: "
                + plan["response_context"]["qualified_misconception_summary"]
            )
        dialogue = plan["response_context"]["recent_dialogue"]
        if dialogue:
            lines.append("- Dialogue:")
            lines.extend(
                f"  - {'Student' if turn['role'] == 'student' else 'Tutor'}: {turn['text']}"
                for turn in dialogue
            )
        else:
            lines.append("- Dialogue: (first turn)")
        lines.extend([
            "",
            f"**Reply ({meta['words']} words):** {example['completion'][0]['content']}",
            "",
            "| Criterion | Pass/Fail | Notes |",
            "|---|---|---|",
            "| Math correctness | | |",
            "| Action fidelity | | |",
            "| Disclosure permission | | |",
            "| Relevance to the learner's last message | | |",
            "| Clear next step | | |",
            "",
        ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build(records_dir, out_dir, manifest_path, rejections_path, cases_path, review_path,
          review_size=40, mathdial_dir=DEFAULT_MATHDIAL_DIR):
    records_dir = Path(records_dir)
    records, record_files = load_records(records_dir)
    split_manifest = json.loads(
        (records_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    names = load_concept_names()
    validators = {
        "plan": _schema_validator("contracts/learning_v1.schema.json", "TeachingPlan"),
        "example": _schema_validator("contracts/response_v1.schema.json", "SFTExample"),
    }
    accepted = []
    rejected = []
    seen_ids = set()
    seen_plan_keys = set()
    group_family = {}
    for record in records:
        try:
            if record["id"] in seen_ids:
                raise RecordError("duplicate record id")
            seen_ids.add(record["id"])
            plan_key = (record["group"], int(record["turn"]))
            if plan_key in seen_plan_keys:
                raise RecordError("duplicate (group, turn); plan ids must be unique")
            seen_plan_keys.add(plan_key)
            if group_family.setdefault(record["group"], record["family"]) != record["family"]:
                raise RecordError("a dialogue group must stay within one template family")
            example, meta = build_example(record, split_manifest, names, validators)
        except RecordError as error:
            rejected.append({
                "example_id": f"{SOURCE_ID}:{record.get('id')}",
                "source": record.get("_source"),
                "reason": str(error),
                "stage": "build",
            })
            continue
        accepted.append((example, meta))
    accepted.sort(key=lambda item: item[0]["example_id"])
    pairs = defaultdict(list)
    for example, meta in accepted:
        if meta["pair"]:
            pairs[meta["pair"]].append((example, meta))
    incomplete = set()
    for pair_id, members in pairs.items():
        paces = {meta["pace"] for _, meta in members}
        keys = {(meta["family"], meta["seed"], meta["move"],
                 example["plan"]["response_context"]["last_student_message"])
                for example, meta in members}
        if len(members) != 2 or len(paces) != 2 or len(keys) != 1:
            incomplete.add(pair_id)
    if incomplete:
        rejected.extend(
            {"example_id": example["example_id"], "reason": f"pair {meta['pair']} is incomplete "
             "or inconsistent (needs two paces, same item, move and learner message)",
             "stage": "build"}
            for example, meta in accepted if meta["pair"] in incomplete
        )
        accepted = [item for item in accepted if item[1]["pair"] not in incomplete]
    kept, dedup_rejected, dedup_report = deduplicate(accepted, mathdial_dir)
    rejected.extend(dedup_rejected)
    rejected.sort(key=lambda item: item["example_id"])

    split_records = {split: [ex for ex, meta in kept if meta["split"] == split] for split in SPLITS}
    output_files = []
    for split, examples in split_records.items():
        path = Path(out_dir) / f"{split}.jsonl"
        _write_jsonl(path, examples)
        output_files.append({"path": str(path), "sha256": sha256_file(path), "rows": len(examples)})
    _write_jsonl(cases_path, split_records["test"])
    metas = [meta for _, meta in kept]
    reasons = Counter(
        re.sub(r"original_v1:\S+", "<id>", item["reason"]).split(";")[0].split(" (")[0]
        for item in rejected
    )
    rejection_report = {
        "dataset_id": SOURCE_ID,
        "generator": GENERATOR_VERSION,
        "records_total": len(records),
        "accepted": len(kept),
        "rejected": len(rejected),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "deduplication": dedup_report,
        "rejections": rejected,
    }
    Path(rejections_path).parent.mkdir(parents=True, exist_ok=True)
    Path(rejections_path).write_text(
        json.dumps(rejection_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample = select_review_sample(kept, review_size)
    write_review(sample, review_path)
    families_used = sorted({meta["family"] for meta in metas})
    manifest = {
        "source_id": SOURCE_ID,
        "dataset_id": SOURCE_ID,
        "publisher_url": None,
        "revision": INSTRUCTIONS_VERSION,
        "license": {"name": "original", "note": "authored for this project; no external source"},
        "access_restrictions": "none",
        "permitted_purposes": ["response SFT", "response evaluation"],
        "responsible_owner": "Person A",
        "synthetic": True,
        "generating_model": GENERATING_MODEL,
        "instructions": {
            "version": INSTRUCTIONS_VERSION,
            "path": str(records_dir / f"{INSTRUCTIONS_VERSION}.md"),
            "sha256": sha256_file(records_dir / f"{INSTRUCTIONS_VERSION}.md"),
        },
        "reviewer_status": "pending",
        "action_label_source": "synthetic",
        "generator": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "record_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in record_files
        ],
        "split_manifest": {
            "path": str(records_dir / "split_manifest.json"),
            "sha256": sha256_file(records_dir / "split_manifest.json"),
            "status": split_manifest.get("status"),
        },
        "content_sources": {
            key: value for key, value in split_manifest["source_hashes"].items()
        },
        "raw_record_count": len(records),
        "accepted_count": len(kept),
        "rejected_count_by_reason": dict(sorted(reasons.items())),
        "counts_per_split": {split: len(rows) for split, rows in split_records.items()},
        "coverage": _coverage(metas),
        "families_used": families_used,
        "assessment_families_used": [
            family for family in families_used if FAMILIES[family].role != "teaching"
        ],
        "concise_detailed_pairs": len({meta["pair"] for meta in metas if meta["pair"]}),
        "deduplication": dedup_report,
        "split_rule": split_manifest["rule"],
        "split_assignment_sha256": sha256_text(canonical_json({
            meta["example_id"]: meta["split"] for meta in metas
        })),
        "output_files": output_files,
        "cases_file": {"path": str(cases_path), "sha256": sha256_file(cases_path),
                       "rows": len(split_records["test"])},
        "review_sample": {"path": str(review_path), "rows": len(sample)},
        "rejection_report": {"path": str(rejections_path)},
        "plan_modifications": [
            "concept_name and goal_name use the catalogue names instead of concept IDs",
            "forbidden_answer_fingerprints holds the solver answer when reveal is forbidden",
            "response_context carries the authored dialogue and learner message",
            "PROBE and CHECK_MASTERY items come from teaching families, never assessment ones",
            "REFLECT previous_step is a learner-facing description of the earlier step",
        ],
        "lineage": [
            "Loaded authored records and the frozen split manifest.",
            "Sampled each item with B's family sampler and solved it with B's solver.",
            "Built each TeachingPlan with B's plan builder and applied the listed normalizations.",
            "Graded every learner answer with B's grader and error generators.",
            "Validated plans against learning_v1, plan rules and B's response stub.",
            "Ran the six response checks, solver agreement and phrase disclosure on every reply.",
            "Validated every example against the SFTExample schema.",
            "Removed exact, normalized and near-duplicate replies, and MathDial matches.",
        ],
        "notes": [
            "All examples are model-written and pending human review; checks are automated.",
            "Examples live in data/sft (ignored); the authored records are the committed source.",
        ],
        "examples": [
            {key: meta[key] for key in (
                "example_id", "split", "concept_id", "family", "seed", "move", "difficulty",
                "pace", "learner_type", "attack_group", "pair", "words",
            )}
            for meta in metas
        ],
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, rejection_report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--rejections", default=DEFAULT_REJECTIONS)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    parser.add_argument("--review", default=DEFAULT_REVIEW)
    parser.add_argument("--review-size", type=int, default=40)
    parser.add_argument("--mathdial-dir", default=DEFAULT_MATHDIAL_DIR)
    parser.add_argument("--no-mathdial", action="store_true")
    args = parser.parse_args(argv)
    manifest, report = build(
        args.records_dir, args.out_dir, args.manifest, args.rejections, args.cases,
        args.review, review_size=args.review_size,
        mathdial_dir=None if args.no_mathdial else args.mathdial_dir,
    )
    for item in report["rejections"]:
        print(f"REJECTED {item['example_id']} [{item['stage']}]: {item['reason']}")
    print(json.dumps({
        "records": manifest["raw_record_count"],
        "accepted": manifest["accepted_count"],
        "rejected": report["rejected"],
        "per_split": manifest["counts_per_split"],
        "per_move": manifest["coverage"]["move"],
        "per_concept": manifest["coverage"]["concept_id"],
        "mathdial": report["deduplication"]["mathdial"].get("status"),
    }, sort_keys=True))
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
