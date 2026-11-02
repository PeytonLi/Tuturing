import copy
from collections import Counter

from packages.response.prompt import compile_prompt
from tools.response.audit_mathdial_sft import audit_example
from tools.response.prepare_sft import build_examples

SOLUTION = "Ada had 20 apples and gave away 3, so 20 - 3 = 17 apples remain.\n 17"
STRUCTURAL_FINDINGS = {
    "context_differs_from_preceding_turns",
    "prompt_differs_from_compiled_plan",
    "completion_differs_from_target_turn",
    "later_turn_text_in_prompt",
}


def _inputs(conversation):
    config = {
        "revision": "rev-test",
        "max_words": 150,
        "split_salt": "mathdial",
        "include_student_solution_as_opening_turn": True,
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
    row = {
        "qid": 11,
        "scenario": 1,
        "question": "Ada has 20 apples and gives 3 away. How many apples remain?",
        "ground_truth": SOLUTION,
        "student_incorrect_solution": "20 + 3 = 23 apples.",
        "student_profile": "Ada is a learner.",
        "teacher_described_confusion": "added instead of subtracting",
        "self-correctness": "Yes",
        "conversation": conversation,
    }
    return {"train": [(0, row)]}, config, mapping


def _by_id(records):
    return {record["example_id"]: record for record in records}


def test_protected_target_that_states_the_final_answer_is_rejected():
    rows, config, mapping = _inputs(
        "Teacher: (focus) Look at the word gives.|EOM|"
        "Ada: So I subtract?|EOM|"
        "Teacher: (probing) Yes, and 20 - 3 = 17, right?|EOM|"
        "Ada: Yes.|EOM|"
        "Teacher: (telling) The answer is 17 apples."
    )
    accepted, rejected, _, _ = build_examples(rows, config, mapping)
    accepted_ids = _by_id(accepted)
    rejected_ids = _by_id(rejected)
    assert "mathdial:train:0000:t00" in accepted_ids
    assert rejected_ids["mathdial:train:0000:t02"]["reason"] == "target_states_protected_answer"
    telling = accepted_ids["mathdial:train:0000:t04"]
    assert telling["plan"]["constraints"]["reveal_allowed"] is True
    assert telling["completion"][0]["content"] == "The answer is 17 apples."


def test_protected_prompt_holding_the_pasted_reference_solution_is_rejected():
    pasted = SOLUTION.replace("\n", " ")
    rows, config, mapping = _inputs(
        f"Teacher: (telling) {pasted}|EOM|"
        "Ada: Oh, I see now.|EOM|"
        "Teacher: (probing) Can you tell me why we subtract?|EOM|"
        "Ada: Because she gives them away.|EOM|"
        "Teacher: (focus) Which word told you that?"
    )
    accepted, rejected, _, _ = build_examples(rows, config, mapping)
    reasons = Counter(record["reason"] for record in rejected)
    assert reasons == {"reference_solution_in_protected_context": 2}
    assert [record["example_id"] for record in accepted] == ["mathdial:train:0000:t00"]
    for record in accepted:
        assert record["plan"]["constraints"]["reveal_allowed"] is True


def test_protected_examples_without_disclosure_keep_their_compiled_prompt():
    rows, config, mapping = _inputs(
        "Teacher: (probing) What does gives away mean for the count?|EOM|"
        "Ada: It goes down.|EOM|"
        "Teacher: (focus) So which operation fits?"
    )
    accepted, rejected, _, _ = build_examples(rows, config, mapping)
    assert rejected == []
    assert len(accepted) == 2
    for record in accepted:
        prompt_text = "\n".join(message["content"] for message in record["prompt"])
        assert record["plan"]["constraints"]["reveal_allowed"] is False
        assert "17" not in prompt_text
        assert "20 - 3" not in prompt_text


def test_audit_flags_injected_future_turn_and_passes_clean_examples():
    future = "This later student message must never appear in an earlier prompt."
    rows, config, mapping = _inputs(
        "Teacher: (probing) What does gives away mean for the count?|EOM|"
        f"Ada: {future}|EOM|"
        "Teacher: (focus) So which operation fits?"
    )
    accepted, _, _, _ = build_examples(rows, config, mapping)
    row = rows["train"][0][1]
    first = _by_id(accepted)["mathdial:train:0000:t00"]
    assert not audit_example(first, row) & STRUCTURAL_FINDINGS
    tampered = copy.deepcopy(first)
    tampered["plan"]["response_context"]["recent_dialogue"].append(
        {"role": "student", "text": future}
    )
    tampered["prompt"] = compile_prompt(tampered["plan"])
    findings = audit_example(tampered, row)
    assert "later_turn_text_in_prompt" in findings
    assert "context_differs_from_preceding_turns" in findings
