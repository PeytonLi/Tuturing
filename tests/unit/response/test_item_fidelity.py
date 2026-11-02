"""Item fidelity: replies that drop or replace the issued problem fail action_faithful.

The substitution texts are Qwen3.8-27B outputs from experiments/response/benchmark_v1
(review sheet supplement, items 1, 2, 20 and 12); the faithful texts are gpt-6.1-sol outputs
from the same benchmark (review sheet, items 1, 2, 4, 6, 15 and 20).
"""

import copy

import pytest

from packages.learning.actions import encode_strategist
from packages.response.fallbacks import render_fallback
from packages.response.item_fidelity import (
    ITEM_FIDELITY_VERSION,
    applies_to,
    item_fidelity,
    quantity_phrases,
    stem_word,
    terms,
)
from packages.response.pipeline import PipelineSettings, ResponsePipeline
from packages.response.validate import VALIDATOR_VERSION, run_checks
from tests.unit.response.pipeline_helpers import (
    ScriptedResponder,
    lane_plan,
    learning_plans,
    make_request,
)

QWEN_DIE_CHECK = (
    "Let’s check your understanding of sample spaces.\n\nConsider a standard six-sided die. "
    "List the complete sample space for a single roll. Then, identify the subset of outcomes "
    "that represent rolling an even number.\n\nPlease attempt this."
)
QWEN_DIE_CHALLENGE = (
    "Let’s test your understanding of sample spaces. Consider a standard six-sided die. The "
    "sample space is {1, 2, 3, 4, 5, 6}. Now, imagine a modified die where the face labeled "
    '"6" is replaced by a face labeled "7". The sample space becomes {1, 2, 3, 4, 5, 7}.\n\n'
    "If you roll this modified die once, what is the probability of rolling an even number?"
    "\n\nPlease attempt to solve this."
)
QWEN_MARBLE_CHALLENGE = (
    "Let’s adjust the scenario to increase complexity while keeping the core concept.\n\n"
    "Imagine a bag contains 3 red, 4 blue, and 5 green marbles. You draw two marbles "
    "**without replacement**.\n\nConstruct a probability tree for the first two draws. Focus "
    "specifically on the branches where the first marble is red.\n\nWhat is the probability "
    "that the second marble is blue, given that the first marble was red?"
)
GPT_COIN_CHECK = (
    "A fair coin is flipped once. List the sample space, identify the outcome that corresponds "
    "to heads, and use this to find the probability of heads.\n\nWill you attempt this and show "
    "your reasoning?"
)
GPT_COIN_CHALLENGE = (
    "A fair coin is flipped once. It has two equally likely outcomes.\n\nFind the probability "
    "of heads. In your reasoning, write the **sample space** (all possible outcomes) and "
    "identify which outcome corresponds to heads.\n\nWill you attempt it and explain your "
    "reasoning?"
)
GPT_COIN_HINT = (
    "Start by listing the two possible outcomes of one coin flip. Can you try listing them?"
)
GPT_COIN_PROBE = (
    "A sample space lists all possible outcomes. For one flip of a fair coin, what outcomes "
    "would you list, and how would you use them to find the probability of heads? Please give "
    "it a try."
)
GPT_SPINNER_HINT = (
    "Count how many sections are left after excluding the 3 blue sections. Then compare that "
    "count to the 10 sections in total."
)


def checks(plan, text):
    return run_checks(plan, text, plan["teaching_action"])


def test_versions_are_bumped():
    assert VALIDATOR_VERSION == "validator_v2"
    assert ITEM_FIDELITY_VERSION == "item_fidelity_v1"


@pytest.mark.parametrize(
    ("fixture", "text"),
    [
        ("teaching_plan_check_mastery", QWEN_DIE_CHECK),
        ("protected_check_mastery", QWEN_DIE_CHECK),
        ("teaching_plan_challenge", QWEN_DIE_CHALLENGE),
    ],
)
def test_qwen_die_substitutions_fail_action_faithful(fixture, text):
    plan = learning_plans()[fixture]
    result = item_fidelity(plan, text)
    assert result.applicable and not result.passed
    assert result.coverage < 0.5
    assert "die" in result.new_objects and "roll" in result.new_objects
    assert "6 die" in result.new_quantities
    assert any("present the issued problem" in reason for reason in result.reasons)
    outcome = checks(plan, text)
    assert outcome["action_faithful"] is False
    assert outcome["math_supported"] is True  # the digit checks alone let it through


def test_qwen_marble_substitution_fails_on_new_objects_and_quantities():
    plan = lane_plan("challenge")
    result = item_fidelity(plan, QWEN_MARBLE_CHALLENGE)
    assert not result.passed
    assert result.new_objects == ("bag", "blue", "marble", "red")
    assert "5 marble" in result.new_quantities
    assert checks(plan, QWEN_MARBLE_CHALLENGE)["action_faithful"] is False


@pytest.mark.parametrize(
    ("fixture", "text"),
    [
        ("teaching_plan_check_mastery", GPT_COIN_CHECK),
        ("protected_check_mastery", GPT_COIN_CHECK),
        ("teaching_plan_challenge", GPT_COIN_CHALLENGE),
        ("teaching_plan_hint", GPT_COIN_HINT),
        ("teaching_plan_probe", GPT_COIN_PROBE),
    ],
)
def test_faithful_frontier_replies_keep_passing(fixture, text):
    plan = learning_plans()[fixture]
    result = item_fidelity(plan, text)
    assert result.applicable and result.passed, result.reasons
    assert result.new_objects == () and result.new_quantities == ()
    assert all(checks(plan, text).values())


def test_spinner_hint_passes_by_referring_to_the_item():
    plan = lane_plan("hint")
    result = item_fidelity(plan, GPT_SPINNER_HINT)
    assert result.passed
    assert set(result.shared) == {"blue", "section"}


def test_every_fixture_fallback_passes_item_fidelity():
    plans = {**learning_plans(), **{name: lane_plan(name) for name in (
        "practice", "hint", "challenge", "check_mastery", "probe",
    )}}
    applicable = 0
    for plan in plans.values():
        text = render_fallback(plan)
        result = item_fidelity(plan, text)
        assert result.passed, (plan["plan_id"], result.reasons)
        applicable += result.applicable
        assert all(checks(plan, text).values())
    assert applicable == 11


def test_hint_may_realize_the_item_through_the_permitted_hint():
    plan = learning_plans()["teaching_plan_hint"]
    text = "Here is a hint: List the two outcomes. What is your next step?"
    result = item_fidelity(plan, text)
    assert result.passed and result.coverage == 0.0
    plan = copy.deepcopy(plan)
    plan["content"]["permitted_hint"] = "Think about which outcomes are possible."
    assert not item_fidelity(plan, text).passed


def test_the_plans_own_family_and_numbers_are_allowed():
    plan = learning_plans()["teaching_plan_practice"]
    tails = "A fair coin is flipped once: the outcomes are heads and tails. What is P(heads)?"
    assert item_fidelity(plan, tails).passed  # tails belongs to the coin family
    two_coins = "Suppose two coins are flipped once. What is the probability of two heads?"
    result = item_fidelity(plan, two_coins)
    assert result.passed  # "two" is in the approved facts; a quantity alone is weak evidence
    plan = copy.deepcopy(plan)
    plan["content"]["approved_facts"] = ["Each outcome is equally likely."]
    assert not item_fidelity(plan, two_coins).passed
    assert item_fidelity(plan, two_coins).new_quantities == ("2 coin",)  # heads: companion
    dialogue = copy.deepcopy(plan)
    dialogue["response_context"]["recent_dialogue"] = [
        {"role": "student", "text": "Is this like rolling a die?"}
    ]
    assert item_fidelity(
        dialogue, "A fair coin is flipped once; unlike a die roll it has two outcomes. "
        "What is the probability of heads?"
    ).passed


def test_referring_moves_may_cite_the_items_numbers_or_three_of_its_words():
    plan = lane_plan("probe")
    plan["content"]["problem_stem"] = (
        "A driver travels 30 miles per hour for 3 hours and 25 miles per hour for 4 hours "
        "to deliver goods to a town every day from Monday to Saturday. How many miles does "
        "the driver travel in a week?"
    )
    plan["constraints"]["forbidden_answer_fingerprints"] = []
    by_number = "Let’s check that calculation. What is 30 times 3?"
    by_words = "How far does the driver travel in one day before you scale up to the week?"
    off_item = "Let’s check your thinking. What is 7 times 8?"
    assert item_fidelity(plan, by_number).passed
    assert item_fidelity(plan, by_words).passed
    assert not item_fidelity(plan, off_item).passed


def test_check_applies_only_to_moves_that_carry_an_item():
    plans = learning_plans()
    assert not item_fidelity(plans["teaching_plan_explain_concept"], QWEN_DIE_CHECK).applicable
    assert not item_fidelity(plans["teaching_plan_reflect"], QWEN_DIE_CHECK).applicable
    assert not item_fidelity(plans["teaching_plan_worked_example"], QWEN_DIE_CHECK).applicable
    assert not applies_to(dict(plans["teaching_plan_probe"], problem_instance_id=None))
    assert item_fidelity(plans["teaching_plan_probe"], None).applicable is False
    assert all(
        applies_to(plans[name]) for name in (
            "teaching_plan_practice", "teaching_plan_challenge", "teaching_plan_check_mastery",
            "teaching_plan_hint", "teaching_plan_probe", "protected_check_mastery",
        )
    )


def test_normalization_helpers():
    assert stem_word("flipped") == stem_word("flips") == stem_word("flipping") == "flip"
    assert stem_word("dice") == "die" and stem_word("drawn") == "draw"
    assert stem_word("probabilities") == "probability"
    assert terms("A fair coin is flipped once. What is the probability of heads?") == {
        "fair", "coin", "flip", "once", "probability", "head",
    }
    assert quantity_phrases("a standard six-sided die and 3 red marbles") == [
        ("6", "die"), ("3", "marble"),
    ]
    assert quantity_phrases("one coin flip with two outcomes") == []
    assert quantity_phrases("two fair coins and three red marbles") == [
        ("2", "coin"), ("3", "marble"),
    ]
    # Instructional nouns, sentence boundaries and companions never form a phrase.
    assert quantity_phrases("separate two things in your head") == []
    assert quantity_phrases("in three moves. A bag contains 4 red marbles.") == [("4", "marble")]
    assert quantity_phrases("two heads in a row, three steps, four ways, five ideas") == []


# Original SFT records (data/response/original_v1) that validator_v2 rejected before the
# quantity rule was narrowed and head/tail became companions; all three embed the stem.
ORIGINAL_RECORDS = {
    "c02d-001": {
        "move": "PROBE",
        "difficulty": "hard",
        "pace": "detailed",
        "stem": (
            "A spinners experiment has 1 favorable outcomes out of 8 equally likely outcomes. "
            "Find the probability."
        ),
        "facts": [
            "A probability is between zero and one.",
            "Equally likely outcomes have equal probability.",
        ],
        "fingerprints": ["1/8"],
        "tutor": (
            "Here is a spinner problem: A spinners experiment has 1 favorable outcomes out of "
            "8 equally likely outcomes. Find the probability."
        ),
        "message": "can i get a hint first?",
        "reply": (
            "Before any hint, I'd like to see your starting point, because the right hint "
            "depends on where you are. Here is the problem again: A spinners experiment has 1 "
            "favorable outcomes out of 8 equally likely outcomes. Find the probability. Read "
            "it once more and separate two things in your head: the collection of all equally "
            "likely outcomes, which is the total, and the subset of those outcomes the event "
            "is asking about, which is the favorable group. Then tell me what you count as the "
            "total number of equally likely outcomes and how many of them you treat as "
            "favorable, and say briefly how you decided?"
        ),
    },
    "c08a-176": {
        "move": "PRACTICE",
        "difficulty": "easy",
        "pace": "detailed",
        "stem": (
            "A dice experiment has 1 favorable outcomes out of 4 equally likely outcomes. "
            "Find the probability."
        ),
        "facts": ["A path probability is the product of its conditional branch probabilities."],
        "fingerprints": ["1/4"],
        "tutor": "We'll practise the multiplication rule and probability trees with dice.",
        "message": "slow down, trees confuse me",
        "reply": (
            "Slowly, then, and we'll keep the tree small in your head. Here is the problem: A "
            "dice experiment has 1 favorable outcomes out of 4 equally likely outcomes. Find "
            "the probability. Picture the experiment as a tree: it starts at a root, splits "
            "into branches at each stage, and every complete route from the root to a tip is "
            "one outcome. A path probability is the product of its conditional branch "
            "probabilities, which means you multiply along a single route to get that route's "
            "chance. Here you don't even need to multiply, because the problem already tells "
            "you that the complete paths are equally likely and how many there are: that "
            "number is the total, and it goes underneath. The favorable count is how many of "
            "those paths the event wants, and it goes on top. So the probability is favorable "
            "paths over all paths. Count the two groups, write the fraction, and give it a try."
        ),
    },
    "p11-d": {
        "move": "PRACTICE",
        "difficulty": "medium",
        "pace": "detailed",
        "stem": (
            "A marbles experiment has 4 favorable outcomes out of 6 equally likely outcomes. "
            "Find the probability."
        ),
        "facts": ["A sample space lists all possible outcomes.", "An event is a set of outcomes."],
        "fingerprints": ["2/3"],
        "tutor": (
            "A sample space lists all possible outcomes, and an event is a set of outcomes "
            "from that list."
        ),
        "message": "ok",
        "reply": (
            "Here is a marbles problem to put that to work, in three moves. A marbles "
            "experiment has 4 favorable outcomes out of 6 equally likely outcomes. Find the "
            "probability. First, list: write down every outcome the experiment can produce, "
            "each once, because a sample space lists all possible outcomes. Second, mark: pick "
            "out the outcomes that belong to the event, since an event is a set of outcomes "
            "from that list, and check that your marked count matches the favorable count the "
            "problem states. Third, compare: the marked count goes on top, the length of the "
            "whole list underneath, as a fraction. Give it a try and send the fraction with "
            "your two counts."
        ),
    },
}


def record_plan(record):
    plan = lane_plan("probe" if record["move"] == "PROBE" else "practice")
    plan["teaching_action"] = {
        "action_id": encode_strategist(record["move"], record["difficulty"], record["pace"]),
        "move": record["move"],
        "difficulty": record["difficulty"],
        "pace": record["pace"],
    }
    plan["content"]["problem_stem"] = record["stem"]
    plan["content"]["approved_facts"] = record["facts"]
    plan["constraints"]["max_words"] = 250
    plan["constraints"]["forbidden_answer_fingerprints"] = record["fingerprints"]
    plan["response_context"] = {
        "last_student_message": record["message"],
        "recent_dialogue": [
            {"role": "tutor", "text": record["tutor"]},
            {"role": "student", "text": record["message"]},
        ],
        "qualified_misconception_summary": None,
    }
    return plan


@pytest.mark.parametrize("record_id", sorted(ORIGINAL_RECORDS))
def test_original_sft_records_with_the_embedded_stem_pass(record_id):
    record = ORIGINAL_RECORDS[record_id]
    plan = record_plan(record)
    result = item_fidelity(plan, record["reply"])
    assert result.applicable and result.passed, result.reasons
    assert result.coverage == 1.0
    assert result.new_objects == () and result.new_quantities == ()
    assert all(checks(plan, record["reply"]).values())


def test_a_counted_scenario_noun_the_plan_lacks_still_fails():
    plan = learning_plans()["teaching_plan_practice"]
    text = (
        "A fair coin is flipped once. What is the probability of heads? Now picture three "
        "red marbles in a bag instead and give it a try."
    )
    result = item_fidelity(plan, text)
    assert not result.passed
    assert "marble" in result.new_objects and "bag" in result.new_objects
    assert "3 marble" in result.new_quantities
    assert checks(plan, text)["action_faithful"] is False
    plan = record_plan(ORIGINAL_RECORDS["p11-d"])
    text = ORIGINAL_RECORDS["p11-d"]["reply"] + " Or try it with two fair dice."
    result = item_fidelity(plan, text)
    assert not result.passed and result.new_quantities == ("2 die",)


def test_substituted_item_is_never_repaired_and_always_falls_back():
    plan = learning_plans()["protected_check_mastery"]
    two_questions = QWEN_DIE_CHECK.replace(
        "Please attempt this.", "Ready? Can you try it now? Please attempt this."
    )
    assert two_questions.count("?") == 2
    responder = ScriptedResponder([two_questions], [QWEN_DIE_CHECK])
    pipeline = ResponsePipeline(responder, PipelineSettings(artifact_id="test_artifact_v1"))
    outcome = pipeline.run(make_request(plan, "request-sub", artifact="test_artifact_v1"))
    assert responder.repair_calls == []
    assert outcome.attempts[0].failed_checks == ("action_faithful", "question_count_valid")
    assert outcome.response["delivery_mode"] == "fallback"
    assert outcome.response["text"] == render_fallback(plan)
    assert "die" not in outcome.response["text"]
