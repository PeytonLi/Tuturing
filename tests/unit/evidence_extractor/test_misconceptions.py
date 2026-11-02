import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from packages.evidence_extractor.extract import extract

ROOT = Path(__file__).parents[3]
CONTEXTS = json.loads(
    (ROOT / "evaluation/response/extraction_contexts_v1.json").read_text(encoding="utf-8")
)


def plan(context_id):
    return CONTEXTS[context_id]["plan"]


def hypotheses(message, context_id):
    result = extract(message, plan(context_id), "r")
    return {
        item["misconception_id"]: [
            message[span["start"]:span["end"]] for span in item["evidence_spans"]
        ]
        for item in result["misconception_hypotheses"]
    }, result


@pytest.mark.parametrize(
    ("message", "context_id", "misconception_id", "span_text"),
    [
        ("50/50, its either red or blue", "a_bag_red", "M01", "either red or blue"),
        ("1/3 because there are 3 colors", "x_spinner_thirds", "M01",
         "because there are 3 colors"),
        ("1/6 because even is one of the outcomes", "x_die_even", "M02",
         "even is one of the outcomes"),
        ("7/3", "a_bag_red", "M03", "7/3"),
        ("15/6 = 2.5", "x_club", "M03", "2.5"),
        ("1 - 2/10 = 8/10", "x_marbles_notblue", "M04", "1 - 2/10"),
        ("P(A|B) is the same as P(B|A) so i just need P(A)", "a_cond_probe", "M05",
         "P(A|B) is the same as P(B|A)"),
        ("6/10, the piano players who also play chess", "x_club", "M05",
         "piano players who"),
        ("12/50, divided by everyone", "x_cold", "M06", "divided by everyone"),
        ("3/5 * 3/5 = 9/25 since the draws are independent", "x_marbles_noreplace", "M07",
         "the draws are independent"),
        ("multiply all the branches: 0.4*0.6*0.5", "x_spinner_tree", "M08",
         "multiply all the branches"),
    ],
)
def test_signatures_are_proposed_with_spans_into_the_message(
    message, context_id, misconception_id, span_text
):
    found, result = hypotheses(message, context_id)
    assert misconception_id in found
    assert any(span_text in text or text in span_text for text in found[misconception_id])
    for item in result["misconception_hypotheses"]:
        assert 0 < item["probability"] < 1
        for span in item["evidence_spans"]:
            assert 0 <= span["start"] < span["end"] <= len(message)


@pytest.mark.parametrize(
    ("message", "context_id"),
    [
        ("50/50 since heads and tails are equally likely", "b_coin_practice"),
        ("50/50", "b_coin_practice"),
        ("even means 2, 4 or 6, so 3 outcomes out of 6 = 1/2", "x_die_even"),
        ("it can't be more than 1 so 0.3", "a_bag_red"),
        ("1 - 3/10 = 7/10", "x_marbles_notblue"),
        ("1 - 0.35 = 0.65", "x_rain"),
        ("P(A|B) isn't the same as P(B|A), so 1/2", "a_cond_probe"),
        ("6/15 since we only look at chess players", "x_club"),
        ("it's replaced so the draws are independent, 2/7 * 2/7 = 4/49", "a_jar_challenge"),
        ("are the draws independent?", "x_marbles_noreplace"),
        ("my friend says its 50/50 cause it's either red or blue but thats wrong right?",
         "a_bag_red"),
        ("5", "b_c06_table_check"),
        ("  9/25  ", "x_marbles_noreplace"),
    ],
)
def test_valid_reasoning_questions_and_attributed_ideas_get_no_hypothesis(message, context_id):
    found, _ = hypotheses(message, context_id)
    assert found == {}


def test_hypotheses_validate_against_the_learning_contract():
    schema = json.loads((ROOT / "contracts/learning_v1.schema.json").read_text("utf-8"))
    validator = Draft202012Validator(
        {"$ref": "#/$defs/Evidence", "$defs": schema["$defs"]},
        format_checker=FormatChecker(),
    )
    result = extract("1 - 1/2 = 1/2, one minus the chance of one head", plan("x_two_coins"), "r")
    assert [item["misconception_id"] for item in result["misconception_hypotheses"]] == ["M04"]
    assert not list(validator.iter_errors(result))
    assert result["correctness"] == "unknown"
