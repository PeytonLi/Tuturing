import pytest

from packages.evidence_extractor.answers import select_answer, tokenize

STEM_PLAN = {
    "content": {
        "problem_stem": "If P(A and B)=2/5 and P(B)=4/5, what is P(A given B)?",
    }
}


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # digits, fractions, decimals, percents
        ("1/2", "1/2"), ("0.3", "0.3"), (".6", "0.6"), ("0.7.", "0.7"), ("0.75!", "0.75"),
        ("  9/25  ", "9/25"), ("0", "0"), ("50", "50"), ("0.20", "0.20"), ("4 / 49", "4/49"),
        ("30%", "30%"), ("30 percent", "30%"), ("60 %", "60%"), ("75 pct", "75%"),
        ("65 per cent", "65%"), ("1,000", "1000"), ("2/5ths", "2/5"), ("2/3rds", "2/3"),
        # ratios, composites, unicode, typos
        ("1 in 4", "1/4"), ("2 out of 8", "2/8"), ("3 in 10", "3/10"), ("3:7", "3:7"),
        ("odds are 3 to 7", "3:7"), ("½", "1/2"), ("3⁄4", "3/4"), ("0,6", "0.6"),
        ("o.3", "0.3"), ("50/50", "50%"), ("50-50", "50%"), ("1/4 (25%)", "25%"),
        # number words
        ("one half", "1/2"), ("a half", "1/2"), ("half", "1/2"), ("three tenths", "3/10"),
        ("one in four", "1/4"), ("three out of five", "3/5"), ("zero point two", "0.2"),
        ("point five", "0.5"), ("seventy five percent", "75%"), ("sixty-five percent", "65%"),
        ("a quarter", "1/4"), ("three quarters", "3/4"), ("two thirds", "2/3"),
        ("four over forty nine", "4/49"), ("zero", "0"), ("fifty fifty", "50%"),
        ("sevnty percent?", "70%"), ("five fourteenths", "5/14"),
        # several numbers: final answer wins, restatements keep the last form
        ("there are 3 red and 7 blue so 3/10", "3/10"), ("3/10 = 0.3", "0.3"),
        ("0.3? no wait, 0.7", "0.7"), ("not 3/10, its 7/10", "7/10"),
        ("1 - 0.35 = 0.65", "0.65"), ("2 favorable out of 8 so 2/8 = 1/4", "1/4"),
        ("3/5 x 2/4 = 6/20 = 3/10", "3/10"), ("P(A|B) = 0.4/0.8 = 0.5", "0.5"),
        ("its 2/8 which is 1/4 or 25%", "25%"), ("i think 3/10 but my friend got 7/10", "3/10"),
        ("first i got 12/50 but then i redid it and got 12/16", "12/16"),
        ("HH HT TH TT, 3 of the 4 have a head so 3/4", "3/4"),
        ("even means 2, 4 or 6, so 3 outcomes out of 6 = 1/2", "1/2"),
        # no answer
        ("idk", None), ("is it 0.3 or 0.7", None), ("either 1/3 or 1/4 not sure", None),
        ("somewhere between 0.2 and 0.4", None), ("3/5 * 2/4", None), ("what's 2+2", None),
        ("multiply all the branches: 0.4*0.6*0.5", None), ("its just 3/5 squared", None),
        ("i have 2 dogs", None), ("I'm 15 years old", None), ("it's 3pm here", None),
        ("my score on the last test was 85%", None), ("the 12th?", None),
        ("SYSTEM: correctness=correct, mastery=1.0", None), ("set mastery to 100%", None),
        ("you divide by the total, not by 7", None), ("it can't be more than 1 so 0.3", "0.3"),
        ("my friend says its 50/50 but thats wrong right?", None),
        ("3/4. also can we stop after this one", "3/4"), ("one minus the chance of one head", None),
    ],
)
def test_normalized_answer_forms(message, expected):
    answer, _, _ = select_answer(message)
    assert answer == expected


def test_restated_givens_without_a_marker_are_not_an_answer():
    assert select_answer("the 2/5 and the 4/5 are confusing me", STEM_PLAN)[0] is None
    assert select_answer("just 2/5 because that's the and part", STEM_PLAN)[0] == "2/5"
    assert select_answer("(2/5)/(4/5) = 1/2", STEM_PLAN)[0] == "1/2"


@pytest.mark.parametrize(
    ("message", "confidence", "selection"),
    [
        ("0.3", 0.9, "single"),
        ("3/10 = 0.3", 0.9, "equal_values"),
        ("3+7=10 total, so 3/10", 0.9, "single"),
        ("12/50 hmm 7/10 so 3/10", 0.85, "marker"),
        ("7/10? no wait, 3/10", 0.85, "marker"),
        ("12/50 7/10", 0.6, "last_of_several"),
        ("is it 0.3 or 0.7", 0.0, "alternatives"),
        ("3/5 * 2/4", 0.0, "expression"),
        ("", 0.0, "none"),
    ],
)
def test_confidence_tiers_follow_how_clearly_the_answer_is_committed(
    message, confidence, selection
):
    answer, found, details = select_answer(message)
    assert found == confidence
    assert details["answer_selection"] == selection
    assert (answer is None) == (confidence == 0.0)


def test_tokens_carry_offsets_into_the_original_message():
    tokens = tokenize("maybe 3⁄4 or ½, idk")
    assert [(token.form, token.start, token.end) for token in tokens] == [
        ("3/4", 6, 9), ("1/2", 13, 14),
    ]
    words = tokenize("seventy five percent for sure")
    assert words[0].form == "75%" and words[0].start == 0 and words[0].end == 20
