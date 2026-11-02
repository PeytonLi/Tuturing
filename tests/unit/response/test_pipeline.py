import copy
import re
import time

import pytest

from packages.response.contracts import schema_errors
from packages.response.errors import GenerationError
from packages.response.fallbacks import render_fallback
from packages.response.hashing import content_hash
from packages.response.pipeline import (
    STUB_PIPELINE_ARTIFACT_ID,
    PipelineSettings,
    ResponsePipeline,
    stub_pipeline,
)
from packages.response.prompt import compile_prompt
from packages.response.repair import is_repairable, repair_messages, trim_to_limits
from packages.response.responders import RepairLimits, ResponderError, ResponderOutput
from packages.response.validate import run_checks
from tests.unit.response.pipeline_helpers import (
    CHECK_NAMES,
    LEARNING_INVALID,
    RESPONSE_INVALID,
    ScriptedResponder,
    lane_plan,
    learning_plans,
    load_json,
    make_request,
)

TEST_ARTIFACT = "test_artifact_v1"
PRACTICE_OK = (
    "Try this one: A bag contains 3 red and 7 blue counters. One counter is drawn at "
    "random. What is the probability of drawing red?"
)
PRACTICE_TWO_QUESTIONS = "Ready? " + PRACTICE_OK.removeprefix("Try this one: ")


def scripted_pipeline(responder, **settings):
    return ResponsePipeline(responder, PipelineSettings(artifact_id=TEST_ARTIFACT, **settings))


def run(pipeline, plan, request_id="request-1", **kwargs):
    return pipeline.run(make_request(plan, request_id, artifact=TEST_ARTIFACT, **kwargs))


def test_every_learning_fixture_plan_yields_a_checked_response():
    plans = learning_plans()
    assert len(plans) == 9
    modes = {}
    for name, plan in plans.items():
        response = stub_pipeline().generate(make_request(plan, f"request-{name}"))
        assert schema_errors(response, "GenerateResponse") == []
        assert set(response["checks"]) == CHECK_NAMES
        assert all(response["checks"].values())
        assert run_checks(plan, response["text"], plan["teaching_action"]) == response["checks"]
        assert response["effective_move"] == plan["teaching_action"]
        assert response["content_hash"] == content_hash(
            response["text"], plan["teaching_action"]
        )
        assert re.fullmatch(r"response-[0-9a-f]{24}", response["response_id"])
        assert response["versions"] == {
            "response_artifact": STUB_PIPELINE_ARTIFACT_ID,
            "prompt": "prompt_v1",
            "validator": "validator_v2",
        }
        modes[name] = response["delivery_mode"]
    assert {plan["teaching_action"]["move"] for plan in plans.values()} == {
        "PROBE", "HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE", "PRACTICE", "CHALLENGE",
        "CHECK_MASTERY", "REFLECT",
    }
    # Template output is always labeled fallback, including the CHECK_MASTERY template,
    # which asks two questions and goes through the one repair (decision 2026-11-02).
    assert set(modes.values()) == {"fallback"}


@pytest.mark.parametrize("source", ["learning_plan", "response_request"])
def test_independent_plan_with_reveal_solution_is_rejected(source):
    if source == "learning_plan":
        plan = load_json(LEARNING_INVALID / "independent_reveal_solution.json")
        request = make_request(plan, "request-reveal")
    else:
        request = load_json(RESPONSE_INVALID / "independent_reveal_request.json")
    with pytest.raises(GenerationError) as info:
        stub_pipeline().generate(request)
    error = info.value
    assert (error.code, error.http_status, error.retryable) == ("INVALID_SCHEMA", 422, False)
    assert "independent plans must not carry a reveal_solution" in error.message
    assert "1/2" not in error.message


RULE_CASES = [
    ("teaching_plan_hint", lambda p: p["content"].update(permitted_hint=None),
     "HINT requires a permitted hint"),
    ("teaching_plan_hint", lambda p: p.update(problem_instance_id=None, template_family_id=None),
     "HINT requires an active item"),
    ("teaching_plan_practice", lambda p: p.update(problem_instance_id=None),
     "PRACTICE requires a problem instance and a template family"),
    ("teaching_plan_challenge", lambda p: p.update(template_family_id=None),
     "CHALLENGE requires a problem instance and a template family"),
    ("teaching_plan_check_mastery", lambda p: p["content"].update(problem_stem=None),
     "CHECK_MASTERY requires a problem stem"),
    ("teaching_plan_probe", lambda p: p.update(template_family_id=None),
     "PROBE with an item requires a problem instance and a template family"),
    ("teaching_plan_worked_example", lambda p: p["content"].update(worked_example=None),
     "WORKED_EXAMPLE requires an approved worked example"),
    ("teaching_plan_explain_concept", lambda p: p["content"].update(approved_facts=[" "]),
     "EXPLAIN_CONCEPT requires approved facts"),
    ("teaching_plan_reflect", lambda p: p["content"].update(previous_step=""),
     "REFLECT requires a previous step"),
    ("teaching_plan_practice", lambda p: p["content"].update(reveal_solution="1/2"),
     "reveal_solution requires reveal_allowed"),
    ("teaching_plan_practice", lambda p: p["teaching_action"].update(action_id=41),
     "teaching action id disagrees with its encoded fields"),
]


@pytest.mark.parametrize(("name", "mutate", "expected"), RULE_CASES)
def test_per_move_content_rules_are_enforced(name, mutate, expected):
    plan = copy.deepcopy(learning_plans()[name])
    mutate(plan)
    with pytest.raises(GenerationError) as info:
        stub_pipeline().generate(make_request(plan))
    assert info.value.code == "INVALID_SCHEMA"
    assert expected in info.value.message


def test_schema_errors_report_paths_without_echoing_values():
    plan = lane_plan("practice")
    secret = "SECRET-ANSWER-" + "7" * 130
    plan["constraints"]["forbidden_answer_fingerprints"] = [secret]
    plan["response_context"]["last_student_message"] = 424242
    request = make_request(plan)
    request["unexpected"] = "learner wrote this privately"
    with pytest.raises(GenerationError) as info:
        stub_pipeline().generate(request)
    message = info.value.message
    assert info.value.code == "INVALID_SCHEMA"
    assert "plan.constraints.forbidden_answer_fingerprints[0]: maxLength" in message
    assert "plan.response_context.last_student_message: type" in message
    assert "$: additionalProperties" in message
    for leaked in ("SECRET-ANSWER", "424242", "learner wrote this"):
        assert leaked not in message


def test_unsupported_versions_and_artifacts_are_rejected():
    plan = learning_plans()["teaching_plan_practice"]
    newer = make_request(plan)
    newer["schema_version"] = "2.0"
    newer_plan = make_request(dict(plan, schema_version="2.0"))
    other_artifact = make_request(plan, artifact="some_other_artifact_v9")
    for request in (newer, newer_plan, other_artifact):
        with pytest.raises(GenerationError) as info:
            stub_pipeline().generate(request)
        assert (info.value.code, info.value.http_status) == ("UNSUPPORTED_VERSION", 400)
        assert not info.value.retryable
    assert STUB_PIPELINE_ARTIFACT_ID in info.value.message
    legacy = stub_pipeline().generate(make_request(plan, artifact="response_stub_v1"))
    assert legacy["versions"]["response_artifact"] == STUB_PIPELINE_ARTIFACT_ID
    with pytest.raises(GenerationError) as info:
        stub_pipeline().generate(["not", "an", "object"])
    assert info.value.code == "INVALID_SCHEMA"


def test_valid_model_draft_is_delivered_as_model_with_usage():
    plan = lane_plan("practice")
    responder = ScriptedResponder([PRACTICE_OK])
    outcome = run(scripted_pipeline(responder), plan)
    response = outcome.response
    assert (response["delivery_mode"], response["text"]) == ("model", PRACTICE_OK)
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 5
    assert response["versions"]["response_artifact"] == TEST_ARTIFACT
    assert responder.repair_calls == []
    call = responder.generate_calls[0]
    assert call["messages"] == compile_prompt(plan)
    assert call["max_tokens"] == 384
    assert [attempt.redacted() for attempt in outcome.attempts] == [
        {"stage": "model", "outcome": "accepted"}
    ]


def test_question_count_failure_gets_exactly_one_repair():
    plan = lane_plan("practice")
    responder = ScriptedResponder([PRACTICE_TWO_QUESTIONS], [PRACTICE_OK])
    outcome = run(scripted_pipeline(responder), plan)
    assert outcome.response["delivery_mode"] == "repaired"
    assert outcome.response["text"] == PRACTICE_OK
    assert all(outcome.response["checks"].values())
    assert outcome.response["effective_move"] == plan["teaching_action"]
    assert outcome.response["usage"]["input_tokens"] == 20
    assert len(responder.repair_calls) == 1
    assert responder.repair_calls[0]["draft"] == PRACTICE_TWO_QUESTIONS
    assert responder.repair_calls[0]["limits"] == RepairLimits(max_words=150, max_questions=1)
    assert [attempt.redacted() for attempt in outcome.attempts] == [
        {"stage": "model", "outcome": "rejected", "failed_checks": ["question_count_valid"]},
        {"stage": "repair", "outcome": "accepted"},
    ]


def test_word_cap_failure_is_repaired():
    plan = lane_plan("practice")
    plan["constraints"]["max_words"] = 30
    long_draft = (
        "Great work so far, and thank you for staying with me through all of these steps "
        "today. " + PRACTICE_OK
    )
    responder = ScriptedResponder([long_draft], [PRACTICE_OK])
    outcome = run(scripted_pipeline(responder), plan)
    assert outcome.attempts[0].failed_checks == ("length_valid",)
    assert outcome.response["delivery_mode"] == "repaired"


def test_truncated_reply_counts_as_a_length_failure():
    plan = lane_plan("practice")
    truncated = ResponderOutput(text=PRACTICE_OK, output_tokens=384, truncated=True)
    responder = ScriptedResponder([truncated], [PRACTICE_OK])
    outcome = run(scripted_pipeline(responder), plan)
    assert outcome.attempts[0].detail == "truncated"
    assert outcome.response["delivery_mode"] == "repaired"


def test_failed_repair_is_not_retried_and_falls_back():
    plan = lane_plan("practice")
    responder = ScriptedResponder([PRACTICE_TWO_QUESTIONS], [PRACTICE_TWO_QUESTIONS])
    outcome = run(scripted_pipeline(responder), plan)
    assert len(responder.repair_calls) == 1
    assert outcome.response["delivery_mode"] == "fallback"
    assert outcome.response["text"] == render_fallback(plan)
    assert [attempt.stage for attempt in outcome.attempts] == ["model", "repair", "fallback"]


def test_repair_can_be_disabled():
    plan = lane_plan("practice")
    responder = ScriptedResponder([PRACTICE_TWO_QUESTIONS])
    outcome = run(scripted_pipeline(responder, max_repairs=0), plan)
    assert responder.repair_calls == []
    assert outcome.response["delivery_mode"] == "fallback"


@pytest.mark.parametrize(
    ("draft", "failed"),
    [
        (
            "A bag contains 3 red and 7 blue counters. The answer is 3/10. What do you get?",
            "disclosure_allowed",
        ),
        (
            "A bag contains 3 red and 7 blue counters. What is the probability of drawing "
            "red? It is close to 0.45.",
            "math_supported",
        ),
        (
            "Ready? A bag contains 3 red and 7 blue counters. The answer is 30%. What next?",
            "disclosure_allowed",
        ),
    ],
)
def test_disclosure_and_math_failures_skip_repair_and_use_the_fallback(draft, failed):
    plan = lane_plan("practice")
    responder = ScriptedResponder([draft])
    outcome = run(scripted_pipeline(responder), plan)
    assert responder.repair_calls == []
    assert failed in outcome.attempts[0].failed_checks
    assert outcome.response["delivery_mode"] == "fallback"
    assert outcome.response["text"] == render_fallback(plan)
    for leaked in ("3/10", "0.45", "30%"):
        assert leaked not in outcome.response["text"]


def test_responder_errors_and_invalid_outputs_fall_back():
    plan = lane_plan("hint")
    failing = ScriptedResponder(error=ResponderError("server down"))
    outcome = run(scripted_pipeline(failing), plan)
    assert outcome.response["delivery_mode"] == "fallback"
    assert outcome.attempts[0].redacted() == {
        "stage": "model", "outcome": "error", "detail": "ResponderError"
    }
    broken = ScriptedResponder([None])
    outcome = run(scripted_pipeline(broken), plan)
    assert outcome.attempts[0].detail == "invalid_output"
    assert outcome.response["delivery_mode"] == "fallback"


def test_slow_responder_is_cut_off_at_the_deadline():
    plan = lane_plan("hint")
    slow = ScriptedResponder([PRACTICE_OK], io_bound=True, delay_s=2.0)
    started = time.monotonic()
    outcome = run(scripted_pipeline(slow), plan, deadline_ms=200)
    assert time.monotonic() - started < 1.5
    assert outcome.attempts[0].outcome == "timeout"
    assert outcome.response["delivery_mode"] == "fallback"
    assert slow.generate_calls[0]["timeout_s"] <= 0.2


def test_internal_ceiling_caps_the_deadline_and_zero_budget_is_rejected():
    plan = lane_plan("practice")
    responder = ScriptedResponder([PRACTICE_OK])
    run(scripted_pipeline(responder), plan, deadline_ms=60_000)
    assert responder.generate_calls[0]["timeout_s"] <= 8.0
    with pytest.raises(GenerationError) as info:
        run(scripted_pipeline(ScriptedResponder([PRACTICE_OK])), plan, deadline_ms=0)
    assert (info.value.code, info.value.http_status, info.value.retryable) == (
        "DEADLINE_EXCEEDED", 504, True
    )
    for bad in ({"internal_deadline_ms": 9000}, {"max_output_tokens": 500}, {"max_repairs": 2}):
        with pytest.raises(ValueError):
            PipelineSettings(artifact_id=TEST_ARTIFACT, **bad)


def test_no_valid_text_raises_a_retryable_generation_unavailable_error():
    plan = lane_plan("hint")
    plan["constraints"]["max_words"] = 3
    responder = ScriptedResponder(error=ResponderError("server down"))
    with pytest.raises(GenerationError) as info:
        run(scripted_pipeline(responder), plan)
    error = info.value
    assert (error.code, error.http_status, error.retryable) == (
        "GENERATION_UNAVAILABLE", 503, True
    )
    assert [attempt.redacted() for attempt in error.attempts] == [
        {"stage": "model", "outcome": "error", "detail": "ResponderError"},
        {"stage": "fallback", "outcome": "unavailable", "detail": "FallbackUnavailable"},
    ]


def test_same_request_gives_the_same_response_id_and_hash():
    for name, plan in learning_plans().items():
        request = make_request(plan, f"request-{name}")
        first = stub_pipeline().generate(copy.deepcopy(request))
        second = stub_pipeline().generate(copy.deepcopy(request))
        first.pop("usage")
        second.pop("usage")
        assert first == second
    other = stub_pipeline().generate(make_request(plan, "request-other"))
    assert other["response_id"] != first["response_id"]
    assert other["content_hash"] == first["content_hash"]


def test_trim_keeps_the_stem_question_and_drops_extra_questions():
    stem = (
        "A box contains 4 red and 6 blue tiles. One tile is drawn at random. "
        "What is the probability of drawing blue?"
    )
    text = f"Work this one independently: {stem} What is your answer?"
    assert trim_to_limits(text, max_words=150, max_questions=1, anchors=(stem,)) == (
        f"Work this one independently: {stem}"
    )
    assert trim_to_limits(
        f"Can you try this? {stem}", max_words=150, max_questions=1, anchors=(stem,)
    ) == stem
    assert "?" not in trim_to_limits(stem, max_words=150, max_questions=0, anchors=(stem,))


def test_trim_drops_filler_before_anchored_sentences_and_the_closing_request():
    stem = "A spinner has 10 equal sections, and 3 are blue."
    text = (
        "You are doing really well with all of these spinner questions today. "
        f"{stem} Give it a try and tell me your answer."
    )
    trimmed = trim_to_limits(text, max_words=20, max_questions=1, anchors=(stem,))
    assert trimmed == f"{stem} Give it a try and tell me your answer."
    assert len(trim_to_limits(text, max_words=5, max_questions=1).split()) == 5


def test_only_length_and_question_failures_are_repairable():
    passing = dict.fromkeys(CHECK_NAMES, True)
    assert not is_repairable(passing)
    assert is_repairable(passing, truncated=True)
    assert is_repairable(dict(passing, length_valid=False, question_count_valid=False))
    assert not is_repairable(dict(passing, question_count_valid=False, disclosure_allowed=False))
    assert not is_repairable(dict(passing, math_supported=False))
    assert not is_repairable(dict(passing, action_faithful=False), truncated=True)


def test_repair_messages_quote_the_draft_and_state_the_limits():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    repaired = repair_messages(messages, "draft text", max_words=80, max_questions=1)
    assert repaired[:2] == messages
    assert repaired[2] == {"role": "assistant", "content": "draft text"}
    assert "at most 80 words" in repaired[3]["content"]
    assert "at most one question" in repaired[3]["content"]
    assert "no question marks" in repair_messages(
        messages, "d", max_words=80, max_questions=0
    )[3]["content"]
