import socket

import pytest

from packages.response.fallbacks import render_fallback
from packages.response.pipeline import PipelineSettings, ResponsePipeline
from packages.response.prompt import compile_prompt
from packages.response.responders import (
    OpenAICompatibleResponder,
    RepairLimits,
    ResponderConfigError,
    ResponderError,
    TemplateStubResponder,
    build_responder,
    strip_reasoning,
)
from packages.response.stub import render_template
from tests.unit.response.fake_openai import FakeOpenAIServer, Reply
from tests.unit.response.pipeline_helpers import lane_plan, learning_plans, make_request

KEY_ENV = "TUTOR_TEST_LLM_KEY"
KEY_VALUE = "fake-key-never-logged"
ARTIFACT = "prompt_only_fake_v1"
CHECK_OK = (
    "On your own: a box contains 4 red and 6 blue tiles, and one tile is drawn at random. "
    "What is the probability of drawing blue?"
)


def responder_for(server, **overrides):
    options = {
        "base_url": server.base_url,
        "model": "fake-model",
        "api_key_env": KEY_ENV,
        "environ": {KEY_ENV: KEY_VALUE},
    }
    options.update(overrides)
    return OpenAICompatibleResponder(**options)


def model_pipeline(server, **overrides):
    return ResponsePipeline(
        responder_for(server, **overrides), PipelineSettings(artifact_id=ARTIFACT)
    )


def test_request_shape_greedy_decoding_token_cap_and_usage():
    plan = lane_plan("check_mastery")
    messages = compile_prompt(plan)
    reply = Reply(CHECK_OK, usage={"prompt_tokens": 321, "completion_tokens": 25})
    with FakeOpenAIServer([reply]) as server:
        responder = responder_for(server, extra_body={"chat_template_kwargs": {"x": False}})
        output = responder.generate(messages, plan=plan, max_tokens=384, timeout_s=5)
    assert output.text == CHECK_OK
    assert (output.input_tokens, output.output_tokens, output.truncated) == (321, 25, False)
    (request,) = server.requests
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == f"Bearer {KEY_VALUE}"
    assert request["headers"]["content-type"] == "application/json"
    assert request["body"] == {
        "chat_template_kwargs": {"x": False},
        "model": "fake-model",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 384,
        "stream": False,
    }
    for secret in plan["constraints"]["forbidden_answer_fingerprints"]:
        assert secret not in str(request["body"]["messages"])


def test_api_key_comes_only_from_the_named_environment_variable():
    with pytest.raises(ResponderConfigError) as info:
        OpenAICompatibleResponder(
            base_url="http://127.0.0.1:9/v1", model="m", api_key_env=KEY_ENV, environ={}
        )
    assert KEY_ENV in str(info.value)
    with FakeOpenAIServer([CHECK_OK]) as server:
        responder = responder_for(server, api_key_env=None)
        responder.generate([], plan={}, max_tokens=384, timeout_s=5)
        keyed = responder_for(server)
    assert "authorization" not in server.requests[0]["headers"]
    assert KEY_VALUE not in repr(keyed)
    assert KEY_VALUE not in str(keyed.describe())


@pytest.mark.parametrize(
    "reply",
    [
        Reply(status=500),
        Reply(status=401),
        Reply(raw_body=b"not json"),
        Reply(raw_body=b'{"choices": []}'),
        Reply(status=307, headers={"Location": "http://127.0.0.1:9/elsewhere"}),
    ],
)
def test_server_failures_raise_responder_errors(reply):
    with FakeOpenAIServer([reply]) as server:
        with pytest.raises(ResponderError) as info:
            responder_for(server).generate([], plan={}, max_tokens=384, timeout_s=5)
    assert KEY_VALUE not in str(info.value)
    assert len(server.requests) == 1


def test_unreachable_server_raises_a_responder_error():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    responder = OpenAICompatibleResponder(base_url=f"http://127.0.0.1:{port}/v1", model="m")
    with pytest.raises(ResponderError):
        responder.generate([], plan={}, max_tokens=384, timeout_s=2)


def test_reply_parsing_strips_reasoning_joins_parts_and_flags_truncation():
    replies = [
        Reply("<think>The answer is 3/5.</think>\nOn your own, try it."),
        Reply([{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]),
        Reply("cut off mid", finish_reason="length"),
        Reply(None),
    ]
    with FakeOpenAIServer(replies) as server:
        responder = responder_for(server)
        outputs = [
            responder.generate([], plan={}, max_tokens=384, timeout_s=5) for _ in replies
        ]
    assert [output.text for output in outputs] == [
        "On your own, try it.", "Part one. Part two.", "cut off mid", ""
    ]
    assert [output.truncated for output in outputs] == [False, False, True, False]


def test_strip_reasoning_handles_open_closed_and_missing_blocks():
    assert strip_reasoning("plain reply") == "plain reply"
    assert strip_reasoning("<think>hidden</think> shown") == "shown"
    assert strip_reasoning("prefilled reasoning</think>shown") == "shown"
    assert strip_reasoning("shown <THINK>unfinished reasoning") == "shown"
    assert strip_reasoning("<think>never closed") == ""


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"kind": "openai_compatible", "base_url": "ftp://host/v1", "model": "m"}, "http"),
        ({"kind": "openai_compatible", "base_url": "http://u:p@host/v1", "model": "m"},
         "credentials"),
        ({"kind": "openai_compatible", "base_url": "http://host/v1", "model": " "}, "model"),
        ({"kind": "openai_compatible", "base_url": "http://host/v1", "model": "m",
          "extra_body": {"temperature": 1}}, "temperature"),
        ({"kind": "openai_compatible", "base_url": "http://host/v1", "model": "m",
          "api_key": "inline-secret"}, "unknown"),
        ({"kind": "openai_compatible", "base_url": "http://host/v1", "model": "m",
          "api_key_env": "not a name"}, "api_key_env"),
        ({"kind": "template_stub", "model": "m"}, "unknown"),
        ({"kind": "sampling_llm"}, "responder.kind"),
    ],
)
def test_responder_config_is_validated(spec, message):
    with pytest.raises(ResponderConfigError) as info:
        build_responder(spec, environ={})
    assert message in str(info.value)


def test_template_responder_renders_stub_templates_and_trims_on_repair():
    plan = learning_plans()["teaching_plan_check_mastery"]
    responder = build_responder({"kind": "template_stub"})
    assert isinstance(responder, TemplateStubResponder)
    draft = responder.generate([], plan=plan, max_tokens=384, timeout_s=1)
    assert draft.text == render_template(plan)
    assert (draft.input_tokens, draft.output_tokens) == (0, 0)
    repaired = responder.repair(
        [], draft.text, plan=plan, limits=RepairLimits(150, 1), max_tokens=384, timeout_s=1
    )
    assert repaired.text.count("?") == 1
    assert plan["content"]["problem_stem"] in repaired.text


@pytest.mark.parametrize("leak", ["3/5", "0.6", "60%", "60 percent"])
def test_reply_that_leaks_a_protected_answer_falls_back(leak):
    plan = lane_plan("check_mastery")
    leaking = (
        "A box contains 4 red and 6 blue tiles. One tile is drawn at random, so the chance "
        f"of drawing blue is {leak}. Can you see why?"
    )
    with FakeOpenAIServer([leaking]) as server:
        outcome = model_pipeline(server).run(make_request(plan, artifact=ARTIFACT))
    response = outcome.response
    assert len(server.requests) == 1
    assert "disclosure_allowed" in outcome.attempts[0].failed_checks
    assert response["delivery_mode"] == "fallback"
    assert response["text"] == render_fallback(plan)
    assert leak not in response["text"]
    assert all(response["checks"].values())
    assert response["usage"]["input_tokens"] == 120


def test_model_repair_sends_the_draft_back_once_with_the_limits():
    plan = lane_plan("check_mastery")
    two_questions = "Ready to try one? " + CHECK_OK
    with FakeOpenAIServer([two_questions, CHECK_OK]) as server:
        outcome = model_pipeline(server).run(make_request(plan, artifact=ARTIFACT))
    assert outcome.response["delivery_mode"] == "repaired"
    assert outcome.response["text"] == CHECK_OK
    first, second = (request["body"]["messages"] for request in server.requests)
    assert second[: len(first)] == first
    assert second[len(first)] == {"role": "assistant", "content": two_questions}
    assert "at most 150 words" in second[-1]["content"]


def test_reasoning_is_stripped_before_validation():
    plan = lane_plan("check_mastery")
    with FakeOpenAIServer([f"<think>It is 3/5.</think>{CHECK_OK}"]) as server:
        response = model_pipeline(server).generate(make_request(plan, artifact=ARTIFACT))
    assert (response["delivery_mode"], response["text"]) == ("model", CHECK_OK)


def test_slow_model_server_falls_back_within_the_deadline():
    plan = lane_plan("check_mastery")
    with FakeOpenAIServer([Reply(CHECK_OK, delay_s=2.0)]) as server:
        outcome = model_pipeline(server).run(
            make_request(plan, artifact=ARTIFACT, deadline_ms=300)
        )
    assert outcome.attempts[0].stage == "model"
    assert outcome.attempts[0].outcome in {"timeout", "error"}
    assert outcome.response["delivery_mode"] == "fallback"
    assert outcome.response["usage"]["elapsed_ms"] < 1500
