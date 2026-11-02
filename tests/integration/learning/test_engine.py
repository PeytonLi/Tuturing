import copy
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from apps.learning_service.engine import LearningEngine, ServiceError
from apps.learning_service.schemas import validate
from packages.learning.actions import decode_navigator, encode_strategist
from packages.learning.episode import (
    _public_state,
    update_public_misconceptions,
)
from packages.learning.features import build_features, feature_order
from packages.learning.scheduler import GoalStack


@pytest.fixture
def engine(tmp_path):
    instance = LearningEngine({}, tmp_path / "learning.sqlite")
    yield instance
    instance.close()


def create(engine, key="create", goal="C01"):
    result = engine.create_session(
        dict(schema_version="1.0", request_id=key, goal_concept_id=goal, pace_preference="normal"),
        trace_id="trace",
    )
    validate("CreateSessionResponse", result)
    validate("PublicState", result["public_state"])
    return result["session_id"]


def observe(engine, sid, version=0, evidence=None, key="observe"):
    body = dict(
        schema_version="1.0",
        request_id=key,
        session_id=sid,
        expected_state_version=version,
        evidence=evidence,
    )
    result = engine.observe_and_plan(body, trace_id="trace")
    validate("ObserveResponse", result)
    return result


def commit_body(plan):
    return dict(
        schema_version="1.0",
        session_id=plan["session_id"],
        plan_id=plan["plan_id"],
        based_on_state_version=plan["based_on_state_version"],
        response_id="response-" + plan["plan_id"],
        content_hash="a" * 64,
        effective_move=copy.deepcopy(plan["teaching_action"]),
        delivery_mode="model",
    )


def commit(engine, plan, key="commit"):
    result = engine.commit_plan(commit_body(plan), key, trace_id="trace")
    validate("CommitPlanResponse", result)
    return result


def state(engine, sid):
    with engine.store.transaction() as db:
        return json.loads(
            db.execute("SELECT state_json FROM sessions WHERE session_id=?", (sid,)).fetchone()[0]
        )


def evidence(engine, plan, *, answer=None, assistance="none", request="none", event="e1"):
    stored = state(engine, plan["session_id"])
    item = engine._item(stored["active_item"])
    return dict(
        schema_version="1.0",
        event_id=event,
        session_id=plan["session_id"],
        trace_id="trace",
        source_response_id=stored["last_response_id"],
        problem_instance_id=plan["problem_instance_id"],
        observed_at="2026-11-02T00:00:00Z",
        normalized_answer=str(item["instance"]["answer"]) if answer is None and item else answer,
        correctness="incorrect",
        grade_source="extractor",
        confidence=1.0,
        assistance=assistance,
        explicit_request=request,
        requested_concept_id=None,
        misconception_hypotheses=[],
        raw_message_ref="message",
        extractor_version="test",
        extensions={},
    )


def cancel(engine, plan, key="cancel"):
    return engine.cancel_plan(
        dict(
            schema_version="1.0",
            request_id=key,
            session_id=plan["session_id"],
            plan_id=plan["plan_id"],
        ),
        trace_id="trace",
    )


def assert_error(code, fn):
    with pytest.raises(ServiceError) as caught:
        fn()
    assert caught.value.code == code
    validate("Error", caught.value.as_dict("request", "trace"))
    if code == "STATE_CONFLICT":
        assert caught.value.http_status == 409
    return caught.value


def test_happy_path_idempotency_and_evidence(engine):
    sid = create(engine)
    initial = state(engine, sid)
    result = observe(engine, sid)
    plan = result["teaching_plan"]
    validate("TeachingPlan", plan)
    assert plan["trace_id"] == "trace"
    assert result == observe(engine, sid)
    assert_error("STATE_CONFLICT", lambda: observe(engine, sid, key="other"))
    assert_error("STATE_CONFLICT", lambda: observe(engine, sid, version=1))
    assert state(engine, sid)["teaching_turns"] == 0
    committed = commit(engine, plan)
    assert committed["state_version"] == 1
    ev = evidence(engine, plan)
    next_result = observe(engine, sid, 1, ev, "next")
    assert next_result["kind"] in {"plan", "session_end"}
    updated = state(engine, sid)
    assert updated["beliefs"]["C01"]["mastery"] > initial["beliefs"]["C01"]["mastery"]
    assert updated["beliefs"]["C01"]["independent_correct"] == 1
    assert updated["teaching_turns"] == 1
    if next_result["kind"] == "plan":
        cancel(engine, next_result["teaching_plan"])
        observe(engine, sid, 1, ev, "replan")
        assert state(engine, sid)["beliefs"] == updated["beliefs"]


def test_concurrent_reservations(engine):
    sid = create(engine)
    barrier = Barrier(2)

    def run(key):
        barrier.wait()
        try:
            return observe(engine, sid, key=key)["kind"]
        except ServiceError as error:
            assert error.current_state_version == 0
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(run, ["a", "b"])) == ["STATE_CONFLICT", "plan"]


def test_cancel_keeps_learning_state_and_uses_new_id(engine):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    before = state(engine, sid)
    result = cancel(engine, plan)
    validate("PlanStatus", result)
    assert result["status"] == "canceled"
    assert state(engine, sid) == before
    newer = observe(engine, sid, key="new")["teaching_plan"]
    assert newer["plan_id"] != plan["plan_id"]
    assert state(engine, sid)["state_version"] == 0


def test_expiry_and_restart(tmp_path):
    now = [datetime(2026, 11, 2, tzinfo=UTC)]
    path = tmp_path / "learning.sqlite"
    engine = LearningEngine({}, path, clock=lambda: now[0])
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    before = state(engine, sid)
    engine.close()
    engine = LearningEngine({}, path, clock=lambda: now[0])
    assert state(engine, sid) == before
    assert engine.get_plan(plan["plan_id"], trace_id="trace")["status"] == "pending"
    assert observe(engine, sid) == dict(kind="plan", teaching_plan=plan)
    now[0] += timedelta(seconds=600)
    assert_error("PLAN_EXPIRED", lambda: commit(engine, plan))
    assert engine.get_plan(plan["plan_id"], trace_id="trace")["status"] == "canceled"
    newer = observe(engine, sid, key="new")["teaching_plan"]
    assert newer["plan_id"] != plan["plan_id"]
    now[0] += timedelta(seconds=601)
    assert engine.get_plan(newer["plan_id"], trace_id="trace")["status"] == "canceled"
    assert state(engine, sid)["teaching_turns"] == 0
    engine.close()


@pytest.mark.parametrize("assistance", ["hint", "answer_shown", "example", "unknown"])
def test_assisted_evidence_does_not_move_mastery(engine, assistance):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    before = state(engine, sid)
    observe(engine, sid, 1, evidence(engine, plan, assistance=assistance), "next")
    after = state(engine, sid)
    assert after["beliefs"]["C01"]["mastery"] == before["beliefs"]["C01"]["mastery"]
    assert after["beliefs"]["C01"]["independent_correct"] == 0


class HintStrategist:
    policy_version = "test_hint_v1"

    def act(self, observation, mask, context):
        if context["active_item"]:
            return encode_strategist("HINT", context["active_item_difficulty"], "normal")
        return encode_strategist("PRACTICE", "easy", "normal")


def test_committed_assistance_blocks_false_independent_claim(tmp_path):
    engine = LearningEngine({}, tmp_path / "db", strategist=HintStrategist())
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    second = observe(engine, sid, 1, evidence(engine, first, answer="nonsense"), "second")[
        "teaching_plan"
    ]
    assert second["assistance"] == "hint"
    commit(engine, second, "commit2")
    before = state(engine, sid)["beliefs"]["C01"]["mastery"]
    observe(engine, sid, 2, evidence(engine, second, event="e2"), "third")
    assert state(engine, sid)["beliefs"]["C01"]["mastery"] == before
    engine.close()


def test_commit_validation_replay_and_no_answer_in_errors(engine):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    body = commit_body(plan)
    bad = copy.deepcopy(body)
    bad["effective_move"]["pace"] = "concise"
    error = assert_error("INVALID_SCHEMA", lambda: engine.commit_plan(bad, "bad", trace_id="t"))
    item = engine._item(
        json.loads(
            engine.store.connection.execute(
                "SELECT item_json FROM plans WHERE plan_id=?", (plan["plan_id"],)
            ).fetchone()[0]
        )
    )
    assert str(item["instance"]["answer"]) not in error.message
    bad = {**body, "content_hash": "invalid"}
    assert_error("INVALID_SCHEMA", lambda: engine.commit_plan(bad, "bad", trace_id="t"))
    result = commit(engine, plan)
    assert result == commit(engine, plan)
    assert result == commit(engine, plan, "another-key")
    changed = {**body, "content_hash": "b" * 64}
    assert_error("STATE_CONFLICT", lambda: engine.commit_plan(changed, "commit", trace_id="t"))
    assert_error("STATE_CONFLICT", lambda: engine.commit_plan(changed, "new", trace_id="t"))
    assert_error("STATE_CONFLICT", lambda: cancel(engine, plan))
    assert state(engine, sid)["teaching_turns"] == 1


@pytest.mark.parametrize("reason", ["stop", "budget"])
def test_ended_session_rejects_mutations(tmp_path, reason):
    engine = LearningEngine({"teaching_budget": 1 if reason == "budget" else 16}, tmp_path / "db")
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    ev = evidence(engine, plan, request="stop" if reason == "stop" else "none")
    result = observe(engine, sid, 1, ev, "end")
    validate("SessionEnd", result["session_end"])
    assert result["session_end"]["reason"] == (
        "learner_stop" if reason == "stop" else "teaching_budget"
    )
    assert result == observe(engine, sid, 1, ev, "end")
    assert_error("SESSION_ENDED", lambda: observe(engine, sid, 2, ev, "late"))
    assert_error("SESSION_ENDED", lambda: commit(engine, plan, "late"))
    assert_error("SESSION_ENDED", lambda: cancel(engine, plan))
    engine.close()


def test_grade_and_unknowns(engine):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    with engine.store.transaction() as db:
        data = json.loads(
            db.execute(
                "SELECT item_json FROM plans WHERE plan_id=?", (plan["plan_id"],)
            ).fetchone()[0]
        )
    answer = str(engine._item(data)["instance"]["answer"])
    for index, (item_id, value, correctness, source) in enumerate(
        [
            (plan["problem_instance_id"], answer, "correct", "deterministic"),
            (plan["problem_instance_id"], "100", "incorrect", "deterministic"),
            (plan["problem_instance_id"], "nonsense", "unknown", "none"),
            ("missing", answer, "unknown", "none"),
        ]
    ):
        body = dict(
            schema_version="1.0",
            request_id=f"grade{index}",
            problem_instance_id=item_id,
            normalized_answer=value,
        )
        result = engine.grade(body, trace_id="t")
        validate("GradeResponse", result)
        assert result["correctness"] == correctness
        assert result["grade_source"] == source
        assert result == engine.grade(body, trace_id="t")
    assert_error("INVALID_SCHEMA", lambda: engine.get_plan("missing", trace_id="t"))
    assert_error("INVALID_SCHEMA", lambda: observe(engine, "missing", key="unknown-session"))
    assert_error("STATE_CONFLICT", lambda: observe(engine, "missing"))


def test_evidence_references_and_null_rejected(engine):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    assert_error("INVALID_SCHEMA", lambda: observe(engine, sid, 1, key="null"))
    for field in ("session_id", "source_response_id", "problem_instance_id"):
        ev = evidence(engine, plan)
        ev[field] = "wrong"
        assert_error("INVALID_SCHEMA", lambda: observe(engine, sid, 1, ev, "bad"))
    assert state(engine, sid)["applied_evidence"] == []


def test_incorrect_grader_overrides_extractor(engine):
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    ev = evidence(engine, plan, answer="100")
    ev["correctness"] = "correct"
    observe(engine, sid, 1, ev, "next")
    belief = state(engine, sid)["beliefs"]["C01"]
    assert belief["mastery"] < 0.35
    assert belief["independent_incorrect"] == 1


def test_public_misconception_features_match_episode_and_service(engine):
    sid = create(engine, goal="C06")
    first_plan = observe(engine, sid)["teaching_plan"]
    commit(engine, first_plan)
    committed_state = state(engine, sid)
    item = engine._item(committed_state["active_item"])
    assert item is not None
    assert "M05" in item["family"].error_generators
    error_answer = item["family"].generate_error("M05", item["instance"])
    next_result = observe(
        engine,
        sid,
        1,
        evidence(engine, first_plan, answer=str(error_answer), event="misconception"),
        "next",
    )
    assert next_result["kind"] == "plan"

    service_state = state(engine, sid)
    expected_misconceptions = update_public_misconceptions(
        {},
        family=item["family"],
        instance=item["instance"],
        normalized_answer=str(error_answer),
        correctness="incorrect",
        move=first_plan["teaching_action"]["move"],
    )
    assert service_state["public_misconceptions"] == expected_misconceptions
    beliefs = engine._beliefs(service_state)
    stack = GoalStack(
        stack=service_state["stack"],
        visit_index=service_state["visit_index"],
        last_pair_visit={
            tuple(key): value for key, value in service_state["last_pair_visit"]
        },
        pair_visits=defaultdict(
            int, {tuple(key): value for key, value in service_state["pair_visits"]}
        ),
    )
    service_item = engine._item(service_state["active_item"])
    service_public, _ = engine._context(service_state, beliefs, service_item, stack)
    navigation = service_state["navigation_action_id"]
    operation, index = (
        decode_navigator(navigation) if navigation is not None else (None, None)
    )
    episode_public = _public_state(
        beliefs,
        current=service_state["current_concept_id"],
        goal=service_state["goal_concept_id"],
        turn=service_state["teaching_turns"],
        active_item=service_item,
        active_option_type=operation,
        active_target=f"C{index + 1:02d}" if index is not None else None,
        previous_move=service_state["previous_move"],
        option_turns=service_state["option_turns"],
        success_streak=service_state["success_streak"],
        failure_streak=service_state["failure_streak"],
        hint_count=service_state["hint_count"],
        explicit_skip=service_state["requests"]["skip"],
        explicit_faster=service_state["requests"]["faster"],
        explicit_slower=service_state["requests"]["slower"],
        stack=stack.stack,
        misconceptions=expected_misconceptions,
        pace_lean_faster=service_state["pace_estimator"]["lean_faster"],
        pace_lean_slower=service_state["pace_estimator"]["lean_slower"],
    )
    episode_public["remaining_budget"] = (
        service_state["teaching_budget"] - service_state["teaching_turns"]
    )
    service_features = build_features(service_public)
    episode_features = build_features(episode_public)
    assert service_features == episode_features
    misconception_start = feature_order().index("misconception_probability_M01")
    misconception_end = feature_order().index("misconception_probability_M25") + 1
    assert service_features[misconception_start:misconception_end] == [
        expected_misconceptions.get(f"M{index:02d}", 0.0)
        for index in range(1, 26)
    ]
    m05_index = feature_order().index("misconception_probability_M05")
    assert service_features[m05_index] > 0.0

    cancel(engine, next_result["teaching_plan"], "cancel-for-decay")

    class PracticeStrategist:
        policy_version = "test_practice_v1"

        def act(self, observation, mask, context):
            return encode_strategist("PRACTICE", context["active_item_difficulty"], "normal")

    engine.strategist = PracticeStrategist()
    decay_result = observe(
        engine,
        sid,
        1,
        evidence(
            engine,
            first_plan,
            answer=str(error_answer),
            event="misconception",
        ),
        "replan-for-decay",
    )
    assert decay_result["kind"] == "plan"
    decay_plan = decay_result["teaching_plan"]
    commit(engine, decay_plan, "commit-decay")
    second_state = state(engine, sid)
    second_item = engine._item(second_state["active_item"])
    assert second_item is not None
    assert "M05" in second_item["family"].error_generators
    expected_after_correct = update_public_misconceptions(
        expected_misconceptions,
        family=second_item["family"],
        instance=second_item["instance"],
        normalized_answer=str(second_item["instance"]["answer"]),
        correctness="correct",
        move=decay_plan["teaching_action"]["move"],
    )
    observe(
        engine,
        sid,
        2,
        evidence(
            engine,
            decay_plan,
            answer=str(second_item["instance"]["answer"]),
            event="correct",
        ),
        "decay",
    )
    assert state(engine, sid)["public_misconceptions"] == expected_after_correct


def test_certified_goal_ends_before_another_plan(engine):
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    second = observe(engine, sid, 1, evidence(engine, first), "second")["teaching_plan"]
    assert second["template_family_id"] != first["template_family_id"]
    commit(engine, second, "commit2")
    result = observe(engine, sid, 2, evidence(engine, second, event="e2"), "third")
    assert result["session_end"]["reason"] == "demonstrated_completion"
    assert result["session_end"]["public_state"]["teaching_turns"] == 2


@pytest.mark.parametrize("explicit_request,pace", [("faster", "concise"), ("slower", "detailed")])
def test_requested_pace(engine, explicit_request, pace):
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    result = observe(engine, sid, 1, evidence(engine, first, request=explicit_request), "next")
    assert result["teaching_plan"]["teaching_action"]["pace"] == pace


@pytest.mark.parametrize("explicit_request", ["skip", "change_goal"])
def test_requested_goal_and_proposed_navigation(engine, explicit_request):
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    ev = evidence(engine, first, request=explicit_request)
    ev["requested_concept_id"] = "C08"
    next_plan = observe(engine, sid, 1, ev, "next")["teaching_plan"]
    assert next_plan["goal_concept_id"] == "C08"
    before_commit = state(engine, sid)
    assert before_commit["current_concept_id"] == "C01"
    if explicit_request == "skip":
        assert next_plan["concept_id"] == "C08"
        assert next_plan["navigation_action"]["operation"] == "SKIP_TO_TARGET"
    commit(engine, next_plan, "commit2")
    assert state(engine, sid)["current_concept_id"] == next_plan["concept_id"]


class InvalidStrategist:
    policy_version = "invalid_test_v1"

    def act(self, observation, mask, context):
        return -1


def test_policy_failure_retains_evidence_without_caching(engine):
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    ev = evidence(engine, first)
    original = engine.strategist
    engine.strategist = InvalidStrategist()
    assert_error("LEARNING_UNAVAILABLE", lambda: observe(engine, sid, 1, ev, "next"))
    before = state(engine, sid)
    assert before["beliefs"]["C01"]["independent_correct"] == 1
    assert before["teaching_turns"] == 1
    engine.strategist = original
    observe(engine, sid, 1, ev, "next")
    assert state(engine, sid)["beliefs"] == before["beliefs"]


def test_empty_and_invalid_requests(engine):
    for body in (None, [], {}, {"schema_version": "1.0"}):
        assert_error("INVALID_SCHEMA", lambda: engine.create_session(body, trace_id="t"))
    assert_error("INVALID_SCHEMA", lambda: create(engine, goal="C15"))


def test_idempotency_after_restart_and_atomic_event_log(tmp_path):
    path = tmp_path / "db"
    engine = LearningEngine({}, path)
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    result = commit(engine, plan)
    engine.close()
    engine = LearningEngine({}, path)
    assert create(engine) == sid
    assert commit(engine, plan) == result
    assert (
        engine.get_plan(plan["plan_id"], trace_id="t")["response_id"]
        == commit_body(plan)["response_id"]
    )
    with engine.store.transaction() as db:
        events = db.execute(
            "SELECT seq, kind FROM events WHERE session_id=? ORDER BY seq", (sid,)
        ).fetchall()
    assert [(row["seq"], row["kind"]) for row in events] == [
        (1, "session_created"),
        (2, "plan_reserved"),
        (3, "teaching_committed"),
    ]
    engine.close()


class ProbeStrategist:
    policy_version = "probe_test_v1"

    def act(self, observation, mask, context):
        return encode_strategist("PROBE", "easy", "normal")


def test_assessment_usage_survives_restart_and_cancellation(tmp_path):
    path = tmp_path / "families.sqlite"
    engine = LearningEngine({}, path, strategist=ProbeStrategist())
    try:
        sid = create(engine)
        ev = None
        selected = []
        for turn in range(2):
            plan = observe(engine, sid, turn, ev, f"observe-{turn}")["teaching_plan"]
            selected.append(plan["template_family_id"])
            commit(engine, plan, f"commit-{turn}")
            ev = evidence(engine, plan, answer="not-an-answer", event=f"e-{turn}")
        assert selected == ["c01_assess_01", "c01_assess_04"]
        before = state(engine, sid)
        assert before["used_assessment"] == dict.fromkeys(selected, 1)
        assert before["assessment_family_reuse_counts"] == {}
        third = observe(engine, sid, 2, ev, "third")
        engine.close()
        engine = LearningEngine({}, path, strategist=ProbeStrategist())
        assert observe(engine, sid, 2, ev, "third") == third
        cancel(engine, third["teaching_plan"])
        assert state(engine, sid)["used_assessment"] == before["used_assessment"]
        assert state(engine, sid)["assessment_family_reuse_counts"] == {}
        plan = observe(engine, sid, 2, ev, "replacement")["teaching_plan"]
        assert plan["template_family_id"] == selected[0]
        commit(engine, plan, "commit-replacement")
        assert state(engine, sid)["used_assessment"] == {selected[0]: 2, selected[1]: 1}
        assert state(engine, sid)["assessment_family_reuse_counts"] == {"C01": 1}
        ev = evidence(engine, plan, answer="not-an-answer", event="e-third")
        plan = observe(engine, sid, 3, ev, "fourth")["teaching_plan"]
        assert plan["template_family_id"] == selected[1]
        commit(engine, plan, "commit-fourth")
        assert state(engine, sid)["used_assessment"] == dict.fromkeys(selected, 2)
        assert state(engine, sid)["assessment_family_reuse_counts"] == {"C01": 2}
    finally:
        engine.close()


@pytest.mark.parametrize("legacy_location", ["session", "pending_plan"])
def test_legacy_assessment_family_lists(engine, legacy_location):
    sid = create(engine)
    ev = None
    version = 0
    if legacy_location == "pending_plan":
        plan = observe(engine, sid)["teaching_plan"]
    with engine.store.transaction() as db:
        if legacy_location == "session":
            stored = json.loads(
                db.execute("SELECT state_json FROM sessions WHERE session_id=?", (sid,)).fetchone()[
                    0
                ]
            )
        else:
            stored = json.loads(
                db.execute(
                    "SELECT state_json FROM plans WHERE plan_id=?", (plan["plan_id"],)
                ).fetchone()[0]
            )
        stored["used_assessment"] = ["c01_assess_01"]
        del stored["assessment_family_reuse_counts"]
        if legacy_location == "session":
            engine._save(db, stored)
        else:
            db.execute(
                "UPDATE plans SET state_json=? WHERE plan_id=?",
                (json.dumps(stored), plan["plan_id"]),
            )
    if legacy_location == "pending_plan":
        commit(engine, plan)
        version = 1
        ev = evidence(engine, plan, answer="not-an-answer")
    engine.strategist = ProbeStrategist()
    result = observe(engine, sid, version, ev, "next")["teaching_plan"]
    assert result["template_family_id"] == "c01_assess_04"
    commit(engine, result, "next-commit")
    assert state(engine, sid)["used_assessment"] == {"c01_assess_01": 1, "c01_assess_04": 1}
    assert state(engine, sid)["assessment_family_reuse_counts"] == {}
