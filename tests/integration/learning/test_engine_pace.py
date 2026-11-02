import json

from tests.integration.learning.test_engine import (
    commit,
    create,
    engine,  # noqa: F401
    evidence,
    observe,
    state,
)


def _with_cues(ev, cues):
    return {**ev, "extensions": {"pace_cues": cues}}


def test_create_session_stores_estimator(engine):  # noqa: F811
    sid = create(engine)
    stored = state(engine, sid)
    assert stored["pace_estimator"]["pace"] == "normal"
    assert stored["requests"]["faster"] is False


def test_faster_steps_one_and_flag_resets(engine):  # noqa: F811
    sid = create(engine)
    first = observe(engine, sid)["teaching_plan"]
    commit(engine, first)
    second = observe(engine, sid, 1, evidence(engine, first, request="faster"), "o2")
    plan = second["teaching_plan"]
    assert plan["teaching_action"]["pace"] == "concise"
    stored = state(engine, sid)
    assert stored["pace"] == "concise"
    assert stored["pace_estimator"]["lean_faster"] == 1.0
    commit(engine, plan, "c2")
    observe(engine, sid, 2, evidence(engine, plan, event="e2"), "o3")
    stored = state(engine, sid)
    assert stored["requests"]["faster"] is False
    assert stored["pace_estimator"]["lean_faster"] < 1.0
    assert stored["pace"] == "concise"


def test_pace_cues_round_trip(engine):  # noqa: F811
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    version = 1
    slow = {"reply_seconds": 120.0, "message_words": 50, "confusion": True, "why_question": True}
    paces = []
    for index in range(2):
        commit(engine, plan, f"c{index}")
        ev = _with_cues(evidence(engine, plan, answer="999", event=f"e{index}"), slow)
        result = observe(engine, sid, version, ev, f"o{index}")
        if result["kind"] != "plan":
            break
        plan = result["teaching_plan"]
        version += 1
        paces.append(state(engine, sid)["pace"])
    assert paces == ["normal", "detailed"]
    assert state(engine, sid)["pace_estimator"]["belief"][2] > 0.5


def test_missing_or_malformed_cues_are_ignored(engine):  # noqa: F811
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    ev = _with_cues(evidence(engine, plan), "not a dict")
    result = observe(engine, sid, 1, ev, "o2")
    assert result["kind"] in {"plan", "session_end"}
    assert state(engine, sid)["pace"] == "normal"


def test_legacy_session_without_estimator_still_loads(engine):  # noqa: F811
    sid = create(engine)
    plan = observe(engine, sid)["teaching_plan"]
    commit(engine, plan)
    with engine.store.transaction() as db:
        stored = json.loads(
            db.execute("SELECT state_json FROM sessions WHERE session_id=?", (sid,)).fetchone()[0]
        )
        stored.pop("pace_estimator")
        db.execute(
            "UPDATE sessions SET state_json=? WHERE session_id=?", (json.dumps(stored), sid)
        )
    result = observe(engine, sid, 1, evidence(engine, plan, request="slower"), "o2")
    assert result["teaching_plan"]["teaching_action"]["pace"] == "detailed"
    assert state(engine, sid)["pace_estimator"]["pace"] == "detailed"
