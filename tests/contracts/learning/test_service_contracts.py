import copy

import pytest
from jsonschema import Draft202012Validator, ValidationError

from apps.learning_service.engine import HTTP_STATUS, LearningEngine, ServiceError, build_strategist
from apps.learning_service.schemas import SCHEMAS, validate
from packages.policies.network import StrategistActorCritic, save_checkpoint


def test_local_schemas_are_202012():
    for schema in SCHEMAS.values():
        Draft202012Validator.check_schema({"$defs": SCHEMAS, **schema})


@pytest.mark.parametrize(
    "name,body",
    [
        (
            "CreateSessionRequest",
            dict(
                schema_version="1.0",
                request_id="r",
                goal_concept_id="C01",
                pace_preference="normal",
            ),
        ),
        (
            "ObserveRequest",
            dict(
                schema_version="1.0",
                request_id="r",
                session_id="s",
                expected_state_version=0,
                evidence=None,
            ),
        ),
        ("CancelRequest", dict(schema_version="1.0", request_id="r", session_id="s", plan_id="p")),
        (
            "GradeRequest",
            dict(
                schema_version="1.0",
                request_id="r",
                problem_instance_id="p",
                normalized_answer="1/2",
            ),
        ),
    ],
)
def test_strict_local_requests(name, body):
    validate(name, body)
    for changed in [{**body, "extra": True}, {**body, "schema_version": "1.1"}]:
        with pytest.raises(ValidationError):
            validate(name, changed)
    for field in body:
        changed = copy.deepcopy(body)
        del changed[field]
        with pytest.raises(ValidationError):
            validate(name, changed)


@pytest.mark.parametrize(
    "version,code",
    [("2.0", "UNSUPPORTED_VERSION"), ("1.2", "INVALID_SCHEMA"), (None, "INVALID_SCHEMA")],
)
def test_version_errors(tmp_path, version, code):
    engine = LearningEngine({}, tmp_path / "db")
    with pytest.raises(ServiceError) as caught:
        engine.create_session(
            dict(
                schema_version=version,
                request_id="r",
                goal_concept_id="C01",
                pace_preference="normal",
            ),
            trace_id="t",
        )
    assert caught.value.code == code
    validate("Error", caught.value.as_dict("r", "t"))
    engine.close()


def test_error_http_map_and_rules_factory():
    assert HTTP_STATUS == dict(
        INVALID_SCHEMA=422,
        UNSUPPORTED_VERSION=400,
        UNAUTHORIZED=401,
        FORBIDDEN=403,
        STATE_CONFLICT=409,
        PLAN_EXPIRED=409,
        SESSION_ENDED=409,
        RATE_LIMITED=429,
        GENERATION_UNAVAILABLE=503,
        LEARNING_UNAVAILABLE=503,
        DEADLINE_EXCEEDED=504,
    )
    assert build_strategist({"strategist": "rules"}).policy_version == "rules_strategist_v1"


def test_checkpoint_factory_uses_greedy_masked_policy(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(
        StrategistActorCritic(seed=0),
        checkpoint,
        {
            "feature_spec_version": "features_v1",
            "action_spec_version": "actions_v1",
            "seed": 0,
            "init": "scratch",
        },
    )
    strategist = build_strategist({"strategist": {"checkpoint": checkpoint}})
    assert strategist.greedy is True
    mask = [False] * 72
    mask[4] = True
    assert strategist.act([0.0] * 140, mask, {}) == 4
    engine = LearningEngine({"strategist": {"checkpoint": checkpoint}}, tmp_path / "db")
    session = engine.create_session(
        dict(
            schema_version="1.0",
            request_id="create",
            goal_concept_id="C01",
            pace_preference="normal",
        ),
        trace_id="t",
    )
    plan = engine.observe_and_plan(
        dict(
            schema_version="1.0",
            request_id="observe",
            session_id=session["session_id"],
            expected_state_version=0,
            evidence=None,
        ),
        trace_id="t",
    )
    assert plan["teaching_plan"]["versions"]["strategist"] == "network_strategist_v1"
    engine.close()
