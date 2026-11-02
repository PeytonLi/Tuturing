import pytest

from apps.learning_service.engine import LearningEngine
from apps.learning_service.schemas import validate
from packages.learning.actions import strategist_mask
from packages.learning.scheduler import GoalStack
from tests.integration.learning.test_engine import create, observe, state

pytest.importorskip("torch")
network = pytest.importorskip("packages.policies.network")


def test_checkpoint_engine_plan_is_masked_and_versioned(tmp_path):
    checkpoint = tmp_path / "random.pt"
    network.save_checkpoint(
        network.StrategistActorCritic(seed=0),
        checkpoint,
        {
            "feature_spec_version": "features_v1",
            "action_spec_version": "actions_v1",
            "seed": 0,
            "init": "scratch",
        },
    )
    engine = LearningEngine(
        {"strategist": {"checkpoint": str(checkpoint)}}, tmp_path / "runtime" / "learning.sqlite3"
    )
    try:
        sid = create(engine)
        result = observe(engine, sid)
        plan = result["teaching_plan"]
        validate("TeachingPlan", plan)
        stored = state(engine, sid)
        _, context = engine._context(stored, engine._beliefs(stored), None, GoalStack())
        mask = strategist_mask(context)
        assert mask[plan["teaching_action"]["action_id"]]
        assert plan["versions"]["strategist"] == "network_strategist_v1"
        assert engine.strategist.greedy is True
    finally:
        engine.close()
