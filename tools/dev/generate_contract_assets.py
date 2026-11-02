from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ID = {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"^[\u0000-\u007F]+$"}
PROBABILITY = {"type": "number", "minimum": 0, "maximum": 1}
INT = {"type": "integer"}


def ref(name: str) -> dict:
    return {"$ref": f"#/$defs/{name}"}


def nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def array(item: dict, minimum: int | None = None, maximum: int | None = None) -> dict:
    result = {"type": "array", "items": item}
    if minimum is not None:
        result["minItems"] = minimum
    if maximum is not None:
        result["maxItems"] = maximum
    return result


def obj(properties: dict, required: list[str] | None = None, *, extras: bool = False) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": extras,
    }


def enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


def record_schema(props: dict, required: list[str] | None = None, *, extras: bool = False) -> dict:
    return obj(props, required, extras=extras)


def learning_defs() -> dict:
    evidence_span = obj(
        {"start": {"type": "integer", "minimum": 0}, "end": {"type": "integer", "minimum": 0}}
    )
    hypothesis = obj(
        {
            "misconception_id": ID,
            "probability": PROBABILITY,
            "evidence_spans": array(ref("EvidenceSpan")),
        }
    )
    concept_belief = obj(
        {
            "concept_id": ID,
            "mastery": PROBABILITY,
            "uncertainty": PROBABILITY,
            "independent_correct": {"type": "integer", "minimum": 0},
            "independent_incorrect": {"type": "integer", "minimum": 0},
            "assisted_attempts": {"type": "integer", "minimum": 0},
            "last_evidence_event_id": nullable(ID),
        }
    )
    navigation = obj(
        {
            "action_id": {"type": "integer", "minimum": 0, "maximum": 75},
            "operation": enum(
                "STAY",
                "ADVANCE_ONE",
                "SKIP_TO_TARGET",
                "REVIEW_PREREQUISITE",
                "RETURN_TO_GOAL",
                "END_SESSION",
            ),
            "target_concept_id": nullable(ID),
        }
    )
    teaching = obj(
        {
            "action_id": {"type": "integer", "minimum": 0, "maximum": 71},
            "move": enum(
                "PROBE",
                "HINT",
                "EXPLAIN_CONCEPT",
                "WORKED_EXAMPLE",
                "PRACTICE",
                "CHALLENGE",
                "CHECK_MASTERY",
                "REFLECT",
            ),
            "difficulty": enum("easy", "medium", "hard"),
            "pace": enum("concise", "normal", "detailed"),
        }
    )
    teaching_plan = obj(
        {
            "schema_version": {"const": "1.0"},
            "plan_id": ID,
            "session_id": ID,
            "trace_id": ID,
            "based_on_state_version": {"type": "integer", "minimum": 0},
            "option_id": ID,
            "navigation_action": ref("NavigationAction"),
            "teaching_action": ref("TeachingAction"),
            "actor_eligible": {"type": "boolean"},
            "concept_id": ID,
            "goal_concept_id": ID,
            "problem_instance_id": nullable(ID),
            "template_family_id": nullable(ID),
            "assessment_mode": enum("none", "practice", "independent"),
            "assistance": enum("none", "hint", "example", "answer_shown"),
            "content_bundle_id": ID,
            "content": obj(
                {
                    "concept_name": {"type": "string"},
                    "goal_name": {"type": "string"},
                    "problem_stem": nullable({"type": "string"}),
                    "approved_facts": array({"type": "string"}),
                    "permitted_hint": nullable({"type": "string"}),
                    "worked_example": nullable({"type": "string"}),
                    "reveal_solution": nullable({"type": "string"}),
                    "previous_step": nullable({"type": "string"}),
                }
            ),
            "constraints": obj(
                {
                    "reveal_allowed": {"type": "boolean"},
                    "max_words": {"type": "integer", "minimum": 1},
                    "max_questions": {"type": "integer", "minimum": 0},
                    "must_request_student_attempt": {"type": "boolean"},
                    "forbidden_answer_fingerprints": array(ID),
                }
            ),
            "response_context": obj(
                {
                    "last_student_message": {"type": "string"},
                    "recent_dialogue": array(
                        obj({"role": enum("student", "tutor"), "text": {"type": "string"}})
                    ),
                    "qualified_misconception_summary": nullable({"type": "string"}),
                }
            ),
            "versions": obj(
                {
                    "graph": ID,
                    "content": ID,
                    "state_estimator": ID,
                    "navigator": ID,
                    "strategist": ID,
                    "mask": ID,
                    "feature": ID,
                }
            ),
            "expires_at": {"type": "string", "format": "date-time"},
        }
    )
    public_state = obj(
        {
            "session_id": ID,
            "state_version": {"type": "integer", "minimum": 0},
            "current_concept_id": ID,
            "goal_concept_id": ID,
            "teaching_turns": {"type": "integer", "minimum": 0, "maximum": 16},
            "status": enum("active", "ended"),
            "end_reason": nullable(ref("EndReason")),
        }
    )
    reward = obj(
        {
            "terminal_gain": {"type": "number"},
            "teaching_cost": {"type": "number"},
            "disclosure_penalty": {"type": "number"},
            "repetition_penalty": {"type": "number"},
            "dropout_penalty": {"type": "number"},
            "invalid_action_penalty": {"type": "number"},
            "shaping": {"type": "number"},
        }
    )
    feature_array = array({"type": "number"}, 140, 140)
    defs = {
        "EvidenceSpan": evidence_span,
        "Evidence": obj(
            {
                "schema_version": {"const": "1.0"},
                "event_id": ID,
                "session_id": ID,
                "trace_id": ID,
                "source_response_id": ID,
                "problem_instance_id": nullable(ID),
                "observed_at": {"type": "string", "format": "date-time"},
                "normalized_answer": nullable({"type": "string"}),
                "correctness": enum("correct", "incorrect", "unknown", "not_applicable"),
                "grade_source": enum("deterministic", "extractor", "none"),
                "confidence": PROBABILITY,
                "assistance": enum("none", "hint", "example", "answer_shown", "unknown"),
                "explicit_request": enum(
                    "none", "skip", "faster", "slower", "hint", "answer", "stop", "change_goal"
                ),
                "requested_concept_id": nullable(ID),
                "misconception_hypotheses": array(ref("MisconceptionHypothesis")),
                "raw_message_ref": ID,
                "extractor_version": ID,
                "extensions": obj({}, [], extras=True),
            }
        ),
        "MisconceptionHypothesis": hypothesis,
        "ConceptBelief": concept_belief,
        "StudentBelief": obj(
            {
                "schema_version": {"const": "1.0"},
                "session_id": ID,
                "state_version": {"type": "integer", "minimum": 0},
                "learner_pseudonym": ID,
                "graph_version": ID,
                "feature_version": {"const": "features_v1"},
                "current_concept_id": ID,
                "goal_concept_id": ID,
                "concepts": array(ref("ConceptBelief"), 15, 15),
                "misconceptions": array(ref("BeliefProbability"), 25, 25),
                "pace_preference": enum("concise", "normal", "detailed"),
                "active_option": nullable(ref("NavigationAction")),
                "option_id": nullable(ID),
                "option_turns": {"type": "integer", "minimum": 0},
                "goal_stack": array(ID, 0, 4),
                "teaching_turns": {"type": "integer", "minimum": 0, "maximum": 16},
                "current_problem_instance_id": nullable(ID),
                "hint_level": {"type": "integer", "minimum": 0},
                "recent_evidence_event_ids": array(ID),
                "session_status": enum("active", "pending_plan", "ended"),
                "end_reason": nullable(ref("EndReason")),
                "updated_at": {"type": "string", "format": "date-time"},
            }
        ),
        "BeliefProbability": obj({"misconception_id": ID, "probability": PROBABILITY}),
        "NavigationAction": navigation,
        "TeachingAction": teaching,
        "TeachingPlan": teaching_plan,
        "EndReason": enum(
            "learner_stop",
            "demonstrated_completion",
            "policy_stop",
            "teaching_budget",
            "simulated_dropout",
            "configuration_error",
        ),
        "SessionEnd": obj(
            {
                "schema_version": {"const": "1.0"},
                "session_id": ID,
                "state_version": {"type": "integer", "minimum": 0},
                "reason": ref("EndReason"),
                "summary_content_id": ID,
                "public_state": ref("PublicState"),
            }
        ),
        "CommitPlanRequest": obj(
            {
                "schema_version": {"const": "1.0"},
                "session_id": ID,
                "plan_id": ID,
                "based_on_state_version": {"type": "integer", "minimum": 0},
                "response_id": ID,
                "content_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "effective_move": ref("TeachingAction"),
                "delivery_mode": enum("model", "repaired", "fallback"),
            }
        ),
        "CommitPlanResponse": obj(
            {
                "schema_version": {"const": "1.0"},
                "plan_id": ID,
                "status": {"const": "committed"},
                "state_version": {"type": "integer", "minimum": 0},
                "public_state": ref("PublicState"),
            }
        ),
        "PublicState": public_state,
        "Error": obj(
            {
                "schema_version": {"const": "1.0"},
                "request_id": ID,
                "trace_id": ID,
                "code": enum(
                    "INVALID_SCHEMA",
                    "UNSUPPORTED_VERSION",
                    "UNAUTHORIZED",
                    "FORBIDDEN",
                    "STATE_CONFLICT",
                    "PLAN_EXPIRED",
                    "SESSION_ENDED",
                    "RATE_LIMITED",
                    "GENERATION_UNAVAILABLE",
                    "LEARNING_UNAVAILABLE",
                    "DEADLINE_EXCEEDED",
                ),
                "message": {"type": "string"},
                "retryable": {"type": "boolean"},
                "current_state_version": nullable({"type": "integer", "minimum": 0}),
            }
        ),
        "Scenario": obj(
            {
                "scenario_id": ID,
                "scenario_family_id": ID,
                "split": enum("train", "validation", "test"),
                "goal_concept_id": ID,
                "start_concept_id": ID,
                "profile_id": ID,
                "simulator_config_id": ID,
                "private_initial_state_ref": ID,
                "public_history": array(ref("Evidence")),
                "template_family_allowlist": array(ID),
                "teaching_horizon": {"type": "integer", "minimum": 0},
                "provenance": obj({"author": ID, "reviewed": {"type": "boolean"}, "revision": ID}),
            }
        ),
        "RewardComponents": reward,
        "StrategistTransition": obj(
            {
                "run_id": ID,
                "episode_id": ID,
                "transition_id": ID,
                "turn_index": {"type": "integer", "minimum": 0},
                "option_id": ID,
                "observation": feature_array,
                "action_mask": array({"type": "boolean"}, 72, 72),
                "action_id": {"type": "integer", "minimum": 0, "maximum": 71},
                "old_log_prob": {"type": "number"},
                "old_value": {"type": "number"},
                "reward_components": ref("RewardComponents"),
                "reward": {"type": "number"},
                "next_observation": feature_array,
                "terminated": {"type": "boolean"},
                "truncated": {"type": "boolean"},
                "actor_eligible": {"type": "boolean"},
                "collector_cut": {"type": "boolean"},
                "policy_version": ID,
                "environment_bundle_id": ID,
            }
        ),
        "NavigatorTransition": obj(
            {
                "run_id": ID,
                "episode_id": ID,
                "transition_id": ID,
                "option_id": ID,
                "start_turn": {"type": "integer", "minimum": 0},
                "end_turn": {"type": "integer", "minimum": 0},
                "observation": feature_array,
                "action_mask": array({"type": "boolean"}, 76, 76),
                "action_id": {"type": "integer", "minimum": 0, "maximum": 75},
                "old_log_prob": {"type": "number"},
                "old_value": {"type": "number"},
                "duration": {"type": "integer", "minimum": 0},
                "discounted_option_reward": {"type": "number"},
                "next_observation": feature_array,
                "terminated": {"type": "boolean"},
                "truncated": {"type": "boolean"},
                "actor_eligible": {"type": "boolean"},
                "boundary_reasons": array({"type": "string"}),
                "policy_version": ID,
                "environment_bundle_id": ID,
            }
        ),
    }
    return defs


def response_defs() -> dict:
    learning = learning_defs()
    return {
        "TeachingAction": learning["TeachingAction"],
        "NavigationAction": learning["NavigationAction"],
        "TeachingPlan": learning["TeachingPlan"],
        "GenerateRequest": obj(
            {
                "schema_version": {"const": "1.0"},
                "request_id": ID,
                "trace_id": ID,
                "plan": ref("TeachingPlan"),
                "response_artifact_id": ID,
                "generation_seed": INT,
                "deadline_ms": {"type": "integer", "minimum": 0},
            }
        ),
        "GenerateResponse": obj(
            {
                "schema_version": {"const": "1.0"},
                "request_id": ID,
                "plan_id": ID,
                "response_id": ID,
                "text": {"type": "string"},
                "effective_move": ref("TeachingAction"),
                "delivery_mode": enum("model", "repaired", "fallback"),
                "content_hash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                "checks": obj(
                    {
                        key: {"type": "boolean"}
                        for key in (
                            "schema_valid",
                            "action_faithful",
                            "math_supported",
                            "disclosure_allowed",
                            "length_valid",
                            "question_count_valid",
                        )
                    }
                ),
                "usage": obj(
                    {
                        "input_tokens": {"type": "integer", "minimum": 0},
                        "output_tokens": {"type": "integer", "minimum": 0},
                        "elapsed_ms": {"type": "integer", "minimum": 0},
                    }
                ),
                "versions": obj({"response_artifact": ID, "prompt": ID, "validator": ID}),
            }
        ),
        "SFTExample": obj(
            {
                "example_id": ID,
                "dialogue_id": ID,
                "original_problem_id": ID,
                "template_family_id": ID,
                "split": enum("train", "validation", "test"),
                "source_id": ID,
                "source_revision": ID,
                "license_ref": ID,
                "synthetic": {"type": "boolean"},
                "reviewer_status": enum("approved", "rejected", "pending"),
                "action_label_source": enum("human", "mapped", "synthetic"),
                "plan": ref("TeachingPlan"),
                "prompt": array(
                    obj(
                        {"role": enum("system", "user", "assistant"), "content": {"type": "string"}}
                    )
                ),
                "completion": array(
                    obj({"role": {"const": "assistant"}, "content": {"type": "string"}})
                ),
                "checks": obj(
                    {
                        "math_correct": {"type": "boolean"},
                        "faithful": {"type": "boolean"},
                        "no_early_answer": {"type": "boolean"},
                    }
                ),
            }
        ),
    }


def root_schema(name: str, defs: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://example.invalid/contracts/{name}.schema.json",
        "title": name,
        "$defs": defs,
        "oneOf": [
            {"$ref": f"#/$defs/{key}"}
            for key in defs
            if key
            not in {"EvidenceSpan", "MisconceptionHypothesis", "BeliefProbability", "EndReason"}
        ],
    }


def dump(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def plan(move: str, index: int, *, independent: bool = False) -> dict:
    difficulty = "hard" if move == "CHALLENGE" else "medium"
    difficulty_index = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    action_id = index * 9 + difficulty_index * 3 + 1
    assessment = (
        "independent"
        if independent or move == "CHECK_MASTERY"
        else ("practice" if move in {"PRACTICE", "CHALLENGE", "PROBE"} else "none")
    )
    assistance = "hint" if move == "HINT" else "example" if move == "WORKED_EXAMPLE" else "none"
    return {
        "schema_version": "1.0",
        "plan_id": f"plan-{move.lower()}",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "based_on_state_version": 1,
        "option_id": "option-1",
        "navigation_action": {"action_id": 0, "operation": "STAY", "target_concept_id": "C01"},
        "teaching_action": {
            "action_id": action_id,
            "move": move,
            "difficulty": difficulty,
            "pace": "normal",
        },
        "actor_eligible": True,
        "concept_id": "C01",
        "goal_concept_id": "C01",
        "problem_instance_id": "item-1"
        if move in {"PROBE", "HINT", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"}
        else None,
        "template_family_id": "family-1"
        if move in {"PROBE", "HINT", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"}
        else None,
        "assessment_mode": assessment,
        "assistance": assistance,
        "content_bundle_id": "bundle-1",
        "content": {
            "concept_name": "Outcomes and sample spaces",
            "goal_name": "Outcomes and sample spaces",
            "problem_stem": "A fair coin is flipped once. What is the probability of heads?"
            if move in {"PROBE", "HINT", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"}
            else None,
            "approved_facts": ["A fair coin has two equally likely outcomes."],
            "permitted_hint": "List the two outcomes." if move == "HINT" else None,
            "worked_example": "A different coin has probability 1/2 of tails."
            if move == "WORKED_EXAMPLE"
            else None,
            "reveal_solution": "1/2" if move == "EXPLAIN_CONCEPT" else None,
            "previous_step": "You listed the sample space." if move == "REFLECT" else None,
        },
        "constraints": {
            "reveal_allowed": move == "EXPLAIN_CONCEPT" and not independent,
            "max_words": 150,
            "max_questions": 1,
            "must_request_student_attempt": move
            in {"PROBE", "HINT", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"},
            "forbidden_answer_fingerprints": [],
        },
        "response_context": {
            "last_student_message": "",
            "recent_dialogue": [],
            "qualified_misconception_summary": None,
        },
        "versions": {
            "graph": "graph_proto_v1",
            "content": "content_v1",
            "state_estimator": "bkt_v1",
            "navigator": "rules_v1",
            "strategist": "rules_v1",
            "mask": "mask_v1",
            "feature": "features_v1",
        },
        "expires_at": "2026-11-02T00:00:00Z",
    }


def response(move: str, index: int) -> dict:
    from packages.response.stub import generate

    return generate(
        {
            "schema_version": "1.0",
            "request_id": f"request-{move.lower()}",
            "trace_id": "trace-fixture",
            "plan": plan(move, index),
            "response_artifact_id": "stub_v1",
            "generation_seed": 0,
            "deadline_ms": 1000,
        }
    )


def generate_fixtures(learning: dict, response_schema: dict) -> None:
    moves = [
        "PROBE",
        "HINT",
        "EXPLAIN_CONCEPT",
        "WORKED_EXAMPLE",
        "PRACTICE",
        "CHALLENGE",
        "CHECK_MASTERY",
        "REFLECT",
    ]
    for index, move in enumerate(moves):
        dump(
            ROOT / "contracts/fixtures/learning/valid" / f"teaching_plan_{move.lower()}.json",
            plan(move, index),
        )
        dump(
            ROOT / "contracts/fixtures/response/valid" / f"generate_response_{move.lower()}.json",
            response(move, index),
        )
    dump(
        ROOT / "contracts/fixtures/response/valid/generate_response_hash_pin.json",
        response("PROBE", 0),
    )
    dump(
        ROOT / "contracts/fixtures/learning/valid/protected_check_mastery.json",
        plan("CHECK_MASTERY", 6, independent=True),
    )
    session_end = {
        "schema_version": "1.0",
        "session_id": "session-1",
        "state_version": 4,
        "reason": "learner_stop",
        "summary_content_id": "summary-1",
        "public_state": {
            "session_id": "session-1",
            "state_version": 4,
            "current_concept_id": "C01",
            "goal_concept_id": "C01",
            "teaching_turns": 1,
            "status": "ended",
            "end_reason": "learner_stop",
        },
    }
    dump(ROOT / "contracts/fixtures/learning/valid/session_end.json", session_end)
    error = {
        "schema_version": "1.0",
        "request_id": "request-1",
        "trace_id": "trace-1",
        "code": "STATE_CONFLICT",
        "message": "State version is stale.",
        "retryable": True,
        "current_state_version": 3,
    }
    dump(ROOT / "contracts/fixtures/learning/valid/state_conflict_error.json", error)
    stale_commit = {
        "schema_version": "1.0",
        "session_id": "session-1",
        "plan_id": "plan-1",
        "based_on_state_version": 1,
        "response_id": "response-1",
        "content_hash": "a" * 64,
        "effective_move": {
            "action_id": 3,
            "move": "PROBE",
            "difficulty": "medium",
            "pace": "normal",
        },
        "delivery_mode": "fallback",
    }
    dump(
        ROOT / "contracts/fixtures/learning/valid/stale_state_commit_plan_request.json",
        stale_commit,
    )
    invalid_plan = plan("CHECK_MASTERY", 6, independent=True)
    invalid_plan["content"]["reveal_solution"] = "1/2"
    dump(
        ROOT / "contracts/fixtures/learning/invalid/independent_reveal_solution.json", invalid_plan
    )
    invalid_move = plan("PROBE", 0)
    invalid_move["teaching_action"]["move"] = "UNKNOWN_MOVE"
    dump(ROOT / "contracts/fixtures/learning/invalid/unknown_move.json", invalid_move)
    nan_plan = plan("PROBE", 0)
    nan_plan["content"]["extensions"] = float("nan")
    path = ROOT / "contracts/fixtures/learning/invalid/nan_value.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nan_plan, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    extra = plan("PROBE", 0)
    extra["unexpected"] = "not allowed"
    dump(ROOT / "contracts/fixtures/learning/invalid/extra_field.json", extra)
    invalid_response = response("PROBE", 0)
    invalid_response["checks"]["schema_valid"] = False
    dump(ROOT / "contracts/fixtures/response/invalid/failed_check.json", invalid_response)
    invalid_extra = response("PROBE", 0)
    invalid_extra["unknown"] = True
    dump(ROOT / "contracts/fixtures/response/invalid/extra_field.json", invalid_extra)
    invalid_request_plan = plan("CHECK_MASTERY", 6, independent=True)
    invalid_request_plan["content"]["reveal_solution"] = "1/2"
    invalid_request = {
        "schema_version": "1.0",
        "request_id": "request-1",
        "trace_id": "trace-1",
        "plan": invalid_request_plan,
        "response_artifact_id": "stub_v1",
        "generation_seed": 1,
        "deadline_ms": 1000,
    }
    dump(
        ROOT / "contracts/fixtures/response/invalid/independent_reveal_request.json",
        invalid_request,
    )


def main() -> None:
    learning = root_schema("learning_v1", learning_defs())
    response = root_schema("response_v1", response_defs())
    dump(ROOT / "contracts/learning_v1.schema.json", learning)
    dump(ROOT / "contracts/response_v1.schema.json", response)
    feature_names = (
        [f"mastery_{i:02d}" for i in range(15)]
        + [f"uncertainty_{i:02d}" for i in range(15)]
        + [f"evidence_count_{i:02d}" for i in range(15)]
        + [f"misconception_probability_M{i:02d}" for i in range(1, 26)]
        + [f"current_concept_C{i:02d}" for i in range(1, 16)]
        + [f"goal_concept_C{i:02d}" for i in range(1, 16)]
        + [f"active_target_C{i:02d}" for i in range(1, 16)]
        + [
            f"active_option_{name}"
            for name in (
                "STAY",
                "ADVANCE_ONE",
                "SKIP_TO_TARGET",
                "REVIEW_PREREQUISITE",
                "RETURN_TO_GOAL",
                "END_SESSION",
            )
        ]
        + [
            f"previous_move_{name}"
            for name in (
                "PROBE",
                "HINT",
                "EXPLAIN_CONCEPT",
                "WORKED_EXAMPLE",
                "PRACTICE",
                "CHALLENGE",
                "CHECK_MASTERY",
                "REFLECT",
                "NONE",
            )
        ]
        + [
            "turn_over_16",
            "turns_in_option_over_4",
            "hint_count_over_4",
            "independent_success_streak_over_3",
            "remaining_budget_over_16",
            "explicit_skip",
            "explicit_faster",
            "explicit_slower",
            "response_failure",
            "goal_stack_depth_over_4",
        ]
    )
    dump(
        ROOT / "contracts/feature_spec_v1.json",
        {
            "schema_version": "features_v1",
            "features": [
                {"index": index, "name": name} for index, name in enumerate(feature_names)
            ],
        },
    )
    dump(
        ROOT / "contracts/action_spec_v1.json",
        {
            "schema_version": "actions_v1",
            "strategist": {
                "moves": [
                    "PROBE",
                    "HINT",
                    "EXPLAIN_CONCEPT",
                    "WORKED_EXAMPLE",
                    "PRACTICE",
                    "CHALLENGE",
                    "CHECK_MASTERY",
                    "REFLECT",
                ],
                "difficulties": ["easy", "medium", "hard"],
                "paces": ["concise", "normal", "detailed"],
                "encoding": "move_index*9 + difficulty_index*3 + pace_index",
                "action_count": 72,
            },
            "navigator": {
                "operations": [
                    "STAY",
                    "ADVANCE_ONE",
                    "SKIP_TO_TARGET",
                    "REVIEW_PREREQUISITE",
                    "RETURN_TO_GOAL",
                ],
                "concept_indices": list(range(15)),
                "encoding": "operation_index*15 + concept_index",
                "end_session_action_id": 75,
                "action_count": 76,
            },
        },
    )
    generate_fixtures(learning, response)


if __name__ == "__main__":
    main()
