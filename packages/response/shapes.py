"""Required-key shape checks for response-lane records."""


class ShapeError(ValueError):
    """Raised when a record omits one or more required fields."""


REQUIRED_KEYS = {
    "TeachingPlan": (
        "schema_version", "plan_id", "session_id", "trace_id",
        "based_on_state_version", "option_id", "navigation_action",
        "teaching_action", "actor_eligible", "concept_id", "goal_concept_id",
        "problem_instance_id", "template_family_id", "assessment_mode",
        "assistance", "content_bundle_id", "content", "constraints",
        "response_context", "versions", "expires_at",
    ),
    "GenerateRequest": (
        "schema_version", "request_id", "trace_id", "plan",
        "response_artifact_id", "generation_seed", "deadline_ms",
    ),
    "GenerateResponse": (
        "schema_version", "request_id", "plan_id", "response_id", "text",
        "effective_move", "delivery_mode", "content_hash", "checks", "usage",
        "versions",
    ),
    "SFTExample": (
        "example_id", "dialogue_id", "original_problem_id",
        "template_family_id", "split", "source_id", "source_revision",
        "license_ref", "synthetic", "reviewer_status", "action_label_source",
        "plan", "prompt", "completion", "checks",
    ),
    "TeachingAction": ("action_id", "move", "difficulty", "pace"),
    "NavigationAction": ("action_id", "operation", "target_concept_id"),
    "Evidence": (
        "schema_version", "event_id", "session_id", "trace_id",
        "source_response_id", "problem_instance_id", "observed_at",
        "normalized_answer", "correctness", "grade_source", "confidence",
        "assistance", "explicit_request", "requested_concept_id",
        "misconception_hypotheses", "raw_message_ref", "extractor_version",
        "extensions",
    ),
}

NESTED_KEYS = {
    "TeachingPlan": {
        "content": (
            "concept_name", "goal_name", "problem_stem", "approved_facts",
            "permitted_hint", "worked_example", "reveal_solution",
            "previous_step",
        ),
        "constraints": (
            "reveal_allowed", "max_words", "max_questions",
            "must_request_student_attempt", "forbidden_answer_fingerprints",
        ),
        "response_context": (
            "last_student_message", "recent_dialogue",
            "qualified_misconception_summary",
        ),
        "versions": (
            "graph", "content", "state_estimator", "navigator", "strategist",
            "mask", "feature",
        ),
        "teaching_action": "TeachingAction",
        "navigation_action": "NavigationAction",
    },
    "GenerateResponse": {
        "checks": (
            "schema_valid", "action_faithful", "math_supported",
            "disclosure_allowed", "length_valid", "question_count_valid",
        ),
        "usage": ("input_tokens", "output_tokens", "elapsed_ms"),
        "versions": ("response_artifact", "prompt", "validator"),
        "effective_move": "TeachingAction",
    },
    "GenerateRequest": {"plan": "TeachingPlan"},
    "SFTExample": {
        "plan": "TeachingPlan",
        "checks": ("math_correct", "faithful", "no_early_answer"),
    },
}

TEACHING_ENUMS = {
    "move": {
        "PROBE", "HINT", "EXPLAIN_CONCEPT", "WORKED_EXAMPLE", "PRACTICE",
        "CHALLENGE", "CHECK_MASTERY", "REFLECT",
    },
    "difficulty": {"easy", "medium", "hard"},
    "pace": {"concise", "normal", "detailed"},
}


def missing_keys(record, kind):
    """Return sorted dotted paths for required fields absent from a plain dict."""
    if kind not in REQUIRED_KEYS:
        raise ValueError(f"unknown record kind: {kind}")
    missing = []

    def visit_fields(value, fields, prefix):
        if not isinstance(value, dict):
            missing.append(prefix.rstrip("."))
            return
        missing.extend(
            f"{prefix}{key}" for key in fields if key not in value
        )

    def visit(value, record_kind, prefix=""):
        if not isinstance(value, dict):
            if prefix:
                missing.append(prefix.rstrip("."))
            else:
                missing.extend(REQUIRED_KEYS[record_kind])
            return
        nested = NESTED_KEYS.get(record_kind, {})
        for key in REQUIRED_KEYS[record_kind]:
            path = f"{prefix}{key}"
            if key not in value:
                missing.append(path)
                continue
            child = nested.get(key)
            if isinstance(child, str):
                visit(value[key], child, path + ".")
            elif isinstance(child, tuple):
                visit_fields(value[key], child, path + ".")

    visit(record, kind)
    return sorted(missing)


def validate_shape(record, kind):
    missing = missing_keys(record, kind)
    if missing:
        raise ShapeError(f"{kind} missing required keys: {', '.join(missing)}")
    if kind == "TeachingAction":
        _validate_teaching_action(record)
    elif kind == "TeachingPlan":
        _validate_teaching_action(record["teaching_action"])
    elif kind in {"GenerateRequest", "SFTExample"}:
        _validate_teaching_action(record["plan"]["teaching_action"])
    elif kind == "GenerateResponse":
        _validate_teaching_action(record["effective_move"])
    return record


def _validate_teaching_action(action):
    for field, allowed in TEACHING_ENUMS.items():
        if action.get(field) not in allowed:
            raise ShapeError(
                f"TeachingAction.{field} must be one of {', '.join(sorted(allowed))}"
            )


def is_valid(record, kind):
    try:
        validate_shape(record, kind)
    except (ShapeError, TypeError):
        return False
    return True
