"""Rule-based extraction of observable student evidence."""

import hashlib
import re
from datetime import UTC, datetime

from packages.response.numbers import extract_numbers

EXTRACTOR_VERSION = "extractor_rules_v1"

_NON_ANSWER = re.compile(
    r"(?i)^\s*(idk|i don'?t know|dunno|no idea|not sure)\b"
)
_SELF_REPORT = re.compile(r"(?i)\bi (already )?know (this|it|that)\b")
_INJECTION = re.compile(
    r"(?i)\b(ignore|disregard|forget)\b.{0,40}\b(rules|instructions|previous|system)\b"
    r"|\bmark me (as )?correct\b|\byou are now\b"
)
_HEDGED = re.compile(r"(?i)\?|\bmaybe\b|\bis it\b|\bi think\b")
_REQUESTS = (
    ("stop", re.compile(r"\b(stop|quit|i'?m done|end (the )?session|exit)\b", re.I)),
    ("skip", re.compile(r"\bskip\b|\bmove on\b|\bnext (topic|concept|one)\b|\bskip to\b", re.I)),
    (
        "change_goal",
        re.compile(
            r"\b(change|switch)( the)? (goal|topic)\b|\bdifferent (goal|topic)\b"
            r"|\blearn about something else\b",
            re.I,
        ),
    ),
    (
        "answer",
        re.compile(
            r"\b(just )?(tell|give|show) me the (answer|solution)\b"
            r"|\bwhat'?s the answer\b|\bjust tell me\b|\breveal\b",
            re.I,
        ),
    ),
    ("hint", re.compile(r"\bhint\b|\bclue\b|\bi'?m stuck\b|\bhelp me\b", re.I)),
    (
        "faster",
        re.compile(
            r"\bfaster\b|\bspeed (it )?up\b|\bquicker\b|\bhurry\b|\btoo slow\b",
            re.I,
        ),
    ),
    ("slower", re.compile(r"\bslower\b|\bslow down\b|\btoo fast\b", re.I)),
)
_CONCEPT_KEYWORDS = (
    ("bayes", "C10"),
    ("total probability", "C09"),
    ("conditional", "C06"),
    ("independen", "C07"),
    ("complement", "C03"),
    ("binomial", "C14"),
    ("bernoulli", "C13"),
    ("expected value", "C15"),
    ("addition rule", "C05"),
    ("multiplication rule", "C08"),
    ("probability tree", "C08"),
    ("tree", "C08"),
    ("union", "C04"),
    ("intersection", "C04"),
    ("permutation", "C11"),
    ("combination", "C12"),
    ("choose", "C12"),
    ("equally likely", "C02"),
    ("sample space", "C01"),
)


def _normalized_number(token):
    raw = token.raw.strip()
    cleaned = re.sub(r"\s+", "", raw).replace(",", "")
    if token.kind == "fraction":
        numerator, denominator = cleaned.split("/", 1)
        if numerator.startswith("."):
            numerator = "0" + numerator
        elif numerator.startswith("-."):
            numerator = "-0" + numerator[1:]
        return f"{numerator}/{denominator}"
    if token.kind == "percent":
        number = re.sub(r"(?:%|percent)$", "", cleaned, flags=re.IGNORECASE)
        if number.startswith("."):
            number = "0" + number
        elif number.startswith("-."):
            number = "-0" + number[1:]
        return number + "%"
    if cleaned.startswith("."):
        return "0" + cleaned
    if cleaned.startswith("-."):
        return "-0" + cleaned[1:]
    return cleaned


def _numeric_fingerprint(value):
    if not isinstance(value, str):
        return None
    tokens = extract_numbers(value)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if (value[:token.start] + value[token.end:]).replace(",", "").strip():
        return None
    return token.value


def _concept_request(message, explicit_request):
    if explicit_request not in {"skip", "change_goal"}:
        return None
    lowered = message.casefold()
    for keyword, concept_id in _CONCEPT_KEYWORDS:
        if keyword in lowered:
            return concept_id
    return None


def _grade(normalized_answer, number_tokens, plan, self_report, injection):
    if normalized_answer is None:
        return "not_applicable", "none"
    constraints = plan.get("constraints", {})
    fingerprints = constraints.get("forbidden_answer_fingerprints") or []
    if (
        len(number_tokens) == 1
        and fingerprints
        and plan.get("problem_instance_id") is not None
        and not self_report
        and not injection
    ):
        answer_value = number_tokens[-1].value
        fingerprint_values = {
            value for value in (_numeric_fingerprint(item) for item in fingerprints)
            if value is not None
        }
        return (
            ("correct" if answer_value in fingerprint_values else "incorrect"),
            "extractor",
        )
    return "unknown", "extractor"


def extract(
    message,
    last_plan,
    last_response_id,
    *,
    event_id=None,
    session_id=None,
    trace_id=None,
    observed_at=None,
    raw_message_ref=None,
):
    plan = last_plan or {}
    session_id = session_id or plan.get("session_id") or "unknown_session"
    trace_id = trace_id or plan.get("trace_id") or "unknown_trace"
    observed_at = observed_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_message_ref = raw_message_ref or (
        "msg_" + hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
    )
    event_id = event_id or (
        "ev_" + hashlib.sha256(
            f"{session_id}|{last_response_id}|{message}".encode()
        ).hexdigest()[:16]
    )

    non_answer = not message.strip() or bool(_NON_ANSWER.search(message))
    number_tokens = [] if non_answer else extract_numbers(message)
    normalized_answer = (
        _normalized_number(number_tokens[-1]) if number_tokens else None
    )
    confidence = 0.0 if not number_tokens else (0.9 if len(number_tokens) == 1 else 0.5)
    explicit_request = next(
        (name for name, pattern in _REQUESTS if pattern.search(message)), "none"
    )
    requested_concept = _concept_request(message, explicit_request)
    self_report = bool(_SELF_REPORT.search(message))
    injection = bool(_INJECTION.search(message))
    correctness, grade_source = _grade(
        normalized_answer, number_tokens, plan, self_report, injection
    )
    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "source_response_id": last_response_id,
        "problem_instance_id": plan.get("problem_instance_id"),
        "observed_at": observed_at,
        "normalized_answer": normalized_answer,
        "correctness": correctness,
        "grade_source": grade_source,
        "confidence": confidence,
        "assistance": plan.get("assistance", "unknown"),
        "explicit_request": explicit_request,
        "requested_concept_id": requested_concept,
        "misconception_hypotheses": [],
        "raw_message_ref": raw_message_ref,
        "extractor_version": EXTRACTOR_VERSION,
        "extensions": {
            "evidence_extractor": {
                "non_answer": non_answer,
                "self_report": self_report,
                "injection_suspected": injection,
                "numbers_found": len(number_tokens),
                "hedged": bool(_HEDGED.search(message)),
            }
        },
    }
