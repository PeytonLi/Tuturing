"""Rule-based extraction of observable student evidence (PRD 6.3, 9.2, 23.2).

extract() proposes an Evidence record from one learner message and the last
committed TeachingPlan. The extractor never grades: correctness is "unknown"
when the message carries an answer and "not_applicable" otherwise, with
grade_source "none"; B's deterministic grader verifies normalized_answer.
Self-reports ("I know this") and instructions inside the message ("mark me
correct") are untrusted text and only set flags under
extensions.evidence_extractor. confidence is the extractor's confidence that
normalized_answer is the answer the learner committed to (0.9 one clear value,
0.85 a final value after a conclusion marker, 0.6 the last of several numbers,
0.0 no answer); B treats grades below 0.80 as unknown for mastery updates.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from packages.evidence_extractor import answers, misconceptions, requests

EXTRACTOR_VERSION = "extractor_rules_v2"

_NON_ANSWER = re.compile(
    r"(?i)\b(?:idk+|i dunno|dunno|no idea|no clue|not sure|unsure|i (?:don'?t|dont|do not) "
    r"(?:know|knwo|get it|understand)|beats me|i'?m lost|im lost|lost|huh|confused|"
    r"i'?m stuck|im stuck|stuck|no idea where to start)\b|^\W*(?:\?+|\.{2,}|…)\W*$"
)
_SELF_REPORT = re.compile(
    r"(?i)\bi (?:already |totally |definitely |think i )?(?:know|knew|understand|get|got) "
    r"(?:this|it|that|these|those|how|all of this|the idea)\b|\b(?:this|that) is (?:so |"
    r"pretty |too |really )?easy\b|\bgot it\b|\bmakes sense\b|\bi understand\b|\bi see\b|"
    r"\bwe did this in\b|\bi'?ve done\b|\bi'?m (?:pretty |really )?good at\b|\btrust me\b|"
    r"\bi know how to\b|\bi already know\b|\bi learned this\b|^\W*easy\b|\bi knew that\b"
)
_INJECTION = re.compile(
    r"(?i)\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}\b(?:rules?|instructions?|"
    r"previous|system|plan|prompt|guidelines?)\b|\bmark (?:me|my answer|it|this) (?:as )?"
    r"(?:correct|right)\b|\bcount (?:this|it|me) as (?:right|correct)\b|\byou are now\b|"
    r"\bsystem:|\badmin override\b|\bas your (?:developer|admin|creator|operator)\b|"
    r"\bnew instructions?\b|\bpretend\b|\bact as\b|\bprint your (?:system )?prompt\b|"
    r"</?(?:tutor|student|system|assistant|user)>|\b(?:correctness|mastery|grade_source|"
    r"reveal_solution|state_version)\b[\"']?\s*[=:]|\bset mastery\b|\bfull marks\b|"
    r"\bupdate mastery\b|"
    r"\bgrade (?:it|me|this) (?:as )?correct\b|\btutor note:|\bteacher mode\b|\bjailbreak\b|"
    r"\bend the check\b|\byou must say\b"
)
_HEDGED = re.compile(
    r"(?i)\?|\bmaybe\b|\bis it\b|\bi think\b|\bi guess\b|\bprobably\b|\bpossibly\b|"
    r"\bnot sure\b|\bperhaps\b|\bcould it be\b|\bwould it be\b|\baround\b|\babout\b|\bidk\b|"
    r"\bi wanna say\b|\bmy guess\b|\bright\?"
)
# Help actions and sources: outside assistance regardless of who supplied the number.
_ASSISTANCE_ACTION = re.compile(
    r"(?i)\b(?:googled|google|looked (?:it|this|that) up|look (?:it|this) up|searched|"
    r"search(?:ed)? (?:it|online|the web)|wikipedia|chatgpt|chat gpt|\bgpt\b|gemini|copilot|"
    r"siri|alexa|calculator|wolfram|answer (?:key|sheet)|solutions? manual|back of the book|"
    r"copied|cheated|online|the internet|group chat|photomath|symbolab|stack ?exchange|"
    r"reddit|(?:helped|told|showed|gave) me|got help|with help)\b"
)
_ASSISTANCE_NEGATION = re.compile(
    r"(?i)\b(?:didn'?t|didnt|did not|don'?t|dont|do not|no|not|without|never|haven'?t|"
    r"havent|wasn'?t|wont|won'?t|isn'?t|nobody|no one)\b(?:\s+\w+){0,4}\s*$"
)


def _assistance(plan, message, selected_token):
    """Plan assistance, or "unknown" when the learner reports outside help.

    Outside help means a help action or source ("googled", "the answer key says"),
    or a number attributed to another person that the learner adopts as their
    answer. Mentioning someone else's different answer is not assistance.
    """
    plan_value = plan.get("assistance", "unknown")
    reported = selected_token is not None and "attributed" in selected_token.flags
    for match in _ASSISTANCE_ACTION.finditer(message):
        before = message[max(0, match.start() - 40):match.start()]
        if _ASSISTANCE_NEGATION.search(before):
            continue
        reported = True
        break
    value = "unknown" if reported and plan_value == "none" else plan_value
    return value, reported


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

    normalized_answer, confidence, answer_details = answers.select_answer(message, plan)
    explicit_request, requested_concept, request_details = requests.detect(message)
    hypotheses, misconception_rules = misconceptions.propose(
        message, plan, answer_details["all_tokens"], normalized_answer
    )
    assistance, assistance_reported = _assistance(
        plan, message, answer_details["selected_token"]
    )
    self_report = bool(_SELF_REPORT.search(message))
    injection = bool(_INJECTION.search(message))
    non_answer = normalized_answer is None and (
        not message.strip() or bool(_NON_ANSWER.search(message))
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
        "correctness": "unknown" if normalized_answer is not None else "not_applicable",
        "grade_source": "none",
        "confidence": confidence,
        "assistance": assistance,
        "explicit_request": explicit_request,
        "requested_concept_id": requested_concept,
        "misconception_hypotheses": hypotheses,
        "raw_message_ref": raw_message_ref,
        "extractor_version": EXTRACTOR_VERSION,
        "extensions": {
            "evidence_extractor": {
                "non_answer": non_answer,
                "self_report": self_report,
                "injection_suspected": injection,
                "numbers_found": answer_details["numbers_found"],
                "hedged": bool(_HEDGED.search(message)),
                "assistance_reported": assistance_reported,
                "plan_assistance": plan.get("assistance"),
                "answer_candidates": answer_details["answer_candidates"],
                "answer_selection": answer_details["answer_selection"],
                "request_candidates": request_details["request_candidates"],
                "concept_mentions": request_details["concept_mentions"],
                "misconception_rules": misconception_rules,
            }
        },
    }
