"""Deterministic response checks against an issued teaching plan.

``action_faithful`` combines the number-based heuristics below with the digit-independent
item-fidelity check in ``packages.response.item_fidelity`` (validator_v2): a reply that
drops or replaces the issued problem fails it, and the pipeline never repairs that failure.
"""

import re
from fractions import Fraction

from packages.response.item_fidelity import item_faithful
from packages.response.numbers import extract_numbers
from packages.response.shapes import is_valid

VALIDATOR_VERSION = "validator_v2"
HINT_FRAME_WORDS = 25
_STOPWORDS = {
    "about", "after", "again", "also", "because", "been", "being", "below",
    "between", "could", "does", "doing", "down", "each", "from", "have",
    "here", "into", "just", "more", "most", "only", "other", "over", "same",
    "some", "such", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "very", "what", "when",
    "where", "which", "while", "with", "would", "your",
}


def word_count(text):
    return len(text.split())


def content_words(text):
    return {
        word for word in re.findall(r"[a-z]+", text.lower())
        if len(word) >= 4 and word not in _STOPWORDS
    }


def _plan_texts(plan, include_solution=False):
    content = plan.get("content", {})
    values = [
        content.get("problem_stem"),
        *(content.get("approved_facts") or []),
        content.get("permitted_hint"),
        content.get("worked_example"),
        content.get("previous_step"),
    ]
    if include_solution:
        values.append(content.get("reveal_solution"))
    return [value for value in values if isinstance(value, str) and value]


def _number_values(text):
    return {token.value for token in extract_numbers(text)}


def _shared_words(text, source):
    source_text = " ".join(source) if not isinstance(source, str) else source
    return content_words(text) & content_words(source_text)


def _same_action(left, right):
    return isinstance(left, dict) and isinstance(right, dict) and all(
        left.get(key) == right.get(key)
        for key in ("move", "difficulty", "pace", "action_id")
    )


def _action_heuristic(plan, text, action):
    content = plan.get("content", {})
    move = action.get("move")
    stem = content.get("problem_stem")
    if move in {"PRACTICE", "CHECK_MASTERY", "CHALLENGE"} or (
        move == "PROBE" and stem is not None
    ):
        if not isinstance(stem, str) or not _number_values(stem) <= _number_values(text):
            return False
    if move == "PROBE" and stem is None and "?" not in text:
        return False
    if plan.get("constraints", {}).get("must_request_student_attempt"):
        if not re.search(
            r"\b(try|your turn|give it a go|what do you get|your answer|attempt)\b",
            text,
            re.IGNORECASE,
        ) and "?" not in text:
            return False
    if move == "HINT":
        hint = content.get("permitted_hint")
        if not isinstance(hint, str) or word_count(text) > word_count(hint) + HINT_FRAME_WORDS:
            return False
    elif move == "REFLECT":
        previous = content.get("previous_step")
        if not isinstance(previous, str):
            return False
        numbers = _number_values(previous)
        if numbers:
            if not numbers & _number_values(text):
                return False
        elif len(_shared_words(text, previous)) < 2:
            return False
    elif move == "WORKED_EXAMPLE":
        example = content.get("worked_example")
        if not isinstance(example, str) or not _number_values(example) <= _number_values(text):
            return False
    elif move == "EXPLAIN_CONCEPT":
        facts = content.get("approved_facts") or []
        if not facts:
            return False
        concept = content.get("concept_name")
        if not (
            isinstance(concept, str)
            and concept
            and concept.casefold() in text.casefold()
        ) and len(_shared_words(text, facts)) < 3:
            return False
    # Digit-independent: the reply must keep the issued item and not introduce another one.
    return item_faithful(plan, text, move)


def _schema_valid(plan, text, effective_move):
    return (
        is_valid(plan, "TeachingPlan")
        and is_valid(effective_move, "TeachingAction")
        and isinstance(text, str)
        and bool(text.strip())
    )


def _math_supported(plan, text):
    pool = _number_values(
        " ".join(_plan_texts(plan, plan.get("constraints", {}).get("reveal_allowed", False)))
    )
    return all(
        token.value in pool or (token.value.denominator == 1 and 0 <= token.value <= 10)
        for token in extract_numbers(text)
    )


def _fingerprint_number(fingerprint):
    tokens = extract_numbers(fingerprint)
    if len(tokens) != 1:
        return None
    token = tokens[0]
    residue = fingerprint[:token.start] + fingerprint[token.end:]
    residue = residue.replace(",", "").strip()
    if residue:
        return None
    return token


def _discloses_fingerprint(text_token, fingerprint_token):
    value = fingerprint_token.value
    if text_token.kind == "fraction":
        return text_token.value == value
    if text_token.kind == "decimal":
        return text_token.value == value or (
            2 <= text_token.decimals <= 4
            and abs(text_token.value - value)
            < Fraction(1, 2 * 10**text_token.decimals)
        )
    if text_token.kind == "percent":
        return text_token.value == value or (
            text_token.decimals <= 2
            and abs(text_token.value - value)
            < Fraction(1, 2 * 10 ** (2 + text_token.decimals))
        )
    return text_token.kind == "int" and text_token.value == value


def _disclosure_allowed(plan, text):
    constraints = plan.get("constraints", {})
    if constraints.get("reveal_allowed"):
        return True
    text_tokens = extract_numbers(text)
    normalized_text = " ".join(text.casefold().split())
    for fingerprint in constraints.get("forbidden_answer_fingerprints", []):
        numeric = _fingerprint_number(fingerprint)
        if numeric is not None:
            if any(_discloses_fingerprint(token, numeric) for token in text_tokens):
                return False
        else:
            phrase = " ".join(fingerprint.casefold().split())
            if phrase and phrase in normalized_text:
                return False
    return True


def run_checks(plan, text, effective_move):
    shape_ok = _schema_valid(plan, text, effective_move)
    action_faithful = False
    if shape_ok:
        action_faithful = _same_action(
            effective_move, plan.get("teaching_action", {})
        ) and _action_heuristic(plan, text, effective_move)
    checks = {
        "schema_valid": bool(shape_ok),
        "action_faithful": bool(action_faithful),
        "math_supported": bool(_math_supported(plan, text)) if isinstance(text, str) else False,
        "disclosure_allowed": (
            bool(_disclosure_allowed(plan, text)) if isinstance(text, str) else False
        ),
        "length_valid": bool(
            isinstance(text, str)
            and word_count(text) <= plan.get("constraints", {}).get("max_words", -1)
        ),
        "question_count_valid": bool(
            isinstance(text, str)
            and text.count("?") <= plan.get("constraints", {}).get("max_questions", -1)
        ),
    }
    return checks
