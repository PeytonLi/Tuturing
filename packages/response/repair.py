"""The single permitted repair for word-cap and question-count failures (PRD 14.1).

A draft that fails only ``length_valid`` and/or ``question_count_valid`` may be repaired
once while keeping the same item and action. Disclosure, math, action and schema failures
are never repaired; the pipeline uses a verified fallback instead. Every repaired text is
checked again before it can be delivered.
"""

from __future__ import annotations

import re

from packages.response.validate import content_words, word_count

REPAIR_VERSION = "repair_v1"
REPAIRABLE_CHECKS = frozenset({"length_valid", "question_count_valid"})
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def failed_checks(checks: dict) -> list[str]:
    return [name for name, passed in checks.items() if not passed]


def is_repairable(checks: dict, *, truncated: bool = False) -> bool:
    """True when the only problems are length or question count.

    A reply cut off at the output token cap counts as a length failure.
    """
    failed = set(failed_checks(checks))
    if truncated:
        failed.add("length_valid")
    return bool(failed) and failed <= REPAIRABLE_CHECKS


def split_sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_BREAK.split(" ".join(text.split())) if part]


def trim_to_limits(
    text: str, *, max_words: int, max_questions: int, anchors: tuple[str, ...] = ()
) -> str:
    """Deterministically shorten ``text`` to the word and question limits.

    Sentences that share most of their content words with ``anchors`` (the problem stem,
    hint, example or previous step) are kept in preference to filler and extra questions,
    and the final sentence, which usually asks the learner to act, is dropped last. The
    result can still fail the checks; the caller must validate it.
    """
    anchor_words = set()
    for anchor in anchors:
        if isinstance(anchor, str):
            anchor_words |= content_words(anchor)

    def anchored(sentence: str) -> bool:
        words = content_words(sentence)
        return bool(words) and 2 * len(words & anchor_words) >= len(words)

    sentences = split_sentences(text)
    if max_questions <= 0:
        sentences = [sentence.replace("?", ".") for sentence in sentences]
    else:
        while sum(sentence.count("?") for sentence in sentences) > max_questions:
            questions = [index for index, sentence in enumerate(sentences) if "?" in sentence]
            loose = [index for index in questions if not anchored(sentences[index])]
            del sentences[(loose or questions)[-1]]

    while sum(word_count(sentence) for sentence in sentences) > max_words and len(sentences) > 1:
        removable = [
            index for index, sentence in enumerate(sentences[:-1]) if not anchored(sentence)
        ]
        if not removable:
            break
        del sentences[max(removable, key=lambda index: (word_count(sentences[index]), -index))]

    result = " ".join(sentences)
    if word_count(result) > max_words:
        result = " ".join(result.split()[:max_words])
    return result


def repair_messages(
    messages: list[dict], draft: str, *, max_words: int, max_questions: int
) -> list[dict]:
    """Chat messages asking a model to rewrite its own draft within the limits."""
    if max_questions <= 0:
        question_rule = "Do not ask any question and use no question marks."
    elif max_questions == 1:
        question_rule = "Ask at most one question, using at most one question mark."
    else:
        question_rule = (
            f"Ask at most {max_questions} questions, using at most {max_questions} "
            "question marks."
        )
    instruction = (
        "Your previous reply broke the response limits. Rewrite it in at most "
        f"{max_words} words. {question_rule} Keep the same problem, the same numbers and "
        "the same teaching move, and keep any request for the learner to try. Do not add "
        "numbers, do not reveal an answer, and reply with the rewritten tutor message only."
    )
    return [
        *messages,
        {"role": "assistant", "content": draft},
        {"role": "user", "content": instruction},
    ]
