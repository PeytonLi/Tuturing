"""Shared text helpers for the rule-based evidence extractor.

Learner messages are untrusted and messy (PRD 23.2). These helpers tokenize a
message into words with offsets and recognize common typos of a small keyword
vocabulary without a dictionary: adjacent transpositions ("hnit"), dropped
vowels ("skp"), a dropped letter in a long word ("answr", "condtional") and a
doubled letter ("thinkk"). Real words that sit one edit from a keyword are
listed in NOT_TYPOS so they are never rewritten.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9']+")
_VOWELS = frozenset("aeiouy")

# Real words within one edit of a keyword; never treated as a typo of it.
NOT_TYPOS = frozenset({
    "lower", "aster", "later", "slow", "fast", "step", "steps", "stem", "last", "past",
    "ship", "slip", "skin", "skim", "spin", "mint", "hunt", "hind", "lint", "tint", "hilt",
    "shop", "stoop", "stomp", "stop", "hint", "skip", "next", "nest", "text", "help", "held",
    "heap", "halt", "hell", "answer", "faster", "slower", "quit", "quite", "quiet", "exit",
    "edit", "quilt", "hits", "hit", "hat", "hot", "sip", "ski", "kip", "tip", "top", "clue",
    "glue", "blue", "clues", "pass", "mass", "bass", "pays", "past", "paste", "sever", "seven",
    "fever", "lever", "never", "ever", "event", "events", "evens", "given", "seventh",
})


def words(text):
    """Yield (word, start, end) for each alphanumeric run in text."""
    for match in WORD_RE.finditer(text):
        yield match.group(0), match.start(), match.end()


def _is_transposition(word, keyword):
    if len(word) != len(keyword):
        return False
    diffs = [index for index, (a, b) in enumerate(zip(word, keyword)) if a != b]
    return (
        len(diffs) == 2
        and diffs[1] == diffs[0] + 1
        and word[diffs[0]] == keyword[diffs[1]]
        and word[diffs[1]] == keyword[diffs[0]]
    )


def _is_single_deletion(short, long):
    if len(short) != len(long) - 1:
        return False
    for index in range(len(long)):
        if long[:index] + long[index + 1:] == short:
            return True
    return False


def _is_single_substitution(word, keyword):
    if len(word) != len(keyword):
        return False
    return sum(1 for a, b in zip(word, keyword) if a != b) == 1


def fuzzy_equal(word, keyword):
    """True when word is keyword or a common typo of it (see module docstring)."""
    word = word.lower().replace("'", "")
    keyword = keyword.lower()
    if word == keyword:
        return True
    if word in NOT_TYPOS or len(word) < 3:
        return False
    if len(word) >= 4 and _is_transposition(word, keyword):
        return True
    if _is_single_deletion(word, keyword):
        if not _VOWELS & set(word):
            return True
        return len(keyword) >= 6
    if len(keyword) >= 7 and _is_single_substitution(word, keyword):
        return True
    if len(word) == len(keyword) + 1:
        # A doubled letter ("thinkk", "idkk") or one stray letter in a long word.
        if any(word[:index] + word[index + 1:] == keyword and (
            index > 0 and word[index] == word[index - 1]
        ) for index in range(len(word))):
            return True
        return len(keyword) >= 7 and _is_single_deletion(keyword, word)
    return False


def canonicalize(text, vocabulary):
    """Lowercase text with typos of vocabulary words rewritten to the word itself.

    Offsets are not preserved; use the result only for pattern matching, never
    for spans into the original message.
    """
    pieces = []
    cursor = 0
    for word, start, end in words(text):
        pieces.append(text[cursor:start].lower())
        lowered = word.lower()
        replacement = lowered
        if lowered.replace("'", "") not in vocabulary:
            for keyword in vocabulary:
                if fuzzy_equal(lowered, keyword):
                    replacement = keyword
                    break
        elif "'" in lowered and lowered.replace("'", "") in vocabulary:
            replacement = lowered.replace("'", "")
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:].lower())
    return "".join(pieces)


def preceded_by(text, position, pattern, window=40):
    """True when the text just before position matches pattern at its end."""
    start = max(0, position - window)
    return re.search(pattern + r"\s*$", text[start:position], re.I) is not None


def followed_by(text, position, pattern, window=40):
    """True when the text right after position starts with pattern."""
    return re.match(r"\s*" + pattern, text[position:position + window], re.I) is not None
