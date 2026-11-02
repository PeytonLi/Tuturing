"""Digit-independent item fidelity: did the reply keep the issued problem?

The number-based action heuristics in ``validate.py`` cannot see an item substitution when
the stem has no digits: in benchmark_v1, "A fair coin is flipped once" replaced by a
six-sided die passed every check. For moves that present or refer to an item this module adds
two signals computed on normalized content words (lowercase, punctuation stripped, stopwords
dropped, light suffix stemming so flipped, flips and flip agree):

1. **Presence**: the share of the stem's content words that the reply reuses. Moves that
   present the item (PRACTICE, CHALLENGE, CHECK_MASTERY) must reuse at least
   ``PRESENT_MIN_COVERAGE``; moves that refer to it (HINT, PROBE with an item) at least
   ``REFER_MIN_COVERAGE``, or ``REFER_MIN_SHARED`` words, or one of the stem's numbers,
   or (HINT only) at least ``HINT_MIN_COVERAGE`` of the permitted hint's words, since the
   approved hint is the plan's own realization of the item.
2. **No competing scenario**: the reply must not introduce a scenario family (a die, cards,
   a spinner, a bag of marbles, a colour, see ``SCENARIO_FAMILIES``) or a quantity phrase
   such as ``3 red marbles`` or ``six-sided die`` that the stem, approved content, worked
   example, concept names and recent dialogue do not already contain. A family the plan
   mentions is allowed as a whole, so "tails" is fine in a reply about a coin that only
   named heads.

Both are heuristics over the plan's own words; they never prove the mathematics correct.
The thresholds were calibrated on experiments/response/benchmark_v1 (see
docs/runbooks/response_service.md): every fallback rendering and every fixture reply that
kept its item passes, and the three Qwen3.8-27B substitutions fail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ITEM_FIDELITY_VERSION = "item_fidelity_v1"
PRESENTING_MOVES = frozenset({"PRACTICE", "CHALLENGE", "CHECK_MASTERY"})
REFERRING_MOVES = frozenset({"HINT", "PROBE"})
PRESENT_MIN_COVERAGE = 0.5
REFER_MIN_COVERAGE = 0.25
REFER_MIN_SHARED = 3
HINT_MIN_COVERAGE = 0.5

COLOURS = frozenset({"red", "blue", "green", "yellow", "white", "black", "purple", "pink"})
# Each family is a random device or drawn object and the words that come with it. A reply may
# use a family only when the plan already uses one of its members.
SCENARIO_FAMILIES = {
    "coin": frozenset({"coin", "head", "tail", "flip", "toss"}),
    "die": frozenset({"die", "dice", "roll"}),
    "card": frozenset({
        "card", "deck", "suit", "heart", "spade", "club", "diamond", "ace", "king", "queen",
        "jack", "joker",
    }),
    "spinner": frozenset({"spinner", "spin"}),
    "raffle": frozenset({"raffle", "lottery", "ticket"}),
    **{name: frozenset({name}) for name in (
        "marble", "ball", "bead", "counter", "tile", "token", "chip", "cube", "sock",
        "button", "candy", "sticker", "dart", "wheel", "bag", "jar", "box", "urn",
    )},
    **{colour: frozenset({colour}) for colour in sorted(COLOURS)},
}
_VERBS = frozenset({"flip", "toss"})  # allowed with their family, never evidence alone
FLAGGABLE = frozenset().union(*SCENARIO_FAMILIES.values()) - _VERBS
NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "done", "down",
    "during", "each", "either", "else", "ever", "every", "few", "for", "from", "further",
    "get", "got", "had", "has", "have", "having", "he", "her", "here", "hers", "him",
    "himself", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "let", "lets", "many", "may", "me", "might", "more", "most", "much",
    "must", "my", "neither", "never", "no", "nor", "not", "now", "of", "off", "on", "one", "only",
    "or", "other", "our", "out", "over", "own", "please", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "us", "very",
    "was", "we", "were", "what", "when", "where", "whether", "which", "while", "who",
    "whom", "why", "will", "with", "would", "yes", "yet", "you", "your", "yours",
})
_IRREGULAR = {
    "dice": "die", "drawn": "draw", "chosen": "choose", "children": "child",
    "people": "person", "feet": "foot", "teeth": "tooth", "mice": "mouse", "men": "man",
    "women": "woman", "candies": "candy", "spun": "spin", "dealt": "deal",
}
_WORD = re.compile(r"[a-z]+")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")


def stem_word(word: str) -> str:
    """Light, deterministic suffix stripping: flipped, flips and flipping all give flip."""
    word = _IRREGULAR.get(word, word)
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeioulsz":
        word = word[:-1]
    return word


def terms(text: object) -> set[str]:
    """Normalized content words of ``text`` (empty for a non-string)."""
    if not isinstance(text, str):
        return set()
    result = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS or len(word) < 3:
            continue
        stemmed = stem_word(word)
        if len(stemmed) >= 3 and stemmed not in _STOPWORDS:
            result.add(stemmed)
    return result


def _tokens(text: str) -> list[str]:
    return [NUMBER_WORDS.get(token, token) for token in _TOKEN.findall(text.lower())]


def _numbers(text: str) -> set[str]:
    return {token for token in _tokens(text) if token[0].isdigit()}


def allowed_texts(plan: dict) -> list[str]:
    """Every text the plan supplies; the reply may reuse any object or number from these."""
    content = plan.get("content", {})
    context = plan.get("response_context", {})
    values = [
        content.get("problem_stem"),
        *(content.get("approved_facts") or []),
        content.get("permitted_hint"),
        content.get("worked_example"),
        content.get("previous_step"),
        content.get("concept_name"),
        content.get("goal_name"),
        context.get("qualified_misconception_summary"),
        context.get("last_student_message"),
    ]
    if plan.get("constraints", {}).get("reveal_allowed"):
        values.append(content.get("reveal_solution"))
    for entry in context.get("recent_dialogue") or []:
        if isinstance(entry, dict):
            values.append(entry.get("text"))
    return [value for value in values if isinstance(value, str) and value]


def quantity_phrases(text: str) -> list[tuple[str, str]]:
    """``(number, object)`` pairs such as ``("3", "marble")`` from ``3 red marbles``."""
    tokens = _tokens(text)
    found = []
    for index, token in enumerate(tokens):
        if not token[0].isdigit():
            continue
        for candidate in tokens[index + 1:index + 4]:
            if candidate[0].isdigit():
                break
            stemmed = stem_word(candidate)
            if stemmed in FLAGGABLE and stemmed not in COLOURS:
                found.append((token, stemmed))
                break
    return found


@dataclass(frozen=True)
class ItemFidelity:
    applicable: bool
    passed: bool
    coverage: float = 1.0
    shared: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    new_objects: tuple[str, ...] = ()
    new_quantities: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def applies_to(plan: dict, move: str | None = None) -> bool:
    """True for PRACTICE, CHALLENGE, CHECK_MASTERY, HINT and PROBE with an item and a stem."""
    move = move or plan.get("teaching_action", {}).get("move")
    stem = plan.get("content", {}).get("problem_stem")
    has_stem = isinstance(stem, str) and bool(stem.strip())
    if move in PRESENTING_MOVES or move == "HINT":
        return has_stem
    if move == "PROBE":
        return has_stem and bool(plan.get("problem_instance_id"))
    return False


def allowed_objects(allowed_terms: set[str]) -> set[str]:
    """Members of every scenario family the plan touches."""
    result = set()
    for members in SCENARIO_FAMILIES.values():
        if members & allowed_terms:
            result |= members
    return result


def item_fidelity(plan: dict, text: object, move: str | None = None) -> ItemFidelity:
    """Evaluate the reply against the issued item; ``applicable`` is False for other moves."""
    move = move or plan.get("teaching_action", {}).get("move")
    if not applies_to(plan, move) or not isinstance(text, str):
        return ItemFidelity(applicable=False, passed=True)
    stem_text = plan["content"]["problem_stem"]
    stem_terms = terms(stem_text)
    reply_terms = terms(text)
    shared = stem_terms & reply_terms
    coverage = len(shared) / len(stem_terms) if stem_terms else 1.0
    reasons = []
    if move in PRESENTING_MOVES:
        if coverage < PRESENT_MIN_COVERAGE:
            reasons.append(
                f"reply reuses {len(shared)} of {len(stem_terms)} stem words; a {move} must "
                f"present the issued problem (at least {PRESENT_MIN_COVERAGE:.0%})"
            )
    else:
        refers = (
            coverage >= REFER_MIN_COVERAGE
            or len(shared) >= REFER_MIN_SHARED
            or bool(_numbers(stem_text) & _numbers(text))
        )
        if not refers and move == "HINT":
            hint_terms = terms(plan["content"].get("permitted_hint"))
            refers = bool(hint_terms) and (
                len(hint_terms & reply_terms) / len(hint_terms) >= HINT_MIN_COVERAGE
            )
        if not refers:
            reasons.append(
                f"reply reuses {len(shared)} of {len(stem_terms)} stem words and none of its "
                f"numbers; a {move} must refer to the issued problem"
            )

    allowed = " ".join(allowed_texts(plan))
    permitted = allowed_objects(terms(allowed))
    allowed_numbers = _numbers(allowed)
    new_objects = sorted((reply_terms & FLAGGABLE) - permitted)
    new_quantities = sorted({
        f"{number} {noun}" for number, noun in quantity_phrases(text)
        if number not in allowed_numbers or noun not in permitted
    })
    if new_objects:
        reasons.append(
            "reply introduces objects the plan does not contain: " + ", ".join(new_objects)
        )
    if new_quantities:
        reasons.append(
            "reply introduces quantities the plan does not contain: "
            + ", ".join(new_quantities)
        )
    return ItemFidelity(
        applicable=True,
        passed=not reasons,
        coverage=coverage,
        shared=tuple(sorted(shared)),
        missing=tuple(sorted(stem_terms - shared)),
        new_objects=tuple(new_objects),
        new_quantities=tuple(new_quantities),
        reasons=tuple(reasons),
    )


def item_faithful(plan: dict, text: object, move: str | None = None) -> bool:
    return item_fidelity(plan, text, move).passed
