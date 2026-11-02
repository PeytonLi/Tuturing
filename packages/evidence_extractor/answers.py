"""Find the numeric answer a learner commits to in a message.

The parser proposes a normalized answer string for B's deterministic grader;
it never grades. Steps:

1. Tokenize numbers: digits with fractions, decimals, percents ("30%", "75 pct",
   "65 per cent"), "n in d", "n out of d", "n over d", "n of d", ratios
   ("3:7", "odds are 3 to 7"), unicode fractions, decimal commas ("0,6"), the
   letter-o typo ("o.3"), "50/50", and number words ("three tenths", "a quarter",
   "zero point two", "seventy five percent", "one in four").
2. Flag tokens that are not answers: counts ("2 dogs", "3 red"), field values
   ("mastery=1.0", "set mastery to 100%"), negated or operand numbers ("not
   3/10", "divide by 7", "more than 1"), denominators left without a numerator
   ("favorable out of 8"), test scores, numbers attributed to someone else when
   the learner also gives their own, and numbers the learner rejects.
3. Group tokens joined by operators into expressions; the number after "=" is
   the result. An expression without a result is not an answer ("3/5 * 2/4").
4. Choose: one distinct value wins (last form kept); otherwise ranges and
   alternatives with different values give no answer; a final number preceded
   by a conclusion marker ("so", "=", "wait,", "i got") wins; restated problem
   givens without a marker give no answer; otherwise the last number wins with
   low confidence.

Normalized forms: integers as digits ("1,000" -> "1000"); decimals with a
leading zero and the learner's digits; fractions "n/d" unreduced; percents
"n%"; ratios "a:b"; "50/50" and "fifty fifty" -> "50%".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from packages.evidence_extractor.text import canonicalize, followed_by, preceded_by, words

_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+"
_DIGIT_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_NUMBER})"
    rf"(?:\s*/\s*(?P<den>\d+)(?:ths|rds|nds|th|st)?)?"
    rf"(?P<pct>\s*%|\s*(?:percent|pct|per cent))?(?!\w|\.\d)",
    re.IGNORECASE,
)
_COMPOSITE_RE = re.compile(
    r"(?<![\w.])(?P<num>\d+(?:\.\d+)?)\s+(?:in|out of|outta|outof|over|of)\s+"
    r"(?:all |the |all the |every )?(?P<den>\d+)(?!\w|\.\d)",
    re.IGNORECASE,
)
_FRAGMENT_RE = re.compile(
    r"\b(?:out of|outta|over|in)\s+(?:all |the |all the )?(?P<den>\d+)(?!\w|\.\d)",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(r"(?<![\w./])(?P<a>\d+)\s*(?::|\s+to\s+)\s*(?P<b>\d+)(?!\w|[./]\d)")
_ODDS_CONTEXT = re.compile(r"\b(?:odds|ratio|chances?)\b", re.IGNORECASE)
_EVEN_ODDS_RE = re.compile(
    r"(?<![\w.])(?:50\s*[/\-:]\s*50|fifty[\s-]+fifty)(?!\w)", re.IGNORECASE
)
_DECIMAL_COMMA_RE = re.compile(r"(?<![\d.,])(?P<int>\d+),(?P<frac>\d{1,2})(?![\d,])")
_LETTER_O_RE = re.compile(r"(?<![A-Za-z0-9.])[oO]\.(?P<frac>\d+)(?!\w)")
_VULGAR = {
    "½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4), "⅕": (1, 5),
    "⅖": (2, 5), "⅗": (3, 5), "⅘": (4, 5), "⅙": (1, 6), "⅚": (5, 6), "⅐": (1, 7),
    "⅛": (1, 8), "⅜": (3, 8), "⅝": (5, 8), "⅞": (7, 8), "⅑": (1, 9), "⅒": (1, 10),
}

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_DENOMS = {
    "half": 2, "halves": 2, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4,
    "fourth": 4, "fourths": 4, "fifth": 5, "fifths": 5, "sixth": 6, "sixths": 6,
    "seventh": 7, "sevenths": 7, "eighth": 8, "eighths": 8, "ninth": 9, "ninths": 9,
    "tenth": 10, "tenths": 10, "eleventh": 11, "elevenths": 11, "twelfth": 12,
    "twelfths": 12, "thirteenth": 13, "thirteenths": 13, "fourteenth": 14,
    "fourteenths": 14, "fifteenth": 15, "fifteenths": 15, "sixteenth": 16,
    "sixteenths": 16, "twentieth": 20, "twentieths": 20, "hundredth": 100,
    "hundredths": 100,
}
_NUMBER_WORDS = frozenset(
    list(_UNITS) + list(_TENS) + list(_DENOMS)
    + ["percent", "pct", "point", "hundred", "over", "out", "of", "in", "a", "an"]
)
_CONNECTORS = frozenset({
    "or", "and", "so", "is", "its", "it's", "then", "but", "which", "percent", "pct", "per",
    "out", "in", "over", "of", "to", "right", "i", "im", "i'm", "lol", "maybe", "idk",
    "please", "pls", "plz", "because", "bc", "cuz", "since", "if", "for", "as", "the",
    "that", "this", "was", "were", "not", "no", "yes", "yeah", "ok", "okay", "thanks",
    "thx", "though", "tho", "probably", "perhaps", "possibly", "i'd", "id", "u", "you",
    "we", "can", "could", "also", "now", "after", "when", "what", "whats", "what's",
    "would", "should", "wait", "nope", "hmm", "um", "umm", "uh", "uhh", "haha",
    "exactly", "approximately", "roughly", "about", "around", "like", "x", "times",
    "divided", "minus", "plus", "squared", "cubed", "correct", "incorrect", "wrong",
    "mark", "grade", "earlier", "before", "again", "either", "too", "at", "on", "by",
    "final", "answer", "ans",
})
_FIELD_RE = (
    r"\b(?:mastery|correctness|confidence|grade|score|reward|version|state_version|level|"
    r"hint_level|turns?|probability_of_mastery|p_mastery|budget|points?|xp|streak)"
    r"\s*(?:=|:|to|at|is|of)"
)
_NEGATION_RE = r"\b(?:not|isn'?t|isnt|not by|rather than|instead of|than|at most|at least)"
_OPERAND_RE = r"\b(?:by|times|multiplied by|divided by|minus|plus)|[-−–+*×÷]"
_POSTFIX_RE = r"(?:squared|cubed|to the power|times itself|\^)"
_SCORE_RE = re.compile(r"\b(?:score|scored|exam|quiz|test)\b", re.IGNORECASE)
_ATTRIBUTION_RE = re.compile(
    r"\b(?:(?:my|the|a|our|his|her|their|some|another) "
    r"(?:friends?|teacher|tutor|brother|sister|mom|mum|dad|parents?|classmates?|cousin|"
    r"roommate|buddy|book|textbook|answer key|key|calculator|internet|group chat)"
    r"|someone|somebody|chatgpt|chat gpt|gpt|google|siri|alexa|wikipedia|gemini|copilot|"
    r"the answer key|the answer sheet|answer key|answer sheet)\b"
    r"[^.;!?]{0,25}?\b(?:said|says|say|told|got|gave|gives|thinks?|answered|wrote|has|had|"
    r"showed|helped|suggested|claims?|reckons?)\b",
    re.IGNORECASE,
)
_REJECT_RE = re.compile(
    r"\b(?:that'?s|thats|that is|which is|but it'?s|but its|it'?s|its|this is) "
    r"(?:wrong|incorrect|not right|not correct)\b|\bwrong,? right\?|\bcan'?t be right\b|"
    r"\bisn'?t right\b|\bdoesn'?t seem right\b",
    re.IGNORECASE,
)
_OPERATOR_GLUE = re.compile(
    r"^[\s()]*(?:[-−–+*×÷/]|x|times|minus|plus|multiplied by|divided by|over|"
    r"multiplied with)[\s()]*$",
    re.IGNORECASE,
)
_EQUALS_GLUE = re.compile(
    r"^[\s()]*(?:=|==|equals|is equal to|gives|->|=>|which is|which gives|which equals|"
    r"which makes|that is|i\.e\.|, i\.e\.|,? which is|,? so (?:that'?s|thats|it'?s|its)?)"
    r"[\s()]*$",
    re.IGNORECASE,
)
_OR_GLUE = re.compile(r"^[\s,]*(?:or|or maybe|or perhaps|or possibly|/)\s*$", re.IGNORECASE)
_RANGE_GLUE = re.compile(r"^[\s,]*(?:and|to|-|–)\s*$", re.IGNORECASE)
_MARKER_RE = (
    r"(?:\bso\b|=|\bequals\b|\bis\b|\bits\b|\bit'?s\b|\banswer is\b|\banswer:|\bans:|"
    r"\bi get\b|\bi got\b|\bgot\b|\bgives\b|\bwhich is\b|\bwhich gives\b|\bmakes\b|"
    r"\bwait\b,?|\bactually\b,?|\bi mean\b,?|\bsorry\b,?|\bno wait\b,?|\bthen\b|"
    r"\btherefore\b|\bthus\b|->|=>|\bfinal answer\b:?|\bchanged? (?:it|my answer) to\b|"
    r"\bno,|\bcorrection\b:?|\bshould be\b|\bmust be\b|\bhas to be\b|\bto\b|\bbe\b)"
)
_BETWEEN_RE = r"\b(?:between|from|somewhere between|anywhere between|in the range)"
_QUESTION_WORDS = ("are", "is", "do", "does", "did", "can", "could", "should", "would",
                   "will", "why", "what", "how", "was", "were")


@dataclass
class Candidate:
    start: int
    end: int
    value: Fraction
    form: str
    kind: str  # int, decimal, fraction, percent, ratio, words
    source: str = "digits"
    flags: set = field(default_factory=set)
    group: int = -1
    result: bool = False
    marker: bool = False

    @property
    def excluded(self):
        return bool(self.flags & {
            "count", "field", "negated", "operand", "fragment", "score", "rejected",
            "expression", "postfix",
        })


def _normalize_digits(num_text):
    cleaned = num_text.replace(",", "")
    if cleaned.startswith("."):
        cleaned = "0" + cleaned
    elif cleaned.startswith("-."):
        cleaned = "-0" + cleaned[1:]
    return cleaned


def _digit_tokens(text):
    tokens = []
    for match in _DIGIT_RE.finditer(text):
        num = _normalize_digits(match.group("num"))
        den = match.group("den")
        pct = match.group("pct")
        if den is not None:
            if int(den) == 0:
                continue
            tokens.append(Candidate(
                match.start(), match.end(), Fraction(num) / int(den), f"{num}/{den}",
                "fraction",
            ))
        elif pct:
            tokens.append(Candidate(
                match.start(), match.end(), Fraction(num) / 100, f"{num}%", "percent",
            ))
        else:
            kind = "decimal" if "." in num else "int"
            tokens.append(Candidate(match.start(), match.end(), Fraction(num), num, kind))
    return tokens


def _special_tokens(text):
    tokens = []
    for match in _EVEN_ODDS_RE.finditer(text):
        tokens.append(Candidate(
            match.start(), match.end(), Fraction(1, 2), "50%", "percent", "even_odds",
        ))
    for match in _COMPOSITE_RE.finditer(text):
        num, den = match.group("num"), match.group("den")
        if int(den) == 0:
            continue
        tokens.append(Candidate(
            match.start(), match.end(), Fraction(num) / int(den), f"{num}/{den}",
            "fraction", "composite",
        ))
    whole = text.strip()
    for match in _RATIO_RE.finditer(text):
        a, b = int(match.group("a")), int(match.group("b"))
        if a + b == 0:
            continue
        if match.group(0).strip() != whole and not _ODDS_CONTEXT.search(
            text[max(0, match.start() - 30):match.start()]
        ):
            continue
        tokens.append(Candidate(
            match.start(), match.end(), Fraction(a, a + b), f"{a}:{b}", "ratio", "ratio",
        ))
    for match in _DECIMAL_COMMA_RE.finditer(text):
        form = f"{match.group('int')}.{match.group('frac')}"
        tokens.append(Candidate(
            match.start(), match.end(), Fraction(form), form, "decimal", "decimal_comma",
        ))
    for match in _LETTER_O_RE.finditer(text):
        form = "0." + match.group("frac")
        tokens.append(Candidate(
            match.start(), match.end(), Fraction(form), form, "decimal", "letter_o",
        ))
    for index, char in enumerate(text):
        if char in _VULGAR:
            num, den = _VULGAR[char]
            tokens.append(Candidate(
                index, index + 1, Fraction(num, den), f"{num}/{den}", "fraction", "vulgar",
            ))
    return tokens


def _word_tokens(text):
    """Number-word phrases on the (typo-canonicalized) word sequence."""
    canonical = canonicalize(text, _NUMBER_WORDS)
    sequence = [(word.replace("'", ""), start, end) for word, start, end in words(canonical)]
    offsets = _offset_map(text, canonical)
    tokens = []
    index = 0
    while index < len(sequence):
        parsed = _parse_word_number(sequence, index, text, offsets)
        if parsed is None:
            index += 1
            continue
        token, index = parsed
        tokens.append(token)
    return tokens


def _offset_map(original, canonical):
    """Map canonical word spans back to original spans (words align one to one)."""
    original_words = list(words(original))
    canonical_words = list(words(canonical))
    if len(original_words) != len(canonical_words):
        return None
    return {
        (c_start, c_end): (o_start, o_end)
        for (_, o_start, o_end), (_, c_start, c_end) in zip(original_words, canonical_words)
    }


def _cardinal(sequence, index):
    """Parse a cardinal number word phrase; return (value, next_index) or None."""
    if index >= len(sequence):
        return None
    word = sequence[index][0]
    if word in ("a", "an"):
        return 1, index + 1
    if word == "hundred":
        return 100, index + 1
    if word in _UNITS:
        value, index = _UNITS[word], index + 1
    elif word in _TENS:
        value, index = _TENS[word], index + 1
        if index < len(sequence) and sequence[index][0] in _UNITS and (
            0 < _UNITS[sequence[index][0]] < 10
        ):
            value += _UNITS[sequence[index][0]]
            index += 1
    else:
        return None
    if index < len(sequence) and sequence[index][0] == "hundred":
        value *= 100
        index += 1
        if index < len(sequence) and sequence[index][0] == "and":
            index += 1
        rest = _cardinal(sequence, index)
        if rest is not None and rest[0] < 100:
            value += rest[0]
            index = rest[1]
    return value, index


def _parse_word_number(sequence, index, text, offsets):
    def span(first, last):
        start, end = sequence[first][1], sequence[last][2]
        if offsets is not None:
            start = offsets[(sequence[first][1], sequence[first][2])][0]
            end = offsets[(sequence[last][1], sequence[last][2])][1]
        return start, end

    word = sequence[index][0]
    if word == "point":
        digits = []
        cursor = index + 1
        while cursor < len(sequence) and sequence[cursor][0] in _UNITS and (
            _UNITS[sequence[cursor][0]] < 10
        ):
            digits.append(str(_UNITS[sequence[cursor][0]]))
            cursor += 1
        if digits:
            form = "0." + "".join(digits)
            start, end = span(index, cursor - 1)
            return Candidate(start, end, Fraction(form), form, "decimal", "words"), cursor
        return None
    if word in _DENOMS and word in ("half", "quarter"):
        # Bare "half" or "quarter" at the end of a clause means one of them.
        start, end = span(index, index)
        if not followed_by(text, end, r"(?:of|the|a|an|an?|way|time|hour|past)\b"):
            den = _DENOMS[word]
            return Candidate(start, end, Fraction(1, den), f"1/{den}", "fraction", "words"), (
                index + 1
            )
        return None
    parsed = _cardinal(sequence, index)
    if parsed is None:
        return None
    value, cursor = parsed
    article = word in ("a", "an")
    if cursor < len(sequence):
        next_word = sequence[cursor][0]
        if next_word in _DENOMS:
            den = _DENOMS[next_word]
            start, end = span(index, cursor)
            return Candidate(
                start, end, Fraction(value, den), f"{value}/{den}", "fraction", "words",
            ), cursor + 1
        if next_word == "point" and not article:
            digits = []
            inner = cursor + 1
            while inner < len(sequence) and sequence[inner][0] in _UNITS and (
                _UNITS[sequence[inner][0]] < 10
            ):
                digits.append(str(_UNITS[sequence[inner][0]]))
                inner += 1
            if digits:
                form = f"{value}." + "".join(digits)
                start, end = span(index, inner - 1)
                return Candidate(start, end, Fraction(form), form, "decimal", "words"), inner
        if next_word in ("percent", "pct") and not article:
            start, end = span(index, cursor)
            return Candidate(
                start, end, Fraction(value, 100), f"{value}%", "percent", "words",
            ), cursor + 1
        if next_word in ("in", "over", "outta") or (
            next_word == "out" and cursor + 1 < len(sequence) and sequence[cursor + 1][0] == "of"
        ):
            inner = cursor + (2 if next_word == "out" else 1)
            if inner < len(sequence) and sequence[inner][0] in ("all", "the", "every"):
                inner += 1
            rest = _cardinal(sequence, inner)
            if rest is not None and rest[0] > 0 and not article:
                start, end = span(index, rest[1] - 1)
                return Candidate(
                    start, end, Fraction(value, rest[0]), f"{value}/{rest[0]}", "fraction",
                    "words",
                ), rest[1]
    if article or value >= 10 or cursor - index > 1:
        return None
    # A bare "zero" or "one" counts only as a whole-message or clause-final answer.
    start, end = span(index, cursor - 1)
    if word not in ("zero", "one"):
        return None
    before_ok = preceded_by(text, start, r"(?:^|\bis|=|\banswer|\bits|\bit's|\bget|\bgot|\bbe)")
    after_ok = followed_by(text, end, r"(?:$|[.!?,;])")
    if before_ok and after_ok:
        return Candidate(start, end, Fraction(value), str(value), "int", "words"), cursor
    return None


def _merge(tokens):
    """Drop tokens nested inside longer ones; keep order by position."""
    ordered = sorted(tokens, key=lambda token: (token.start, -(token.end - token.start)))
    merged = []
    for token in ordered:
        if merged and token.start < merged[-1].end:
            continue
        merged.append(token)
    return merged


def tokenize(text):
    """All numeric tokens in text, longest match first, in message order."""
    text = text.replace("⁄", "/")
    tokens = _special_tokens(text) + _digit_tokens(text) + _word_tokens(text)
    return _merge(tokens)


def _stem_values(stem):
    if not stem:
        return set()
    return {token.value for token in tokenize(stem)}


def _flag_tokens(text, tokens, stem):
    stem_has_score = bool(stem and _SCORE_RE.search(stem))
    stem_values = _stem_values(stem)
    attributed_clauses = [
        (match.start(), _clause_end(text, match.end()))
        for match in _ATTRIBUTION_RE.finditer(text)
    ]
    for token in tokens:
        before = text[:token.start]
        after = text[token.end:]
        if token.kind == "int" and token.source == "digits":
            match = re.match(r"\s+([A-Za-z][A-Za-z']*)", after)
            if match and match.group(1).lower() not in _CONNECTORS and not (
                _FRAGMENT_RE.match(text, token.end) or re.match(r"\s*%", after)
            ):
                token.flags.add("count")
        if preceded_by(text, token.start, _FIELD_RE):
            token.flags.add("field")
        if preceded_by(text, token.start, _NEGATION_RE):
            token.flags.add("negated")
        if preceded_by(text, token.start, r"\bby") or preceded_by(
            text, token.start, r"\b(?:divided|multiplied) by"
        ):
            token.flags.add("operand")
        if followed_by(text, token.end, _POSTFIX_RE):
            token.flags.add("postfix")
        if token.source == "digits" and token.kind == "int":
            fragment = re.search(
                r"\b(?:out of|outta|over|in)\s+(?:all |the |all the )?$", before, re.I
            )
            if fragment:
                token.flags.add("fragment")
        clause_start = _clause_start(text, token.start)
        if not stem_has_score and _SCORE_RE.search(text[clause_start:token.start]):
            token.flags.add("score")
        if any(start <= token.start < end for start, end in attributed_clauses):
            token.flags.add("attributed")
        if _REJECT_RE.search(text[token.end:_clause_end(text, token.end, hard=True)]):
            token.flags.add("rejected")
        if token.value in stem_values:
            token.flags.add("given")
    return tokens


def _clause_start(text, position):
    boundary = max(
        text.rfind(".", 0, position), text.rfind(";", 0, position), text.rfind("!", 0, position),
        text.rfind("?", 0, position),
    )
    but = re.search(r"\bbut\b(?!.*\bbut\b)", text[:position], re.I)
    if but and but.start() > boundary:
        boundary = but.start()
    return boundary + 1


def _clause_end(text, position, hard=False):
    candidates = [index for index in (
        text.find(".", position), text.find(";", position), text.find("!", position),
        text.find("?", position),
    ) if index >= 0]
    if not hard:
        but = re.search(r"\bbut\b", text[position:], re.I)
        if but:
            candidates.append(position + but.start())
    return min(candidates) if candidates else len(text)


def _structure(text, tokens):
    """Group operator-joined tokens into expressions and mark results and markers."""
    group_id = 0
    for index, token in enumerate(tokens):
        if token.group < 0:
            token.group = group_id
            group_id += 1
        if index + 1 >= len(tokens):
            break
        nxt = tokens[index + 1]
        glue = text[token.end:nxt.start]
        if _OPERATOR_GLUE.match(glue):
            nxt.group = token.group
        elif _EQUALS_GLUE.match(glue):
            nxt.result = True
            nxt.marker = True
            nxt.group = group_id
            group_id += 1
    groups = {}
    for token in tokens:
        groups.setdefault(token.group, []).append(token)
    for members in groups.values():
        if len(members) > 1:
            for member in members:
                member.flags.add("expression")
    for index, token in enumerate(tokens):
        if preceded_by(text, token.start, _MARKER_RE, window=20):
            token.marker = True
        if preceded_by(text, token.start, r"\b(?:just|exactly|only)", window=12) and index > 0:
            token.marker = True
    return tokens


def _is_question_about(text):
    stripped = text.strip().lower()
    return stripped.endswith("?") and stripped.split(" ", 1)[0] in _QUESTION_WORDS


def select_answer(text, plan=None):
    """Return (normalized_answer, confidence, details) for a learner message."""
    stem = ((plan or {}).get("content") or {}).get("problem_stem")
    tokens = _structure(text, _flag_tokens(text, tokenize(text), stem))
    candidates = [token for token in tokens if not token.excluded]
    if any("attributed" not in token.flags for token in candidates):
        candidates = [token for token in candidates if "attributed" not in token.flags]
    details = {
        "numbers_found": len(tokens),
        "answer_candidates": [token.form for token in candidates],
        "answer_selection": "none",
        "all_tokens": tokens,
        "selected_token": None,
    }
    if not candidates:
        if tokens and all("expression" in token.flags for token in tokens):
            details["answer_selection"] = "expression"
        return None, 0.0, details
    distinct = {token.value for token in candidates}
    if len(distinct) == 1:
        details["answer_selection"] = "single" if len(candidates) == 1 else "equal_values"
        details["selected_token"] = candidates[-1]
        return candidates[-1].form, 0.9, details
    first, last = candidates[0], candidates[-1]
    if preceded_by(text, first.start, _BETWEEN_RE, window=24) and any(
        _RANGE_GLUE.match(text[a.end:b.start]) for a, b in zip(candidates, candidates[1:])
    ):
        details["answer_selection"] = "range"
        return None, 0.0, details
    if last.marker or last.result:
        details["answer_selection"] = "marker"
        details["selected_token"] = last
        return last.form, 0.85, details
    if any(_OR_GLUE.match(text[a.end:b.start]) for a, b in zip(candidates, candidates[1:])):
        details["answer_selection"] = "alternatives"
        return None, 0.0, details
    if all("given" in token.flags for token in candidates):
        details["answer_selection"] = "givens"
        return None, 0.0, details
    if _is_question_about(text):
        details["answer_selection"] = "question"
        return None, 0.0, details
    details["answer_selection"] = "last_of_several"
    details["selected_token"] = last
    return last.form, 0.6, details
