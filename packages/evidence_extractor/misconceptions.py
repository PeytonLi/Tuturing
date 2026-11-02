"""Misconception hypotheses with evidence spans (PRD 6.3, 9.2).

Each rule looks for the reasoning signature of one active misconception from
data/original_content/misconceptions.yaml and quotes the phrase as a span of
zero-based code point offsets into the original message. A hypothesis is a
proposal with a probability well below one; B decides what it means, and a
quoted span is never treated as proof. No hypothesis is proposed when the
learner attributes the idea to someone else, rejects it, or merely asks about
it.

Rules use the plan's problem stem to avoid labeling valid reasoning: a fair
coin really is 50/50 (M01), subtracting the complement the stem names is
correct (M04), draws with replacement are independent (M07).
"""

from __future__ import annotations

import re
from fractions import Fraction

from packages.evidence_extractor.answers import _ATTRIBUTION_RE, _REJECT_RE, tokenize

ACTIVE = ("M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08")
_QUESTION_START = re.compile(
    r"^\s*(?:are|is|do|does|did|can|could|should|would|will|why|what|how|was|were)\b",
    re.IGNORECASE,
)

_M01_PATTERNS = [
    r"\beither\b[^.;?!]{1,40}?\bor\b[^.;?!,]{0,20}",
    r"\b(?:same|equal) (?:chance|chances|probability|odds|likelihood)\b",
    r"\bequally likely\b",
    r"\b(?:because|since|cause|cuz|as) there (?:are|is) (?:\d+|two|three|four|five|six) "
    r"(?:colors|colours|options|choices|possibilities|outcomes|results|kinds|types|"
    r"sections|things|ways|events)\b",
    r"\b(?:each|every|all|both) (?:of the |the )?\w+ (?:is|are|has|have|get|gets) "
    r"(?:the )?(?:equally likely|same chance|same probability|equal chance|as likely)\b",
    r"\bjust as likely\b", r"\bone of (?:\d+|two|three|four|five) (?:colors|colours|options|"
    r"choices|possibilities)\b",
]
_M01_EXTRA_SPANS = re.compile(r"(?<![\w.])(?:50\s*[/\-:]\s*50|fifty[\s-]+fifty)(?!\w)", re.I)
_M02_PATTERNS = [
    r"\b(?:is|its|it's|are|be|counts as|count as) (?:just |only |still )?(?:one|a single|1) "
    r"(?:single )?(?:outcome|result|way|possibility|event|option)\b",
    r"\bonly one way\b", r"\bone of the outcomes\b", r"\bone outcome\b",
    r"\b(?:an? )?event is (?:just |only )?(?:one|a single)\b",
    r"\b(?:just|only) (?:one|1) (?:outcome|result|way)\b",
]
_M04_PATTERNS = [
    r"(?<![\w.])(?:1|one)\s?(?:-|−|–|minus)\s?(?P<sub>\d+(?:\.\d+)?(?:\s*/\s*\d+)?%?)",
    r"\bone minus\b[^.;?!=]{0,40}",
    r"\bsubtract(?:ed|ing)?\b[^.;?!]{0,30}\bfrom (?:1|one)\b",
    r"\b(?:1|one) take away\b[^.;?!]{0,20}",
]
_M05_PATTERNS = [
    r"\bP\([^)]*\|[^)]*\)\s*(?:is|=|equals|is the same as|same as)\s*(?:the same as )?"
    r"P\([^)]*\|[^)]*\)",
    r"\b(?:\w+) given (?:\w+) is the same as (?:\w+) given (?:\w+)\b",
    r"\b(?:doesn'?t|does not|dont|don'?t) matter which way (?:round|around)\b",
    r"\b(?:is|are|its|it's) (?:the )?same (?:thing )?as\b[^.;?!]{0,30}\bgiven\b",
    r"\b(?:just )?(?:flip|reverse|swap)(?:ped|ing)? (?:the )?(?:condition|conditional)\b",
]
_M05_NEGATED = re.compile(
    r"\b(?:isn'?t|is not|aren'?t|are not|not) (?:the )?same\b|\bdifferent from\b", re.I
)
_M06_PATTERNS = [
    r"\b(?:out of|over|divided by|divide by|by) (?:all|everyone|everybody|the total|"
    r"the whole|the whole thing|all the|all of the|the entire|everything)\b[^.;?!]{0,20}",
    r"\b(?:total|whole) (?:number|count|group|class|population)\b[^.;?!]{0,10}",
]
_M07_PATTERNS = [
    r"\b(?:are|is|they'?re|they are|its|it's|being|were|must be|would be) "
    r"(?:all |both |still |also )?independent\b",
    r"\bdon'?t affect each other\b", r"\bdo not affect each other\b",
    r"\b(?:doesn'?t|does not|dont|don'?t|won'?t|will not) (?:change|affect|matter for|"
    r"influence|impact) (?:the |my )?(?:second|next|other|2nd|later|following)\b",
    r"\bsince (?:they'?re|they are|its|it's) independent\b", r"\bindependent,? so\b",
    r"\b(?:squared|times itself)\b", r"\bnothing changes\b", r"\bsame (?:probability|chance) "
    r"(?:again|both times|each time|the second time)\b",
]
_M07_STEM_INDEPENDENT = re.compile(
    r"\b(?:replaced|replacement|put back|returned|independent|independently|fair coin[s]? "
    r"(?:is|are) flipped twice|two fair coins|two coins|two dice|each roll)\b", re.I
)
_M07_STEM_DEPENDENT = re.compile(r"\b(?:without replacement|not replaced|given|two-way|"
                                 r"not put back)\b", re.I)
_M08_PATTERNS = [
    r"\bmultipl(?:y|ied|ying) (?:all|every|both|each)(?: of)? the branches\b",
    r"\bmultipl(?:y|ied|ying) (?:the )?(?P<a>\w+) branch (?:by|with|and|times) (?:the )?"
    r"(?P<b>\w+) branch\b",
    r"\b(?:\w+) branch (?:times|x|\*) (?:the )?(?:\w+) branch\b",
    r"\bmultipl(?:y|ied|ying) (?:all |both )?(?:the )?(?:paths|branches|outcomes) together\b",
    r"\badd(?:ed|ing)? (?:up )?(?:the )?(?:branches|paths)\b",
]
_M08_TREE_CONTEXT = re.compile(r"\b(?:tree|branch|branches|path|paths|after|then|second|"
                               r"two|both|without replacement)\b", re.I)
_PRODUCT_RE = re.compile(
    r"(?P<a>\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*(?:\*|x|×|times)\s*(?P<b>\d+(?:\.\d+)?(?:\s*/\s*"
    r"\d+)?)(?:\s*(?:\*|x|×|times)\s*(?:\d+(?:\.\d+)?(?:\s*/\s*\d+)?))*",
    re.IGNORECASE,
)
_STEM_CONDITION = re.compile(
    r"\b(?:a|an|the|one) (?:(?P<adj>\w+) (?:player|student|person|people|member|customer)|"
    r"(?:person|student|people|someone) with (?:a |an |the )?(?P<with>\w+)|(?P<noun>\w+)) "
    r"(?:is|are|was|were) (?:chosen|selected|picked|drawn)\b",
    re.IGNORECASE,
)
_LEARNER_CONDITION = re.compile(
    r"\b(?:the |those |all the |all |of the \d+ |of the )?(?P<group>\w+) (?:players|people|"
    r"students|ones|members) (?:who|that|which)\b|\b(?:people|students|those|ones|players) "
    r"with (?:a |an |the )?(?P<with>\w+)\b|\bwith (?:a |an |the )?(?P<with2>\w+) (?:have|"
    r"has|had|also)\b",
    re.IGNORECASE,
)
_COUNT_WORD = re.compile(r"(?<![\w.])(\d+)\s+(?:are\s+|is\s+)?([A-Za-z]+)")
_NOT_TARGET = re.compile(
    r"\b(?:not|isn'?t|is not|does not|doesn'?t|do not|don'?t|no)\s+(?:a |an |the )?"
    r"(?:land(?:ing)? on (?:a )?(?:section that is )?)?(\w+)",
    re.IGNORECASE,
)
_STEM_TOTAL = re.compile(
    r"(?<![\w.])(\d+)\s+(?:equal |equally likely |total |identical )?(?:sections|outcomes|"
    r"cards|counters|marbles|tiles|beads|students|people|balls|socks|items|members|sides)\b",
    re.IGNORECASE,
)
_STEM_PROBABILITY = re.compile(
    r"\b(?:probability|chance|likelihood)\b[^.;]{0,40}?\bis\s+(\d+(?:\.\d+)?%?|\d+/\d+)",
    re.IGNORECASE,
)


def _span(match, group=0):
    return {"start": match.start(group), "end": match.end(group)}


def _add(hypotheses, misconception_id, probability, spans, rule):
    entry = hypotheses.setdefault(
        misconception_id, {"probability": probability, "spans": [], "rules": []}
    )
    entry["probability"] = max(entry["probability"], probability)
    for span in spans:
        if span not in entry["spans"]:
            entry["spans"].append(span)
    entry["rules"].append(rule)


def _fraction(text):
    try:
        text = text.replace(" ", "")
        if text.endswith("%"):
            return Fraction(text[:-1]) / 100
        if "/" in text:
            num, den = text.split("/", 1)
            return Fraction(num) / int(den) if int(den) else None
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _expected_complement(stem):
    """The probability the stem's 'not X' target should be subtracted from one."""
    if not stem:
        return None, False
    complement_typed = bool(
        re.search(r"\b(?:not|at least|does not|doesn'?t|isn'?t|no |neither)\b", stem, re.I)
    )
    probability = _STEM_PROBABILITY.search(stem)
    if probability and complement_typed:
        return _fraction(probability.group(1)), complement_typed
    target = _NOT_TARGET.search(stem)
    if not target:
        return None, complement_typed
    word = target.group(1).lower()
    pairs = [(int(count), noun.lower()) for count, noun in _COUNT_WORD.findall(stem)]
    count = next((number for number, noun in pairs if noun == word), None)
    total_match = _STEM_TOTAL.search(stem)
    if total_match:
        total = int(total_match.group(1))
    else:
        total = sum(number for number, noun in pairs
                    if noun not in {"equal", "equally", "favorable", "of", "and", "or"})
    if count is None or not total:
        return None, complement_typed
    return Fraction(count, total), complement_typed


def _stem_sibling_probabilities(stem):
    values = [token.value for token in tokenize(stem or "")
              if token.kind in {"decimal", "percent", "fraction"}]
    siblings = set()
    for index, first in enumerate(values):
        for second in values[index + 1:]:
            if first + second == 1 and first != second:
                siblings.add(first)
                siblings.add(second)
    return siblings


def propose(message, plan, answer_tokens, selected_answer):
    """Return (hypotheses list, rule names) for the Evidence record."""
    plan = plan or {}
    content = plan.get("content") or {}
    stem = content.get("problem_stem") or ""
    concept = plan.get("concept_id")
    rules = []
    attributed = bool(_ATTRIBUTION_RE.search(message))
    rejected = bool(_REJECT_RE.search(message))
    question = bool(_QUESTION_START.match(message)) and message.strip().endswith("?") and (
        selected_answer is None
    )
    if attributed or rejected or question:
        reason = "attributed" if attributed else "rejected" if rejected else "question"
        return [], [f"suppressed:{reason}"]

    hypotheses = {}
    candidates = [token for token in answer_tokens if not token.excluded]
    answer_value = next(
        (token.value for token in reversed(candidates) if token.form == selected_answer), None
    )

    # M01 treating unequal outcomes as equally likely
    stem_lower = stem.lower()
    fair_two = "fair coin" in stem_lower or "fair die" in stem_lower or "fair dice" in stem_lower
    stem_fractions = {token.value for token in tokenize(stem) if token.kind == "fraction"}
    for pattern in _M01_PATTERNS:
        for match in re.finditer(pattern, message, re.I):
            if fair_two and answer_value in {Fraction(1, 2), Fraction(1, 6), None}:
                continue
            if answer_value is not None and answer_value in stem_fractions:
                continue
            spans = [_span(match)]
            spans += [_span(extra) for extra in _M01_EXTRA_SPANS.finditer(message)]
            _add(hypotheses, "M01", 0.6, spans, "M01:equal_likelihood")

    # M02 confusing an event with a single outcome
    for pattern in _M02_PATTERNS:
        for match in re.finditer(pattern, message, re.I):
            _add(hypotheses, "M02", 0.55, [_span(match)], "M02:single_outcome")

    # M03 assigning probability above one
    above = [token for token in answer_tokens
             if token.kind in {"fraction", "decimal", "percent"} and token.value > 1
             and not token.flags & {"field", "negated", "score", "count"}]
    if above:
        chosen = next((token for token in reversed(above) if token.form == selected_answer),
                      above[-1])
        _add(hypotheses, "M03", 0.65,
             [{"start": chosen.start, "end": chosen.end}], "M03:above_one")

    # M04 subtracting the wrong complement
    expected, complement_typed = _expected_complement(stem)
    for pattern in _M04_PATTERNS:
        for match in re.finditer(pattern, message, re.I):
            subtrahend = _fraction(match.group("sub")) if "sub" in match.groupdict() and (
                match.group("sub")
            ) else None
            if expected is not None and subtrahend is not None:
                if subtrahend == expected:
                    continue
                probability = 0.6
            elif complement_typed:
                probability = 0.4
            else:
                probability = 0.3
            _add(hypotheses, "M04", probability, [_span(match)], "M04:one_minus")

    # M05 reversing conditional probabilities
    for pattern in _M05_PATTERNS:
        for match in re.finditer(pattern, message, re.I):
            if _M05_NEGATED.search(match.group(0)):
                continue
            _add(hypotheses, "M05", 0.55, [_span(match)], "M05:reversal_phrase")
    condition = _STEM_CONDITION.search(stem)
    if condition:
        stem_condition = (condition.group("adj") or condition.group("with")
                          or condition.group("noun") or "").lower()
        for match in _LEARNER_CONDITION.finditer(message):
            learner_condition = (match.group("group") or match.group("with")
                                 or match.group("with2") or "").lower()
            if (
                learner_condition
                and learner_condition != stem_condition
                and re.search(r"\b" + re.escape(learner_condition) + r"\b", stem, re.I)
            ):
                _add(hypotheses, "M05", 0.5, [_span(match)], "M05:conditioned_on_other_group")

    # M06 using the wrong conditional denominator
    conditional_context = concept == "C06" or bool(
        re.search(r"\bgiven\b|\bchosen\b|\bselected\b|\|", stem, re.I)
    )
    if conditional_context:
        for pattern in _M06_PATTERNS:
            for match in re.finditer(pattern, message, re.I):
                _add(hypotheses, "M06", 0.55, [_span(match)], "M06:whole_denominator")

    # M07 assuming independence without evidence
    stem_dependent = bool(_M07_STEM_DEPENDENT.search(stem))
    if stem_dependent or not _M07_STEM_INDEPENDENT.search(stem):
        probability = 0.6 if stem_dependent else 0.45
        for pattern in _M07_PATTERNS:
            for match in re.finditer(pattern, message, re.I):
                _add(hypotheses, "M07", probability, [_span(match)], "M07:independence_claim")

    # M08 multiplying along incompatible tree paths
    if concept == "C08" or _M08_TREE_CONTEXT.search(stem):
        for pattern in _M08_PATTERNS:
            for match in re.finditer(pattern, message, re.I):
                _add(hypotheses, "M08", 0.55, [_span(match)], "M08:branch_phrase")
        siblings = _stem_sibling_probabilities(stem)
        mentions_branch = bool(re.search(r"\b(?:branch|branches|path|paths)\b", message, re.I))
        for match in _PRODUCT_RE.finditer(message):
            factors = {_fraction(match.group("a")), _fraction(match.group("b"))}
            factors.discard(None)
            if siblings and len(factors & siblings) >= 2:
                _add(hypotheses, "M08", 0.55, [_span(match)], "M08:sibling_product")
            elif mentions_branch:
                _add(hypotheses, "M08", 0.45, [_span(match)], "M08:branch_product")

    result = []
    for misconception_id in ACTIVE:
        if misconception_id in hypotheses:
            entry = hypotheses[misconception_id]
            result.append({
                "misconception_id": misconception_id,
                "probability": entry["probability"],
                "evidence_spans": sorted(entry["spans"], key=lambda span: span["start"]),
            })
            rules.extend(entry["rules"])
    return result, rules
