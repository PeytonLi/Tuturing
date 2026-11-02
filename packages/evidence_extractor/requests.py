"""Explicit learner requests and the concept they name (PRD 6.3, 9.2).

A request is something the learner asks for in their own words. Mentions of a
past hint, negated requests ("don't skip this") and attempts to set internal
fields ("mark me correct") are not requests. Priority when several appear:
stop > change_goal > skip > answer > hint > slower > faster.

requested_concept_id is filled only for skip and change_goal, from a concept
named in the message as a destination ("skip to Bayes", "can we do conditional
probability"). A concept the learner is leaving ("skip complements") is not a
destination. Matching tolerates common typos through text.canonicalize.
"""

from __future__ import annotations

import re

from packages.evidence_extractor.text import canonicalize

PRIORITY = ("stop", "change_goal", "skip", "answer", "hint", "slower", "faster")

_NEG = (
    r"(?:don'?t|dont|do not|(?<!if )not|never|no|without|didn'?t|didnt|won'?t|wont|rather not)"
)
_LEARN = (
    r"(?:learn|do|study|practice|practise|work on|try|cover|go over|focus on|look at|"
    r"start|review|see|talk about|explore|tackle)"
)
_FRAME = (
    r"(?:teach me|show me|i want to|i wanna|i'?d like to|id like to|i would like to|"
    r"i'?d rather|id rather|i would rather|i'?d prefer to|i prefer to|let'?s|lets|can we|"
    r"could we|can i|could i|may we|shall we|we should|i need to|i want|i wanna|"
    r"i'?d love to)"
)

_REQUEST_PATTERNS = {
    "stop": [
        r"\bstop\b(?!\s+(?:being|with|\w+ing\b))",
        r"\bquit\b", r"\bexit\b", r"\b(?:bye|goodbye|cya|ttyl|gtg|g2g|bye bye)\b",
        r"\b(?:gotta|got to|have to|need to|should|i'?ll|must|going to|gonna) (?:go|leave|"
        r"head out|log off|sign off)\b",
        r"\bi'?m done\b", r"\bim done\b", r"\bwe'?re done\b",
        r"\b(?:that'?s|thats|that is) (?:enough|it|all)\b", r"\benough for (?:today|now)\b",
        r"\bdone for (?:today|now|tonight)\b", r"\bcall it a (?:day|night)\b",
        r"\bend (?:the |this |our )?(?:session|lesson|chat|tutoring|class)\b",
        r"\b(?:log|sign) off\b", r"\bsee you\b", r"\btalk (?:to you )?later\b",
        r"\bno more (?:questions|problems|today)\b", r"\bi want to stop\b",
        r"\bleave it (?:here|there)\b", r"\blet'?s (?:finish|end) (?:here|now)\b",
    ],
    "skip": [
        r"\bskip\b(?!\s+the (?:explanation|explanations|intro|details|fluff|lecture|theory))",
        r"\bmove on\b", r"\bmove onto\b", r"\bmoving on\b",
        r"\bjump (?:ahead|to|forward|straight to)\b", r"\bgo (?:on|ahead|straight) to\b",
        r"\bon to the next\b", r"\bgo to the next\b", r"\bskip ahead\b",
        r"(?<!what's )(?<!whats )(?<!what is )(?<!the )(?<!my )(?<!your )(?<!be on )"
        r"(?<!comes )(?<!is )(?<!are )(?<!until )(?<!till )(?<!to )(?<!on the )"
        r"\bnext\b(?!\s+(?:week|time|year|day|class|test|to|step|question is|one is)\b)",
        r"(?:^|[.!?,;]\s*|\band\s+)(?:i )?pass(?: on (?:this|it|that)(?: one)?)?\W*$",
        r"\bforget this (?:one|question|problem)\b", r"\bgive me (?:a |the )?(?:next|new|"
        r"different|another) (?:one|question|problem)\b",
    ],
    "change_goal": [
        r"\b(?:switch|change|changing|switching)\b[^.?!]{0,15}?\b(?:topics?|goals?|subjects?|"
        r"concepts?|lessons?)\b",
        r"\b(?:switch|change) to\b",
        r"\b(?:different|another|other|new) (?:topic|goal|concept|subject|lesson)\b",
        r"\bnew goal\b",
        rf"\b(?:{_LEARN}|teach me|about)\b[^.?!]{{0,12}}\b(?:something|anything) else\b",
        r"\b(?:i'?d|id|i would) rather (?:not )?(?:do|learn|study|practice|work on|cover)"
        r"[^.?!]{0,30}\binstead\b",
    ],
    "answer": [
        r"\b(?:tell|give|show|send|spill|reveal) (?:me |us )?(?:the |your |ur |an? |da |teh )?"
        r"(?:correct |right |final |full )?(?:answer|solution|result|answers|solutions)\b",
        r"\bwhat(?:'?s| is) (?:the |da |teh )?(?:correct |right |final )?(?:answer|solution|"
        r"result)\b",
        r"\bjust tell me\b", r"\breveal\b", r"\bi give up\b", r"\bgive up\b",
        r"\bsolve (?:it|this|that|the problem|the question)( for me)?\b",
        r"\bdo it for me\b", r"\b(?:answer|solution)\s*(?:pls|please|plz)\b",
        r"^\W*(?:the )?(?:answer|solution)\W*$",
        r"\bwant (?:the|an|to know the|to see the) (?:answer|solution)\b",
        r"\bshow (?:me )?(?:how to solve|the (?:working|steps|solution|answer))\b",
        r"\bwhat(?:'?s| is) it\b(?=\W*$)",
    ],
    "hint": [
        r"\bhints?\b", r"\bclue\b", r"\bhelp\b", r"\bnudge\b",
        r"\bwhere (?:do|should|would) i (?:start|begin)\b", r"\bhow (?:do|should) i (?:start|"
        r"begin)\b", r"\b(?:what'?s|whats|what is) the first step\b",
        r"\bget (?:me )?started\b", r"\bpoint me in\b", r"\bguide me\b",
        r"\bwalk me through\b", r"\bpointer\b", r"\bstuck,? (?:can|could) (?:i|you)\b",
        r"\ba (?:little|small|tiny) (?:push|nudge|help|clue)\b",
    ],
    "slower": [
        r"\bslower\b", r"\bslow(?:ly)? down\b", r"\bmore slowly\b", r"\btoo fast\b",
        r"\bso fast\b", r"\bnot so fast\b", r"\bone step at a time\b", r"\bstep by step\b",
        r"\bmore detail(?:s|ed)?\b", r"\bin (?:more )?depth\b", r"\btake it slow\b",
        r"\bgoing too fast\b", r"\bexplain (?:it |that |this )?(?:more|slowly|in detail|"
        r"more carefully|properly)\b", r"\bbreak it down\b", r"\bsmaller steps\b",
        r"\bslow it down\b", r"\blonger (?:explanation|explanations|answers)\b",
    ],
    "faster": [
        r"\bfaster\b", r"\bquicker\b", r"\bspeed (?:it |this |things )?up\b", r"\bhurry\b",
        r"\btoo slow\b", r"\bso slow\b", r"\bmore brief\b", r"\bbriefer\b", r"\bbe brief\b",
        r"\bbriefly\b", r"\bbrief(?:er)? (?:answers|explanations|please|pls)\b",
        r"\bshorter\b", r"\bless (?:talking|talk|explaining|explanation|explanations|text|"
        r"words|detail|detailed|chat|chatting)\b", r"\btl;?dr\b", r"\bpick up the pace\b",
        r"\b(?:taking|takes|take) forever\b", r"\bget (?:to|on with) (?:it|the point)\b",
        r"\bcut to the chase\b", r"\bskip the (?:explanation|explanations|intro|details|"
        r"fluff|lecture|theory)\b", r"\bkeep it short\b", r"\bshort (?:answers|version)\b",
        r"\bquick(?:ly)? (?:please|pls)\b",
    ],
}
_COMPILED = {
    name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for name, patterns in _REQUEST_PATTERNS.items()
}

# Phrases that cancel a match when they immediately precede or surround it.
_GUARDS = {
    "stop": [
        r"\b(?:you|did you|why did you|why do you|why'?d you|" + _NEG + r"|please don'?t) "
        r"(?:just |ever |always |suddenly )?$",
    ],
    "skip": [_NEG + r"(?: want to| wanna| need to)?(?: just)? $", r"\bis $", r"\bare $"],
    "change_goal": [_NEG + r"(?: want to| wanna)? $"],
    "answer": [
        _NEG + r"(?: want to| wanna| need to| have to)?(?: just)? $",
        r"\b(?:please|pls|plz) don'?t $", r"\bwithout $",
    ],
    "hint": [
        r"\b(?:the|that|this|your|last|previous|first|second|with the|with that|without an?|"
        r"without the|after the|thanks for the|thx for the|thanks for that|ty for the|no|"
        r"not|a bad|that'?s a|what a|some|used the|took the|got the|read the|saw the|"
        r"i have no|have no|i'?ve got no|i'?ve no) $",
        _NEG + r"(?: really| even| actually)? (?:need|want)(?: an?| any| the| more| another|"
        r" your)? $",
        r"\b(?:didn'?t|didnt|doesn'?t|doesnt|did not|does not|won'?t|wont|hasn'?t|hadn'?t)"
        r"(?:\s+\w+){0,2} $",
    ],
    "slower": [_NEG + r" $"],
    "faster": [_NEG + r" $"],
}
_HINT_AFTER_GUARDS = [
    r"^ with my\b", r"^ with (?!this|the problem|the question|it\b|this one|the first step)",
    r"^ (?:didn'?t|didnt|doesn'?t|doesnt|did not|does not|won'?t|wont) ",
    r"^s? (?:didn'?t|didnt|doesn'?t|doesnt|did not|does not|were|was|are|is) ",
    r"^ (?:me )?(?:with|on) my (?!answer|work|attempt)",
]
_COMPILED_GUARDS = {
    name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for name, patterns in _GUARDS.items()
}
_COMPILED_HINT_AFTER = [re.compile(pattern, re.IGNORECASE) for pattern in _HINT_AFTER_GUARDS]

# Concept lexicon: longest phrases first so "total probability" beats "probability tree".
_CONCEPTS = [
    ("law of total probability", "C09"), ("total probability", "C09"),
    ("conditional probability", "C06"), ("conditional probabilities", "C06"),
    ("multiplication rule", "C08"), ("probability trees", "C08"), ("probability tree", "C08"),
    ("tree diagrams", "C08"), ("tree diagram", "C08"),
    ("addition rule", "C05"), ("or rule", "C05"),
    ("expected value", "C15"), ("expected values", "C15"), ("expectation", "C15"),
    ("sample spaces", "C01"), ("sample space", "C01"), ("outcomes and events", "C01"),
    ("equally likely", "C02"), ("probability bounds", "C02"),
    ("bayes theorem", "C10"), ("bayes rule", "C10"), ("bayes", "C10"), ("bayesian", "C10"),
    ("set notation", "C04"), ("venn diagrams", "C04"), ("venn diagram", "C04"),
    ("venn", "C04"), ("unions", "C04"), ("union", "C04"), ("intersections", "C04"),
    ("intersection", "C04"),
    ("independent events", "C07"), ("independence", "C07"),
    ("counting with order", "C11"), ("permutations", "C11"), ("permutation", "C11"),
    ("arrangements", "C11"),
    ("counting without order", "C12"), ("combinations", "C12"), ("combination", "C12"),
    ("choose", "C12"),
    ("bernoulli trials", "C13"), ("bernoulli trial", "C13"), ("bernoulli", "C13"),
    ("binomial probability", "C14"), ("binomial distribution", "C14"), ("binomial", "C14"),
    ("complements", "C03"), ("complement", "C03"), ("complementary events", "C03"),
    ("conditional", "C06"), ("trees", "C08"), ("tree", "C08"),
]
_CONCEPT_VOCABULARY = frozenset(
    word for phrase, _ in _CONCEPTS for word in phrase.split()
) | frozenset({"probability", "theorem", "instead", "another", "please"})
_REQUEST_VOCABULARY = frozenset({
    "hint", "hints", "skip", "stop", "faster", "slower", "answer", "solution", "help",
    "next", "quit", "exit", "slow", "fast", "clue", "please", "another", "the", "think",
    "know", "instead", "topic", "switch", "change", "learn", "move", "give", "tell",
    "show", "what", "whats", "down", "speed", "hurry", "quicker", "brief", "shorter",
    "explain", "detail", "reveal", "solve", "goal", "different", "something", "else",
})
VOCABULARY = _CONCEPT_VOCABULARY | _REQUEST_VOCABULARY

_SKIPPED_BEFORE = re.compile(
    r"\b(?:skip|skipping|skipped|know|knows|knew|understand|understood|get|got|done with|"
    r"finished|hate|over|bored of|bored with|sick of|tired of)\s+(?:the |this |these |"
    r"that |those |all the |all of the )?$",
    re.IGNORECASE,
)
_SKIPPED_AFTER = re.compile(
    r"^\s*(?:is|are|was|were|seems?|seemed|r)\s+(?:so |too |really |pretty |very )?"
    r"(?:easy|boring|simple|done|fine|clear|old|ok|okay|basic|trivial|obvious)\b",
    re.IGNORECASE,
)


def _matches(name, text):
    """Matched request phrases for one request type after guards."""
    found = []
    for pattern in _COMPILED[name]:
        for match in pattern.finditer(text):
            before = text[max(0, match.start() - 40):match.start()]
            if any(guard.search(before) for guard in _COMPILED_GUARDS.get(name, [])):
                continue
            if name == "hint" and match.group(0).lower() in {"hint", "hints", "help", "clue"}:
                after = text[match.end():match.end() + 40]
                if any(guard.search(after) for guard in _COMPILED_HINT_AFTER):
                    continue
            found.append(match)
    return found


def concept_mentions(text):
    """Concept phrases in canonical text: [(concept_id, start, end, destination)]."""
    mentions = []
    taken = []
    for phrase, concept_id in _CONCEPTS:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", text):
            if any(start < match.end() and match.start() < end for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            before = text[max(0, match.start() - 40):match.start()]
            after = text[match.end():match.end() + 40]
            destination = not (_SKIPPED_BEFORE.search(before) or _SKIPPED_AFTER.match(after))
            mentions.append((concept_id, match.start(), match.end(), destination))
    mentions.sort(key=lambda item: item[1])
    return mentions


def detect(message):
    """Return (explicit_request, requested_concept_id, details) for a learner message."""
    text = canonicalize(message, VOCABULARY)
    matched = {}
    for name in PRIORITY:
        hits = _matches(name, text)
        if hits:
            matched[name] = hits[0].group(0).strip()
    mentions = concept_mentions(text)
    destinations = [item for item in mentions if item[3]]
    # A learn-frame followed by a concept is a goal change even without topic words.
    if "change_goal" not in matched and destinations:
        frame = re.compile(
            rf"\b{_FRAME}\b(?:\s+(?:to\s+)?{_LEARN})?(?:\s+about)?\s+(?:the |some |a bit of |"
            r"more )?$",
            re.IGNORECASE,
        )
        instead = re.compile(r"^\s*(?:\w+\s+){0,2}instead\b", re.IGNORECASE)
        for concept_id, start, end, _ in destinations:
            before = text[max(0, start - 60):start]
            if frame.search(before) or instead.match(text[end:end + 30]):
                if not any(guard.search(before[-30:])
                           for guard in _COMPILED_GUARDS["change_goal"]):
                    matched["change_goal"] = text[max(0, start - 20):end].strip()
                    break
    request = next((name for name in PRIORITY if name in matched), "none")
    concept = None
    if request in {"skip", "change_goal"} and destinations:
        concept = destinations[0][0]
    details = {
        "request_candidates": matched,
        "concept_mentions": [
            {"concept_id": concept_id, "destination": destination}
            for concept_id, _, _, destination in mentions
        ],
    }
    return request, concept, details
