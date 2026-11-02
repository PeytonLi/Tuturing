"""Pace stepping and a small per-session pace estimator.

The estimator state is a plain JSON-serializable dict so it can live in the
learning-service session state and in episode diagnostics.
"""

from __future__ import annotations

import math

from packages.learning.actions import PACES

BASE_PRIOR = (0.2, 0.6, 0.2)
PRIOR_ANCHOR = 0.5
REQUEST_MASS = 0.8
DECAY = 0.1
GAIN = 1.5
HOLD = 2

W_STREAK = 0.6
W_FAIL = 0.6
W_HINT = 0.4
W_ANSWER = 0.6
W_CUE = 0.4

FAST_SECONDS = 8.0
SLOW_SECONDS = 40.0
SHORT_WORDS = 3
LONG_WORDS = 25

_STEP = {"faster": -1, "slower": 1}


def step_pace(current: str, request: str | None) -> str:
    """Move one pace step: faster toward concise, slower toward detailed."""
    if current not in PACES:
        current = "normal"
    if request not in _STEP:
        return current
    index = PACES.index(current) + _STEP[request]
    return PACES[min(len(PACES) - 1, max(0, index))]


def _onehot(pace: str) -> list[float]:
    return [1.0 if candidate == pace else 0.0 for candidate in PACES]


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        return list(BASE_PRIOR)
    return [value / total for value in values]


def initial_state(pace: str = "normal") -> dict:
    if pace not in PACES:
        pace = "normal"
    prior = _normalize(
        [
            (1 - PRIOR_ANCHOR) * base + PRIOR_ANCHOR * hot
            for base, hot in zip(BASE_PRIOR, _onehot(pace), strict=True)
        ]
    )
    return {
        "prior": prior,
        "belief": list(prior),
        "pace": pace,
        "request": None,
        "lead": None,
        "lead_count": 0,
        "lean_faster": 0.0,
        "lean_slower": 0.0,
    }


def ensure_state(state: dict | None, pace: str = "normal") -> dict:
    """Return a usable estimator state, filling gaps from older stored sessions."""
    base = initial_state(pace)
    if not isinstance(state, dict):
        return base
    merged = {**base, **state}
    belief = merged.get("belief")
    if not isinstance(belief, list) or len(belief) != len(PACES):
        merged["belief"] = list(base["prior"])
    return merged


def _sides(belief: list[float], current: str) -> tuple[float, float, float]:
    index = PACES.index(current) if current in PACES else 1
    return sum(belief[:index], 0.0), belief[index], sum(belief[index + 1 :], 0.0)


def _signal(
    *,
    correct: bool | None,
    used_hint: bool,
    asked_answer: bool,
    success_streak: int,
    failure_streak: int,
    cues: dict | None,
) -> float:
    """Signed weak evidence: negative leans concise, positive leans detailed."""
    signal = 0.0
    if correct is True and not used_hint and success_streak >= 2:
        signal -= W_STREAK
    if correct is False and failure_streak >= 2:
        signal += W_FAIL
    if used_hint:
        signal += W_HINT
    if asked_answer:
        signal += W_ANSWER
    if isinstance(cues, dict):
        seconds = cues.get("reply_seconds")
        if isinstance(seconds, int | float) and not isinstance(seconds, bool):
            if seconds < FAST_SECONDS:
                signal -= W_CUE
            elif seconds > SLOW_SECONDS:
                signal += W_CUE
        words = cues.get("message_words")
        if isinstance(words, int | float) and not isinstance(words, bool):
            if words <= SHORT_WORDS:
                signal -= W_CUE
            elif words >= LONG_WORDS:
                signal += W_CUE
        if cues.get("skim"):
            signal -= W_CUE
        if cues.get("confusion"):
            signal += W_CUE
        if cues.get("why_question"):
            signal += W_CUE
    return signal


def _direction(belief: list[float], current: str) -> str | None:
    faster, here, slower = _sides(belief, current)
    if faster > here and faster > slower:
        return "faster"
    if slower > here and slower > faster:
        return "slower"
    return None


def update(
    state: dict | None,
    *,
    request: str | None = None,
    correct: bool | None = None,
    used_hint: bool = False,
    asked_answer: bool = False,
    success_streak: int = 0,
    failure_streak: int = 0,
    cues: dict | None = None,
    current: str | None = None,
) -> dict:
    """Fold one learner turn into the estimator and return the new state."""
    current = current or (state or {}).get("pace", "normal")
    old = ensure_state(state, current)
    if current not in PACES:
        current = "normal"
    prior = old["prior"]
    belief = list(old["belief"])
    request = request if request in _STEP else None
    if request is not None:
        target = _onehot(step_pace(current, request))
        belief = [
            (1 - REQUEST_MASS) * value + REQUEST_MASS * hot
            for value, hot in zip(belief, target, strict=True)
        ]
    signal = _signal(
        correct=correct,
        used_hint=used_hint,
        asked_answer=asked_answer,
        success_streak=success_streak,
        failure_streak=failure_streak,
        cues=cues,
    )
    if signal:
        weights = [math.exp(-GAIN * signal), 1.0, math.exp(GAIN * signal)]
        belief = _normalize([value * weight for value, weight in zip(belief, weights, strict=True)])
    belief = _normalize(
        [(1 - DECAY) * value + DECAY * base for value, base in zip(belief, prior, strict=True)]
    )
    direction = request or _direction(belief, current)
    if request is not None:
        lead, lead_count = None, 0
    elif direction is None:
        lead, lead_count = None, 0
    elif direction == old.get("lead"):
        lead, lead_count = direction, int(old.get("lead_count", 0)) + 1
    else:
        lead, lead_count = direction, 1
    faster, _, slower = _sides(belief, current)
    return {
        "prior": list(prior),
        "belief": belief,
        "pace": current,
        "request": request,
        "lead": lead,
        "lead_count": lead_count,
        "lean_faster": 1.0 if request == "faster" else faster,
        "lean_slower": 1.0 if request == "slower" else slower,
    }


def select(state: dict | None, current: str) -> tuple[str, dict]:
    """Choose the next pace with hysteresis and return it with the updated state."""
    state = ensure_state(state, current)
    request = state.get("request")
    if request in _STEP:
        pace = step_pace(current, request)
        return pace, {**state, "pace": pace, "lead": None, "lead_count": 0}
    lead = state.get("lead")
    if lead in _STEP and int(state.get("lead_count", 0)) >= HOLD:
        pace = step_pace(current, lead)
        if pace != current:
            faster, _, slower = _sides(state["belief"], pace)
            return pace, {
                **state,
                "pace": pace,
                "lead": None,
                "lead_count": 0,
                "lean_faster": faster,
                "lean_slower": slower,
            }
    return current, {**state, "pace": current}


def choose_pace(state: dict | None, current: str) -> str:
    return select(state, current)[0]


def lean_faster(state: dict | None) -> float:
    return float(min(1.0, max(0.0, (state or {}).get("lean_faster", 0.0))))


def lean_slower(state: dict | None) -> float:
    return float(min(1.0, max(0.0, (state or {}).get("lean_slower", 0.0))))
