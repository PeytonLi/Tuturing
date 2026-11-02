"""Simulated per-turn pace cues for the learner's reply.

Assumptions set for this prototype; these are not empirically calibrated effects.
"""

from __future__ import annotations

import math

from packages.learning.actions import PACES
from packages.simulator.randomness import EventRandomBank
from packages.simulator.student import PrivateStudentState

PACE_CUES_STREAM = "pace_cues"
DEFAULT_PACE_CUES: dict = {
    "strength": 1.0,
    "mismatch_weight": 0.5,
    "reply_seconds": {"median": 14.0, "sigma": 0.55, "log_shift": 0.35},
    "message_words": {"mean": 6.0, "log_shift": 0.4, "skim_max_words": 2},
    "skim": {"base_logit": -1.6, "slope": 0.9},
    "confusion": {"base_logit": -2.0, "slope": 0.8},
    "why_question": {"base_logit": -2.2, "slope": 0.8},
}


def pace_cue_parameters(config: dict | None) -> dict:
    """Return the pace_cues block merged over code defaults (one level deep)."""
    supplied = (config or {}).get("pace_cues") or {}
    merged: dict = {}
    for key, default in DEFAULT_PACE_CUES.items():
        value = supplied.get(key, default)
        merged[key] = {**default, **value} if isinstance(default, dict) else float(value)
    return merged


def detail_drive(preferred_pace: str, tutor_pace: str, mismatch_weight: float) -> float:
    """Latent pull toward more detail: negative favours concise, positive favours detailed.

    The preferred pace sets the baseline; a tutor pace longer than preferred pushes the learner
    further toward concise cues, and a shorter one pushes toward detailed cues.
    """
    for value in (preferred_pace, tutor_pace):
        if value not in PACES:
            raise ValueError(f"unknown pace {value!r}; expected one of {PACES}")
    preferred = PACES.index(preferred_pace)
    tutor = PACES.index(tutor_pace)
    return (preferred - 1) + mismatch_weight * (preferred - tutor)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def sample_pace_cues(
    student: PrivateStudentState,
    pace: str,
    *,
    scenario: dict,
    episode_seed: int,
    turn: int,
    bank: EventRandomBank,
    config: dict,
) -> dict:
    """Sample the pace cue dict for one learner turn after the tutor used ``pace``."""
    parameters = pace_cue_parameters(config)
    strength = parameters["strength"]
    drive = strength * detail_drive(student.preferred_pace, pace, parameters["mismatch_weight"])
    rng = bank.rng(scenario["scenario_id"], episode_seed, turn, PACE_CUES_STREAM)

    seconds = parameters["reply_seconds"]
    reply_seconds = float(
        rng.lognormal(
            math.log(float(seconds["median"])) + float(seconds["log_shift"]) * drive,
            float(seconds["sigma"]),
        )
    )
    words = parameters["message_words"]
    word_rate = max(0.1, float(words["mean"]) * math.exp(float(words["log_shift"]) * drive))
    message_words = 1 + int(rng.poisson(word_rate))
    flags = {}
    for name, sign in (("skim", -1.0), ("confusion", 1.0), ("why_question", 1.0)):
        flag = parameters[name]
        probability = _sigmoid(float(flag["base_logit"]) + sign * float(flag["slope"]) * drive)
        flags[name] = bool(rng.random() < probability)
    if flags["skim"]:
        message_words = min(message_words, int(words["skim_max_words"]))
    return {
        "reply_seconds": round(reply_seconds, 3),
        "message_words": message_words,
        "skim": flags["skim"],
        "confusion": flags["confusion"],
        "why_question": flags["why_question"],
    }
