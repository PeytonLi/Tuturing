from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GAEOutput:
    advantages: list[float]
    returns: list[float]
    deltas: list[float]


def duration_aware_gae(
    rewards: list[float],
    values: list[float],
    durations: list[int],
    next_values: list[float],
    terminated: list[bool],
    truncated: list[bool] | None = None,
    continuation: list[bool] | None = None,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> GAEOutput:
    count = len(rewards)
    if not all(len(items) == count for items in (values, durations, next_values, terminated)):
        raise ValueError("all transition arrays must have equal length")
    truncated = truncated or [False] * count
    continuation = continuation or [index + 1 < count for index in range(count)]
    if len(truncated) != count or len(continuation) != count:
        raise ValueError("all transition arrays must have equal length")
    if any(duration < 0 for duration in durations):
        raise ValueError("transition durations cannot be negative")
    deltas = [0.0] * count
    advantages = [0.0] * count
    for index in range(count):
        duration = durations[index]
        bootstrap = 0.0 if terminated[index] else gamma**duration * next_values[index]
        deltas[index] = rewards[index] + bootstrap - values[index]
    for index in range(count - 1, -1, -1):
        duration = durations[index]
        can_chain = continuation[index] and not terminated[index] and not truncated[index]
        next_advantage = advantages[index + 1] if can_chain and index + 1 < count else 0.0
        advantages[index] = (
            deltas[index] + ((gamma * lam) ** duration) * next_advantage
            if can_chain
            else deltas[index]
        )
    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return GAEOutput(advantages, returns, deltas)
