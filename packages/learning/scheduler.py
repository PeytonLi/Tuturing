from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def boundary_reasons(context: dict) -> list[str]:
    reasons: set[str] = set()
    if context.get("session_start"):
        reasons.add("session_start")
    if int(context.get("option_turns", 0)) >= 4:
        reasons.add("option_duration")
    if int(context.get("independent_success_streak", 0)) >= 2:
        reasons.add("independent_successes")
    if int(context.get("independent_failure_streak", 0)) >= 2:
        reasons.add("independent_failures")
    if context.get("explicit_skip"):
        reasons.add("explicit_skip")
    if context.get("explicit_goal_change"):
        reasons.add("goal_change")
    if context.get("content_family_exhausted"):
        reasons.add("family_exhausted")
    return sorted(reasons)


@dataclass
class GoalStack:
    maximum_depth: int = 4
    cooldown_visits: int = 2
    stack: list[str] = field(default_factory=list)
    visit_index: int = 0
    last_pair_visit: dict[tuple[str, str], int] = field(default_factory=dict)
    pair_visits: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def visit_pair(self, source: str, target: str) -> bool:
        pair = (source, target)
        previous = self.last_pair_visit.get(pair, -self.cooldown_visits)
        allowed = self.visit_index - previous >= self.cooldown_visits
        self.visit_index += 1
        if allowed:
            self.last_pair_visit[pair] = self.visit_index - 1
            self.pair_visits[pair] += 1
        return allowed

    def push(self, goal: str, source: str | None = None) -> bool:
        if len(self.stack) >= self.maximum_depth:
            return False
        if source is not None and not self.visit_pair(source, goal):
            return False
        self.stack.append(goal)
        return True

    def pop(self) -> str | None:
        return self.stack.pop() if self.stack else None


def skip_diagnostic_due(context: dict, turn_index: int) -> bool:
    deadline = context.get("skip_diagnostic_deadline")
    return (
        deadline is not None
        and turn_index <= int(deadline)
        and not context.get("skip_diagnostic_done", False)
    )
