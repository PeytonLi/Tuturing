from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import yaml

from packages.learning.actions import (
    decode_navigator,
    decode_strategist,
    encode_strategist,
    navigator_mask,
    strategist_mask,
)
from packages.learning.belief import ConceptBelief, update_belief
from packages.learning.content.templates.families import TemplateFamily, families_for
from packages.learning.features import build_features
from packages.learning.grader import grade_answer
from packages.learning.scheduler import GoalStack, boundary_reasons, skip_diagnostic_due
from packages.policies.rules import (
    AlwaysHintStrategist,
    AlwaysWorkedExampleStrategist,
    RandomStrategist,
    RulesNavigator,
    RulesStrategist,
)
from packages.simulator.randomness import EventRandomBank
from packages.simulator.response_adapter import realize_plan
from packages.simulator.reward import (
    RewardComponents,
    expected_score,
    terminal_components,
    turn_components,
)
from packages.simulator.scenarios import ROOT, load_scenarios
from packages.simulator.student import (
    DEFAULT_SIMULATOR_CONFIG,
    PrivateStudentState,
    StudentOutcome,
    apply_teaching_effect,
    dropout_hazard,
    dropout_hazard_components,
    probability_correct,
    sample_student,
    sample_student_outcome,
)

HORIZON = 16
GAMMA = 0.99
ENABLED_CONCEPTS = ("C01", "C02", "C03", "C06", "C08")


def load_graph() -> dict:
    document = yaml.safe_load(
        (ROOT / "data/original_content/concepts.yaml").read_text(encoding="utf-8")
    )
    concepts = {item["concept_id"]: item for item in document["concepts"]}
    edges = document["graphs"]["graph_proto_v1"]
    prerequisites: dict[str, list[str]] = {}
    successors: dict[str, list[str]] = {}
    for edge in edges:
        source = edge["source_concept_id"]
        target = edge["dependent_concept_id"]
        prerequisites.setdefault(target, []).append(source)
        successors.setdefault(source, []).append(target)
    ancestors: dict[str, list[str]] = {}
    for concept_id in concepts:
        found: set[str] = set()
        pending = list(prerequisites.get(concept_id, []))
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent)
                pending.extend(prerequisites.get(parent, []))
        ancestors[concept_id] = sorted(found)
    return {
        "enabled_concepts": list(ENABLED_CONCEPTS),
        "prerequisites": prerequisites,
        "successors": successors,
        "ancestors": ancestors,
        "concepts": concepts,
    }


@dataclass
class EpisodeResult:
    episode_id: str
    scenario_id: str
    profile_id: str
    split: str
    end_reason: str
    turns: int
    gain: float
    pre_score: float
    post_score: float
    dropout: bool
    move_histogram: dict[str, int]
    strategist_transitions: list[dict]
    navigator_transitions: list[dict]
    private_diagnostics: list[dict]
    terminal_components: dict[str, float]
    final_beliefs: dict[str, ConceptBelief] = field(repr=False)
    assessment_family_reuse_count: int = 0

    @property
    def transitions(self) -> list[dict]:
        return [*self.strategist_transitions, *self.navigator_transitions]


def _scenario_requests(scenario: dict) -> dict:
    requests = {
        "skip": False,
        "faster": False,
        "slower": False,
        "stop": False,
        "answer": False,
        "goal": None,
    }
    for evidence in scenario.get("public_history", []):
        request = evidence.get("explicit_request")
        if request == "skip":
            requests["skip"] = True
            requests["goal"] = evidence.get("requested_concept_id")
        elif request in {"faster", "slower", "stop", "answer", "change_goal"}:
            requests[request] = True
            if request == "change_goal":
                requests["goal"] = evidence.get("requested_concept_id")
    return requests


def update_public_misconceptions(
    current: dict[str, float],
    *,
    family: TemplateFamily,
    instance: dict,
    normalized_answer: Fraction | str | int | float | None,
    correctness: str,
    move: str,
) -> dict[str, float]:
    updated = dict(current)
    if correctness == "incorrect" and normalized_answer is not None:
        answer = Fraction(normalized_answer)
        for misconception in sorted(family.error_generators):
            if answer != family.generate_error(misconception, instance):
                continue
            if move == "PROBE" or updated.get(misconception, 0.0) >= 0.5:
                updated[misconception] = 0.85
            else:
                updated[misconception] = max(updated.get(misconception, 0.0), 0.6)
    elif correctness == "correct":
        for misconception in family.error_generators:
            if misconception not in updated:
                continue
            updated[misconception] /= 2
            if updated[misconception] < 0.2:
                del updated[misconception]
    return updated


def _public_state(
    beliefs: dict[str, ConceptBelief],
    *,
    current: str,
    goal: str,
    turn: int,
    active_item: dict | None,
    active_option_type: str | None,
    active_target: str | None,
    previous_move: str,
    option_turns: int,
    success_streak: int,
    failure_streak: int,
    hint_count: int,
    explicit_skip: bool,
    explicit_faster: bool,
    explicit_slower: bool,
    stack: list[str],
    misconceptions: dict[str, float] | None = None,
) -> dict:
    concepts = []
    for index in range(1, 16):
        concept_id = f"C{index:02d}"
        belief = beliefs[concept_id]
        concepts.append(
            {
                "mastery": belief.mastery,
                "uncertainty": belief.uncertainty,
                "independent_correct": belief.independent_correct,
                "independent_incorrect": belief.independent_incorrect,
            }
        )
    return {
        "concepts": concepts,
        "misconceptions": [
            float((misconceptions or {}).get(f"M{index:02d}", 0.0))
            for index in range(1, 26)
        ],
        "current_concept_id": current,
        "goal_concept_id": goal,
        "active_target_concept_id": active_target,
        "active_option_type": active_option_type,
        "previous_move": previous_move,
        "turn": turn,
        "turns_in_option": option_turns,
        "hint_count": hint_count,
        "independent_success_streak": success_streak,
        "remaining_budget": max(0, HORIZON - turn),
        "explicit_skip": explicit_skip,
        "explicit_faster": explicit_faster,
        "explicit_slower": explicit_slower,
        "response_failure": False,
        "stack_depth": len(stack),
        "goal_stack": list(stack),
        "active_item": active_item is not None,
        "active_item_difficulty": active_item["family"].difficulty if active_item else "medium",
        "problem_instance_id": active_item["item_id"] if active_item else None,
        "mastery": beliefs[current].mastery,
        "mastery_by_concept": {concept_id: beliefs[concept_id].mastery for concept_id in beliefs},
    }


def _concept_graph_context(graph: dict, beliefs: dict[str, ConceptBelief], goal: str) -> dict:
    return {
        "enabled_concepts": graph["enabled_concepts"],
        "operations": {
            "successors": graph["successors"],
            "ancestors": graph["ancestors"],
        },
        "goal_ancestors": set(graph["ancestors"].get(goal, [])),
        "successors": graph["successors"],
    }


def _weakest_ancestor(current: str, graph: dict, beliefs: dict[str, ConceptBelief]) -> str | None:
    candidates = [
        concept_id
        for concept_id in graph["ancestors"].get(current, [])
        if concept_id in ENABLED_CONCEPTS and beliefs[concept_id].mastery < 0.80
    ]
    return (
        min(candidates, key=lambda concept_id: (beliefs[concept_id].mastery, concept_id))
        if candidates
        else None
    )


def _active_context(
    public: dict,
    *,
    current: str,
    goal: str,
    beliefs: dict[str, ConceptBelief],
    active_item: dict | None,
    stack: GoalStack,
    graph: dict,
    requests: dict,
    success_streak: int,
    failure_streak: int,
    meaningful_error: bool,
    unsuccessful_hints: int,
    no_diagnostic_yet: bool,
    skip_deadline: int | None,
    skip_diagnostic_done: bool,
    turn: int,
    pace: str,
    allow_policy_stop: bool,
) -> dict:
    return {
        **public,
        **_concept_graph_context(graph, beliefs, goal),
        "goal_concept_id": goal,
        "current_concept_id": current,
        "goal_stack": list(stack.stack),
        "weakest_ancestor": _weakest_ancestor(current, graph, beliefs),
        "mastery": beliefs[current].mastery,
        "mastery_by_concept": {
            concept_id: belief.mastery for concept_id, belief in beliefs.items()
        },
        "recent_independent_correct": bool(
            beliefs[current].recent_independent and beliefs[current].recent_independent[-1][0]
        ),
        "active_item": active_item is not None,
        "problem_instance_id": active_item["item_id"] if active_item else None,
        "active_item_difficulty": active_item["family"].difficulty if active_item else "medium",
        "remaining_hints": bool(active_item and active_item["hint_level"] < 3),
        "hint_level": active_item["hint_level"] if active_item else 0,
        "previous_step": public.get("previous_move") not in (None, "NONE"),
        "pace_preference": pace,
        "explicit_skip": bool(requests["skip"]),
        "explicit_goal_change": bool(requests.get("goal_change", False)),
        "goal_certified": beliefs[goal].certified,
        "current_certified": beliefs[current].certified,
        "remediation_certified": bool(stack.stack and beliefs[current].certified),
        "independent_success_streak": success_streak,
        "independent_failure_streak": failure_streak,
        "meaningful_error": meaningful_error,
        "unsuccessful_hint_count": unsuccessful_hints,
        "no_diagnostic_yet": no_diagnostic_yet,
        "unsupported_advancement": False,
        "wants_certification": False,
        "turn_index": turn,
        "allow_policy_stop": allow_policy_stop,
        "learner_stop": bool(requests.get("stop")),
        "skip_diagnostic_deadline": skip_deadline,
        "skip_diagnostic_done": skip_diagnostic_done,
        "skip_diagnostic_due": skip_diagnostic_due(
            {
                "skip_diagnostic_deadline": skip_deadline,
                "skip_diagnostic_done": skip_diagnostic_done,
            },
            turn,
        ),
    }


def _next_item(
    concept_id: str,
    move: str,
    scenario: dict,
    episode_seed: int,
    turn: int,
    bank: EventRandomBank,
    assessment_family_uses: Counter[str],
    assessment_family_reuse_counts: Counter[str],
    used_teaching_families: set[str],
) -> tuple[TemplateFamily, dict, str]:
    allowlist = set(scenario["template_family_allowlist"])
    is_assessment = move in {"PROBE", "CHECK_MASTERY"}
    if is_assessment:
        assessment_split = "train" if scenario["split"] == "train" else scenario["split"]
        families = [
            family
            for family in families_for(concept_id, role="assessment", split=assessment_split)
            if family.template_family_id in allowlist
        ]
        if not families:
            raise ValueError(f"assessment family exhausted for {concept_id} ({scenario['split']})")
        unused_families = [
            family
            for family in families
            if assessment_family_uses[family.template_family_id] == 0
        ]
        if unused_families:
            selected = min(unused_families, key=lambda family: family.template_family_id)
        else:
            selected = min(
                families,
                key=lambda family: (
                    assessment_family_uses[family.template_family_id],
                    family.template_family_id,
                ),
            )
            assessment_family_reuse_counts[concept_id] += 1
        assessment_family_uses[selected.template_family_id] += 1
    else:
        selected = None
        families = [
            family
            for family in families_for(concept_id, role="teaching", split=scenario["split"])
            if family.template_family_id in allowlist
            and family.template_family_id not in used_teaching_families
        ]
        if not families:
            families = [
                family
                for family in families_for(concept_id, role="teaching", split=scenario["split"])
                if family.template_family_id in allowlist
            ]
        if not families:
            families = families_for(concept_id, role="teaching", split="train")
        if not families:
            raise ValueError(f"teaching family unavailable for {concept_id}")
        choice_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "family_choice")
        selected = families[int(choice_rng.integers(0, len(families)))]
    item_rng = bank.rng(scenario["scenario_id"], episode_seed, turn, "item_parameters")
    item_seed = int(item_rng.integers(0, 2**31))
    instance = selected.sample(item_seed)
    item_id = f"{scenario['scenario_id']}-turn-{turn:02d}-item"
    return selected, instance, item_id


def _problem_stem(family: TemplateFamily, instance: dict) -> str:
    numerator = instance["numerator"]
    denominator = instance["denominator"]
    if family.item_kind == "tree":
        return (
            f"A probability tree path has branch probabilities {numerator}/{denominator}. "
            "Find the path probability."
        )
    if family.item_kind == "two_way_table":
        return (
            f"A two-way table has {numerator} outcomes in the intersection and "
            f"{denominator} in the relevant group. Find the probability."
        )
    return (
        f"A {family.item_kind} experiment has {numerator} favorable outcomes out of "
        f"{denominator} equally likely outcomes. Find the probability."
    )


def _build_plan(
    *,
    scenario: dict,
    episode_id: str,
    turn: int,
    current: str,
    goal: str,
    navigation_action_id: int,
    teaching_action_id: int,
    active_item: dict | None,
    beliefs: dict[str, ConceptBelief],
    hint_level: int,
    previous_step: str | None,
    assessment_independent: bool,
    actor_eligible: bool,
    forced_answer: bool,
) -> dict:
    move, difficulty, pace = decode_strategist(teaching_action_id)
    navigation_operation, target_index = decode_navigator(navigation_action_id)
    target_concept_id = f"C{target_index + 1:02d}" if target_index is not None else None
    concept_name = current
    if move in {"PROBE", "PRACTICE", "CHALLENGE", "CHECK_MASTERY", "HINT"} and active_item:
        instance = active_item["instance"]
        family = active_item["family"]
        problem_stem = _problem_stem(family, instance)
        family_id = family.template_family_id
        item_id = active_item["item_id"]
    else:
        problem_stem = None
        family_id = None
        item_id = None
    family = active_item["family"] if active_item else None
    content: dict = {
        "concept_name": concept_name,
        "goal_name": goal,
        "problem_stem": problem_stem,
        "approved_facts": list(family.approved_facts)
        if family
        else ["Probability is a ratio between zero and one."],
        "permitted_hint": family.hint_ladder[min(hint_level, 2)]
        if move == "HINT" and family
        else None,
        "worked_example": None,
        "reveal_solution": None,
        "previous_step": previous_step if move == "REFLECT" else None,
    }
    if move == "WORKED_EXAMPLE":
        example_family = (
            family or families_for(current, role="teaching", split=scenario["split"])[0]
        )
        sample_seed = int(
            hashlib.sha256(
                f"{scenario['scenario_id']}:{episode_id}:{turn}:example".encode()
            ).hexdigest()[:8],
            16,
        )
        instance = example_family.sample(sample_seed)
        example = example_family.worked_example(sample_seed, instance["answer"])
        content["worked_example"] = f"{example['prompt']} The answer is {example['answer']}."
    if forced_answer and not assessment_independent:
        content["reveal_solution"] = (
            f"{active_item['instance']['answer'].numerator}/{active_item['instance']['answer'].denominator}"
            if active_item
            else "See the worked solution."
        )
    reveal_allowed = bool(forced_answer and not assessment_independent)
    assistance = (
        "hint"
        if move == "HINT"
        else "example"
        if move == "WORKED_EXAMPLE"
        else "answer_shown"
        if reveal_allowed
        else "none"
    )
    if assessment_independent:
        assistance = "none"
    return {
        "schema_version": "1.0",
        "plan_id": f"{episode_id}-plan-{turn:02d}",
        "session_id": episode_id,
        "trace_id": f"{episode_id}-trace-{turn:02d}",
        "based_on_state_version": turn,
        "option_id": f"{episode_id}-option-{navigation_action_id}-{turn // 4}",
        "navigation_action": {
            "action_id": navigation_action_id,
            "operation": navigation_operation,
            "target_concept_id": target_concept_id,
        },
        "teaching_action": {
            "action_id": teaching_action_id,
            "move": move,
            "difficulty": difficulty,
            "pace": pace,
        },
        "actor_eligible": actor_eligible,
        "concept_id": current,
        "goal_concept_id": goal,
        "problem_instance_id": item_id,
        "template_family_id": family_id,
        "assessment_mode": "independent"
        if assessment_independent
        else "practice"
        if move in {"PROBE", "PRACTICE", "CHALLENGE"}
        else "none",
        "assistance": assistance,
        "content_bundle_id": f"bundle-{family_id or current}",
        "content": content,
        "constraints": {
            "reveal_allowed": reveal_allowed,
            "max_words": {"concise": 80, "normal": 150, "detailed": 250}[pace],
            "max_questions": 1,
            "must_request_student_attempt": move
            in {"PROBE", "HINT", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"},
            "forbidden_answer_fingerprints": [],
        },
        "response_context": {
            "last_student_message": "",
            "recent_dialogue": [],
            "qualified_misconception_summary": None,
        },
        "versions": {
            "graph": "graph_proto_v1",
            "content": "content_proto_v1",
            "state_estimator": "bkt_v1",
            "navigator": "rules_navigator_v1",
            "strategist": "rules_strategist_v3",
            "mask": "mask_v1",
            "feature": "features_v1",
        },
        "expires_at": "2026-11-02T00:00:00Z",
    }


def _navigator_graph(
    graph: dict, goal: str, current: str, beliefs: dict[str, ConceptBelief]
) -> dict:
    return {
        "enabled_concepts": graph["enabled_concepts"],
        "operations": {
            "successors": graph["successors"],
            "ancestors": graph["ancestors"],
            "weakest_ancestor": _weakest_ancestor(current, graph, beliefs),
        },
    }


def _navigator_observation(public: dict) -> list[float]:
    return build_features(public, navigator=True)


def _make_navigator_transition(
    *,
    run_id: str,
    episode_id: str,
    option_id: str,
    start_turn: int,
    end_turn: int,
    observation: list[float],
    action_mask: list[bool],
    action_id: int,
    discounted_reward: float,
    next_observation: list[float],
    terminated: bool,
    truncated: bool,
    boundary: list[str],
    environment_bundle_id: str,
) -> dict:
    duration = end_turn - start_turn
    return {
        "run_id": run_id,
        "episode_id": episode_id,
        "transition_id": f"{episode_id}-navigator-{start_turn:02d}-{end_turn:02d}",
        "option_id": option_id,
        "start_turn": start_turn,
        "end_turn": end_turn,
        "observation": observation,
        "action_mask": action_mask,
        "action_id": action_id,
        "old_log_prob": 0.0,
        "old_value": 0.0,
        "duration": duration,
        "discounted_option_reward": discounted_reward,
        "next_observation": next_observation,
        "terminated": terminated,
        "truncated": truncated,
        "actor_eligible": False,
        "boundary_reasons": list(boundary),
        "policy_version": "rules_navigator_v1",
        "environment_bundle_id": environment_bundle_id,
    }


def run_episode(
    scenario: dict,
    *,
    run_id: str = "local",
    episode_seed: int = 0,
    strategist_name: str = "rules",
    strategist: object | None = None,
    horizon: int = HORIZON,
    bank: EventRandomBank | None = None,
    simulator_config: str | Path = DEFAULT_SIMULATOR_CONFIG,
) -> EpisodeResult:
    if not 1 <= horizon <= HORIZON:
        raise ValueError("episode horizon must be between one and sixteen turns")
    random_bank = bank or EventRandomBank()
    graph = load_graph()
    sim_config = yaml.safe_load(
        (ROOT / simulator_config).read_text(encoding="utf-8")
    )
    environment_bundle_id = Path(simulator_config).stem
    private = sample_student(
        scenario, episode_seed, config_path=simulator_config, bank=random_bank
    )
    beliefs = {f"C{index:02d}": ConceptBelief() for index in range(1, 16)}
    requests = _scenario_requests(scenario)
    public_misconceptions: dict[str, float] = {}
    if requests["goal"] in ENABLED_CONCEPTS:
        goal = requests["goal"]
    else:
        goal = scenario["goal_concept_id"]
    current = scenario["start_concept_id"]
    pace = "concise" if requests["faster"] else "detailed" if requests["slower"] else "normal"
    stack = GoalStack()
    active_item: dict | None = None
    previous_move = "NONE"
    previous_step = None
    success_streak = failure_streak = option_turns = hint_count = 0
    unsuccessful_hints = 0
    meaningful_error = False
    no_diagnostic_yet = True
    skip_deadline: int | None = None
    skip_diagnostic_done = False
    assessment_family_uses: Counter[str] = Counter()
    assessment_family_reuse_counts: Counter[str] = Counter()
    used_teaching_families: set[str] = set()
    repeated_pattern_evidence: dict[tuple[str, ...], int] = {}
    learner_evidence_count = 0
    if strategist is None:
        if strategist_name == "rules":
            strategist = RulesStrategist()
        elif strategist_name == "random":
            strategist = RandomStrategist(episode_seed)
        elif strategist_name == "always_hint":
            strategist = AlwaysHintStrategist()
        elif strategist_name == "always_worked_example":
            strategist = AlwaysWorkedExampleStrategist()
        else:
            raise ValueError(f"unknown strategist policy {strategist_name}")
    navigator = RulesNavigator()
    episode_id = f"{scenario['scenario_id']}-seed-{episode_seed}"
    pre_score = expected_score(
        private, goal, scenario["split"], config_path=simulator_config
    )
    initial_public = _public_state(
        beliefs,
        current=current,
        goal=goal,
        turn=0,
        active_item=None,
        active_option_type=None,
        active_target=None,
        previous_move=previous_move,
        option_turns=0,
        success_streak=0,
        failure_streak=0,
        hint_count=0,
        explicit_skip=requests["skip"],
        explicit_faster=requests["faster"],
        explicit_slower=requests["slower"],
        stack=stack.stack,
        misconceptions=public_misconceptions,
    )
    navigator_context = _active_context(
        initial_public,
        current=current,
        goal=goal,
        beliefs=beliefs,
        active_item=None,
        stack=stack,
        graph=graph,
        requests=requests,
        success_streak=0,
        failure_streak=0,
        meaningful_error=False,
        unsuccessful_hints=0,
        no_diagnostic_yet=True,
        skip_deadline=None,
        skip_diagnostic_done=False,
        turn=0,
        pace=pace,
        allow_policy_stop=False,
    )
    navigator_state = {
        **initial_public,
        "enabled_concepts": graph["enabled_concepts"],
        "goal_ancestors": graph["ancestors"].get(goal, []),
        "goal_stack": list(stack.stack),
        "goal_certified": beliefs[goal].certified,
        "allow_policy_stop": False,
        "learner_stop": requests["stop"],
    }
    nav_mask = navigator_mask(navigator_state, _navigator_graph(graph, goal, current, beliefs))
    nav_observation = _navigator_observation(initial_public)
    nav_action = navigator.act(nav_observation, nav_mask, navigator_context)
    option_id = f"{episode_id}-option-0"
    pending_navigator = {
        "option_id": option_id,
        "start_turn": 0,
        "observation": nav_observation,
        "mask": nav_mask,
        "action_id": nav_action,
        "discounted_reward": 0.0,
        "boundary": ["session_start"],
    }
    navigator_transitions: list[dict] = []
    strategist_transitions: list[dict] = []
    diagnostics: list[dict] = []
    moves = Counter()
    turns = 0
    end_reason = "teaching_budget"
    dropout = False
    terminal_reward_components = RewardComponents()
    terminal_added = False

    def apply_navigation(action_id: int, turn: int) -> bool:
        nonlocal current, skip_deadline, skip_diagnostic_done
        operation, target_index = decode_navigator(action_id)
        target = f"C{target_index + 1:02d}" if target_index is not None else None
        if operation == "END_SESSION":
            return True
        if operation == "SKIP_TO_TARGET" and target:
            current = target
            skip_deadline = turn + 2
            skip_diagnostic_done = False
            requests["skip"] = False
        elif operation == "ADVANCE_ONE" and target:
            current = target
        elif operation == "REVIEW_PREREQUISITE" and target:
            if stack.push(scenario["goal_concept_id"], current):
                current = target
        elif operation == "RETURN_TO_GOAL" and target:
            if stack.stack:
                stack.pop()
            current = target
        return False

    immediate_end = apply_navigation(nav_action, 0)
    if immediate_end:
        end_reason = (
            "learner_stop"
            if requests["stop"]
            else "demonstrated_completion"
            if beliefs[goal].certified
            else "policy_stop"
        )
        pre_score_post = expected_score(
            private, goal, scenario["split"], config_path=simulator_config
        )
        terminal_reward_components = terminal_components(pre_score, pre_score_post)
        navigator_transitions.append(
            _make_navigator_transition(
                run_id=run_id,
                episode_id=episode_id,
                option_id=option_id,
                start_turn=0,
                end_turn=0,
                observation=nav_observation,
                action_mask=nav_mask,
                action_id=nav_action,
                discounted_reward=terminal_reward_components.total,
                next_observation=nav_observation,
                terminated=True,
                truncated=False,
                boundary=["session_start"],
                environment_bundle_id=environment_bundle_id,
            )
        )
        terminal_added = True

    for turn in range(horizon):
        if immediate_end:
            break
        turns = turn + 1
        public = _public_state(
            beliefs,
            current=current,
            goal=goal,
            turn=turn,
            active_item=active_item,
            active_option_type=decode_navigator(pending_navigator["action_id"])[0],
            active_target=(
                f"C{decode_navigator(pending_navigator['action_id'])[1] + 1:02d}"
                if decode_navigator(pending_navigator["action_id"])[1] is not None
                else None
            ),
            previous_move=previous_move,
            option_turns=option_turns,
            success_streak=success_streak,
            failure_streak=failure_streak,
            hint_count=hint_count,
            explicit_skip=requests["skip"],
            explicit_faster=requests["faster"],
            explicit_slower=requests["slower"],
            stack=stack.stack,
            misconceptions=public_misconceptions,
        )
        context = _active_context(
            public,
            current=current,
            goal=goal,
            beliefs=beliefs,
            active_item=active_item,
            stack=stack,
            graph=graph,
            requests=requests,
            success_streak=success_streak,
            failure_streak=failure_streak,
            meaningful_error=meaningful_error,
            unsuccessful_hints=unsuccessful_hints,
            no_diagnostic_yet=no_diagnostic_yet,
            skip_deadline=skip_deadline,
            skip_diagnostic_done=skip_diagnostic_done,
            turn=turn,
            pace=pace,
            allow_policy_stop=False,
        )
        if active_item:
            context["active_item_difficulty"] = active_item["family"].difficulty
        strategist_observation = build_features(public)
        strat_mask = strategist_mask(context)
        action_id = strategist.act(strategist_observation, strat_mask, context)
        move, difficulty, selected_pace = decode_strategist(action_id)
        actor_eligible = True
        if move in {"PROBE", "PRACTICE", "CHALLENGE", "CHECK_MASTERY"}:
            family, instance, item_id = _next_item(
                current,
                move,
                scenario,
                episode_seed,
                turn,
                random_bank,
                assessment_family_uses,
                assessment_family_reuse_counts,
                used_teaching_families,
            )
            active_item = {
                "family": family,
                "instance": instance,
                "item_id": item_id,
                "hint_level": 0,
            }
            if family.role == "teaching":
                used_teaching_families.add(family.template_family_id)
        elif move == "HINT" and active_item:
            active_item["hint_level"] = min(3, active_item["hint_level"] + 1)
            hint_count += 1
        elif move == "WORKED_EXAMPLE" and active_item is None:
            teaching_families = families_for(current, role="teaching", split=scenario["split"])
            selected = teaching_families[0]
            sample_rng = random_bank.rng(
                scenario["scenario_id"], episode_seed, turn, "example_item_parameters"
            )
            instance = selected.sample(int(sample_rng.integers(0, 2**31)))
            active_item = {
                "family": selected,
                "instance": instance,
                "item_id": f"{scenario['scenario_id']}-turn-{turn:02d}-example",
                "hint_level": 0,
            }
        independent_check = move == "CHECK_MASTERY"
        forced_answer = bool(requests.get("answer", False))
        if forced_answer:
            action_id = encode_strategist("EXPLAIN_CONCEPT", difficulty, selected_pace)
            move = "EXPLAIN_CONCEPT"
            actor_eligible = False
            requests["answer"] = False
        plan = _build_plan(
            scenario=scenario,
            episode_id=episode_id,
            turn=turn,
            current=current,
            goal=goal,
            navigation_action_id=pending_navigator["action_id"],
            teaching_action_id=action_id,
            active_item=active_item,
            beliefs=beliefs,
            hint_level=active_item["hint_level"] if active_item else 0,
            previous_step=previous_step,
            assessment_independent=independent_check,
            actor_eligible=actor_eligible,
            forced_answer=forced_answer,
        )
        patterns: list[tuple[str, ...]] = []
        has_item_action = move in {
            "PROBE",
            "HINT",
            "PRACTICE",
            "CHALLENGE",
            "CHECK_MASTERY",
        }
        if active_item and (has_item_action or forced_answer):
            patterns.append(("item", active_item["item_id"]))
            if move == "HINT":
                patterns.append(("hint", active_item["item_id"], str(active_item["hint_level"])))
        unproductive_repetition = any(
            pattern in repeated_pattern_evidence
            and learner_evidence_count <= repeated_pattern_evidence[pattern]
            for pattern in patterns
        )
        for pattern in patterns:
            repeated_pattern_evidence[pattern] = learner_evidence_count
        realized = realize_plan(
            plan, request_id=f"{episode_id}-request-{turn:02d}", generation_seed=episode_seed
        )
        if not all(realized["checks"].values()):
            raise AssertionError("response stub returned failed safety checks")
        item_moves = {"PROBE", "PRACTICE", "CHALLENGE", "CHECK_MASTERY", "HINT"}
        if (move in item_moves or (forced_answer and active_item is not None)) and active_item:
            item = active_item
            outcome = sample_student_outcome(
                private,
                scenario=scenario,
                episode_seed=episode_seed,
                turn=turn,
                concept_id=current,
                difficulty=difficulty,
                item_id=item["item_id"],
                assistance=plan["assistance"],
                family=item["family"],
                instance=item["instance"],
                graph=graph,
                bank=random_bank,
                config=sim_config,
            )
            grader_result = (
                grade_answer(outcome.normalized_answer, item["instance"], item["family"])
                if outcome.normalized_answer is not None
                else "unknown"
            )
            evidence_correctness = (
                grader_result if grader_result in {"correct", "incorrect"} else "unknown"
            )
            public_misconceptions = update_public_misconceptions(
                public_misconceptions,
                family=item["family"],
                instance=item["instance"],
                normalized_answer=outcome.normalized_answer,
                correctness=evidence_correctness,
                move=move,
            )
            if outcome.normalized_answer is not None:
                learner_evidence_count += 1
            family_id = item["family"].template_family_id
            update_belief(
                beliefs[current],
                evidence_correctness,
                assistance=outcome.assistance,
                move=move,
                family_id=family_id,
                event_id=f"{episode_id}-evidence-{turn:02d}",
            )
            if outcome.independent and evidence_correctness in {"correct", "incorrect"}:
                if evidence_correctness == "correct":
                    success_streak += 1
                    failure_streak = 0
                    private.consecutive_independent_incorrect = 0
                    meaningful_error = False
                else:
                    failure_streak += 1
                    success_streak = 0
                    private.consecutive_independent_incorrect += 1
                    meaningful_error = True
                private.last_answer_correct = evidence_correctness == "correct"
                private.last_answer_independent = True
                private.last_answer_item_id = item["item_id"]
            elif outcome.assistance != "none" and evidence_correctness in {"correct", "incorrect"}:
                unsuccessful_hints += 1
            if outcome.explicit_request == "hint":
                meaningful_error = True
            if move == "PROBE":
                no_diagnostic_yet = False
            if move == "CHECK_MASTERY" and skip_deadline is not None:
                skip_diagnostic_done = True
            if move == "HINT" and evidence_correctness == "correct":
                unsuccessful_hints = 0
            private.active_item_id = item["item_id"]
        else:
            outcome = StudentOutcome(
                "not_applicable", None, plan["assistance"], "none", None, False
            )
            evidence_correctness = "unknown"
            family_id = None
        mastery_before = dict(private.mastery)
        previous_answer_correct = private.last_answer_correct
        previous_answer_item = private.last_answer_item_id
        apply_teaching_effect(
            private,
            concept_id=current,
            move=move,
            difficulty=difficulty,
            turn=turn,
            scenario=scenario,
            episode_seed=episode_seed,
            graph=graph,
            bank=random_bank,
            config=sim_config,
            content_realized=all(realized["checks"].values()),
            item_id=active_item["item_id"] if active_item else None,
        )
        if outcome.independent and evidence_correctness in {"correct", "incorrect"}:
            private.last_answer_correct = evidence_correctness == "correct"
            private.last_answer_independent = True
            private.last_answer_item_id = active_item["item_id"] if active_item else None
        elif outcome.assistance != "none":
            private.last_answer_independent = False
        if move == "WORKED_EXAMPLE" and active_item and active_item["item_id"].endswith("-example"):
            active_item = None
        previous_step = f"{move} on {current}"
        previous_move = move
        moves[move] += 1
        option_turns += 1
        turn_reward = turn_components(
            disclosure=(
                plan["content"]["reveal_solution"] is not None
                and not plan["constraints"]["reveal_allowed"]
            ),
            unproductive_repetition=unproductive_repetition,
        )
        dropout_rng = random_bank.rng(scenario["scenario_id"], episode_seed, turn, "dropout")
        stop_rng = random_bank.rng(scenario["scenario_id"], episode_seed, turn, "stop_request")
        pace_mismatch = selected_pace != private.preferred_pace
        pace_request = None
        if pace_mismatch:
            pace_request_rng = random_bank.rng(
                scenario["scenario_id"], episode_seed, turn, "pace_mismatch_request"
            )
            if float(pace_request_rng.random()) < 0.5:
                if private.preferred_pace == "concise":
                    pace_request = "faster"
                elif private.preferred_pace == "detailed":
                    pace_request = "slower"
                else:
                    pace_request = "slower" if selected_pace == "concise" else "faster"
                requests["faster"] = pace_request == "faster"
                requests["slower"] = pace_request == "slower"
                pace = "concise" if pace_request == "faster" else "detailed"
        hazard_components = dropout_hazard_components(private, selected_pace, sim_config)
        hazard = dropout_hazard(private, selected_pace, sim_config)
        dropout = float(dropout_rng.random()) < hazard or float(stop_rng.random()) < hazard
        dropout_cause = None
        if dropout:
            cause_total = sum(hazard_components.values())
            cause_rng = random_bank.rng(
                scenario["scenario_id"], episode_seed, turn, "dropout_cause"
            )
            cause_draw = float(cause_rng.random()) * cause_total
            cumulative = 0.0
            for cause, probability in hazard_components.items():
                cumulative += probability
                if cause_draw < cumulative:
                    dropout_cause = cause
                    break
        terminal = False
        if dropout:
            end_reason = "simulated_dropout"
            terminal = True
        elif turns >= horizon:
            end_reason = "teaching_budget"
            terminal = True
        public_after = _public_state(
            beliefs,
            current=current,
            goal=goal,
            turn=turn + 1,
            active_item=active_item,
            active_option_type=decode_navigator(pending_navigator["action_id"])[0],
            active_target=(
                f"C{decode_navigator(pending_navigator['action_id'])[1] + 1:02d}"
                if decode_navigator(pending_navigator["action_id"])[1] is not None
                else None
            ),
            previous_move=previous_move,
            option_turns=option_turns,
            success_streak=success_streak,
            failure_streak=failure_streak,
            hint_count=hint_count,
            explicit_skip=requests["skip"],
            explicit_faster=requests["faster"],
            explicit_slower=requests["slower"],
            stack=stack.stack,
            misconceptions=public_misconceptions,
        )
        next_observation = build_features(public_after)
        if terminal:
            post_score = expected_score(
                private, goal, scenario["split"], config_path=simulator_config
            )
            terminal_reward_components = terminal_components(pre_score, post_score, dropout=dropout)
            for key, value in terminal_reward_components.to_dict().items():
                setattr(turn_reward, key, getattr(turn_reward, key) + value)
            if abs(turn_reward.total - sum(turn_reward.to_dict().values())) > 1e-12:
                raise AssertionError("reward components do not sum exactly")
            terminal_added = True
        reward = turn_reward.total
        strat_transition_id = f"{episode_id}-strategist-{turn:02d}"
        strategist_transitions.append(
            {
                "run_id": run_id,
                "episode_id": episode_id,
                "transition_id": strat_transition_id,
                "turn_index": turn,
                "option_id": pending_navigator["option_id"],
                "observation": strategist_observation,
                "action_mask": strat_mask,
                "action_id": action_id,
                "old_log_prob": 0.0,
                "old_value": 0.0,
                "reward_components": turn_reward.to_dict(),
                "reward": reward,
                "next_observation": next_observation,
                "terminated": terminal,
                "truncated": False,
                "actor_eligible": actor_eligible,
                "collector_cut": False,
                "policy_version": getattr(strategist, "policy_version", "policy_v1"),
                "environment_bundle_id": environment_bundle_id,
            }
        )
        diagnostics.append(
            {
                "transition_id": strat_transition_id,
                "scenario_id": scenario["scenario_id"],
                "episode_seed": episode_seed,
                "turn": turn,
                "student_outcome": {
                    "correctness": evidence_correctness,
                    "normalized_answer": outcome.normalized_answer,
                    "assistance": outcome.assistance,
                    "explicit_request": outcome.explicit_request,
                    "misconception_used": outcome.misconception_used,
                },
                "private_mastery_before": mastery_before,
                "private_mastery_after": dict(private.mastery),
                "private_active_misconceptions": sorted(private.active_misconceptions),
                "dropout_hazard_components": hazard_components,
                "dropout_cause": dropout_cause,
                "pace_mismatch": pace_mismatch,
                "pace_request_emitted": pace_request is not None,
                "pace_request": pace_request,
                "previous_answer_correct": previous_answer_correct,
                "previous_answer_item_id": previous_answer_item,
                "teaching_delta": private.mastery[current] - mastery_before[current],
                "p_correct_before": probability_correct(
                    PrivateStudentState(
                        mastery=mastery_before,
                        learning_rates=private.learning_rates,
                        guess=private.guess,
                        slip=private.slip,
                        profile_id=private.profile_id,
                        active_misconceptions=set(private.active_misconceptions),
                        susceptible_concepts=private.susceptible_concepts,
                    ),
                    current,
                    difficulty,
                    graph,
                ),
            }
        )
        pending_navigator["discounted_reward"] += (GAMMA ** (option_turns - 1)) * reward
        if terminal:
            boundary = ["teaching_budget"] if end_reason == "teaching_budget" else [end_reason]
            navigator_transitions.append(
                _make_navigator_transition(
                    run_id=run_id,
                    episode_id=episode_id,
                    option_id=pending_navigator["option_id"],
                    start_turn=pending_navigator["start_turn"],
                    end_turn=turn + 1,
                    observation=pending_navigator["observation"],
                    action_mask=pending_navigator["mask"],
                    action_id=pending_navigator["action_id"],
                    discounted_reward=pending_navigator["discounted_reward"],
                    next_observation=next_observation,
                    terminated=True,
                    truncated=False,
                    boundary=boundary,
                    environment_bundle_id=environment_bundle_id,
                )
            )
            break
        reasons_context = {
            "option_turns": option_turns,
            "independent_success_streak": success_streak,
            "independent_failure_streak": failure_streak,
            "explicit_skip": requests["skip"],
            "explicit_goal_change": requests.get("goal_change", False),
            "content_family_exhausted": bool(
                move in {"PRACTICE", "CHALLENGE"}
                and {
                    family.template_family_id
                    for family in families_for(current, role="teaching", split=scenario["split"])
                    if family.template_family_id in set(scenario["template_family_allowlist"])
                }
                <= used_teaching_families
            ),
        }
        boundary = boundary_reasons(reasons_context)
        if not boundary:
            continue
        navigator_transitions.append(
            _make_navigator_transition(
                run_id=run_id,
                episode_id=episode_id,
                option_id=pending_navigator["option_id"],
                start_turn=pending_navigator["start_turn"],
                end_turn=turn + 1,
                observation=pending_navigator["observation"],
                action_mask=pending_navigator["mask"],
                action_id=pending_navigator["action_id"],
                discounted_reward=pending_navigator["discounted_reward"],
                next_observation=next_observation,
                terminated=False,
                truncated=False,
                boundary=boundary,
                environment_bundle_id=environment_bundle_id,
            )
        )
        boundary_success_streak = success_streak
        boundary_failure_streak = failure_streak
        option_turns = 0
        if "family_exhausted" in boundary:
            used_teaching_families.clear()
        next_public = _public_state(
            beliefs,
            current=current,
            goal=goal,
            turn=turn + 1,
            active_item=active_item,
            active_option_type=None,
            active_target=None,
            previous_move=previous_move,
            option_turns=0,
            success_streak=0,
            failure_streak=0,
            hint_count=hint_count,
            explicit_skip=requests["skip"],
            explicit_faster=requests["faster"],
            explicit_slower=requests["slower"],
            stack=stack.stack,
            misconceptions=public_misconceptions,
        )
        next_context = _active_context(
            next_public,
            current=current,
            goal=goal,
            beliefs=beliefs,
            active_item=active_item,
            stack=stack,
            graph=graph,
            requests=requests,
            success_streak=boundary_success_streak,
            failure_streak=boundary_failure_streak,
            meaningful_error=meaningful_error,
            unsuccessful_hints=unsuccessful_hints,
            no_diagnostic_yet=no_diagnostic_yet,
            skip_deadline=skip_deadline,
            skip_diagnostic_done=skip_diagnostic_done,
            turn=turn + 1,
            pace=pace,
            allow_policy_stop=False,
        )
        state = {
            **next_public,
            "enabled_concepts": graph["enabled_concepts"],
            "goal_ancestors": graph["ancestors"].get(goal, []),
            "goal_stack": list(stack.stack),
            "independent_success_streak": boundary_success_streak,
            "independent_failure_streak": boundary_failure_streak,
            "current_certified": beliefs[current].certified,
            "goal_certified": beliefs[goal].certified,
            "allow_policy_stop": False,
            "learner_stop": requests["stop"],
        }
        next_mask = navigator_mask(state, _navigator_graph(graph, goal, current, beliefs), boundary)
        next_obs = _navigator_observation(next_public)
        next_action = navigator.act(next_obs, next_mask, next_context)
        next_option_id = f"{episode_id}-option-{turn + 1}"
        pending_navigator = {
            "option_id": next_option_id,
            "start_turn": turn + 1,
            "observation": next_obs,
            "mask": next_mask,
            "action_id": next_action,
            "discounted_reward": 0.0,
            "boundary": boundary,
        }
        if apply_navigation(next_action, turn + 1):
            end_reason = (
                "learner_stop"
                if requests["stop"]
                else "demonstrated_completion"
                if beliefs[goal].certified
                else "policy_stop"
            )
            post_score = expected_score(
                private, goal, scenario["split"], config_path=simulator_config
            )
            terminal_reward_components = terminal_components(pre_score, post_score)
            terminal_added = True
            navigator_transitions.append(
                _make_navigator_transition(
                    run_id=run_id,
                    episode_id=episode_id,
                    option_id=next_option_id,
                    start_turn=turn + 1,
                    end_turn=turn + 1,
                    observation=next_obs,
                    action_mask=next_mask,
                    action_id=next_action,
                    discounted_reward=terminal_reward_components.total,
                    next_observation=next_obs,
                    terminated=True,
                    truncated=False,
                    boundary=boundary,
                    environment_bundle_id=environment_bundle_id,
                )
            )
            break
        success_streak = failure_streak = 0

    if not terminal_added:
        raise AssertionError("episode ended without its single terminal reward")
    post_score = expected_score(
        private, goal, scenario["split"], config_path=simulator_config
    )
    gain = post_score - pre_score
    return EpisodeResult(
        episode_id=episode_id,
        scenario_id=scenario["scenario_id"],
        profile_id=scenario["profile_id"],
        split=scenario["split"],
        end_reason=end_reason,
        turns=turns,
        gain=gain,
        pre_score=pre_score,
        post_score=post_score,
        dropout=dropout,
        move_histogram=dict(sorted(moves.items())),
        strategist_transitions=strategist_transitions,
        navigator_transitions=navigator_transitions,
        private_diagnostics=diagnostics,
        terminal_components=terminal_reward_components.to_dict(),
        final_beliefs=beliefs,
        assessment_family_reuse_count=sum(assessment_family_reuse_counts.values()),
    )


def simulate_episodes(
    *,
    episodes: int,
    split: str = "train",
    seed: int = 20250115,
    strategist_name: str = "rules",
    strategist: object | None = None,
    scenarios: list[dict] | None = None,
    run_id: str = "local",
    simulator_config: str | Path = DEFAULT_SIMULATOR_CONFIG,
) -> list[EpisodeResult]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    candidates = scenarios if scenarios is not None else load_scenarios()
    candidates = [scenario for scenario in candidates if scenario["split"] == split]
    if not candidates:
        raise ValueError(f"no scenarios available for split {split}")
    results = []
    for index in range(episodes):
        scenario = candidates[index % len(candidates)]
        episode_seed = seed + index
        results.append(
            run_episode(
                scenario,
                run_id=run_id,
                episode_seed=episode_seed,
                strategist_name=strategist_name,
                strategist=strategist,
                horizon=min(int(scenario["teaching_horizon"]), HORIZON),
                simulator_config=simulator_config,
            )
        )
    return results


def write_run(results: list[EpisodeResult], output_dir: str | Path, *, manifest: dict) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    strategist_transitions = [
        transition for result in results for transition in result.strategist_transitions
    ]
    navigator_transitions = [
        transition for result in results for transition in result.navigator_transitions
    ]
    private_diagnostics = [
        diagnostic for result in results for diagnostic in result.private_diagnostics
    ]
    all_transitions = [*strategist_transitions, *navigator_transitions]
    all_transitions.sort(
        key=lambda item: (
            item["episode_id"],
            item.get("turn_index", item.get("start_turn", 0)),
            item["transition_id"],
        )
    )
    for filename, records in (
        ("transitions.jsonl", all_transitions),
        ("strategist_transitions.jsonl", strategist_transitions),
        ("navigator_transitions.jsonl", navigator_transitions),
        ("private_diagnostics.jsonl", private_diagnostics),
    ):
        (destination / filename).write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    moves = Counter()
    for result in results:
        moves.update(result.move_histogram)
    profile_summaries = {}
    for profile_id in sorted({result.profile_id for result in results}):
        group = [result for result in results if result.profile_id == profile_id]
        profile_summaries[profile_id] = {
            "episodes": len(group),
            "mean_gain": sum(result.gain for result in group) / len(group),
            "mean_turns": sum(result.turns for result in group) / len(group),
            "dropout_rate": sum(result.dropout for result in group) / len(group),
        }
    summary = {
        "run_id": manifest["run_id"],
        "episodes": len(results),
        "mean_gain": sum(result.gain for result in results) / len(results),
        "mean_turns": sum(result.turns for result in results) / len(results),
        "dropout_rate": sum(result.dropout for result in results) / len(results),
        "assessment_family_reuse_count": sum(
            result.assessment_family_reuse_count for result in results
        ),
        "move_histogram": dict(sorted(moves.items())),
        "end_reason_counts": dict(sorted(Counter(result.end_reason for result in results).items())),
        "profile_summaries": profile_summaries,
    }
    manifest["episode_count"] = len(results)
    manifest["episodes"] = [
        {
            "episode_id": result.episode_id,
            "scenario_id": result.scenario_id,
            "profile_id": result.profile_id,
            "seed": result.episode_id.rsplit("-", 1)[-1],
            "turns": result.turns,
            "end_reason": result.end_reason,
            "pre_score": result.pre_score,
            "post_score": result.post_score,
            "reward_components": result.terminal_components,
        }
        for result in results
    ]
    manifest["transitions_sha256"] = hashlib.sha256(
        (destination / "transitions.jsonl").read_bytes()
    ).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
