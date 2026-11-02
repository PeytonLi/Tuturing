"""Person B's audited consumer of A's public generation and extraction entry points."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict

from apps.response_service.config import build_pipeline, load_config
from packages.evidence_extractor.extract import EXTRACTOR_VERSION, extract
from packages.learning.grader import grade_answer, normalize_for_instance
from packages.response.hashing import content_hash
from packages.response.item_fidelity import item_fidelity
from packages.response.validate import run_checks
from packages.simulator.student import StudentOutcome
from packages.simulator.text_renderer import RENDERER_VERSION, render_checked

CRITICAL_FLAGS = frozenset(
    {
        "wrong_move",
        "item_changed",
        "answer_disclosed",
        "validator_failure",
        "extractor_disagreement",
        "verbalizer_rejected",
    }
)


class CallLimitExceeded(RuntimeError):
    pass


class CallBudget:
    def __init__(self, maximum: int = 0):
        if maximum < 0:
            raise ValueError("max-calls must be nonnegative")
        self.maximum = maximum
        self.calls = 0
        self.exceeded = False
        self._lock = threading.Lock()

    def consume(self) -> None:
        with self._lock:
            if self.calls >= self.maximum:
                self.exceeded = True
                raise CallLimitExceeded("model call budget exhausted; no further call sent")
            self.calls += 1


class BudgetedResponder:
    def __init__(self, responder, budget: CallBudget):
        self.responder = responder
        self.budget = budget

    def __getattr__(self, name):
        return getattr(self.responder, name)

    def generate(self, *args, **kwargs):
        self.budget.consume()
        return self.responder.generate(*args, **kwargs)

    def repair(self, *args, **kwargs):
        self.budget.consume()
        return self.responder.repair(*args, **kwargs)


class ServiceRealizer:
    """Use the same config, prompt, repair and verified fallback as A's serve command."""

    def __init__(self, config_path: str, *, budget: CallBudget):
        self.config = load_config(config_path)
        self.pipeline = build_pipeline(self.config)
        self.budget = budget
        if self.config.responder["kind"] != "template_stub":
            self.pipeline.responder = BudgetedResponder(self.pipeline.responder, budget)
        self.last_attempts: list[dict] = []
        self.description = self.pipeline.describe()

    def __call__(self, plan: dict, *, request_id: str, generation_seed: int = 0) -> dict:
        result = self.pipeline.run(
            {
                "schema_version": "1.0",
                "request_id": request_id,
                "trace_id": plan["trace_id"],
                "plan": plan,
                "response_artifact_id": self.config.artifact_id,
                "generation_seed": generation_seed,
                "deadline_ms": self.config.internal_deadline_ms,
            }
        )
        if self.budget.exceeded:
            raise CallLimitExceeded(
                "model call budget exhausted; run aborted, not a fallback score"
            )
        self.last_attempts = [
            {**attempt.redacted(), **({"text": attempt.text} if attempt.text is not None else {})}
            for attempt in result.attempts
        ]
        return result.response


class DegradedRealizer:
    """Deliberate deterministic fault injection AFTER A's verified stub, never a model."""

    def __init__(self, base: ServiceRealizer, rate: float = 0.2):
        if not 0 <= rate <= 1:
            raise ValueError("degradation rate must be in [0, 1]")
        self.base, self.rate = base, rate
        self.description = {"kind": "fake_wrong_move", "rate": rate, "base": base.description}

    @property
    def last_attempts(self):
        return self.base.last_attempts

    def __call__(self, plan, *, request_id, generation_seed=0):
        response = copy.deepcopy(
            self.base(plan, request_id=request_id, generation_seed=generation_seed)
        )
        key = f"{request_id}|{generation_seed}|bridge_wrong_move_v1"
        draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
        if draw < self.rate:
            response["effective_move"]["move"] = (
                "HINT" if plan["teaching_action"]["move"] != "HINT" else "EXPLAIN_CONCEPT"
            )
            response["text"] = "Let us talk about something else instead."
            response["content_hash"] = content_hash(response["text"], response["effective_move"])
        return response


class TextPath:
    def __init__(
        self,
        realizer: Callable,
        *,
        structured: bool = False,
        verbalizer: Callable[[str], str] | None = None,
        renderer: Callable = render_checked,
    ):
        self.realizer, self.structured = realizer, structured
        self.verbalizer, self.renderer = verbalizer, renderer
        self.records: list[dict] = []
        self.current: dict = {}

    def realize(self, plan, **kwargs):
        return self.realizer(plan, **kwargs)

    def check_content(self, plan, realized, *, item=None):
        audit_plan = copy.deepcopy(plan)
        if (
            item is not None
            and plan["teaching_action"]["move"] != "WORKED_EXAMPLE"
            and not plan["constraints"]["reveal_allowed"]
        ):
            # The solver answer is ONLY for checks, never added to the generator's prompt.
            audit_plan["constraints"]["forbidden_answer_fingerprints"] = [
                str(item["instance"]["answer"])
            ]
        checks = run_checks(audit_plan, realized["text"], realized["effective_move"])
        item_check = item_fidelity(plan, realized["text"])
        item_changed = item_check.applicable and not item_check.passed
        flags = {
            "wrong_move": not checks["action_faithful"],
            "item_changed": item_changed,
            "answer_disclosed": not checks["disclosure_allowed"],
            "validator_failure": not all(checks.values()),
            "fallback_used": realized["delivery_mode"] == "fallback",
            "draft_validator_failure": any(
                attempt.get("outcome") == "rejected"
                for attempt in getattr(self.realizer, "last_attempts", [])
            ),
        }
        verified = all(checks.values()) and not item_changed
        self.current = {
            "trace_id": plan["trace_id"],
            "planned_move": copy.deepcopy(plan["teaching_action"]),
            "realized_move": copy.deepcopy(realized["effective_move"]),
            "plan": copy.deepcopy(plan),
            "response": copy.deepcopy(realized),
            "checks": checks,
            "item_fidelity": asdict(item_check),
            "flags": flags,
            "content_fidelity_verified": verified,
            "attempts": copy.deepcopy(getattr(self.realizer, "last_attempts", [])),
            "renderer_version": RENDERER_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
        }
        return {**realized, "checks": {**checks, "item_unchanged": not item_changed}}

    def observe(self, outcome, plan, realized, *, turn, item):
        message, rejected = (
            self.renderer(outcome, key=plan["trace_id"], verbalizer=self.verbalizer)
            if not self.structured
            else (None, False)
        )
        evidence = (
            None
            if self.structured
            else extract(
                message,
                plan,
                realized["response_id"],
                observed_at="2000-01-01T00:00:00Z",
                event_id=f"{plan['trace_id']}-evidence",
            )
        )
        observed = outcome
        disagreements = []
        if evidence is not None:
            answer = evidence["normalized_answer"] if evidence["confidence"] >= 0.8 else None
            correctness = (
                grade_answer(answer, item["instance"], item["family"]) if item else "unknown"
            )
            if normalize_for_instance(answer) != normalize_for_instance(outcome.normalized_answer):
                disagreements.append("normalized_answer")
            sampled_grade = (
                grade_answer(outcome.normalized_answer, item["instance"], item["family"])
                if item
                else "unknown"
            )
            if correctness != sampled_grade:
                disagreements.append("graded_correctness")
            for name in ("assistance", "explicit_request"):
                if evidence[name] != getattr(outcome, name):
                    disagreements.append(name)
            observed = StudentOutcome(
                correctness,
                answer,
                evidence["assistance"],
                evidence["explicit_request"],
                outcome.misconception_used,
                answer is not None and evidence["assistance"] == "none",
            )
        self.current.update(
            {
                "turn": turn,
                "sampled_outcome": asdict(outcome),
                "student_message": message,
                "extracted_evidence": evidence,
                "observed_outcome": asdict(observed),
                "disagreement_fields": disagreements,
                "solver_answer": str(item["instance"]["answer"]) if item else None,
            }
        )
        self.current["flags"].update(
            {
                "extractor_disagreement": bool(disagreements),
                "verbalizer_rejected": rejected,
            }
        )
        self.current["review_hash"] = hashlib.sha256(
            json.dumps(
                [
                    plan,
                    realized["text"],
                    realized["effective_move"],
                    message,
                    self.current["flags"],
                    self.current["sampled_outcome"],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self.records.append(self.current)
        return observed


def llm_verbalizer(config_path: str, budget: CallBudget):
    """A separate frozen responder phrases a sampled outcome; it cannot choose one."""
    config = load_config(config_path)
    pipeline = build_pipeline(config)
    responder = BudgetedResponder(pipeline.responder, budget)

    def phrase(prompt: str) -> str:
        return responder.generate(
            [{"role": "user", "content": prompt}], plan={}, max_tokens=128, timeout_s=8
        ).text

    return phrase
