"""Verified response generation (PRD 8.2 step 4, 9.4 and 14).

``ResponsePipeline.run`` turns a GenerateRequest into a GenerateResponse:

1. Verify the request: supported major versions, the response_v1 GenerateRequest schema,
   the PRD 9.3 plan rules (an independent plan with a reveal_solution is rejected) and
   the requested response artifact.
2. Compile the prompt with ``packages.response.prompt.compile_prompt`` and call the
   responder. Output is buffered in full before anything is checked.
3. Run the six response checks. A draft that fails only the word cap or question count
   gets at most one repair with the same item and action.
4. Otherwise use a verified fallback from ``packages.response.fallbacks``.
5. Otherwise raise a retryable ``GENERATION_UNAVAILABLE`` error.

Text is returned only after all six checks pass on exactly that text. Responder calls are
bounded by ``min(deadline_ms, internal_deadline_ms)``, never more than the PRD 14.3 eight
second ceiling.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

from packages.response.contracts import (
    SUPPORTED_MAJOR_VERSION,
    major_version,
    schema_errors,
)
from packages.response.errors import GenerationError, is_id
from packages.response.fallbacks import FALLBACK_VERSION, render_fallback
from packages.response.hashing import canonical_json, content_hash
from packages.response.plan_checks import plan_errors
from packages.response.prompt import PROMPT_VERSION, compile_prompt
from packages.response.repair import REPAIR_VERSION, failed_checks, is_repairable
from packages.response.responders import (
    MAX_OUTPUT_TOKENS,
    RepairLimits,
    ResponderConfigRejected,
    ResponderOutput,
    TemplateStubResponder,
)
from packages.response.types import GenerateRequest, GenerateResponse
from packages.response.validate import VALIDATOR_VERSION, run_checks

PIPELINE_VERSION = "pipeline_v1"
INTERNAL_DEADLINE_CEILING_MS = 8000
STUB_PIPELINE_ARTIFACT_ID = "response_stub_pipeline_v1"
LEGACY_STUB_ARTIFACT_ID = "response_stub_v1"  # what packages.simulator.response_adapter sends
MAX_REPORTED_PROBLEMS = 5


class ResponderTimeout(RuntimeError):
    """A responder call did not finish inside the generation deadline."""


@dataclass(frozen=True)
class PipelineSettings:
    artifact_id: str
    compatible_artifact_ids: tuple[str, ...] = ()
    internal_deadline_ms: int = INTERNAL_DEADLINE_CEILING_MS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    max_repairs: int = 1

    def __post_init__(self):
        if not is_id(self.artifact_id):
            raise ValueError("artifact_id must be a nonempty ASCII ID of at most 128 characters")
        accepted = tuple(dict.fromkeys((self.artifact_id, *self.compatible_artifact_ids)))
        if not all(is_id(value) for value in accepted):
            raise ValueError("compatible_artifact_ids must be ASCII IDs")
        object.__setattr__(self, "compatible_artifact_ids", accepted)
        if not 0 < self.internal_deadline_ms <= INTERNAL_DEADLINE_CEILING_MS:
            raise ValueError(
                f"internal_deadline_ms must be in (0, {INTERNAL_DEADLINE_CEILING_MS}]"
            )
        if not 0 < self.max_output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens must be in (0, {MAX_OUTPUT_TOKENS}]")
        if self.max_repairs not in (0, 1):
            raise ValueError("max_repairs must be 0 or 1")


@dataclass
class Attempt:
    """One step of a generation, for logs and access-controlled traces.

    ``text`` holds a rejected draft for the optional evaluation trace only; ``redacted``
    is what may be logged.
    """

    stage: str
    outcome: str
    failed_checks: tuple[str, ...] = ()
    detail: str | None = None
    text: str | None = field(default=None, repr=False)

    def redacted(self) -> dict:
        record = {"stage": self.stage, "outcome": self.outcome}
        if self.failed_checks:
            record["failed_checks"] = list(self.failed_checks)
        if self.detail:
            record["detail"] = self.detail
        return record


@dataclass(frozen=True)
class GenerationOutcome:
    response: GenerateResponse
    attempts: tuple[Attempt, ...]


def response_id_for(request: GenerateRequest, artifact_id: str) -> str:
    """Deterministic response ID: the same request to the same artifact gets the same ID."""
    seed = canonical_json(
        {
            "artifact": artifact_id,
            "plan_id": request["plan"]["plan_id"],
            "request_id": request["request_id"],
        }
    )
    return "response-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _call_with_deadline(function, timeout_s: float):
    """Run ``function`` in a daemon thread and stop waiting after ``timeout_s``."""
    box = {}

    def target():
        try:
            box["value"] = function()
        except BaseException as error:  # handed back to the caller below
            box["error"] = error

    worker = threading.Thread(target=target, name="responder-call", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ResponderTimeout("responder exceeded the generation deadline")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _problems(prefix: str, problems: list[str]) -> str:
    shown = problems[:MAX_REPORTED_PROBLEMS]
    more = len(problems) - len(shown)
    suffix = f"; and {more} more" if more > 0 else ""
    return f"{prefix}: " + "; ".join(shown) + suffix


class ResponsePipeline:
    def __init__(self, responder, settings: PipelineSettings, *, clock=time.monotonic):
        self.responder = responder
        self.settings = settings
        self._clock = clock

    @property
    def versions(self) -> dict:
        return {
            "response_artifact": self.settings.artifact_id,
            "prompt": PROMPT_VERSION,
            "validator": VALIDATOR_VERSION,
        }

    def describe(self) -> dict:
        """Versions and limits, safe to expose on a health endpoint or in a manifest."""
        describe = getattr(self.responder, "describe", None)
        return {
            "artifact_id": self.settings.artifact_id,
            "compatible_artifact_ids": list(self.settings.compatible_artifact_ids),
            "responder": describe() if callable(describe) else {"kind": "custom"},
            "versions": {
                "pipeline": PIPELINE_VERSION,
                "prompt": PROMPT_VERSION,
                "validator": VALIDATOR_VERSION,
                "fallback": FALLBACK_VERSION,
                "repair": REPAIR_VERSION,
            },
            "limits": {
                "internal_deadline_ms": self.settings.internal_deadline_ms,
                "max_output_tokens": self.settings.max_output_tokens,
                "max_repairs": self.settings.max_repairs,
                # "greedy" (temperature 0), "provider_default_temperature" (field omitted, so
                # not deterministic) or "deterministic" for the template stub.
                "decoding": getattr(self.responder, "decoding_label", "greedy"),
            },
        }

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        return self.run(request).response

    def verify(self, request: GenerateRequest) -> None:
        """Raise ``GenerationError`` unless the request can be served by this pipeline."""
        if not isinstance(request, dict):
            raise GenerationError(
                "INVALID_SCHEMA", "GenerateRequest must be a JSON object", retryable=False
            )
        plan = request.get("plan")
        versions = (
            ("schema_version", request.get("schema_version")),
            ("plan.schema_version", plan.get("schema_version") if isinstance(plan, dict) else None),
        )
        for label, version in versions:
            major = major_version(version)
            if major is not None and major != SUPPORTED_MAJOR_VERSION:
                raise GenerationError(
                    "UNSUPPORTED_VERSION",
                    f"{label} major version is not supported; supported major version is "
                    f"{SUPPORTED_MAJOR_VERSION}",
                    retryable=False,
                )
        problems = schema_errors(request, "GenerateRequest")
        if problems:
            raise GenerationError(
                "INVALID_SCHEMA",
                _problems("GenerateRequest failed schema validation", problems),
                retryable=False,
            )
        problems = plan_errors(request["plan"])
        if problems:
            raise GenerationError(
                "INVALID_SCHEMA", _problems("TeachingPlan violates plan rules", problems),
                retryable=False,
            )
        if request["response_artifact_id"] not in self.settings.compatible_artifact_ids:
            raise GenerationError(
                "UNSUPPORTED_VERSION",
                "response_artifact_id is not served here; this service serves "
                f"{self.settings.artifact_id}",
                retryable=False,
            )

    def run(self, request: GenerateRequest) -> GenerationOutcome:
        started = self._clock()
        self.verify(request)
        plan = request["plan"]
        budget_ms = min(request["deadline_ms"], self.settings.internal_deadline_ms)
        if budget_ms <= 0:
            raise GenerationError(
                "DEADLINE_EXCEEDED", "deadline_ms leaves no time to generate", retryable=True
            )
        deadline = started + budget_ms / 1000
        action = dict(plan["teaching_action"])
        attempts: list[Attempt] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        delivered = self._from_responder(plan, action, deadline, attempts, usage)
        if delivered is None:
            delivered = self._from_fallback(plan, action, attempts)
        if delivered is None:
            raise GenerationError(
                "GENERATION_UNAVAILABLE",
                "no response passed validation and no verified fallback realizes the plan",
                retryable=True,
                attempts=attempts,
            )
        text, delivery_mode, checks = delivered
        elapsed_ms = max(0, round((self._clock() - started) * 1000))
        response: GenerateResponse = {
            "schema_version": "1.0",
            "request_id": request["request_id"],
            "plan_id": plan["plan_id"],
            "response_id": response_id_for(request, self.settings.artifact_id),
            "text": text,
            "effective_move": action,
            "delivery_mode": delivery_mode,
            "content_hash": content_hash(text, action),
            "checks": checks,
            "usage": {**usage, "elapsed_ms": elapsed_ms},
            "versions": self.versions,
        }
        if not all(checks.values()) or schema_errors(response, "GenerateResponse"):
            raise GenerationError(
                "GENERATION_UNAVAILABLE",
                "assembled response failed its contract",
                retryable=True,
                attempts=attempts,
            )
        return GenerationOutcome(response=response, attempts=tuple(attempts))

    def _call(self, stage, function, deadline, attempts, usage) -> ResponderOutput | None:
        remaining = deadline - self._clock()
        if remaining <= 0:
            attempts.append(Attempt(stage, "skipped", detail="deadline"))
            return None
        try:
            if getattr(self.responder, "io_bound", True):
                output = _call_with_deadline(lambda: function(remaining), remaining)
            else:
                output = function(remaining)
        except ResponderTimeout:
            attempts.append(Attempt(stage, "timeout"))
            return None
        except ResponderConfigRejected as error:
            # The provider refused the configured request format. The learner still gets the
            # fallback, but the trail names the rejected parameter so operators can see it.
            attempts.append(Attempt(stage, "config_error", detail=error.summary))
            return None
        except Exception as error:  # every responder failure falls through to the fallback
            attempts.append(Attempt(stage, "error", detail=type(error).__name__))
            return None
        if not isinstance(output, ResponderOutput) or not isinstance(output.text, str):
            attempts.append(Attempt(stage, "error", detail="invalid_output"))
            return None
        usage["input_tokens"] += max(0, int(output.input_tokens))
        usage["output_tokens"] += max(0, int(output.output_tokens))
        if self._clock() > deadline:
            attempts.append(Attempt(stage, "timeout", detail="late"))
            return None
        return output

    def _judge(self, stage, output, plan, action, attempts):
        checks = run_checks(plan, output.text, action)
        if all(checks.values()) and not output.truncated:
            attempts.append(Attempt(stage, "accepted"))
            return checks, True
        attempts.append(
            Attempt(
                stage,
                "rejected",
                failed_checks=tuple(failed_checks(checks)),
                detail="truncated" if output.truncated else None,
                text=output.text,
            )
        )
        return checks, False

    def _from_responder(self, plan, action, deadline, attempts, usage):
        messages = compile_prompt(plan)
        max_tokens = self.settings.max_output_tokens
        draft = self._call(
            "model",
            lambda timeout: self.responder.generate(
                messages, plan=plan, max_tokens=max_tokens, timeout_s=timeout
            ),
            deadline,
            attempts,
            usage,
        )
        if draft is None:
            return None
        # A responder labels its own output: "model" for a language model, "fallback" for
        # the template stub. A repaired template is still template text, so it stays "fallback".
        primary_mode = getattr(self.responder, "delivery_mode", "model")
        checks, accepted = self._judge("model", draft, plan, action, attempts)
        if accepted:
            return draft.text, primary_mode, checks
        if self.settings.max_repairs < 1 or not is_repairable(checks, truncated=draft.truncated):
            return None
        limits = RepairLimits(
            max_words=plan["constraints"]["max_words"],
            max_questions=plan["constraints"]["max_questions"],
        )
        repaired = self._call(
            "repair",
            lambda timeout: self.responder.repair(
                messages,
                draft.text,
                plan=plan,
                limits=limits,
                max_tokens=max_tokens,
                timeout_s=timeout,
            ),
            deadline,
            attempts,
            usage,
        )
        if repaired is None:
            return None
        checks, accepted = self._judge("repair", repaired, plan, action, attempts)
        repaired_mode = "repaired" if primary_mode == "model" else primary_mode
        return (repaired.text, repaired_mode, checks) if accepted else None

    def _from_fallback(self, plan, action, attempts):
        try:
            text = render_fallback(plan)
        except Exception as error:  # FallbackUnavailable or an unrenderable plan
            attempts.append(Attempt("fallback", "unavailable", detail=type(error).__name__))
            return None
        checks = run_checks(plan, text, action)
        if not all(checks.values()):
            attempts.append(
                Attempt("fallback", "rejected", failed_checks=tuple(failed_checks(checks)))
            )
            return None
        attempts.append(Attempt("fallback", "accepted"))
        return text, "fallback", checks


def stub_pipeline(artifact_id: str = STUB_PIPELINE_ARTIFACT_ID) -> ResponsePipeline:
    """The CPU stub: existing stub templates behind the full verified pipeline.

    It also accepts requests addressed to the legacy stub artifact that B's adapter names.
    """
    settings = PipelineSettings(
        artifact_id=artifact_id, compatible_artifact_ids=(LEGACY_STUB_ARTIFACT_ID,)
    )
    return ResponsePipeline(TemplateStubResponder(), settings)
