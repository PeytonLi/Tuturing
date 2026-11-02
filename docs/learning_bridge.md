# Person B text-system bridge (SIMULATED)

This measures a language/simulator integration gap, not human learning. No live model
was called for the checked-in validation smokes. Validation and fake replies cannot
clear PRD G8. Keep the test split sealed until a real-text evaluation is approved.

## Paths and pairing

`tools/learning/bridge.py` runs paired episodes with a rules Navigator and either a
rules Strategist or a frozen checkpoint (`--policy checkpoint:artifacts/strategist.pt`,
or the checkpoint path alone). Both arms use the same scenario, seed, policy pair,
simulator configuration and `EventRandomBank`. Seeds vary across episodes; cycling
the same scenario does not create new scenario families.

The structured arm consumes sampled `StudentOutcome` objects directly. The text arm
phrases those same types of sampled outcomes with the frozen
`student_templates_bridge_v1` renderer, calls A's public `extract`, and feeds the
extracted answer/assistance/request through B's grader and existing BKT path.
Answer extractions below 0.80 confidence abstain from answer evidence (the raw
extraction stays in the trace). The
renderer never selects correctness, resamples a student event, or awards learning.
An optional LLM verbalizer can only select an exact member of a finite set of
approved phrasings of the already-sampled answer/request. Invalid wording is
rejected, logged, and replaced by the frozen template. It cannot add reasoning or
change numbers. Do not interpret plausible prose as causal evidence.

Both bridge arms use A's **verified stub pipeline**, not the legacy unchecked
`packages.response.stub.generate` directly. This isolates text/extractor effects:
the stub pipeline performs A's normal prompt/repair/fallback checks and may use a
verified fallback even for template replies. The default `run_episode` without a
`text_path` remains the legacy structured route. The sole episode integration is
an optional realizer/check/observe hook; no policy features or contracts change.

When replies differ, later actions, private mastery, outcomes and episode lengths
may diverge. The keyed random bank still supplies common random events at matching
scenario/seed/turn/event keys. Do not force later student outcomes to stay equal:
that would conceal the effect being measured.

## Fidelity rule and trace review

PRD `docs/adaptive_ai_tutor_prd.md:895` says `content_fidelity` is zero for
unsupported or contradictory rendered content and at most one for verified content.
Line 899 requires independent checks rather than the generator's pass flag.

Implemented rule: independently rerun A's public `run_checks` on the delivered
text and effective action. **Any failed check or A item-fidelity failure sets
`content_realized=False`, hence `content_fidelity=0`.** Otherwise use the
declared simulator `stub_realized` coefficient and its seeded fidelity-failure
process (primary configuration: 1.0, failure probability 0.0). Zero fidelity
prevents both direct mastery gain and misconception deactivation. A valid fallback
can teach; a rejected draft is not itself the delivered content.

For protected non-example items, the independent audit adds B's solver answer to
a private copy of the plan's forbidden fingerprints, never the generator prompt.
WORKED_EXAMPLE is exempt from that extra answer fingerprint: showing the example's
own answer is intentional. A's public `packages.response.item_fidelity.item_fidelity`
(introduced with validator_v2) supplies item-identity diagnostics, including
coverage, new objects, new quantities and reasons. It also applies to HINT.
This check is part of A's `action_faithful`; the bridge records its separate
`item_changed` flag and diagnostics as well. These checks are heuristic, not a
semantic/math oracle: false positives and missed substitutions remain possible.

Each trace records the complete plan, delivered reply/effective move, draft
attempts, independent checks, sampled outcome, rendered message, extracted
evidence, B's regraded outcome, and disagreement fields. Flags include wrong
move, item changed, answer disclosed, validator failure, fallback used, rejected
draft, extractor disagreement and rejected verbalizer. Extractor disagreement is
an interpretation flag, not automatically a tutor-content fidelity failure.

Only turns with a `CRITICAL_FLAGS` flag from either arm require G8 review and go
to `<output>.traces.jsonl.gz`. Fallback use and rejected drafts remain informational
counts: alone they do not require review. These files contain only simulated text.
If the full critical file exceeds 1,000,000 compressed bytes, the committed file is
a deterministic hash-ranked sample of up to 200 critical turns per run. JSON records
the sample and full SHA-256 hashes and row counts. Sampling never reduces the G8
review requirement: reviewers must still cover the full critical set.

Full per-turn dumps are never written beside the committed report. They are stored
under the gitignored `experiments/learning/evals/bridge/<sha256>.turns.jsonl.gz`;
JSON records only their SHA-256. An oversized full critical file is retained there
as `<sha256>.traces.jsonl.gz`, using the full hash recorded in JSON. Do not add
these ignored audit files to Git.
Read traces with `gzip -dc <output>.traces.jsonl.gz`. For example, decompress into
an editor and review the actual plan, reply and extraction, not just the flag.

G8 (PRD line 1152) is PASS only for at least 300 explicitly unsealed **test**
pairs with at least one delivered model/repaired-model reply in **each** text
episode, all critical-flagged traces human-reviewed, no reviewed
critical failures, and mean(text gain - structured gain) >= -0.02. Reports publish
1,000-resample percentile bootstrap 95% intervals for each path and for the
paired episode differences. A CI crossing -0.02 is published, not an additional
PRD gate. Intervals condition on the simulator, policy and scenario bank, not
human effects or new scenario families. Automated critical candidates are not
human verdicts: reviewers can dismiss documented false positives.

A review file is a JSON array with one entry for every critical-flagged trace:

```json
[
  {
    "path": "text",
    "trace_id": "copy-from-trace",
    "review_hash": "copy-from-trace",
    "reviewed": true,
    "reviewed_by": "human reviewer name",
    "critical_failure": false,
    "notes": "Describe the inspected failure or why this flag is noncritical."
  }
]
```

Pass `--review-file work/reviews.json` on an identical run. The hash binds reviews
to the inspected plan/text/outcome; mismatched, duplicate and foreign reviews are
refused. A rerun with a nondeterministic hosted backend can differ; re-review it
instead of reusing stale decisions. There is no automatic human approval.

The misconception-deactivation fidelity guard does not change the default
structured episode (`text_path=None`): the primary stub coefficient is 1.0 and
its failure probability is zero, so every structured teaching turn has positive
fidelity. Regression tests compare gains, move histograms and private diagnostics
against the pre-guard function for two fixed seeds. Nondefault simulator configs
with zero fidelity are intentionally subject to the guard even without a text path.

## Local runs (no network, no credentials)

```bash
uv sync --extra learning
uv run python -m tools.learning.bridge --episodes 300 --split validation --reply stub
uv run python -m tools.learning.bridge --episodes 300 --split validation \
  --reply degraded --degradation-rate 0.2 --output docs/results/bridge_degraded
uv run python -m tools.learning.bridge --smoke-suite --episodes 300 \
  --split validation --policy rules --output evaluation/learning/results/bridge_smoke_v1
uv run ruff check .
uv run pytest -q
```

The degraded fake deterministically injects a wrong move and irrelevant text on
roughly 20% of turns after A's stub pipeline. Both arms still use the same random
event bank. Stub-vs-stub must have exactly zero paired gain difference; the fake
must be detected even when its numeric gain gap remains within 0.02.

## Approved live runs through A's public pipeline

The bridge calls `apps.response_service.config.load_config/build_pipeline` in
process: the same responder, prompt compiler, repair, validator and verified
fallback path as A's server. No HTTP service process or service bearer token is
needed. No A-owned files are edited. It is not a direct free-form chat client.

First estimate locally, without constructing the hosted responder or sending any
requests (base URL, model and key need not be present for this dry run):

```bash
uv run python -m tools.learning.bridge --episodes 300 --split validation \
  --reply openai --dry-run --output work/bridge_call_estimate
```

Only after explicit spending approval, bind credentials from the environment:

```bash
export TUTOR_LLM_BASE_URL=https://api.openai.com/v1
export TUTOR_LLM_MODEL=gpt-6.1-sol
export TUTOR_LLM_API_KEY="$OPENAI_API_KEY"
uv run python -m tools.learning.bridge --episodes 300 --split validation \
  --reply openai --response-config configs/response/serve_openai_compatible.yaml \
  --max-calls 9600 --output docs/results/bridge_openai_validation
```

`gpt-6.1-sol` is the requested model ID, not a claim of verified account access
or compatibility. A's current client sends Chat Completions with greedy decoding
(`temperature=0`), `max_tokens`, and an eight-second deadline. Its provider
compatibility has not been tested here. B's decision scorer's
`TUTOR_LLM_OMIT_TEMPERATURE` / `TUTOR_LLM_MAX_TOKENS_FIELD` flags do **not** alter
A's response client. If the chosen provider requires different decoding fields,
A must expose that configuration; do not silently patch A or bypass the pipeline.

For an **already approved, ready** Fireworks Qwen on-demand deployment:

```bash
export TUTOR_LLM_BASE_URL=https://api.fireworks.ai/inference/v1
export TUTOR_LLM_MODEL='accounts/fireworks/models/<qwen-model-id>#accounts/<account>/deployments/<ready-deployment-id>'
export TUTOR_LLM_API_KEY="$FIREWORKS_API_KEY"
```

Use the same `--reply openai` command with a different output prefix. Replace the
model/deployment placeholders with the frozen IDs supplied by the deployment
owner. This bridge never creates, scales, or deletes a deployment; on-demand
compute can incur charges while idle. Provisioning and cleanup need separate
approval. Do not assume that serverless training implies serverless inference.

For A's tuned model later, point those variables at A's frozen OpenAI-compatible
endpoint/model revision and pass A's released response config with its distinct
artifact ID via `--response-config`. The report records pipeline/artifact/prompt/
validator versions and model identity. An unpinned prompt-only config is an
engineering run, not a frozen model release.

Optional wording-only LLM: pass `--verbalizer-config` with an A-format service
config, a positive shared `--max-calls` cap, and separately approved spending.
The deterministic renderer is the default. Both model paths share one call budget.

## Call budget

300 episodes at the current 16-turn horizon have a maximum of **4,800 text turns**.
Structured replies, the local fake, validators and the rule extractor cost zero
model calls. A live reply takes one primary call per text turn, plus at most one
repair: **at most 9,600 calls** per paired run. The optional wording verbalizer
adds one per text turn: **at most 14,400 total** with replies and repairs. Actual
stops/dropouts shorten runs; the dry-run's stub trajectory is a nominal estimate,
not a guarantee that the text policy follows the same trajectory.

`--max-calls` defaults to zero. Live paths require an explicit positive cap.
Primary, repair and verbalizer calls consume it before dispatch; a timeout still
counts. Exhausting the cap aborts the run, writes `<output>.aborted.json` and
does not publish a partial paired score as a completed evaluation. This is a
request-count bound, not a dollar/token budget or spending approval. Estimate
provider pricing separately. Default live response output is capped at 384
tokens and verbalizer output at 128. No automatic paid retries/deployments run.

For final G8 evaluation, add `--split test --unseal-test` **only after approval**;
the warning and split are recorded. No test-split rollout was run for the smokes.

## Observed A-side integration gaps (not changed here)

- Legacy template PROBE wording can fail A's `action_faithful` attempt heuristic;
  the public pipeline repairs/falls back. Bridge counts distinguish rejected
  drafts, final delivery failures and fallback use. Template replies commonly
  report fallback delivery; that alone does not mean a critical failure.
- A's extractor inherits `plan.assistance` even for a no-answer hint request.
  Simulator help-seeking requests can have `assistance=none` after a HINT plan;
  extraction then reports `hint`. These semantic disagreements are retained for
  review, not overwritten to force zero flag counts.
- A's item-fidelity validator was merged during this work. The bridge now calls
  that public entry point instead of maintaining a separate exact-stem heuristic.
  It detects word/quantity/scenario substitutions, not mathematical equivalence.
- Hosted provider/schema/model support is untested; real runs may fall back on
  provider errors. Inspect fallback and attempt counts, not just gain. Credentials
  and paid access were not required or exercised for these simulated smokes.
  An episode with only fallback deliveries is not counted as a real-text G8 pair.
