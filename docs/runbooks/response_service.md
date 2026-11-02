# Response service runbook (Person A)

The response service turns a `TeachingPlan` issued by Person B into one checked tutor
message. It never decides what to teach: it verifies the plan, compiles the prompt, calls a
responder, runs the six response checks, repairs once for length or question count only,
and otherwise delivers a verified fallback or a retryable error (PRD 8.2 step 4, 9.4, 14).

Two responders exist behind one interface, in three shipped profiles:

| Responder | Artifact | Config | What it does |
|---|---|---|---|
| `template_stub` | `response_stub_pipeline_v1` | `configs/response/serve_stub.yaml` | The deterministic CPU templates in `packages/response/stub.py` behind the verified pipeline. No model, no network, no credential. |
| `openai_compatible` | `prompt_only_openai_compatible_v1` | `configs/response/serve_openai_compatible.yaml` | Any OpenAI-compatible `/chat/completions` server (Qwen3.8-27B on a Fireworks deployment, local llama.cpp `llama-server`, vLLM). Greedy decoding (`temperature: 0`), `max_tokens`, `reasoning_effort: none`: the body benchmark_v1 used for Qwen. |
| `openai_compatible` | `prompt_only_gpt61sol_v1` | `configs/response/serve_gpt61sol.yaml` | OpenAI `gpt-6.1-sol`: `max_completion_tokens`, temperature omitted (the model accepts only its default, so decoding is not deterministic), `reasoning_effort: low`. |

## Loading the stub (Person B)

B can use the stub three ways. All three realize the same plan with the same checks.

1. **In process through the contract adapter (unchanged).**
   `packages/simulator/response_adapter.py` calls `packages.response.stub.generate`, which
   is backward compatible: same signature, same `response_id` and `content_hash` as before,
   `delivery_mode` always `fallback`. Nothing in B's simulator changes.
2. **In process through the verified pipeline.**
   ```python
   from packages.response.pipeline import stub_pipeline
   response = stub_pipeline().generate(generate_request)   # GenerateRequest -> GenerateResponse
   ```
   `generate_request["response_artifact_id"]` must be `response_stub_pipeline_v1` or
   `response_stub_v1`. Invalid plans raise `packages.response.errors.GenerationError` with the
   PRD 9.5 code (`INVALID_SCHEMA` for an independent plan with a `reveal_solution`).
   `delivery_mode` is always `fallback` for template output, including the CHECK_MASTERY
   template, which asks two questions and goes through the one permitted repair: a template
   is a verified fallback realization, never model text (PRD 8.4; decision of 2026-11-02).
   Only a language-model responder yields `model` or `repaired`.
3. **Over HTTP.** Start the service and post `GenerateRequest` objects:
   ```bash
   uv run python -m tools.response.serve --config configs/response/serve_stub.yaml
   # in another shell
   curl -s http://127.0.0.1:8081/internal/v1/health
   curl -s -X POST http://127.0.0.1:8081/internal/v1/generate \
     -H 'Content-Type: application/json' \
     --data @contracts/fixtures/response/invalid/independent_reveal_request.json   # -> 422
   ```
   The stub config allows unauthenticated requests on a loopback address only. With
   `TUTOR_RESPONSE_SERVICE_TOKEN` set, every request needs
   `Authorization: Bearer $TUTOR_RESPONSE_SERVICE_TOKEN`.

The pipeline's `response_id` is deterministic for the same `request_id`, `plan_id` and
artifact; the legacy stub derives it from `request_id` alone, so the two differ for the same
request. `content_hash` is SHA-256 of the text plus the canonical effective action in both.

## One reproducible command

```bash
uv run python -m tools.response.publish_stub --out artifacts/response
```

writes `artifacts/response/response_stub_pipeline_v1/manifest.json`, `manifest.sha256` and
`fixtures/<plan>.json` (one `GenerateResponse` per fixture in
`contracts/fixtures/learning/valid`). The manifest records the prompt, validator, fallback,
repair and template versions, a hash of the fallback template sources and of their rendered
output on the fixtures, source hashes, the accepted `response_artifact_id` values, fallback
coverage per move, the rejected independent-reveal fixture, the source commit and the
limitations. `artifacts/` is ignored by Git; B compares `manifest.sha256` to pin a version.
Everything in the manifest except `created_at` and `source` is deterministic.

## Service reference

- `POST /internal/v1/generate`: `GenerateRequest` in, `GenerateResponse` out (both in
  `contracts/response_v1.schema.json`). All six checks are true on every 200 response.
- `GET /internal/v1/health` (alias `/healthz`): artifact id, versions, responder kind and
  auth mode. No authentication.
- Auth: bearer token read from the environment variable named by `service.auth.token_env`.
  A non-loopback bind always requires it. Wrong or missing token: `401 UNAUTHORIZED`.
- Limits: bodies above `service.max_request_bytes` (256 KB, PRD 9.5) are refused before
  they are read; `Content-Length` is required. `generation.internal_deadline_ms` (at most
  8000, PRD 14.3) caps the responder call together with the request's `deadline_ms`.
  `service.max_concurrent` is the admission limit; beyond it the service answers
  `429 RATE_LIMITED` (retryable). Requests for the same `session_id` run serially.
- Idempotency: a repeated `request_id` with identical content returns the cached response
  (same `response_id` and `content_hash`, PRD 21.1 item 5); the same `request_id` with
  different content returns `409 STATE_CONFLICT`. The cache is in memory and bounded by
  `service.idempotency_cache_size`; durable staging belongs to the public API.
- `tools.response.serve --check` validates a config, resolves the named environment
  variables and prints the startup summary without binding a port.

### Errors

Error bodies are `Error` objects (`contracts/learning_v1.schema.json`) with no plan content,
learner text or prompt internals in `message`.

| Situation | Status | Code | Retryable |
|---|---|---|---|
| Missing or wrong bearer token | 401 | `UNAUTHORIZED` | no |
| Body over 256 KB, no `Content-Length`, invalid JSON, NaN or Infinity | 422 | `INVALID_SCHEMA` | no |
| `GenerateRequest` schema failure or PRD 9.3 plan rule violation (including an independent plan with a `reveal_solution`) | 422 | `INVALID_SCHEMA` | no |
| Unsupported `schema_version` major, or a `response_artifact_id` this service does not serve | 400 | `UNSUPPORTED_VERSION` | no |
| Same `request_id` with different content | 409 | `STATE_CONFLICT` | no |
| Admission limit reached | 429 | `RATE_LIMITED` | yes |
| No text passed validation and no verified fallback realizes the plan | 503 | `GENERATION_UNAVAILABLE` | yes |
| `deadline_ms` of 0 | 504 | `DEADLINE_EXCEEDED` | yes |

Unknown paths return `404 {"message": "not found"}`.

### Logging (PRD 14.3 and 23)

Each request writes one JSON line on the `tuturing.response_service` logger with the event,
status, error code, `request_id`, `trace_id`, `plan_id`, `session_id`, move, difficulty,
pace, delivery mode, the six check results, token counts, elapsed milliseconds, versions
and a redacted attempt trail (stage and outcome only). Learner text, plan content, prompts,
rejected drafts, response text and tokens are never logged. The integration tests assert
this with sentinel strings.

## Configuration

All configs share one layout (`apps/response_service/config.py` validates it; unknown keys,
placeholders and version pins that disagree with the code are rejected at startup):

```yaml
schema_version: 1
service:    {host, port, auth: {token_env, allow_unauthenticated_loopback},
             max_request_bytes, max_concurrent, idempotency_cache_size}
artifact:   {id, compatible_ids}
responder:  {kind: template_stub}
            # or
            {kind: openai_compatible, base_url_env | base_url, model_env | model,
             api_key_env, api_key_required, request_timeout_s,
             decoding: {token_param, temperature, reasoning_effort}, extra_body}
generation: {internal_deadline_ms, max_output_tokens, max_repairs}
versions:   {prompt, validator, fallback}
```

Environment variables used by `serve_openai_compatible.yaml`: `TUTOR_LLM_BASE_URL`,
`TUTOR_LLM_MODEL`, `TUTOR_LLM_API_KEY` (optional for a local server because
`api_key_required: false`) and `TUTOR_RESPONSE_SERVICE_TOKEN`. `serve_gpt61sol.yaml` pins
the base URL and model and reads `OPENAI_API_KEY` (the name used in
`~/.config/tuturing/llm.env`; `set -a; . ~/.config/tuturing/llm.env; set +a` exports it
without printing) and `TUTOR_RESPONSE_SERVICE_TOKEN`; it requires the service token even on
loopback because every request spends money. Keys are read from the environment at startup
and are never printed, logged or written. The client refuses redirects so the key cannot
follow one to another host.

A local llama.cpp server works with
`llama-server -m <model.gguf> --port 8080` and `TUTOR_LLM_BASE_URL=http://127.0.0.1:8080/v1`
(delete `reasoning_effort` from the decoding block if the server rejects it).

### Decoding per model

`responder.decoding` uses the vocabulary of `configs/response/benchmark_v1.yaml`:

| Key | Values | Default | Sent as |
|---|---|---|---|
| `token_param` | `max_tokens` or `max_completion_tokens` | `max_tokens` | that field, set to `generation.max_output_tokens` (at most 384, PRD 14.1) |
| `temperature` | a number in [0, 2], or `null` | `0` (greedy) | `temperature`; `null` omits the field for models that accept only their default |
| `reasoning_effort` | a short lowercase name (`none`, `low`, ...) or `null` | `null` | `reasoning_effort` when set |

The output cap stays in `generation.max_output_tokens`. For a reasoning model the cap sent as
`max_completion_tokens` covers reasoning plus the visible reply, which is stricter than PRD
14.1; a reply cut off at the cap is repaired once or falls back. `extra_body` adds any other
provider field (for example `chat_template_kwargs`), but a config that sets a field twice is
rejected at startup with an error naming where the field belongs: `temperature`,
`max_tokens`, `max_completion_tokens` and `reasoning_effort` belong in `decoding`; `model`,
`messages`, `stream` and `n` are set by the responder; `top_p` and `seed` are sampling
controls serving does not take. `/healthz` and `serve --check` report the decoding in use
(`greedy`, `provider_default_temperature` or, for the stub, `deterministic`).

### Provider configuration errors are never silent

A provider that refuses the configured request format no longer looks like an ordinary
fallback. `OpenAICompatibleResponder` classifies HTTP errors:

| Provider answer | Classified as | Attempt outcome |
|---|---|---|
| 400 or 422 for a parameter (for example `temperature` or `max_tokens` unsupported) | `invalid_parameter`, with the provider's `param` and `code` | `config_error` |
| 401, 403, 404 (bad key, no access, wrong URL or model) | `unauthorized`, `forbidden`, `not_found` | `config_error` |
| 400 `context_length_exceeded` or a content-filter code | request-specific | `error` |
| 429, 5xx, timeouts, network failures | transient | `error` or `timeout` |

The learner still receives the verified fallback. The request's log line is written at
WARNING with a top-level `responder_config_error`, for example
`HTTP 400 invalid_parameter param=temperature code=unsupported_value`, and the attempt trail
carries the same detail. Only the status, reason, parameter name and error code are logged:
never the provider's message, learner text or the key. `GET /internal/v1/health` reports
`status: degraded` with `responder_config_errors` (count, last detail, last time) once any
rejection has occurred since startup.

To test a profile against its provider before serving, ask explicitly:

```bash
uv run python -m tools.response.serve --config configs/response/serve_gpt61sol.yaml --check --probe
```

`--probe` sends one request in the configured format (same token field, temperature,
`reasoning_effort` and `extra_body`) with a synthetic two-message prompt, prints `accepted`
or the classified rejection with the provider's message after removing key-like strings, and
exits 3 if the provider refuses. It costs a few tokens, so it never runs without `--probe`
and never in tests; `--check` alone opens no connection.

## Smoke gate

```bash
uv sync --extra learning
uv run pytest -q
uv run ruff check .
uv run python -m tools.response.validate_contracts
uv run python -m tools.learning.validate_contracts
uv run python -m tools.response.serve --config configs/response/serve_stub.yaml --check
```

Tests: `tests/unit/response/test_pipeline.py`, `test_responders.py` (against a local fake
OpenAI-compatible server; no real API is ever called), `test_service_config.py`,
`test_publish_stub.py` and `tests/integration/response/test_service_http.py`.

## Item fidelity (validator_v2)

benchmark_v1 found that Qwen3.8-27B replaced the issued coin problem with a six-sided die on
both CHECK_MASTERY fixtures and the CHALLENGE fixture and passed every check, because the
coin stem has no digits for the number heuristics to anchor on. validator_v2 folds a
digit-independent item-fidelity check (`packages/response/item_fidelity.py`) into
`action_faithful` for the moves that carry an item: PRACTICE, CHALLENGE, CHECK_MASTERY, HINT
and PROBE with an item. The six-check contract is unchanged, and because `action_faithful`
is never repairable, a substituted item always falls back to the verified rendering.

Both signals work on normalized content words (lowercase, punctuation stripped, stopwords
dropped, light suffix stemming):

- **Presence.** PRACTICE, CHALLENGE and CHECK_MASTERY must reuse at least 50% of the stem's
  content words (a heavy paraphrase of a protected check falls back by design, PRD 14.2).
  HINT and PROBE must reuse at least 25% of them, or three of them, or one of the stem's
  numbers; a HINT may instead reuse at least 50% of the permitted hint's words, since the
  approved hint is the plan's own realization of the item.
- **No competing scenario.** The reply may not introduce a scenario family (coin, die,
  cards, spinner, raffle, each container or drawn object such as bag, jar, box, urn, marble,
  ball, bead, counter, tile, and each colour) or a quantity phrase such as `3 red marbles`
  or `six-sided die` that the stem, approved facts, hint, worked example, previous step,
  concept names, misconception summary and recent dialogue do not already contain. A family
  the plan mentions is allowed as a whole, so "tails" is fine in a reply about a coin that
  only named heads. A number counts as a quantity only when a scenario object noun follows
  it directly or after modifiers such as colours, "fair" or "sided" within the same
  sentence; instructional nouns ("two things", "three moves") never count, and "head",
  "tail", "flip" and "toss" are companions of the coin family, never evidence on their own
  ("in your head"). The 371 original SFT examples in `data/response/original_v1` all pass
  (`tools.response.build_original_sft` accepts 371 of 371); before that narrowing, three
  replies that embed the stem verbatim were rejected for "in your head" and "in three
  moves".

Calibration on the benchmark_v1 review-sheet texts (the only responses with text in the
repository; MathDial plans were approximated from the sheet's stem and last learner message,
without the rest of the dialogue):

| Set | Applicable replies | Fail item fidelity |
|---|---:|---:|
| Fallback renderer, B's 9 fixture plans and the 5 smoke item plans | 11 | 0 |
| gpt-6.1-sol, B fixtures and smoke | 11 | 0 |
| Qwen3.8-27B, B fixtures and smoke | 11 | 4 (the three die substitutions and the smoke CHALLENGE marble substitution) |
| gpt-6.1-sol, MathDial (false positives) | 15 | 0 |
| Qwen3.8-27B, MathDial (false positives) | 15 | 0 |

Before the number rule for referring moves, 3 of 15 gpt-6.1-sol and 2 of 15 Qwen MathDial
replies were flagged: all referred to the item through its numbers only ("What is 30 times
3?"). The check is a heuristic over the plan's own words: it cannot see a substitution that
reuses the stem's vocabulary, and the scenario lexicon covers probability devices, not
arbitrary story objects. `tools.response.benchmark` reruns under validator_v2 will show the
die substitutions as `action_faithful` failures.

## Limitations

- The template stub cannot adapt wording to the learner message; the model profile is
  prompt-only until an SFT artifact exists (D10 is pending Xueming's confirmation).
- The checks are heuristics over approved content; they do not prove free-form mathematics
  correct (PRD 14.1). Rejected drafts are kept only in memory for the redacted attempt
  trail; an access-controlled evaluation trace store is not implemented yet.
- The idempotency cache is per process and in memory. Durable request staging, crash
  recovery and the public API retry demonstration live in `apps/public_api`.
- The `prompt_only_openai_compatible_v1` artifact id does not pin the model; a frozen release
  must set `model` in the config and use a new artifact id.
- `prompt_only_gpt61sol_v1` cannot decode greedily (the model rejects `temperature: 0`), so
  two requests with different `request_id` values for the same plan may get different text;
  PRD 14.1 treats that as a separately reported stochastic experiment. Only the idempotency
  cache makes a retry of the same `request_id` identical.
- A provider config rejection is reported on every affected request, but nothing stops the
  service from sending the next request in the same format; fix the config and restart.
