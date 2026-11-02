# Response service runbook (Person A)

The response service turns a `TeachingPlan` issued by Person B into one checked tutor
message. It never decides what to teach: it verifies the plan, compiles the prompt, calls a
responder, runs the six response checks, repairs once for length or question count only,
and otherwise delivers a verified fallback or a retryable error (PRD 8.2 step 4, 9.4, 14).

Two responders exist behind one interface:

| Responder | Artifact | Config | What it does |
|---|---|---|---|
| `template_stub` | `response_stub_pipeline_v1` | `configs/response/serve_stub.yaml` | The deterministic CPU templates in `packages/response/stub.py` behind the verified pipeline. No model, no network, no credential. |
| `openai_compatible` | `prompt_only_openai_compatible_v1` | `configs/response/serve_openai_compatible.yaml` | Any OpenAI-compatible `/chat/completions` server (local llama.cpp `llama-server`, vLLM, or a hosted provider), greedy decoding, 384-token cap, standard library HTTP client. |

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
   `delivery_mode` is `model` when the template passes as is, `repaired` when the one
   permitted repair was needed (the CHECK_MASTERY template asks two questions), `fallback`
   otherwise.
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

Both configs share one layout (`apps/response_service/config.py` validates it; unknown keys,
placeholders and version pins that disagree with the code are rejected at startup):

```yaml
schema_version: 1
service:    {host, port, auth: {token_env, allow_unauthenticated_loopback},
             max_request_bytes, max_concurrent, idempotency_cache_size}
artifact:   {id, compatible_ids}
responder:  {kind: template_stub}
            # or
            {kind: openai_compatible, base_url_env | base_url, model_env | model,
             api_key_env, api_key_required, request_timeout_s, extra_body}
generation: {internal_deadline_ms, max_output_tokens, max_repairs}
versions:   {prompt, validator, fallback}
```

Environment variables used by `serve_openai_compatible.yaml`: `TUTOR_LLM_BASE_URL`,
`TUTOR_LLM_MODEL`, `TUTOR_LLM_API_KEY` (optional for a local server because
`api_key_required: false`) and `TUTOR_RESPONSE_SERVICE_TOKEN`. Keys are read from the
environment at startup and are never printed, logged or written. The client refuses
redirects so the key cannot follow one to another host.

A local llama.cpp server works with
`llama-server -m <model.gguf> --port 8080` and `TUTOR_LLM_BASE_URL=http://127.0.0.1:8080/v1`.

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
