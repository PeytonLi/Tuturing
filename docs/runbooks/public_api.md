# Runbook: public API (Person A)

The public API (`apps/public_api`) is the learner-facing HTTP service of PRD 9.5. It sequences
Person B's learning service and Person A's response generation, keeps the durable request cache
and response staging of PRD 8.2 to 8.3, and never decides what to teach. Contract:
`contracts/public_v1.openapi.yaml` (placement rationale in `docs/decisions/D11_public_contract_placement.md`).

## Run locally (CPU only)

All commands run from the repository root with Python 3.12 and `uv sync --extra learning`.

1. Start Person B's learning service (its runbook is `docs/learning_service.md`):

   ```bash
   export TUTOR_LEARNING_SERVICE_TOKEN=dev      # local demo token only; never commit a real one
   uv run python -m tools.learning.serve --config configs/learning/rules.yaml --port 8100
   ```

2. Start the public API against it. Tokens are `caller=token` pairs in one environment variable;
   the config names the variable and never holds values:

   ```bash
   export TUTOR_LEARNING_SERVICE_TOKEN=dev
   export TUTOR_PUBLIC_API_TOKENS="demo=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
   uv run python -m apps.public_api --config configs/response/public_api_local.yaml
   ```

   `configs/response/public_api_local.yaml` defaults to `learning.mode: standin` (Person A's
   transaction stand-in from `tests/integration/response/learning_standin.py`, no second
   process) and `response.mode: in_process` (the CPU stub `packages.response.stub`). To use B's
   service set `learning.mode: http` and `learning.base_url: http://127.0.0.1:8100`. The SQLite
   request store defaults to `work/public_api/requests.sqlite` (ignored by Git).

3. Talk to it (`TOKEN` is the value after `demo=`):

   ```bash
   BASE=http://127.0.0.1:8080
   curl -sS $BASE/v1/sessions -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     --data '{"request_id":"c1","goal_concept_id":"C01","pace_preference":"normal"}'
   # -> 201 {"session_id":"s-...","public_state":{...,"state_version":0}}
   curl -sS $BASE/v1/sessions/$SID/turns -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     --data '{"request_id":"t1","expected_state_version":0,"source_response_id":null,"message":""}'
   # -> 200 {"request_id":"t1","response":{"response_id","plan_id","text","delivery_mode"},"public_state":{...}}
   curl -sS $BASE/v1/sessions/$SID/turns -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     --data '{"request_id":"t2","expected_state_version":1,"source_response_id":"<response_id of t1>","message":"is it 1/2?"}'
   curl -sS $BASE/v1/sessions/$SID -H "Authorization: Bearer $TOKEN"          # public state and transcript
   curl -sS $BASE/v1/sessions/$SID/stop -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     --data '{"request_id":"s1","expected_state_version":2}'                  # -> {"request_id","session_end"}
   ```

   The first turn sends `source_response_id: null` and an empty message; every later turn names
   the last committed `response_id` and the current `state_version`. Resend the identical body
   with the same `request_id` to retry; the reply is byte-identical and no learning update or
   generation is repeated. A `request_id` reused with a different body gets `409 STATE_CONFLICT`.

## Ten-session CPU smoke workflow (PRD 11.2, gate G4)

With B's service running on port 8100:

```bash
export TUTOR_LEARNING_SERVICE_TOKEN=dev
uv run python -m tools.response.smoke_sessions --learning-url http://127.0.0.1:8100 \
  --report work/public_api/smoke_report.json
```

Without B's service (stand-in): `uv run python -m tools.response.smoke_sessions --report work/public_api/smoke_report.json`.

The workflow creates ten scripted sessions, injects a dropped connection on one turn of every
session and retries it, checks that the retry returns the same `response_id` and that A's staged
`content_hash` equals the hash B recorded for the committed plan, stops each session, reads the
transcript, and writes a JSON report (counts, delivery modes, moves, end reasons, retry results,
latency; message lengths only, never learner text). Exit code 0 means every session completed,
every retry was identical and no public error occurred.

## Tests

```bash
uv run pytest -q tests/integration/response          # public API recovery, limits, HTTP, smoke
uv run python -m tools.response.validate_contracts   # response + public contract and fixtures
uv run python -m tools.response.make_public_fixtures # regenerate public fixtures after a change
```

`tests/integration/response/test_public_api.py` covers: create; first turn; learner turn; retry
after an injected dropped connection (same `response_id` and `content_hash`); `request_id` reuse
with different content; competing requests on one state version; crash after B's commit and
recovery through plan status; crash after storing the commit; generation failure (retryable,
plan stays pending) then retry; plan expiry; stop and session end; oversize and malformed
requests over HTTP; no internal fields in any public body; learner text never in logs.
`test_smoke_workflow.py` starts B's real service in-process on a free localhost port with a
temporary database and a throwaway token, runs the ten sessions through it, and is skipped if
no port can be bound.

## Behavior worth knowing

- Durable state: every request's progress (reserved plan, staged response and hash, commit
  attempt, commit result, final body) is written to SQLite before the next step, so a retry
  after any crash resumes without regenerating. A response is returned only after B committed it.
- Recovery after a crash between B's commit and A's record of it: A queries
  `GET /internal/v1/plans/{plan_id}`; B's status carries only the hash and response ID, so A
  derives the public state (`state_version + 1`, `teaching_turns + 1`, the plan's concept) and
  logs a warning. See D11.
- One request per session at a time: a second request while another is in flight gets
  `409 STATE_CONFLICT` (`retryable: true`); after the first commits, a stale version is a
  terminal `409`.
- A pending plan whose response was never committed is canceled with B when a newer request for
  the session arrives; the old request then answers `409 PLAN_EXPIRED` on retry.
- Stop before any committed response is `409 STATE_CONFLICT`: B accepts evidence, hence a stop,
  only after a committed response. Clients can simply abandon such a session.
- Logs carry trace, request, session and plan IDs, statuses and codes; never learner text.
- Limits: 65,536-byte bodies, 8,000-character messages, 10 s request deadline, 8 s generation
  ceiling, one generation retry (PRD 14.3).

## Not done yet

- No rate limiting (`RATE_LIMITED` is declared, never emitted) and no per-session serial queue
  across processes; the session lock is per process.
- The response service over HTTP (`response.mode: http`) is coded to `GenerateRequest` and
  `GenerateResponse` only; `apps/response_service` is built on another branch.
- Public response streaming is deferred (PRD 8.3).
