# Internal learning service: Person A hand-off

This service owns the learner's state. It chooses what to teach next. It grades answers and tracks
mastery. It does not generate tutor text. A renders B's plan, stages the exact text, commits delivery,
and only then shows it to the learner.

The runtime is CPU-only. It uses rules by default. To load a checkpoint, set
`strategist: {checkpoint: path/to/policy.pt}` in a learning config. The Navigator stays rules-based.
This is a prototype, not a production deployment or a claim about real learning outcomes.

## Start it

Run from the repository root:

```bash
uv sync --extra learning
export TUTOR_LEARNING_SERVICE_TOKEN=dev
uv run python -m tools.learning.serve --config configs/learning/rules.yaml --port 8100
```

`dev` is only a local demo token. Use a private token for real service calls. Do not commit it.
The server refuses to start if the variable is missing or empty.

The default host is `127.0.0.1`. `--host` changes it. `--db path/to/learning.sqlite3` overrides
the config's database path. The default is `work/learning.sqlite3`. Parent directories are created
automatically. This ignored file holds sessions, plans, evidence, events, and retry results.
Restart with the same file to resume. Stop with Ctrl-C.

## Headers and response format

Every call needs:

- `Authorization: Bearer <token>` matching the server's environment token.
- `X-Trace-Id`: an ASCII string of 1–128 characters.

Every POST also needs:

- `Content-Type: application/json`.
- `Idempotency-Key`: an ASCII string of 1–128 characters. It must equal `request_id` when the body
  has that field. Commit has no `request_id`; its key comes from this header alone.
- `Content-Length`: curl and most HTTP clients add this. Chunked requests are not supported.

Bodies must be valid JSON objects, at most 65,536 bytes. Unknown fields are rejected. Use
`schema_version: "1.0"`. All responses use `Content-Type: application/json`.
Success is HTTP 200, including creation. Unknown GET/POST routes return HTTP 422 `INVALID_SCHEMA`.
Missing or wrong auth, or a missing or invalid trace ID, returns 401 before the body is read.

The examples below were captured from a local run on 2026-11-02. Responses are abbreviated:
fields not shown still exist. The IDs, hash, actions, and statuses came from that run.
On a new database, use your returned IDs and hash, not these recorded IDs. Plans expire after
600 seconds by default. Keep these common settings for the curl examples:

```bash
BASE=http://127.0.0.1:8100/internal/v1
TOKEN=dev
TRACE=docs-trace
SID=s-409d3bbe032645a19203c760c244f921
PLAN=$SID-plan-1
ITEM=$SID-1-turn-01-item
```

## 1. Create a session

`POST /sessions` initializes state at version 0. It does not choose a plan or consume a turn.
Enabled goals are C01, C02, C03, C06, and C08. Pace is `concise`, `normal`, or `detailed`.

```bash
curl -sS "$BASE/sessions" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-create' \
  --data '{"schema_version":"1.0","request_id":"docs-create","goal_concept_id":"C01","pace_preference":"normal"}'
```

```json
{"schema_version":"1.0","session_id":"s-409d3bbe032645a19203c760c244f921","public_state":{"state_version":0,"status":"active","teaching_turns":0}}
```

## 2. Observe the learner and reserve a plan

`POST /observe-and-plan` accepts evidence, checks end rules, and reserves one pending plan.
Evidence is null before the first committed response only.

```bash
curl -sS "$BASE/observe-and-plan" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-observe' \
  --data "{\"schema_version\":\"1.0\",\"request_id\":\"docs-observe\",\"session_id\":\"$SID\",\"expected_state_version\":0,\"evidence\":null}"
```

```json
{"kind":"plan","teaching_plan":{"plan_id":"s-409d3bbe032645a19203c760c244f921-plan-1","based_on_state_version":0,"teaching_action":{"action_id":1,"move":"PROBE","difficulty":"easy","pace":"normal"},"assessment_mode":"independent","assistance":"none","expires_at":"2026-11-02T21:23:47.716921+00:00"}}
```

Render the full `teaching_plan`, not this abbreviated example. A plan may contain internal content
and constraints. Do not return the whole plan to the public client.

If the session is complete, the result is instead
`{"kind":"session_end","session_end":{...}}`. The session-end record includes `reason`,
`state_version`, `summary_content_id`, and `public_state`. Do not request another teaching action.

## 3. Commit staged delivery

`POST /commit-plan` marks the exact staged response as delivered. This is the only operation
that consumes a teaching turn. Generate and validate text first. Save it durably before committing.

```bash
curl -sS "$BASE/commit-plan" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-commit' \
  --data "{\"schema_version\":\"1.0\",\"session_id\":\"$SID\",\"plan_id\":\"$PLAN\",\"based_on_state_version\":0,\"response_id\":\"docs-response-1\",\"content_hash\":\"ef878ea4693373c00c4749f02df689614b0efeef4ec81b03fe28ba5fccf4db13\",\"effective_move\":{\"action_id\":1,\"move\":\"PROBE\",\"difficulty\":\"easy\",\"pace\":\"normal\"},\"delivery_mode\":\"model\"}"
```

```json
{"schema_version":"1.0","plan_id":"s-409d3bbe032645a19203c760c244f921-plan-1","status":"committed","state_version":1,"public_state":{"teaching_turns":1,"status":"active"}}
```

The recorded hash is an example from the transport run, not a constant to reuse. A must compute
SHA-256 over the exact staged text's UTF-8 bytes. Do not hash the JSON envelope. Do not add or remove
whitespace or newlines after hashing. In Python: `hashlib.sha256(text.encode("utf-8")).hexdigest()`.
B checks the hash format and stores it; B cannot inspect A's staged text. A still owns response
validation and action fidelity. `delivery_mode` is `model`, `repaired`, or `fallback`.

## 4. Query a plan's status

`GET /plans/{plan_id}` helps recover after a lost response or a crash. It also expires old pending
plans. A canceled or expired plan has status `canceled`. No idempotency key or body is needed.

```bash
curl -sS "$BASE/plans/$PLAN" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE"
```

```json
{"schema_version":"1.0","plan_id":"s-409d3bbe032645a19203c760c244f921-plan-1","status":"committed","content_hash":"ef878ea4693373c00c4749f02df689614b0efeef4ec81b03fe28ba5fccf4db13","response_id":"docs-response-1"}
```

For a committed plan, return A's saved text for that response. Never regenerate it.

## 5. Grade an issued problem

`POST /grade` grades a normalized answer for a problem B issued. It does not update belief.
Unknown instances or unparseable answers return `unknown` with `grade_source: "none"`.

```bash
curl -sS "$BASE/grade" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-grade' \
  --data "{\"schema_version\":\"1.0\",\"request_id\":\"docs-grade\",\"problem_instance_id\":\"$ITEM\",\"normalized_answer\":\"0\"}"
```

```json
{"schema_version":"1.0","problem_instance_id":"s-409d3bbe032645a19203c760c244f921-1-turn-01-item","correctness":"incorrect","grade_source":"deterministic"}
```

## 6. Send the next learner message as evidence

This is the same observe endpoint, now with evidence tied to the committed response.
This recorded example represents a learner who gave no normalized answer.
All Evidence fields shown are required, including nulls and empty lists.

```bash
curl -sS "$BASE/observe-and-plan" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-observe-2' \
  --data "{
    \"schema_version\":\"1.0\",\"request_id\":\"docs-observe-2\",\"session_id\":\"$SID\",
    \"expected_state_version\":1,
    \"evidence\":{
      \"schema_version\":\"1.0\",\"event_id\":\"docs-evidence-1\",\"session_id\":\"$SID\",
      \"trace_id\":\"docs-trace\",\"source_response_id\":\"docs-response-1\",
      \"problem_instance_id\":\"$ITEM\",\"observed_at\":\"2026-11-02T00:00:00Z\",
      \"normalized_answer\":null,\"correctness\":\"unknown\",\"grade_source\":\"none\",
      \"confidence\":1.0,\"assistance\":\"none\",\"explicit_request\":\"none\",
      \"requested_concept_id\":null,\"misconception_hypotheses\":[],
      \"raw_message_ref\":\"docs-message-1\",\"extractor_version\":\"extractor_v1\",\"extensions\":{}
    }
  }"
```

```json
{"kind":"plan","teaching_plan":{"plan_id":"s-409d3bbe032645a19203c760c244f921-plan-2","based_on_state_version":1,"teaching_action":{"action_id":37,"move":"PRACTICE","difficulty":"easy","pace":"normal"}}}
```

## 7. Cancel an unseen plan

`POST /cancel-plan` cancels a plan that was never shown to a learner. This example cancels the
second plan above, not the committed first plan. No teaching turn or state version is consumed.
Evidence already accepted by observe remains applied.

```bash
curl -sS "$BASE/cancel-plan" \
  -H "Authorization: Bearer $TOKEN" -H "X-Trace-Id: $TRACE" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: docs-cancel' \
  --data "{\"schema_version\":\"1.0\",\"request_id\":\"docs-cancel\",\"session_id\":\"$SID\",\"plan_id\":\"$SID-plan-2\"}"
```

```json
{"schema_version":"1.0","plan_id":"s-409d3bbe032645a19203c760c244f921-plan-2","status":"canceled","content_hash":null,"response_id":null}
```

## What A must send and save

1. Keep the session ID and latest committed state version. Send that version as
   `expected_state_version`. There is at most one pending plan per session.
2. Before the first committed response, send `evidence: null`. After a commit, evidence is required.
   Its `session_id` must match. Its `source_response_id` must be the last committed response ID.
   Its `problem_instance_id` must match that plan, including null for a plan without a problem.
   Use a stable `event_id` for the same learner message. B applies it only once.
3. Generate text that follows the plan and its reveal constraints. Save the exact text, response ID,
   and SHA-256 hash durably. `effective_move` must equal the plan's whole `teaching_action`:
   `action_id`, `move`, `difficulty`, and `pace`.
4. Commit before exposing text. If a request times out, retry with the same Idempotency-Key and the
   same body. A retry returns the saved result. A changed body needs a new key; key reuse with a
   different body is a conflict. Never regenerate a committed response.
5. On `PLAN_EXPIRED`, discard the unseen staged response. Ask observe for a fresh plan with a new
   request ID/key and the current version. Reuse the same evidence event ID. Earlier evidence is
   retained and will not be applied twice. Do the same after an intentional cancellation.
6. On `STATE_CONFLICT`, do not blindly retry with new IDs. Check the known plan's status and
   reconcile A's saved version and response. Retry the original request to recover its saved result
   if needed. The error includes `current_state_version` when a known session is involved. This
   service has no separate session-read endpoint. A must retain committed responses and versions.
7. Cancel only plans never shown to a learner. A committed plan cannot be canceled.
8. B changes mastery only when B's own grader finds a correct/incorrect answer and both the evidence
   and committed plan have `assistance: "none"`. Extractor correctness cannot override B's grade.
   Hints, examples, and shown answers do not move mastery. Grade alone never updates state.
9. For a learner stop, send evidence with `explicit_request: "stop"`. B returns session end without
   another teaching plan. A summary-generation failure must not reopen the session.

## Errors

An Error always has `schema_version`, `request_id`, `trace_id`, `code`, `message`, `retryable`, and
`current_state_version` (possibly null). Missing or invalid correlation IDs become `unknown`.
Messages never include answer keys or stack traces. Example from the unauthenticated curl call:

```json
{"schema_version":"1.0","request_id":"docs-unauthorized","trace_id":"docs-trace","code":"UNAUTHORIZED","message":"Valid bearer token and X-Trace-Id are required","retryable":false,"current_state_version":null}
```

The full PRD status map is below. Some codes are reserved and are not currently emitted by this
transport. `retryable` describes retrying unchanged input; conflicts need the recovery steps above.

| Code | HTTP | When it happens | Retryable |
|---|---:|---|---|
| INVALID_SCHEMA | 422 | Invalid JSON/schema, missing or mismatched key, unknown route/session/plan, disabled goal, wrong effective move | No |
| UNSUPPORTED_VERSION | 400 | Schema major version is not 1 | No |
| UNAUTHORIZED | 401 | Wrong/missing bearer token or invalid/missing trace ID | No; fix headers |
| FORBIDDEN | 403 | Reserved: caller lacks permission | No |
| STATE_CONFLICT | 409 | Stale version, competing pending plan, changed idempotent body, or changing a committed plan | No; reconcile first |
| PLAN_EXPIRED | 409 | Commit attempted after the pending timeout | No; reserve a new plan |
| SESSION_ENDED | 409 | A fresh mutation targets an ended session | No |
| RATE_LIMITED | 429 | Reserved: request limit reached | Follow the returned flag; normally yes after waiting |
| GENERATION_UNAVAILABLE | 503 | Reserved for generation failure; B does not generate text | Follow the returned flag |
| LEARNING_UNAVAILABLE | 503 | Unexpected service failure or invalid policy action | Yes |
| DEADLINE_EXCEEDED | 504 | Reserved: service deadline exceeded | Follow the returned flag |

The Error definition is in `contracts/learning_v1.schema.json`. Additional internal request and
response schemas are in `apps/learning_service/schemas.py`. The HTTP layer performs no learning
transactions itself. It uses `LearningEngine` and does not log request bodies.
