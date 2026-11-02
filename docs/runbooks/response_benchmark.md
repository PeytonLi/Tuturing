# Runbook: prompt-only response benchmark

Compares frontier and open-source models, prompt only, on the tutoring response task so
Person A can choose the SFT base model (D10) and the demo responder (PRD 13.5, 14.1, 18.2;
`docs/person_a_tasks.md` item 4). Harness: `tools/response/benchmark.py`; matrix:
`configs/response/benchmark_v1.yaml`; report: `experiments/response/benchmark_v1/`.

## Rerun everything

```bash
uv sync --extra learning \
  && uv run python -m tools.response.prepare_sft --config configs/response/mathdial_v1.yaml \
  && uv run python -m tools.response.benchmark plan \
  && uv run python -m tools.response.benchmark run --models fallback \
  && uv run python -m tools.response.benchmark run --models gpt-6.1-sol --pilot 3 \
  && uv run python -m tools.response.benchmark run --models gpt-6.1-sol \
  && uv run python -m tools.response.benchmark run --models qwen3.8-27b-fw-h100 --manage-deployment \
  && uv run python -m tools.response.benchmark report
```

`plan` makes no model calls: it lists the case sets, which entries will run or why they
are skipped, and the worst-case cost of each paid entry. `run` writes one JSONL record per
case under `experiments/response/benchmark_v1/runs/<model id>/` (git-ignored) and resumes:
cases whose record already matches the compiled prompt and decoding settings are not called
again. `report` rebuilds the CSVs, `report.md` and the blinded review sheet from those
records. Add `--case-sets smoke,b_fixtures` or `--limit 20` to any `run` for a cheap check.

Every run goes through the same steps per case: `packages.response.prompt.compile_prompt`,
one chat completion, `packages.response.validate.run_checks`, then the delivery mode the
serving pipeline would pick (`model`, `repair`, `fallback`, `error`), latency, tokens and
cost. Hosted models that cannot turn thinking off have `<think>` blocks stripped and the
record flags it.

## Keys and the spend cap

The runner reads keys only from `~/.config/tuturing/llm.env` (mode 600, `KEY=value` lines)
and never prints or stores their values. Create it once:

```bash
mkdir -p ~/.config/tuturing && chmod 700 ~/.config/tuturing
cat > ~/.config/tuturing/llm.env <<'EOF'
OPENAI_API_KEY=...
FIREWORKS_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
TUTURING_BENCH_SPEND_CAP_USD=25
EOF
chmod 600 ~/.config/tuturing/llm.env
```

- An entry whose `api_key_env` is missing is skipped with a message; nothing else changes.
- Without `TUTURING_BENCH_SPEND_CAP_USD` no paid call is made at all.
- The cap is cumulative over `experiments/response/benchmark_v1/runs/ledger.jsonl`, which
  records every paid call's usage and actual cost (usage times the list prices in the
  config), every pilot estimate, and every GPU deployment's hours. Before each call the
  runner reserves the worst case (estimated input plus the full output cap) and refuses the
  call if reserved plus spent would pass the cap. To spend more later, raise the cap in the
  keys file; do not delete the ledger. `--max-usd 2` adds a per-invocation limit.
- Paid entries need a pilot (`--pilot 3`) before a full run. The full run extrapolates the
  pilot's mean cost per case (times `estimate_safety_factor`) and, if the remaining budget
  cannot cover every pending case, takes an evenly stratified prefix across case sets and
  moves; below `min_subsample_cases` it stops and reports the estimate instead.

## Qwen3.8-27B on a Fireworks on-demand deployment

`accounts/fireworks/models/qwen3p8-27b` reports `supportsServerless: false`, so D10's model
needs a dedicated deployment for benchmarking and for serving. The entry
`qwen3.8-27b-fw-h100` uses the validated shape "Qwen3.8 27B BF16 1x H100"
(`accounts/fireworks/deploymentShapes/sva2n3mh`, list price $8.00 per GPU hour). Rules the
runner enforces with `--manage-deployment`:

1. It refuses to start unless the cap covers the full `max_minutes` reserve.
2. `minReplicaCount: 0`, `maxReplicaCount: 1`, `scaleToZeroWindow: 300s` (the minimum).
3. Create, wait for `READY`, warm a replica with a one-word request, pilot, estimate from
   measured latency, run what fits the time and money left, then delete in a `finally`
   block and verify with `GET` (404 or state `DELETED`) plus a `showDeleted` listing.
4. The state file `runs/deployments/<entry>.json` records the deployment ID, shape,
   accelerator, timestamps, the verification responses and the estimated GPU cost (wall
   time from the create request to verified deletion, at list price: an upper bound).

Manual controls (free control-plane calls):

```bash
uv run python -m tools.response.fireworks_deployment validate --entry qwen3.8-27b-fw-h100
uv run python -m tools.response.fireworks_deployment list
uv run python -m tools.response.fireworks_deployment status --entry qwen3.8-27b-fw-h100
uv run python -m tools.response.fireworks_deployment delete --entry qwen3.8-27b-fw-h100
```

Run `list` after any interrupted run; if a deployment is still there, run `delete`. The
account must have a payment method on file: on 2026-11-02 `validate` returned
HTTP 400 `code 9: "payment method is required"` and no deployment was created.

## Local open models on this Mac (free)

Entries `qwen3.5-9b-q4km-local`, `qwen2.5-7b-instruct-q4km-local` and
`llama-3.1-8b-instruct-q4km-local` are configured but disabled (no new downloads were
wanted on 2026-11-02). To run one: check free memory (`memory_pressure`, keep 8 GB
free for a Q4_K_M 7-9B model), download the GGUF named in the entry's `weights` block
into `~/models/` from `https://huggingface.co/<repo>/resolve/<revision>/<file>`, start
the server with the entry's `server_command` (bound to 127.0.0.1, context 4096), set
`enabled: true`, run `benchmark run --models <id>`, then stop the server before starting
the next model. `llama-server 0.8.0` here lists the `qwen35` architecture, so the
Qwen3.5-9B proxy for the Qwen3.8 family loads.

## Adding a model or a case set

- Model: copy an entry in the config. Required: `id`, `provider` (`fallback` or
  `openai_compatible`), `base_url`, `model`, `api_key_env` for hosted entries, `decoding`
  (`temperature`, `max_tokens`, `token_param` for providers that need
  `max_completion_tokens`, `extra` for `reasoning_effort` or `chat_template_kwargs`),
  `reasoning.setting` (what was disabled or set, for the report), and `pricing` with
  `basis: tokens` (USD per million tokens plus `source` and `checked`), `gpu_time` (with a
  `deployment` block) or `free`.
- Case set: add a `case_sets` entry with a `path` to a JSONL of `{"example_id", "plan"}`
  records or a directory of TeachingPlan JSON files; `optional: true` lets a missing file
  skip cleanly. `evaluation/response/probability_cases_v1.jsonl` is already listed and is
  picked up as soon as it exists.

## Review

`experiments/response/benchmark_v1/review/review_sheet_blinded.csv` has about 40 cases
with every model's response shuffled and labelled `R1..Rk`; rate each row's columns, then
join with `review_key_do_not_open_before_rating.csv` on `item` and `response_label`.
