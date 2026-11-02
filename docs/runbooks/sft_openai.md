# Runbook: supervised fine-tuning on OpenAI (option B)

This is the fallback while Fireworks refuses dataset uploads on `accounts/peli`
(`docs/runbooks/sft_fireworks.md`). The default plan is one job on the full 60/40 mix,
3 epochs, with OpenAI's defaults otherwise. The base is `gpt-4.1-mini-2025-04-14`.

- Tool: `tools/response/openai_sft.py`.
- Configs: `configs/response/sft_openai_{full,smoke}_v1.yaml`, plus
  `sft_openai_eval_v1.yaml` and `sft_openai_test_v1.yaml` for the benchmark harness.
- Run directories (git-ignored), each with its own ledger:
  `experiments/response/sft_openai_v1/runs/{full,smoke}`.

Nothing is uploaded and no job is created until Xueming approves option B.

## Facts behind the plan (developers.openai.com, checked 2026-11-02)

- Supported models: `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14` and
  `gpt-4.1-nano-2025-04-14`. OpenAI is winding the platform down. New users cannot use
  it, and eligible existing customers can create jobs until January 6, 2027.
- This key lists fine-tuning jobs (none so far) and sees all three snapshots. It has never
  run a job, so whether it may create one shows only at `train --confirm`. A refusal there
  costs nothing.
- Prices per million tokens (standard):

| Model | Training | Fine-tuned input | Fine-tuned output |
|---|---:|---:|---:|
| gpt-4.1-mini | $5 | $0.80 | $3.20 |
| gpt-4.1 | $25 | $3.00 | $12.00 |

- Training is billed as tokens in the training file times epochs. The job reports
  `trained_tokens` when it finishes.
- Format:
  - chat JSONL with at least 10 examples;
  - 65,536 tokens per example for the gpt-4.1 family, truncated from the end;
  - `weight` (0 or 1) on assistant messages only.

  Our files hold the system and user prompt from `compile_prompt`, then the tutor reply
  with `weight: 1`. That reply is the only trained turn.
- OpenAI saves a checkpoint at the end of each epoch and keeps the last three, so all
  three epochs of a 3-epoch job can be evaluated.

## Commands

```bash
uv sync --extra learning
# data (free, local); the full plan uses mix_v1
uv run python -m tools.response.prepare_sft --config configs/response/mathdial_v1.yaml
uv run python -m tools.response.build_original_sft     # then git restore any changed committed file
uv run python -m tools.response.build_sft_mix --config configs/response/sft_mix_v1.yaml
C=configs/response/sft_openai_full_v1.yaml
uv run python -m tools.response.openai_sft --config $C convert
uv run --with tiktoken python -m tools.response.openai_sft --config $C check   # format, tokens, cost

# paid, after approval only
uv run python -m tools.response.openai_sft --config $C upload           # two files, purpose fine-tune
uv run python -m tools.response.openai_sft --config $C train            # dry run: body and estimate
uv run python -m tools.response.openai_sft --config $C train --confirm  # creates the job
uv run python -m tools.response.openai_sft --config $C poll             # every 60 s, then settles
uv run python -m tools.response.openai_sft --config $C fetch            # events, checkpoints, models.json

# validation, checkpoint selection, then the single test run
E=configs/response/sft_openai_eval_v1.yaml
for M in gpt-4.1-mini-base gpt-4.1-mini-sft-full-final gpt-4.1-mini-sft-full-epoch1 \
         gpt-4.1-mini-sft-full-epoch2; do
  uv run python -m tools.response.benchmark --config $E run --models $M --pilot 3
  uv run python -m tools.response.benchmark --config $E run --models $M
done
uv run python -m tools.response.benchmark --config $E report
uv run python -m tools.response.openai_sft --config $C select --key <final|epoch_1|epoch_2> \
  --reason "<PRD 13.5 ordering result>"
T=configs/response/sft_openai_test_v1.yaml
uv run python -m tools.response.benchmark --config $T run --models gpt-4.1-mini-sft-selected --pilot 3
uv run python -m tools.response.benchmark --config $T run --models gpt-4.1-mini-sft-selected
uv run python -m tools.response.benchmark --config $T report
```

## Spend rules

- The keys file's `TUTURING_BENCH_SPEND_CAP_USD` authorizes paid calls. A config's
  `spend_cap_usd` can only lower it. Full run: min(keys file, $40). Smoke: min(keys file,
  $20).
- `check` must pass for the exact file hashes before `upload`.
- `train --confirm` refuses to start unless the committed spend plus the reserve (1.5 ×
  the upper bound) stays under the cap.
- `poll` settles the ledger at `trained_tokens` × the training price. If OpenAI reports
  no `trained_tokens`, the reserve stays as the charge until checked by hand.
- Evaluation calls reserve their worst case before every request, and paid entries need a
  pilot first.
- The evaluation and test configs share the full run's ledger.

## Selection

Select on validation only. The PRD 13.5 order is:

1. zero critical math and protected-disclosure failures;
2. action fidelity, including item fidelity;
3. a human score, which does not exist yet;
4. length and cost.

`select` records the chosen key and the reason in `models.json`. The test config runs that
model once on the 55 frozen probability cases. The prompt-only numbers on main for that set
are fallback 55/55, gpt-6.1-sol 33/55 and Qwen3.8-27B 19/55 delivered as model.
