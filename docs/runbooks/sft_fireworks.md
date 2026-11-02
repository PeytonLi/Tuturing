# Runbook: LoRA SFT of the response model on Fireworks

This runbook trains the D10 base model (`Qwen/Qwen3.8-27B` at revision `1d4bf0f2…`, which
is `accounts/fireworks/models/qwen3p8-27b` on Fireworks) with Fireworks' managed
supervised fine-tuning. It then compares the base model and the LoRA on validation from
one deployment window (PRD 13.4 and 13.5; `docs/person_a_tasks.md` item 6).

- Tool: `tools/response/fireworks_sft.py`.
- Configs: `configs/response/sft_smoke_qwen38_fireworks.yaml` (smoke, original_v1) and
  `configs/response/sft_full_qwen38_fireworks.yaml` (full run, mix_v1).
- Evaluation: `configs/response/sft_smoke_eval_v1.yaml`, run through
  `tools/response/benchmark.py`.

## Status on 2026-11-02

Blocked. `accounts/peli` refuses dataset uploads: `:upload` returns HTTP 403 and
`:getUploadEndpoint` returns 400 code 9, both with "dataset uploads are temporarily
disabled, please contact Fireworks AI to get access". Managed jobs train only on uploaded
datasets, so no job ran. See `experiments/response/sft_smoke_v1/report.md`.

## 1. Data (free, local)

```bash
uv sync --extra learning
uv run python -m tools.response.prepare_sft --config configs/response/mathdial_v1.yaml  # full run only
uv run python -m tools.response.build_original_sft
uv run python -m tools.response.build_sft_mix --config configs/response/sft_mix_v1.yaml # full run only
uv run python -m tools.response.fireworks_sft convert                    # smoke
uv run python -m tools.response.fireworks_sft --config configs/response/sft_full_qwen38_fireworks.yaml convert
```

`convert` writes Fireworks chat JSONL under the config's `run_dir`, which Git ignores.
It also re-exports the original_v1 validation split as the benchmark case set
`experiments/response/sft_smoke_v1/validation_cases_v1.jsonl`. Each record has three
messages:

- the system message, with `weight 0`;
- the user message, with `weight 0`;
- the tutor reply, with `weight 1`.

Fireworks documents that `weight: 0` excludes a message from the loss and any other
number trains it.

The converter refuses an example in four cases:

- its stored prompt is not `compile_prompt(plan)`, which is what serving and the benchmark
  send;
- it has a role other than system and user;
- its reply is empty;
- its reply contains a `<think>` marker.

`build_original_sft` also rewrites `data/manifests/response/original_v1.json` and the
frozen `evaluation/response/probability_cases_v1.jsonl`. If `git status` shows changes to
either file, restore them (`git restore`) unless you mean to refresh the data. The test
split stays frozen.

## 2. Token, mask and cost checks (free; downloads only tokenizer files)

```bash
HF_HUB_OFFLINE=0 uv run --extra learning --extra response-train \
  python -m tools.response.fireworks_sft tokens
uv run --extra learning --extra response-train python -m tools.response.train_sft \
  --config configs/response/sft_smoke_qwen38_fireworks.yaml --check-config
uv run --extra learning --extra response-train python -m tools.response.train_sft \
  --config configs/response/sft_smoke_qwen38_fireworks.yaml --inspect-masks 20 \
  --cases data/sft/original_v1/train.jsonl
```

`tokens` counts every example with the pinned tokenizer, using the same rendering and
mask as `train_sft.build_loss_mask` (`enable_thinking=False`). It writes `tokens.json`
with the optimizer steps and the cost estimate:

- training tokens are dataset tokens times epochs (Fireworks' rule);
- the price is $3.00 per million tokens (Managed Training, LoRA SFT, 16.1B–80B tier);
- the upper bound adds one evaluation pass per epoch plus one;
- the reserve is 1.5 times the upper bound.

The paid step reads the estimate from `tokens.json`, so run `tokens` first.

## 3. Train (paid)

```bash
uv run python -m tools.response.fireworks_sft upload          # free; creates two datasets
uv run python -m tools.response.fireworks_sft train           # dry run: request and estimate
uv run python -m tools.response.fireworks_sft train --confirm # creates the job
uv run python -m tools.response.fireworks_sft poll            # polls every 60 s, then settles
uv run python -m tools.response.fireworks_sft fetch           # metrics, rendered masks, LoRA details
uv run python -m tools.response.fireworks_sft status
```

For the full run, add `--config configs/response/sft_full_qwen38_fireworks.yaml` to every
command.

Spend rules:

- The keys file must define `TUTURING_BENCH_SPEND_CAP_USD`, which authorizes paid calls.
  The config's `spend_cap_usd` (smoke $20, full $40) can only lower that cap.
- Each run directory has its own `ledger.jsonl`.
- `train --confirm` refuses to start unless the committed spend plus the reserve stays
  under the cap. It records the reserve as `sft_job_reserve`.
- When the job ends, `poll` adds an `sft_job_settle` adjustment. The ledger then totals
  the larger of Fireworks' `estimatedCost` and the token estimate.
- `poll` cancels the job (`DELETE`) if Fireworks' `estimatedCost` passes the reserve.

Settings sent to Fireworks (`fireworks.job`):

- LoRA rank 16 and alpha 32;
- learning rate 1e-4, with a linear decay to 0;
- 16 samples per optimizer step (PRD micro-batch 1 × accumulation 16) and 3 epochs. That
  is 51 steps on the smoke data and 84 on the mix;
- warmup of ceil(0.05 × steps);
- weight decay 0.01 and max context 4,096;
- the validation split as `evaluationDataset`.

Managed SFT does not expose these settings, so `fetch` records what was actually used:

- LoRA dropout;
- target modules;
- gradient clipping;
- packing;
- seed;
- the evaluation and checkpoint interval.

For target modules, the model's `peftDetails.targetModules` shows what Fireworks chose.
An earlier managed job on this account (`qwen3p5-9b`) used every attention,
linear-attention and MLP projection. That is broader than D10's 160-module attention-only
scope.

## 4. Evaluate base and LoRA from one deployment (paid, GPU time)

```bash
uv run python -m tools.response.benchmark --config configs/response/sft_smoke_eval_v1.yaml plan
uv run python -m tools.response.benchmark --config configs/response/sft_smoke_eval_v1.yaml \
  run --models qwen3.8-27b-base-hotreload,qwen3.8-27b-sft-smoke-v1 --manage-deployment
uv run python -m tools.response.benchmark --config configs/response/sft_smoke_eval_v1.yaml \
  run --models gpt-6.1-sol --case-sets validation --pilot 3   # optional reference, ~$0.08
uv run python -m tools.response.benchmark --config configs/response/sft_smoke_eval_v1.yaml report
```

No validated deployment shape for `qwen3p8-27b` has the `MULTI_LORA` capability.
`deploymentShapeVersions:match` with `enableAddons: true` returned none, and a
`validateOnly` create with `enableAddons` returned 400. A deployment with
`enableHotReloadLatestAddon: true` on the validated 1x H100 BF16 shape does pass
`validateOnly`. The harness therefore runs in this order:

1. It creates that deployment and runs the base entry on every case.
2. It loads the LoRA with `POST /v1/accounts/<acct>/deployedModels?replaceMergedAddon=true`,
   which merges it into the served weights.
3. It finds a route that answers, trying `{adapter}#{deployment}`, then `{adapter}`, then
   `{deployment}`.
4. It runs the adapter entry on the same cases.
5. It deletes the deployment in a `finally` block and verifies the deletion with `GET`
   and a `showDeleted` listing.

The harness stops if every pilot reply of the adapter equals the base reply, because that
means the LoRA was not applied. The deployment state file records the deployment ID, the
times, the LoRA load time, the route that served and the estimated GPU cost. If an
interrupted run left something behind, use these commands:

```bash
uv run python -m tools.response.fireworks_deployment --config configs/response/sft_smoke_eval_v1.yaml list
uv run python -m tools.response.fireworks_deployment --config configs/response/sft_smoke_eval_v1.yaml \
  delete --entry qwen3.8-27b-base-hotreload
```

Only delete deployments this run created. Other deployments on the shared account belong
to other people.

## 5. Checkpoint selection and the single test run

Select on validation only. The PRD 13.5 order is:

1. zero critical math and protected-disclosure failures;
2. action fidelity, including item fidelity;
3. a human pedagogical score, which does not exist yet;
4. length and cost.

Run the probability test cases (`evaluation/response/probability_cases_v1.jsonl`) once,
for the selected adapter only. Use a config whose only case set is that file. Do not
iterate after seeing test results.
