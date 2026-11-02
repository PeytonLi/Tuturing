# SFT on Fireworks: status

Updated 2026-11-02 23:25 UTC.

## Blocked: no job was submitted, $0.00 spent

Fireworks refuses dataset uploads for `accounts/peli`. A managed SFT job can only train
on an uploaded dataset, so neither job A (original_v1) nor job B (mix_v1) can start, on
`qwen3p8-27b` or on the `qwen3p5-9b` fallback. Changing the base model does not help: the
refusal is for the whole account, not for one model.

| Call (2026-11-02 UTC) | Result |
|---|---|
| `POST /v1/accounts/peli/datasets` (create `tut-sft-smoke-v1-train`, 261 rows, CHAT) at 23:19:53 | 200; the object was created in state `UPLOADING` |
| `POST .../datasets/tut-sft-smoke-v1-train:upload` (multipart, 431,252 bytes) | **403** `dataset uploads are temporarily disabled, please contact Fireworks AI to get access` |
| `POST ...:getUploadEndpoint` (signed-URL path for larger files) | **400**, code 9, same message |
| `POST ...:validateUpload` | **400**, code 9, same message |
| `DELETE` of the empty dataset object at 23:21:50 | 200; `GET` 404 and absent from the dataset listing at 23:21:57 |

The account reads `state READY`, `suspendState UNSUSPENDED`,
`accountVerification NOT_VERIFIED`. Peyton's earlier uploads (August and September 2026)
went through, so the gate is recent. Whether verifying the account clears it is not
documented. No other upload route was tried: the provider's message says to contact them.

What this run left on Fireworks: nothing. There are no datasets, jobs, models or
deployments of ours. The other qwen3p8-27b deployments on `accounts/peli`
(`tutoring-qwen38-fullsess-1793660139`, `kn-qwen27b-1793661352`) were not touched.

## Ready to launch the moment uploads work

- Model: `accounts/fireworks/models/qwen3p8-27b` (`supervisedLoraTunable: true`,
  `useTrainingV2: true`; `supportsServerlessTraining: false` refers to the Training API's
  serverless loop, not to managed jobs).
- Data A (original_v1): train 261, validation 55, converted to Fireworks chat JSONL. System
  and user turns have `weight 0` and the tutor reply has `weight 1`. Tokens with the pinned
  tokenizer: train 90,438 in total (17,410 of them on the reply), validation 18,844, longest
  example 563 tokens.
- Job A: LoRA rank 16, alpha 32, LR 1e-4, 16 samples per step, 3 epochs, which is 51
  optimizer steps. Warmup is 3 steps, then a linear decay. Max context 4,096. The
  validation split is the evaluation dataset.
- Cost A: $3.00 per million training tokens (Managed Training, LoRA SFT, 16.1B to 80B
  tier). Expected $0.81. The upper bound is $1.04 if evaluation tokens are also billed, and
  $1.56 is held in reserve.
- Commands:
  `uv run python -m tools.response.fireworks_sft upload` then
  `uv run python -m tools.response.fireworks_sft train --confirm` then `poll`.
- Job B (mix_v1): being prepared on this branch next; its estimate will be added here.

## What needs a decision (Xueming / Peyton)

1. Peyton, as the `accounts/peli` owner, asks Fireworks to re-enable dataset uploads, or
   verifies the account, then tells us. Relaunch takes about 2 minutes. Training costs
   about $0.81 for A, and B is estimated below.
2. Or Xueming's own account (`accounts/xueming-r9e8nrrxg6ml`) gets a payment method, then
   a key in `~/.config/tuturing/llm.env`. Whether that account may upload is unknown.
3. Not tried: the Fireworks Training API, which runs your own loop and bills per GPU hour.
   It needs a new SDK, and `supportsServerlessTraining` is false for both Qwen models.
   Renting a GPU elsewhere for `train_sft.py` would need a new provider account, and the
   local trainer is not yet validated for this multimodal checkpoint (D10).
