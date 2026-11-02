# SFT smoke and full run on Fireworks: report (2026-11-02)

Saved by the coordinating session from the SFT agent's final report; the agent's harness did not allow it to write report files. `STATUS.md` in this directory is the agent's own committed status.

## Outcome

Nothing trained. Fireworks refuses dataset uploads on `accounts/peli` with HTTP 403 "dataset uploads are temporarily disabled, please contact Fireworks AI to get access" (account `READY`, `UNSUSPENDED`, `accountVerification NOT_VERIFIED`). Every managed fine-tuning job needs an uploaded dataset, so the smoke job (original_v1), the full job (mix_v1) and the Qwen3.5-9B fallback are all blocked. Spend: $0.00 of the $20 smoke cap and $0.00 of the $40 full-run cap. The only object created, an empty dataset, was deleted and verified gone; no job, model or deployment of ours exists on the account.

## What the free checks established

- Managed fine-tuning (`supervisedFineTuningJobs`) applies to `qwen3p8-27b` (`supervisedLoraTunable: true`, `useTrainingV2: true`). `supportsServerlessTraining: false` refers to Fireworks' separate own-training-loop API, so D10's "serverless training pool" wording does not describe this path.
- Price: $3.00 per million training tokens (LoRA, 16.1B to 80B tier), counted as dataset tokens times epochs; no GPU rental for training.
- Exposed settings: LoRA rank, epochs, learning rate, warmup steps, scheduler, batch size, max context, weight decay, evaluation dataset. Not exposed: LoRA dropout, target modules, gradient clipping, packing, seed, evaluation or checkpoint interval. A previous managed job on this account adapted every projection on this architecture, broader than D10's 160 attention-only modules.
- Serving: no validated shape supports multiple LoRA add-ons for this model; a hot-reload deployment on the 1x H100 shape passes the dry run and can serve the base model first and then the merged adapter in one window (untested live).

## Ready to run once uploads work

| Split | Examples | Total tokens | Trained tokens | Longest | Over 4,096 |
|---|---:|---:|---:|---:|---:|
| original_v1 train | 261 | 90,438 | 17,410 | 563 | 0 |
| original_v1 validation | 55 | 18,844 | 3,590 | 495 | 0 |

- Loss-mask inspection passed on 20 of 20 training examples and on 20 more spread across all concepts and moves; 20 of 20 MathDial examples in the mix pass too.
- Smoke job: rank 16, alpha 32, learning rate 1e-4, 16 samples per step, 3 epochs (51 steps), 3 warmup steps, max context 4,096 (`configs/response/sft_smoke_qwen38_fireworks.yaml`); estimated $0.81.
- Full job on mix_v1: 435 train and 92 validation, 214,989 tokens, 84 steps (`configs/response/sft_full_qwen38_fireworks.yaml`); estimated $1.93.
- Evaluation: `configs/response/sft_smoke_eval_v1.yaml` runs base then LoRA on one hot-reload deployment over the 55 validation cases, B's 9 fixtures and 8 smoke cases, and deletes the deployment with verification.
- Commands are in `docs/runbooks/sft_fireworks.md`.

## Open items

1. Peyton, as owner of `accounts/peli`, asks Fireworks to enable dataset uploads or verifies the account.
2. `data/manifests/response/mix_v1.json` on main predates the regenerated original_v1 build (train hash `b1c9e41e…` recorded, `243a6869…` on rebuild); it is being regenerated.
3. Proposed D10 wording updates (not applied): managed LoRA fine-tuning at $3.00 per million training tokens instead of "serverless training pool"; the hosted path adapts all projections; one adapter per hot-reload deployment.
