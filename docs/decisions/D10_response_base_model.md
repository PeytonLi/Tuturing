# D10: Response SFT base-model change

- **Status:** accepted by lead
- **Owner:** Person A

## Decision

By lead decision, this response SFT lane switches from the PRD's
`Qwen/Qwen2.5-7B-Instruct` to `Qwen/Qwen3.8-27B`. The selected checkpoint is
available in Fireworks' serverless training pool, which provides a practical
LoRA SFT path and a matching small-model inference target. This is a model
selection decision, not a claim that the new checkpoint is quality-equivalent
to the PRD model.

The pinned Hugging Face revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. At that revision the checkpoint
is Apache-2.0 licensed and not gated. Its Fireworks model identifier is
`accounts/fireworks/models/qwen3p8-27b`.

Fireworks serverless inference uses the OpenAI-compatible base URL
`https://api.fireworks.ai/inference/v1`. The response evaluator sends plain
HTTP requests through `urllib`; no Fireworks SDK dependency is needed. The
Fireworks Serverless Training service is the intended LoRA SFT path. The
repository's local `train_sft.py` path is not yet verified for this model.

## Architecture and LoRA scope

The checkpoint reports architecture
`Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5`, and pipeline
`image-text-to-text`. It includes both a vision tower and an MTP layer. The
language model has 64 layers under `model.language_model.layers.N`:

- 16 full-attention layers expose `self_attn.q_proj`, `k_proj`, `v_proj`, and
  `o_proj`.
- The other 48 layers use linear attention with `in_proj_qkv`, `in_proj_z`,
  `in_proj_a`, `in_proj_b`, `out_proj`, and `conv1d`.
- All 64 text layers have MLP `gate_proj`, `up_proj`, and `down_proj`.

The PRD target list `[q_proj, k_proj, v_proj, o_proj]` therefore adapts only
the 16 full-attention layers. PEFT suffix/name matching may also match modules
in the MTP layer or vision tower if those module names overlap.

The resolved config retains the PRD target list and explicitly records
`lora_target_modules_scope: decision_pending`. Before any training run, choose
and record one of these approaches:

1. Keep the PRD attention projections and scope matching to
   `model.language_model`; or
2. Expand the targets to include selected linear-attention projections such
   as `in_proj_qkv` and `out_proj`, and/or MLP projections.

That adaptation choice remains open. The current resolved config is not an
approval to start training with an unscoped suffix match.

## Local training, tokenizer, and memory checks

Because this is a multimodal ConditionalGeneration checkpoint,
`AutoModelForCausalLM` may not be the correct local model class. Verify that a
text-only local load works before any GPU training run; this decision does not
claim that the local loading path has been validated.

The pinned tokenizer was inspected with Transformers 5.22.1 and tokenizers
0.23.3 using `--inspect-masks 3`. Its default generation prompt ends with an
open `<think>` marker; the fully rendered assistant response adds an empty
thinking block, which made prompt token IDs fail the full-input prefix check.
`build_loss_mask` now passes `enable_thinking=False` to both template renders
when supported. The resulting prompt still contains a closed empty
`<think>...</think>` block, but its token IDs are a prefix of the full input.
All three sampled cases (HINT, EXPLAIN_CONCEPT, and PROBE) passed the prompt,
target-mask, learner-message, and end-of-turn checks. Re-check the pinned
tokenizer's template with `--inspect-masks` after tokenizer or template
changes.

The PRD's single-GPU 48 GB assumption (section 4.3) must be re-benchmarked for
the 27B bfloat16 checkpoint, or the Fireworks training path should be used.
