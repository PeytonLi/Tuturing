"""SFT helpers aligned with TRL v1.16.0 SFTTrainer/SFTConfig docs
(https://huggingface.co/docs/trl/v1.16.0/en/sft_trainer), retrieved via
Context7 /huggingface/trl/v1.16.0.
"""

import argparse
import json
import math
from pathlib import Path

import yaml

CONFIG_KEYS = (
    "run_type", "base_model", "base_model_revision", "tokenizer_revision",
    "method", "lora_rank", "lora_alpha", "lora_dropout",
    "lora_target_modules", "precision", "max_sequence_tokens",
    "micro_batch_size", "gradient_accumulation_steps", "learning_rate",
    "epochs_max", "warmup_ratio", "weight_decay", "gradient_clip_norm",
    "packing", "loss_scope", "evaluation_interval_optimizer_steps",
    "checkpoint_interval_optimizer_steps", "seed",
)
_MISSING_REVISION = "REQUIRED_RESOLVED_COMMIT"


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _contains_value(value, sought):
    if isinstance(value, dict):
        return any(_contains_value(item, sought) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, sought) for item in value)
    return value == sought


def check_config(cfg):
    errors = [f"missing key: {key}" for key in CONFIG_KEYS if key not in cfg]
    if _contains_value(cfg, _MISSING_REVISION):
        for key in ("base_model_revision", "tokenizer_revision"):
            if cfg.get(key) == _MISSING_REVISION:
                errors.append(f"{key} must replace {_MISSING_REVISION}")
        if not any(_MISSING_REVISION == value for value in cfg.values()):
            errors.append(f"configuration contains {_MISSING_REVISION}")
    if cfg.get("method") != "lora":
        errors.append("method must be lora")
    if cfg.get("loss_scope") != "target_completion_only":
        errors.append("loss_scope must be target_completion_only")
    if cfg.get("packing") is not False:
        errors.append("packing must be false")
    for key in (
        "lora_rank", "lora_alpha", "max_sequence_tokens", "micro_batch_size",
        "gradient_accumulation_steps", "epochs_max",
        "evaluation_interval_optimizer_steps",
        "checkpoint_interval_optimizer_steps",
    ):
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in ("learning_rate", "gradient_clip_norm"):
        value = cfg.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"{key} must be positive")
    for key in ("lora_dropout", "warmup_ratio"):
        value = cfg.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0 <= value < 1
        ):
            errors.append(f"{key} must be in [0, 1)")
    value = cfg.get("weight_decay")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        errors.append("weight_decay must be non-negative")
    seed = cfg.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("seed must be a non-negative integer")
    if cfg.get("precision") not in {"bfloat16", "float16", "float32"}:
        errors.append("precision must be bfloat16, float16, or float32")
    if not isinstance(cfg.get("lora_target_modules"), list) or not cfg.get("lora_target_modules"):
        errors.append("lora_target_modules must be a non-empty list")
    return errors


def build_sft_config_kwargs(cfg, output_dir):
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg["epochs_max"],
        "per_device_train_batch_size": cfg["micro_batch_size"],
        "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
        "learning_rate": cfg["learning_rate"],
        # Transformers 5 removed warmup_ratio; a fractional warmup_steps is
        # interpreted as the same fraction of total optimizer steps.
        "warmup_steps": cfg["warmup_ratio"],
        "weight_decay": cfg["weight_decay"],
        "max_grad_norm": cfg["gradient_clip_norm"],
        "bf16": cfg["precision"] == "bfloat16",
        "max_length": cfg["max_sequence_tokens"],
        "packing": False,
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "eval_strategy": "steps",
        "eval_steps": cfg["evaluation_interval_optimizer_steps"],
        "save_strategy": "steps",
        "save_steps": cfg["checkpoint_interval_optimizer_steps"],
        "seed": cfg["seed"],
        "model_init_kwargs": {
            "revision": cfg["base_model_revision"],
            "dtype": cfg["precision"],
        },
    }


def build_lora_kwargs(cfg):
    return {
        "r": cfg["lora_rank"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": cfg["lora_target_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def to_prompt_completion(example):
    return {"prompt": example["prompt"], "completion": example["completion"]}


def _input_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _find_subsequence(sequence, subsequence):
    if not subsequence:
        return None
    for start in range(len(sequence) - len(subsequence) + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return None


def build_loss_mask(tokenizer, example):
    prompt_messages = example["prompt"]
    completion_messages = example["completion"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False
    )
    full_text = tokenizer.apply_chat_template(
        [*prompt_messages, *completion_messages],
        add_generation_prompt=False,
        tokenize=False,
    )
    end_token = getattr(tokenizer, "im_end_token", None) or getattr(
        tokenizer, "eos_token", None
    )
    if end_token:
        end_position = full_text.rfind(end_token)
        if end_position >= 0:
            suffix_start = end_position + len(end_token)
            if not full_text[suffix_start:].strip():
                full_text = full_text[:suffix_start]
    if not full_text.startswith(prompt_text):
        raise ValueError("full chat rendering does not start with the generation prompt")
    prompt_ids = _input_ids(tokenizer, prompt_text)
    full_ids = _input_ids(tokenizer, full_text)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt token ids are not a prefix of full input token ids")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("completion has no target tokens")
    mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    target_ids = [token for token, loss in zip(full_ids, mask) if loss]
    end_id = getattr(tokenizer, "im_end_token_id", None)
    if end_id is None:
        end_id = getattr(tokenizer, "eos_token_id", None)
    if end_id is None:
        raise ValueError("tokenizer has no end-of-turn token id")
    if target_ids[-1] != end_id:
        raise ValueError("last target token is not the tokenizer end-of-turn token")

    plan = example.get("plan", {})
    content = plan.get("content", {})
    reveal = content.get("reveal_solution")
    reveal_allowed = plan.get("constraints", {}).get("reveal_allowed")
    if not reveal_allowed and reveal and reveal in prompt_text:
        raise ValueError("protected reveal_solution appears in the prompt")
    last_message = plan.get("response_context", {}).get("last_student_message", "")
    if last_message:
        learner_block_start = prompt_text.rfind("<learner_message>")
        learner_block_end = prompt_text.find(
            "</learner_message>", learner_block_start
        )
        if learner_block_start < 0 or learner_block_end < 0:
            raise ValueError("learner message block is absent from the prompt")
        try:
            encoded_offsets = tokenizer(
                prompt_text, add_special_tokens=False, return_offsets_mapping=True
            )
        except TypeError:
            encoded_offsets = None
        offsets = (
            encoded_offsets.get("offset_mapping")
            if isinstance(encoded_offsets, dict)
            else getattr(encoded_offsets, "offset_mapping", None)
        ) if encoded_offsets is not None else None
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets[0]
            and isinstance(offsets[0][0], list)
        ):
            offsets = offsets[0]
        if offsets is not None and len(offsets) == len(prompt_ids):
            message_start = prompt_text.rfind(last_message, learner_block_start, learner_block_end)
            if message_start < 0:
                message_start = learner_block_start
                message_end = learner_block_end + len("</learner_message>")
            else:
                message_end = message_start + len(last_message)
            for index, (start, end) in enumerate(offsets):
                if end > message_start and start < message_end and mask[index]:
                    raise ValueError("learner message has unmasked tokens")
        else:
            learner_ids = _input_ids(tokenizer, last_message)
            learner_start = _find_subsequence(prompt_ids, learner_ids)
            if learner_start is None:
                raise ValueError("learner message tokens are absent from the prompt")
            if any(mask[learner_start:learner_start + len(learner_ids)]):
                raise ValueError("learner message has unmasked tokens")
    return {"input_ids": full_ids, "loss_mask": mask}


def train(cfg):
    errors = check_config(cfg)
    if errors:
        raise ValueError("invalid SFT config: " + "; ".join(errors))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires CUDA; no training was started")
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["tokenizer_revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        revision=cfg["base_model_revision"],
        torch_dtype=getattr(torch, cfg["precision"]),
    )
    train_data = load_dataset("json", data_files=cfg["data"]["train"], split="train")
    train_data = train_data.map(
        to_prompt_completion, remove_columns=train_data.column_names
    )
    eval_data = load_dataset(
        "json", data_files=cfg["data"]["validation"], split="train"
    )
    eval_data = eval_data.map(
        to_prompt_completion, remove_columns=eval_data.column_names
    )
    kwargs = build_sft_config_kwargs(cfg, cfg["output_dir"])
    kwargs.pop("model_init_kwargs", None)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**kwargs),
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=LoraConfig(**build_lora_kwargs(cfg)),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


def _select_examples(path, count):
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                examples.append(json.loads(line))
    selected = []
    seen_moves = set()
    for example in examples:
        move = example.get("plan", {}).get("teaching_action", {}).get("move")
        if move not in seen_moves:
            selected.append(example)
            seen_moves.add(move)
            if len(selected) == count:
                return selected
    for example in examples:
        if example not in selected:
            selected.append(example)
            if len(selected) == count:
                break
    return selected


def inspect_masks(cfg, count, cases_path=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], revision=cfg["tokenizer_revision"]
    )
    path = cases_path or cfg.get("data", {}).get("validation")
    if not path:
        raise ValueError("mask inspection needs --cases or data.validation")
    examples = _select_examples(path, count)
    if len(examples) < count:
        raise ValueError(f"requested {count} cases but only found {len(examples)}")
    for example in examples:
        result = build_loss_mask(tokenizer, example)
        converter = getattr(tokenizer, "convert_ids_to_tokens", None)
        tokens = (
            converter(result["input_ids"])
            if converter
            else [tokenizer.decode([token], skip_special_tokens=False) for token in result["input_ids"]]
        )
        print(" ".join(
            f"[{mask}] {token!r}"
            for mask, token in zip(result["loss_mask"], tokens)
        ))
        target_count = sum(result["loss_mask"])
        print(
            f"summary {example['example_id']}: "
            f"move={example['plan']['teaching_action']['move']} "
            f"prompt_tokens={len(result['loss_mask']) - target_count} "
            f"target_tokens={target_count} checks=passed"
        )
    return len(examples)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/response/sft.yaml")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--inspect-masks", type=int)
    parser.add_argument("--cases")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.check_config:
        errors = check_config(cfg)
        if errors:
            print("\n".join(errors))
            return 1
        print("config OK")
        return 0
    if args.inspect_masks is not None:
        try:
            inspect_masks(cfg, args.inspect_masks, args.cases)
        except Exception as error:
            print(f"mask inspection failed: {error}")
            return 1
        return 0
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
