import json
import re
from pathlib import Path

from packages.response.prompt import compile_prompt
from tools.response.evaluate import evaluate, load_cases
from tools.response.train_sft import (
    build_lora_kwargs,
    build_loss_mask,
    build_sft_config_kwargs,
    check_config,
    load_config,
    matched_lora_targets,
    to_prompt_completion,
)

ROOT = Path(__file__).parents[3]
PLAN_DIR = Path(__file__).parent / "fixtures" / "plans"


def plans():
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in PLAN_DIR.glob("*.json")
    }


def test_smoke_cases_match_the_eight_plan_fixtures():
    smoke_path = ROOT / "evaluation" / "response" / "smoke_cases_v1.jsonl"
    smoke = load_cases(smoke_path)
    fixtures = plans()
    assert len(smoke) == 8
    assert {case["plan"]["teaching_action"]["move"] for case in smoke} == {
        plan["teaching_action"]["move"] for plan in fixtures.values()
    }
    for case in smoke:
        name = case["example_id"].removeprefix("smoke_").removesuffix("_v1")
        assert case["plan"] == fixtures[name]
        assert case["prompt"] == compile_prompt(fixtures[name])
        assert case["source_id"] == "original_fixture"
        assert case["synthetic"] is True
        assert case["reviewer_status"] == "pending"
        assert case["action_label_source"] == "synthetic"


def test_fallback_evaluator_writes_reports_and_all_fixture_checks_pass(tmp_path):
    examples = [
        {"example_id": f"case-{name}", "plan": fixture}
        for name, fixture in plans().items()
    ]
    report, rows = evaluate(examples, "fallback", tmp_path)
    assert report["n_cases"] == 8
    assert report["all_checks_pass_rate"] == 1.0
    assert report["errors"] == 0
    assert all(all(row["checks"].values()) for row in rows)
    assert {path.name for path in tmp_path.iterdir()} == {
        "report.json", "report.md", "results.jsonl"
    }
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 8
    assert json.loads((tmp_path / "report.json").read_text())["n_cases"] == 8


def test_openai_compatible_evaluation_uses_expected_http_request(monkeypatch, tmp_path):
    from tools.response import evaluate as evaluate_module

    fixture = plans()["hint"]
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [
                    {
                        "message": {
                            "content": "A clue: count the non-blue sections. What next?"
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 10},
            }).encode()

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(evaluate_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TUTOR_LLM_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("TUTOR_LLM_API_KEY", "do-not-log-this")
    monkeypatch.setenv("TUTOR_LLM_MODEL", "fake-model")
    report, rows = evaluate(
        [{"example_id": "case-hint", "plan": fixture}],
        "openai_compatible",
        tmp_path,
    )
    request = captured["request"]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert request.get_header("User-agent") == "tuturing-response-eval/1.0"
    assert request.get_header("Authorization") == "Bearer do-not-log-this"
    body = json.loads(request.data)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 384
    assert body["model"] == "fake-model"
    assert report["model"] == "fake-model"
    assert rows[0]["_usage"] == {"prompt_tokens": 12, "completion_tokens": 10}
    result_text = (tmp_path / "results.jsonl").read_text()
    assert "do-not-log-this" not in result_text


def test_evaluate_config_model_defaults_and_environment_overrides(
    monkeypatch, tmp_path
):
    from tools.response import evaluate as evaluate_module

    case_path = tmp_path / "case.jsonl"
    case_path.write_text(
        json.dumps({"example_id": "case-hint", "plan": plans()["hint"]}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"cases: {json.dumps(str(case_path))}\n"
        f"out: {json.dumps(str(tmp_path / 'out'))}\n"
        "small_model:\n"
        "  responder: openai_compatible\n"
        "  base_url: https://config.invalid/v1\n"
        "  model: accounts/fireworks/models/qwen3p8-27b\n"
        "frontier_model:\n"
        "  responder: PLACEHOLDER\n"
        "  base_url: PLACEHOLDER\n"
        "  model: PLACEHOLDER\n",
        encoding="utf-8",
    )
    captured = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "A clue: count carefully. What next?"}}],
            }).encode()

    def fake_urlopen(request, timeout):
        captured.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(evaluate_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TUTOR_LLM_API_KEY", "key-from-env")
    monkeypatch.delenv("TUTOR_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TUTOR_LLM_MODEL", raising=False)
    evaluate_module.main(["--config", str(config_path)])
    default_request = captured[-1][0]
    assert default_request.full_url == "https://config.invalid/v1/chat/completions"
    assert json.loads(default_request.data)["model"] == "accounts/fireworks/models/qwen3p8-27b"
    assert json.loads((tmp_path / "out" / "report.json").read_text())["model"] == (
        "accounts/fireworks/models/qwen3p8-27b"
    )
    assert default_request.get_header("Authorization") == "Bearer key-from-env"

    monkeypatch.setenv("TUTOR_LLM_BASE_URL", "https://env.invalid/v1/")
    monkeypatch.setenv("TUTOR_LLM_MODEL", "env-model")
    evaluate_module.main(["--config", str(config_path)])
    override_request = captured[-1][0]
    assert override_request.full_url == "https://env.invalid/v1/chat/completions"
    assert json.loads(override_request.data)["model"] == "env-model"
    assert json.loads((tmp_path / "out" / "report.json").read_text())["model"] == "env-model"


def test_sft_config_placeholders_missing_keys_and_resolved_config():
    placeholder = load_config("configs/response/sft.yaml")
    errors = check_config(placeholder)
    assert any("base_model_revision must replace" in error for error in errors)
    assert any("tokenizer_revision must replace" in error for error in errors)

    incomplete = dict(placeholder)
    incomplete.pop("epochs_max")
    assert "missing key: epochs_max" in check_config(incomplete)

    resolved = load_config("configs/response/sft_resolved_qwen3.8-27b.yaml")
    assert check_config(resolved) == []
    assert resolved["base_model"] == "Qwen/Qwen3.8-27B"
    assert resolved["fireworks_model"] == "accounts/fireworks/models/qwen3p8-27b"
    assert resolved["lora_target_modules_scope"] == "language_model_attention"
    assert resolved["lora_expected_target_count"] == 160
    assert resolved["lora_target_modules"] == (
        r"^(model\.)?(language_model\.)?layers\.\d+\.[^.]+\."
        r"(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj)$"
    )
    invalid_nested = dict(resolved)
    invalid_nested["package_pins"] = {
        **resolved["package_pins"],
        "unexpected": "REQUIRED_RESOLVED_COMMIT",
    }
    assert any("configuration contains REQUIRED_RESOLVED_COMMIT" in error
               for error in check_config(invalid_nested))

    invalid_regex = dict(resolved, lora_target_modules="[")
    assert any("lora_target_modules must be a valid regex" in error
               for error in check_config(invalid_regex))
    invalid_count = dict(resolved, lora_expected_target_count=True)
    assert any("lora_expected_target_count must be a positive integer" in error
               for error in check_config(invalid_count))


def test_sft_kwargs_and_prompt_completion_projection():
    cfg = load_config("configs/response/sft_resolved_qwen3.8-27b.yaml")
    kwargs = build_sft_config_kwargs(cfg, "some/output")
    assert kwargs["num_train_epochs"] == cfg["epochs_max"]
    assert kwargs["per_device_train_batch_size"] == cfg["micro_batch_size"]
    assert kwargs["gradient_accumulation_steps"] == 16
    assert kwargs["warmup_steps"] == cfg["warmup_ratio"]
    assert kwargs["max_grad_norm"] == cfg["gradient_clip_norm"]
    assert kwargs["max_length"] == 4096
    assert kwargs["completion_only_loss"] is True
    assert kwargs["assistant_only_loss"] is False
    assert kwargs["packing"] is False
    assert kwargs["model_init_kwargs"] == {
        "revision": cfg["base_model_revision"], "dtype": "bfloat16"
    }
    assert build_lora_kwargs(cfg)["target_modules"] == cfg["lora_target_modules"]
    assert build_lora_kwargs(cfg) == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": (
            r"^(model\.)?(language_model\.)?layers\.\d+\.[^.]+\."
            r"(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|out_proj)$"
        ),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    assert to_prompt_completion({
        "prompt": [{"role": "user", "content": "x"}],
        "completion": [{"role": "assistant", "content": "y"}],
        "other": "drop",
    }) == {
        "prompt": [{"role": "user", "content": "x"}],
        "completion": [{"role": "assistant", "content": "y"}],
    }


def test_lora_target_matching_scopes_to_text_attention_projections():
    names = []
    for index in range(64):
        layer = f"model.language_model.layers.{index}"
        if index % 4 == 3:
            names.extend(
                f"{layer}.self_attn.{projection}_proj"
                for projection in ("q", "k", "v", "o")
            )
        else:
            names.extend(
                f"{layer}.linear_attn.{projection}"
                for projection in (
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_a",
                    "in_proj_b",
                    "out_proj",
                    "conv1d",
                )
            )
        names.extend(
            f"{layer}.mlp.{projection}_proj" for projection in ("gate", "up", "down")
        )
    decoys = [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.attn.proj",
        "model.visual.merger.out_proj",
        "mtp.layers.0.self_attn.q_proj",
        "model.language_model.mtp.layers.0.self_attn.k_proj",
    ]
    names.extend(decoys)
    target_regex = load_config(
        "configs/response/sft_resolved_qwen3.8-27b.yaml"
    )["lora_target_modules"]
    matched = matched_lora_targets(names, target_regex)
    assert len(matched) == 160
    assert all(name.startswith("model.language_model.layers.") for name in matched)
    assert not set(matched).intersection(decoys)
    assert not any(".mlp." in name for name in matched)

    text_only_names = [
        f"model.layers.0.self_attn.{projection}_proj"
        for projection in ("q", "k", "v", "o")
    ] + [
        "model.layers.1.linear_attn.in_proj_qkv",
        "model.layers.1.linear_attn.out_proj",
    ]
    assert len(matched_lora_targets(text_only_names, target_regex)) == 6

    suffix_matches = matched_lora_targets(
        ["model.layers.0.self_attn.q_proj", "model.visual.q_proj"],
        ["q_proj"],
    )
    assert suffix_matches == [
        "model.layers.0.self_attn.q_proj",
        "model.visual.q_proj",
    ]


class TinyTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 999
    im_end_token_id = 999

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, *, add_special_tokens):
        ids = []
        for part in re.split(r"(<\|im_end\|>)", text):
            if not part:
                continue
            if part == "<|im_end|>":
                ids.append(self.im_end_token_id)
            else:
                ids.extend(ord(char) for char in part)
        return {"input_ids": ids}


def test_fake_tokenizer_masks_prompt_and_unmasks_completion_target():
    fixture = plans()["practice"]
    example = {
        "plan": fixture,
        "prompt": compile_prompt(fixture),
        "completion": [{"role": "assistant", "content": "Try it now."}],
    }
    result = build_loss_mask(TinyTokenizer(), example)
    first_target = result["loss_mask"].index(1)
    assert first_target > 0
    assert all(token == 0 for token in result["loss_mask"][:first_target])
    assert all(token == 1 for token in result["loss_mask"][first_target:])
    assert result["input_ids"][-1] == TinyTokenizer.eos_token_id
