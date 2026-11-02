import json
from pathlib import Path
import re

import pytest

from packages.response.prompt import compile_prompt
from tools.response.evaluate import evaluate, load_cases
from tools.response.make_smoke_cases import build_smoke_cases
from tools.response.train_sft import (
    build_loss_mask,
    build_lora_kwargs,
    build_sft_config_kwargs,
    check_config,
    load_config,
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
                "choices": [{"message": {"content": "A clue: count the non-blue sections. What next?"}}],
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
    assert request.get_header("User-agent") == "tutoring-response-eval/1.0"
    assert request.get_header("Authorization") == "Bearer do-not-log-this"
    body = json.loads(request.data)
    assert body["temperature"] == 0
    assert body["max_tokens"] == 384
    assert body["model"] == "fake-model"
    assert report["model"] == "fake-model"
    assert rows[0]["_usage"] == {"prompt_tokens": 12, "completion_tokens": 10}
    result_text = (tmp_path / "results.jsonl").read_text()
    assert "do-not-log-this" not in result_text


def test_sft_config_placeholders_missing_keys_and_resolved_config():
    placeholder = load_config("configs/response/sft.yaml")
    errors = check_config(placeholder)
    assert any("base_model_revision must replace" in error for error in errors)
    assert any("tokenizer_revision must replace" in error for error in errors)

    incomplete = dict(placeholder)
    incomplete.pop("epochs_max")
    assert "missing key: epochs_max" in check_config(incomplete)

    resolved = load_config("configs/response/sft_resolved_qwen2.5-7b.yaml")
    assert check_config(resolved) == []
    invalid_nested = dict(resolved)
    invalid_nested["package_pins"] = {
        **resolved["package_pins"],
        "unexpected": "REQUIRED_RESOLVED_COMMIT",
    }
    assert any("configuration contains REQUIRED_RESOLVED_COMMIT" in error
               for error in check_config(invalid_nested))


def test_sft_kwargs_and_prompt_completion_projection():
    cfg = load_config("configs/response/sft_resolved_qwen2.5-7b.yaml")
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
    assert build_lora_kwargs(cfg) == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
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
