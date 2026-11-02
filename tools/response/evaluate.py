"""Evaluate deterministic or OpenAI-compatible response generation on JSONL cases."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import urllib.request

import yaml

from packages.response.fallbacks import FallbackUnavailable, render_fallback
from packages.response.hashing import content_hash
from packages.response.prompt import PROMPT_VERSION, compile_prompt
from packages.response.validate import VALIDATOR_VERSION, run_checks

CHECK_NAMES = (
    "schema_valid", "action_faithful", "math_supported", "disclosure_allowed",
    "length_valid", "question_count_valid",
)


def load_cases(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def openai_compatible(messages, *, env=None, max_tokens=384, timeout=60):
    env = os.environ if env is None else env
    required = (
        "TUTOR_LLM_BASE_URL", "TUTOR_LLM_API_KEY", "TUTOR_LLM_MODEL"
    )
    missing = [name for name in required if not env.get(name)]
    if missing:
        raise RuntimeError("missing required environment variable(s): " + ", ".join(missing))
    url = env["TUTOR_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": env["TUTOR_LLM_MODEL"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {env['TUTOR_LLM_API_KEY']}",
            "Content-Type": "application/json",
            "User-Agent": "tutoring-response-eval/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"]
    return text, body.get("usage")


def _rate(passed, total):
    return passed / total if total else 0.0


def evaluate(cases, responder, out_dir, *, model=None, limit=None):
    selected = list(cases[:limit] if limit is not None else cases)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = 0
    for case in selected:
        plan = case["plan"]
        messages = compile_prompt(plan)
        move = plan["teaching_action"]["move"]
        effective_move = plan["teaching_action"]
        usage = None
        error_text = None
        try:
            if responder == "fallback":
                text = render_fallback(plan)
            elif responder == "openai_compatible":
                text, usage = openai_compatible(messages)
            else:
                raise ValueError(f"unknown responder: {responder}")
        except Exception as error:
            text = ""
            error_text = f"{type(error).__name__}: {error}"
            errors += 1
        checks = run_checks(plan, text, effective_move)
        rows.append({
            "example_id": case.get("example_id"),
            "move": move,
            "text": text,
            "checks": checks,
            "content_hash": content_hash(text, effective_move),
            "_error": error_text,
            "_usage": usage,
        })

    check_totals = Counter()
    move_groups = defaultdict(list)
    for row in rows:
        for name in CHECK_NAMES:
            check_totals[name] += int(row["checks"][name])
        move_groups[row["move"]].append(row)
    per_check = {
        name: _rate(check_totals[name], len(rows)) for name in CHECK_NAMES
    }
    per_move = {}
    for move, group in sorted(move_groups.items()):
        per_move[move] = {
            "n": len(group),
            "all_pass": _rate(
                sum(all(row["checks"].values()) for row in group), len(group)
            ),
            "per_check": {
                name: _rate(
                    sum(row["checks"][name] for row in group), len(group)
                )
                for name in CHECK_NAMES
            },
        }
    report = {
        "responder": responder,
        "model": model or (os.environ.get("TUTOR_LLM_MODEL") if responder == "openai_compatible" else None),
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "n_cases": len(rows),
        "all_checks_pass_rate": _rate(
            sum(all(row["checks"].values()) for row in rows), len(rows)
        ),
        "per_check": per_check,
        "per_move": per_move,
        "errors": errors,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    (out_path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# Response evaluation",
        "",
        f"- Responder: `{responder}`",
        f"- Cases: {len(rows)}",
        f"- All checks pass: {report['all_checks_pass_rate']:.3f}",
        "",
        "| Check | Pass rate |",
        "|---|---:|",
    ]
    markdown.extend(
        f"| {name} | {per_check[name]:.3f} |" for name in CHECK_NAMES
    )
    markdown.extend(["", "## Per move", "", "| Move | N | All pass | " + " | ".join(CHECK_NAMES) + " |"])
    markdown.append("|---|---:|---:|" + "---:|" * len(CHECK_NAMES))
    for move, values in per_move.items():
        cells = " | ".join(f"{values['per_check'][name]:.3f}" for name in CHECK_NAMES)
        markdown.append(
            f"| {move} | {values['n']} | {values['all_pass']:.3f} | {cells} |"
        )
    (out_path / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    with (out_path / "results.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            record = {key: row[key] for key in ("example_id", "move", "text", "checks", "content_hash")}
            record["error"] = row["_error"]
            if row["_usage"] is not None:
                record["usage"] = row["_usage"]
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return report, rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--responder", choices=("fallback", "openai_compatible"))
    parser.add_argument("--cases")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    configured = {}
    if args.config:
        with Path(args.config).open(encoding="utf-8") as stream:
            configured = yaml.safe_load(stream) or {}
    cases_path = args.cases or configured.get("cases")
    out_dir = args.out or configured.get("out")
    responder = args.responder or configured.get("responder")
    if not cases_path or not out_dir or not responder:
        parser.error("provide --cases, --out, and --responder or a config with those values")
    report, _ = evaluate(
        load_cases(cases_path),
        responder,
        out_dir,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
