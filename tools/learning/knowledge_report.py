"""Graphs and a summary table for the hidden-knowledge comparison (simulated results only)."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("evaluation/learning/results/knowledge_v1")
ARMS = [
    ("qwen3p8_27b_base", "Base Qwen 3.8 27B", "#c98b5b"),
    ("strategist_bc", "Our trained Strategist", "#4f8a6e"),
    ("gpt_6_1_sol", "gpt-6.1-sol", "#5b7fc9"),
    ("gpt_6_astra", "gpt-6-astra", "#8a6fc9"),
]
TYPES = [
    "missing_basics", "already_knows", "reverses_conditionals", "wrong_denominator",
    "assumes_independence", "tree_paths", "complement_slip", "event_vs_outcome",
    "strong_but_reverses",
]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarize(report, meta):
    eps = report["episodes"]
    mis = [e for e in eps if e["had_misconception"]]
    known = [e for e in eps if e["profile_id"] == "k_already_knows"]
    usage = report.get("llm_usage", {})
    calls = usage.get("calls", 0)
    fallbacks = sum((usage.get("fallbacks_by_reason") or {}).values())
    wall = meta.get("wall_seconds")
    return {
        "gain": mean(e["gain"] for e in eps),
        "fixed": mean(e["misconception_fixed"] for e in mis),
        "turn_fixed": mean(e["turn_fixed"] for e in mis if e["turn_fixed"] is not None),
        "wasted_known": mean(e["teaching_share"] for e in known),
        "probe_share_mis": mean(e["probe_share"] for e in mis),
        "dropout": mean(e["dropout"] for e in eps),
        "calls": calls,
        "fallback_rate": fallbacks / calls if calls else 0.0,
        "fallbacks": usage.get("fallbacks_by_reason") or {},
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "wall_seconds": wall,
        "seconds_per_session": wall / len(eps) if wall else None,
        "cost_usd": meta.get("cost_usd"),
        "by_type": {
            t: mean(e["gain"] for e in eps if e["profile_id"] == f"k_{t}") for t in TYPES
        },
        "fixed_by_type": {
            t: mean(e["misconception_fixed"] for e in mis if e["profile_id"] == f"k_{t}")
            for t in TYPES
            if any(e["profile_id"] == f"k_{t}" for e in mis)
        },
    }


def bar(ax, arms, values, title, fmt="{:.3f}", ylabel=""):
    names = [a[1] for a in arms]
    colors = [a[2] for a in arms]
    bars = ax.bar(names, values, color=colors)
    for b, v in zip(bars, values, strict=True):
        if v == v:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8, rotation=12)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(RESULTS / "run_meta.json"))
    parser.add_argument("--out", default=str(RESULTS / "figures"))
    args = parser.parse_args()
    meta = json.loads(Path(args.meta).read_text()) if Path(args.meta).exists() else {}
    arms, rows = [], {}
    for key, label, color in ARMS:
        path = RESULTS / f"{key}.json"
        if path.exists():
            arms.append((key, label, color))
            rows[key] = summarize(json.loads(path.read_text()), meta.get(key, {}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = [
        ("gain", "Learning gain (higher is better)", "{:.3f}", "true mastery gain"),
        ("fixed", "Hidden wrong ideas fixed by session end", "{:.0%}", "share of sessions"),
        ("turn_fixed", "Turn the wrong idea was fixed (lower is faster)", "{:.1f}", "turn"),
        ("wasted_known", "Re-teaching a learner who already knows it (lower)", "{:.0%}",
         "share of turns"),
        ("dropout", "Simulated quits (lower is better)", "{:.0%}", "share of sessions"),
        ("fallback_rate", "Turns where the model reply was unusable", "{:.0%}",
         "share of calls"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, title, fmt, yl) in zip(axes.flat, panels, strict=True):
        bar(ax, arms, [rows[a[0]][k] for a in arms], title, fmt, yl)
    fig.suptitle("Hidden-knowledge test set: 9 learner types x 6 sessions (simulated, not human data)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=130)

    for metric, title, fname in [
        ("by_type", "Learning gain by hidden learner type", "gain_by_type.png"),
        ("fixed_by_type", "Wrong idea fixed by learner type", "fixed_by_type.png"),
    ]:
        types = [t for t in TYPES if all(t in rows[a[0]][metric] for a in arms)]
        fig, ax = plt.subplots(figsize=(14, 5))
        width = 0.8 / max(1, len(arms))
        for i, (key, label, color) in enumerate(arms):
            xs = [j + i * width for j in range(len(types))]
            ax.bar(xs, [rows[key][metric][t] for t in types], width, label=label, color=color)
        ax.set_xticks([j + width * (len(arms) - 1) / 2 for j in range(len(types))])
        ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, fontsize=9)
        ax.set_title(title + " (simulated)")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)

    cost_panels = [
        ("cost_usd", "Cost of the full run (USD)", "${:.2f}", "USD"),
        ("seconds_per_session", "Wall time per session (16 run in parallel)", "{:.1f}s", "seconds"),
        ("calls", "Model calls", "{:.0f}", "calls"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (k, title, fmt, yl) in zip(axes.flat, cost_panels, strict=True):
        vals = [rows[a[0]][k] if rows[a[0]][k] is not None else float("nan") for a in arms]
        bar(ax, arms, vals, title, fmt, yl)
    fig.tight_layout()
    fig.savefig(out / "cost_speed.png", dpi=130)

    gap = {}
    if {"qwen3p8_27b_base", "gpt_6_1_sol", "strategist_bc"} <= rows.keys():
        q, g, s = (rows[k]["gain"] for k in ("qwen3p8_27b_base", "gpt_6_1_sol", "strategist_bc"))
        gap = {"gap_closed": (s - q) / (g - q) if g != q else None}
    summary = {"label": "simulated", "arms": rows, **gap}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: {m: v for m, v in r.items() if not isinstance(v, dict)}
                      for k, r in rows.items()}, indent=1, default=str))
    print(gap)


if __name__ == "__main__":
    main()
