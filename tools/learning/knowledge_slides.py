"""One slide-sized graph per knowledge_v1 score, for presenting (simulated results)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_report import RESULTS, TYPES, summarize  # noqa: E402

ARMS = [
    ("qwen3p8_27b_base", "Base Qwen 3.8 27B", "#c98b5b"),
    ("strategist_bc", "Our Strategist (before)", "#9cc3ad"),
    ("strategist_ppo_reteach_02", "Our Strategist + RL (PPO)", "#2f7d55"),
    ("strategist_bc_gpt_distill", "Our Strategist trained on gpt", "#1f5c3f"),
    ("strategist_ppo_distill_control", "Trained on gpt + RL", "#0f3d29"),
    ("strategist_ppo_distill_02", "Trained on gpt + RL + re-teach cost", "#4a6b5a"),
    ("nemotron_lightning_3p5_30b", "Nemotron Lightning 30B", "#b5a24a"),
    ("gpt_6_1_sol", "gpt-6.1-sol (frontier)", "#5b7fc9"),
    ("gpt_6_astra", "gpt-6-astra (frontier)", "#8a6fc9"),
]
FOOTER = "Simulated students: 9 hidden learner types x 6 sessions = 54 sessions, one seed. Not human data."

SLIDES = [
    ("01_learning_gain", "gain", "Learning gain",
     "How much the student's true mastery rose during the session.", "higher is better", "{:.3f}"),
    ("02_wrong_ideas_fixed", "fixed", "Hidden wrong ideas fixed",
     "Of the 42 sessions where the student secretly held a wrong belief, the share where it was gone by the end.",
     "higher is better", "{:.0%}"),
    ("03_turn_of_fix", "turn_fixed", "How fast the wrong idea was fixed",
     "Average turn on which the hidden wrong belief went away.", "lower is better", "{:.1f}"),
    ("04_reteaching_known", "wasted_known", "Re-teaching a student who already knows it",
     "For students who already know the material, the share of turns spent on hints, explanations or worked examples.",
     "lower is better", "{:.0%}"),
    ("05_quits", "dropout", "Simulated students who quit",
     "Share of sessions where the simulated student lost engagement and quit early.",
     "lower is better", "{:.0%}"),
    ("06_unusable_replies", "fallback_rate", "Unusable model replies",
     "Share of model calls whose reply was cut off, unreadable or illegal (the turn fell back to rules).",
     "lower is better", "{:.1%}"),
    ("07_cost", "cost_usd", "Cost to run all 54 sessions",
     "API or GPU cost in US dollars. Our Strategist runs on a CPU for free. Nemotron not shown (price not verified).", "lower is better", "${:.2f}"),
    ("08_time_per_session", "seconds_per_session", "Time per session",
     "Wall-clock seconds per session, 16 sessions run in parallel.", "lower is better", "{:.1f} s"),
]


def load():
    meta = json.loads((RESULTS / "run_meta.json").read_text())
    arms, rows = [], {}
    for key, label, color in ARMS:
        path = RESULTS / f"{key}.json"
        if path.exists():
            arms.append((key, label, color))
            rows[key] = summarize(json.loads(path.read_text()), meta.get(key, {}))
    return arms, rows


def finish(fig, ax, title, subtitle, direction):
    fig.suptitle(title, x=0.03, ha="left", fontsize=22, fontweight="bold", y=0.97)
    fig.text(0.03, 0.885, subtitle, fontsize=13, color="#444")
    fig.text(0.03, 0.84, direction, fontsize=12, color="#2f7d55", style="italic")
    fig.text(0.03, 0.025, FOOTER, fontsize=10, color="#777")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.27, right=0.93, top=0.80, bottom=0.10)


def slide(arms, rows, out, name, key, title, subtitle, direction, fmt):
    arms = [a for a in arms if rows[a[0]][key] is not None]
    labels = [a[1] for a in arms][::-1]
    colors = [a[2] for a in arms][::-1]
    values = [rows[a[0]][key] for a in arms][::-1]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    bars = ax.barh(labels, values, color=colors, height=0.65)
    top = max(values) if max(values) > 0 else 1
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_width() + top * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", fontsize=14)
    base = rows.get("qwen3p8_27b_base", {}).get(key)
    if base:
        ax.axvline(base, color="#c98b5b", ls="--", lw=1.5)
    ax.set_xlim(0, top * 1.15)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=11)
    finish(fig, ax, title, subtitle, direction + ("   (dashed line = base Qwen)" if base else ""))
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def by_type(arms, rows, out, name, key, title, subtitle, fmt_pct):
    types = [t for t in TYPES if all(t in rows[a[0]][key] for a in arms)]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    width = 0.8 / len(arms)
    for i, (k, label, color) in enumerate(arms):
        ax.bar([j + i * width for j in range(len(types))], [rows[k][key][t] for t in types],
               width, label=label, color=color)
    ax.set_xticks([j + width * (len(arms) - 1) / 2 for j in range(len(types))])
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=12)
    if fmt_pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(fontsize=10, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    finish(fig, ax, title, subtitle, "higher is better")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.14)
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def main():
    arms, rows = load()
    out = RESULTS / "figures" / "slides"
    out.mkdir(parents=True, exist_ok=True)
    for spec in SLIDES:
        slide(arms, rows, out, *spec)
    by_type(arms, rows, out, "09_gain_by_type", "by_type", "Learning gain by hidden learner type",
            "Same gain score, split by what the student secretly knew or believed.", False)
    by_type(arms, rows, out, "10_fixed_by_type", "fixed_by_type", "Wrong ideas fixed by type",
            "For each type of hidden wrong belief, the share of sessions where it was fixed.", True)
    print("\n".join(sorted(p.name for p in out.glob("*.png"))))


if __name__ == "__main__":
    main()
