"""Paper figures for the heading study (static, light background, PDF + PNG).

Reads reports/heading_study/results.json and per_query.csv.gz (run analyze_heading_study.py first).
Colors: the dataviz reference categorical palette in fixed order, one hue per question set,
validated with its CVD checker; marker shapes and a legend give identity beyond color.

uv run python scripts/make_paper_figures.py
Output: reports/heading_study/figures/paper/fig{1..5}_*.{pdf,png}
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from structuralrag.utils.config import REPORTS_DIR  # noqa: E402

DIR = REPORTS_DIR / "heading_study"
OUT = DIR / "figures" / "paper"
SETS = ["GovReport (ICT)", "GovReport (LLM-written)", "QASPER (human)", "NQ (human)"]
SHORT = {
    "GovReport (ICT)": "GovReport ICT",
    "GovReport (LLM-written)": "GovReport LLM",
    "QASPER (human)": "QASPER",
    "NQ (human)": "NQ",
}
COLOR = dict(zip(SETS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], strict=True))
MARKER = dict(zip(SETS, ["o", "s", "D", "^"], strict=True))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.facecolor": SURFACE,
        "figure.facecolor": "white",
        "legend.frameon": False,
        "lines.linewidth": 2,
        "pdf.fonttype": 42,
    }
)


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def zero(ax) -> None:
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)


def fig_conditions(res: dict) -> None:
    """Fig 1: gain over flat with real, shuffled and generic headings."""
    rows = {r["set"]: r for r in res["A"]["table"]}
    conds = [("SA real", "Real headings"), ("SA placebo", "Shuffled (placebo)"), ("SA generic", 'Generic "Section n"')]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    x = np.arange(len(SETS))
    width = 0.24
    shades = ["#2a78d6", "#86b6ef", "#c9c8c3"]  # one hue for the condition of interest, neutrals for controls
    for k, ((key, name), col) in enumerate(zip(conds, shades, strict=True)):
        vals = [rows[s][key] - rows[s]["flat"] for s in SETS]
        ax.bar(x + (k - 1) * width, vals, width - 0.03, color=col, label=name, zorder=2)
    zero(ax)
    ax.set_xticks(x, [SHORT[s] for s in SETS])
    ax.set_ylabel("MRR@10 gain over flat")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "fig1_conditions")


def fig_specificity(res: dict) -> None:
    """Fig 2: gain by the target section's heading specificity (the main result)."""
    rows = [r for r in res["H"] if not str(r["gold heading"]).startswith("Spearman")]
    bins = ["generic (w < 0.5)", "shared (0.5-0.9)", "unique (w >= 0.9)"]
    labels = ["Generic\n(w < 0.5)", "Shared\n(0.5–0.9)", "Unique\n(w ≥ 0.9)"]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for k, s in enumerate(SETS):
        r = {x["gold heading"]: x for x in rows if x["set"] == s}
        xs = np.arange(3) + (k - 1.5) * 0.12
        y = np.array([r[b]["gain_vs_flat"] for b in bins])
        lo = y - np.array([r[b]["ci_lo"] for b in bins])
        hi = np.array([r[b]["ci_hi"] for b in bins]) - y
        ax.errorbar(
            xs,
            y,
            yerr=[lo, hi],
            fmt=MARKER[s],
            ms=5,
            color=COLOR[s],
            ecolor=COLOR[s],
            elinewidth=1,
            capsize=0,
            label=SHORT[s],
            zorder=3,
            mec="white",
            mew=0.8,
        )
    zero(ax)
    ax.set_xticks(range(3), labels)
    ax.set_xlabel("Specificity of the target section's heading")
    ax.set_ylabel("MRR@10 gain over flat (95% CI)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    save(fig, "fig2_specificity")


def fig_dose(res: dict) -> None:
    """Fig 3: dose-response as headings are made generic or noisy."""
    dose = pd.DataFrame(res["B"])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, kind, xl in (
        (axes[0], "generic", "Share of headings made generic"),
        (axes[1], "ocr", "OCR noise rate on headings"),
    ):
        for s in SETS:
            d = dose[(dose.set == s) & (dose.corruption == kind)].sort_values("level")
            ax.plot(
                d.level, d.gain_vs_flat, marker=MARKER[s], ms=4.5, color=COLOR[s], label=SHORT[s], mec="white", mew=0.6
            )
        zero(ax)
        ax.set_xlabel(xl)
    axes[0].set_ylabel("MRR@10 gain over flat")
    axes[1].legend(fontsize=7, loc="lower right")
    save(fig, "fig3_dose_response")


def fig_alpha(pq: pd.DataFrame, alphas: list[float]) -> None:
    """Fig 4: path-weight sweep, structure-aware vs the neutral specificity fallback."""
    real = pq[pq.variant == "real"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharex=True)
    for ax, s in zip(axes, SETS, strict=True):
        ds, qs = {
            "GovReport (ICT)": ("govreport", "ict"),
            "GovReport (LLM-written)": ("govreport", "llm"),
            "QASPER (human)": ("qasper", "human"),
            "NQ (human)": ("nq", "human"),
        }[s]
        m = real[(real.dataset == ds) & (real.query_set == qs)].groupby("system").rr.mean()
        ax.plot(alphas, [m[f"sa_a{a}"] for a in alphas], color=COLOR[s], marker="o", ms=3.5, label="Structure-aware")
        ax.plot(
            alphas,
            [m[f"safbm_a{a}"] for a in alphas],
            color=COLOR[s],
            ls="--",
            marker="s",
            ms=3.5,
            label="+ neutral fallback",
        )
        ax.axhline(m["flat"], color=INK2, lw=0.8, ls=":", label="Flat")
        ax.set_title(SHORT[s], fontsize=8, color=INK)
        ax.set_xlabel("Path weight α")
    axes[0].set_ylabel("MRR@10")
    axes[0].legend(fontsize=6, loc="lower left")
    save(fig, "fig4_alpha_sweep")


def fig_pipeline() -> None:
    """Fig 5: method overview."""
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    h = 0.95
    boxes = {  # name: (x, y, width, text, fill)
        "doc": (0.05, 1.0, 1.8, "Document\nsection tree", "#cde2fb"),
        "var": (2.3, 1.9, 2.2, "Heading variant\nreal · shuffled · generic\nOCR · flattened · LLM", "#f0efec"),
        "spec": (2.3, 0.1, 2.2, "Specificity\nw = log(N/df) / log N", "#f0efec"),
        "score": (4.95, 1.0, 2.1, "Score sections\nwithin the document\nflat vs structure-aware", "#cde2fb"),
        "gain": (7.5, 1.9, 2.45, "Gain over flat,\nreal vs placebo", "#f0efec"),
        "bytgt": (7.5, 0.1, 2.45, "Gain by target\nheading specificity", "#f0efec"),
    }
    for x, y, w, text, col in boxes.values():
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec=INK2, lw=0.8)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color=INK)

    def right(b):
        x, y, w, *_ = boxes[b]
        return (x + w + 0.04, y + h / 2)

    def left(b):
        x, y, *_ = boxes[b]
        return (x - 0.04, y + h / 2)

    for a, b in (
        ("doc", "var"),
        ("doc", "spec"),
        ("var", "score"),
        ("spec", "score"),
        ("score", "gain"),
        ("score", "bytgt"),
    ):
        ax.add_patch(FancyArrowPatch(right(a), left(b), arrowstyle="-|>", mutation_scale=8, color=INK2, lw=0.8))
    save(fig, "fig5_pipeline")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = json.loads((DIR / "results.json").read_text())
    pq = pd.read_csv(DIR / "per_query.csv.gz")
    fig_conditions(res)
    fig_specificity(res)
    fig_dose(res)
    fig_alpha(pq, res["alphas"])
    fig_pipeline()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
