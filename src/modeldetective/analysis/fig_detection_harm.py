"""Detection-vs-harm figure (PLAN.md Phase 5 step 3) for one model size.

x = harm (flip rate vs f16 on MMLU; null flip rate shaded), y = queries to reach 80% power on
the primary probe (log2 axis). Variants not detected within the largest reliable budget sit on a
">125" ceiling band. Degenerate variants are drawn hollow. Static PNG + SVG for the README.

    uv run python -m modeldetective.analysis.fig_detection_harm --model qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = "#2a78d6"  # reference palette categorical slot 1
ORDER = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--backend", default="cpu")
    a = ap.parse_args()
    det = json.loads(Path(f"results/sweep/{a.backend}/{a.model}__short_bits_16.json").read_text())
    harm = json.loads(Path(f"results/harm/{a.model}_mmlu_v2.json").read_text())
    null = harm["f16__run2"]

    ceiling = 256  # plotted position for "not detected within the reliable range"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvspan(*null["flip_rate_ci95"], color=GRID, zorder=0, lw=0)
    ax.text(
        null["flip_rate_ci95"][1] + 0.004,
        1.25,
        "f16 vs f16\n(null)",
        color=MUTED,
        fontsize=8,
        va="bottom",
    )
    ax.axhspan(160, 400, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.text(
        0.445,
        300,
        "not detected within 125 queries",
        color=INK2,
        fontsize=8,
        ha="right",
        va="center",
    )

    for q in ORDER:
        v = det["variants"][q]
        c = v["content_only"]
        m80 = c["queries_to_detect_80"]
        y = m80 if m80 is not None else ceiling
        x = harm[f"{q}__run1"]["flip_rate"]
        xlo, xhi = harm[f"{q}__run1"]["flip_rate_ci95"]
        hollow = v["degenerate"]
        ax.plot([xlo, xhi], [y, y], color=SERIES, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(
            [x],
            [y],
            s=70,
            zorder=3,
            facecolor=SURFACE if hollow else SERIES,
            edgecolor=SERIES if hollow else SURFACE,
            linewidth=2,
        )
        label = q + (" (degenerate)" if hollow else "")
        ax.annotate(
            label,
            (x, y),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
        )

    ax.set_yscale("log", base=2)
    ax.set_ylim(0.7, 400)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_xlim(0, 0.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Harm: MMLU answers changed vs f16 (flip rate, 95% CI)", color=INK2, fontsize=9)
    ax.set_ylabel("Queries to detect (80% power, alpha = 0.05)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title(
        f"{a.model}: harm appears before detectability",
        loc="left",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.125,
        0.005,
        "Primary probe: 16 random bits, content-only features, 2,000 responses per pool. "
        "Lower is easier to detect.",
        color=MUTED,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"detection_vs_harm_{a.model}.{ext}", facecolor=SURFACE)
    print(f"wrote {out}/detection_vs_harm_{a.model}.png/.svg")


if __name__ == "__main__":
    main()
