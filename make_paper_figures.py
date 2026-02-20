#!/usr/bin/env python3
"""Generate compact publication figures for directional frequency leakage.

Outputs:
  results/directional/paper/fig_directional_leakage.pdf   (1 figure, 2 subplots)
  results/directional/paper/table_directional.tex          (LaTeX tabular)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV = Path(__file__).resolve().parent / "results/directional/directional_leakage.csv"
OUT = Path(__file__).resolve().parent / "results/directional/paper"

TRADITIONAL = {"jpeg", "webp", "jpegxl"}
# For table: insert \midrule between this block and NICs
TRADITIONAL_TABLE = {"jpeg", "webp", "jpegxl", "tcm", "ftic"}
SKIP = {"identity"}

# Display names for the paper
DISPLAY = {
    "jpeg": "JPEG",
    "webp": "WebP",
    "jpegxl": "JPEG XL",
    "ftic": "FTIC",
    "tcm": "TCM",
    "cheng2020-anchor": "Cheng2020-Anchor",
    "cheng2020-attn": "Cheng2020-Attention",
    "bmshj2018-factorized": "BMSHJ-Fact",
    "bmshj2018-hyperprior": "BMSHJ-Hyper",
    "mbt2018-mean": "MBT2018-M",
    "mbt2018": "MBT2018",
}

# Consistent colors: blues for traditional, warm for neural
COLORS_TRAD = {"jpeg": "#1f77b4", "webp": "#2ca02c", "jpegxl": "#17becf"}
COLORS_NEURAL = {
    "ftic": "#d62728",
    "tcm": "#ff7f0e",
    "cheng2020-anchor": "#9467bd",
    "cheng2020-attn": "#e377c2",
    "bmshj2018-factorized": "#8c564b",
    "bmshj2018-hyperprior": "#bcbd22",
    "mbt2018-mean": "#7f7f7f",
    "mbt2018": "#aec7e8",
}
COLORS = {**COLORS_TRAD, **COLORS_NEURAL}

MARKERS_TRAD = {"jpeg": "s", "webp": "D", "jpegxl": "^"}
MARKERS_NEURAL = {
    "ftic": "o", "tcm": "v",
    "cheng2020-anchor": "P", "cheng2020-attn": "X",
    "bmshj2018-factorized": "p", "bmshj2018-hyperprior": "h",
    "mbt2018-mean": "*", "mbt2018": "d",
}
MARKERS = {**MARKERS_TRAD, **MARKERS_NEURAL}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(size_filter=None, quality_filter=None):
    """Load CSV and build pivot. Optionally restrict to one size/quality for paper."""
    df = pd.read_csv(CSV)
    df = df[~df["model"].isin(SKIP)]
    if size_filter is not None:
        size_str = f"{size_filter}x{size_filter}"
        df = df[df["size"] == size_str]
    if quality_filter is not None:
        # Keep rows with q == quality_filter; for TCM (has p, no q) keep only p=128
        df = df[(df["q"] == quality_filter) | (df["q"].isna() & (df["p"] == 128))]
    pivot = df.pivot_table(index="model", columns="angle_deg", values="L_k")
    return df, pivot


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pivot):
    """Single line plot: L_k vs angle. IEEE single-column width 3.5 in."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
    })

    # IEEE Signal Processing Letters: single column 3.5 in
    fig, ax = plt.subplots(figsize=(3.5, 2.75))

    models = list(pivot.index)
    angles_deg = sorted(pivot.columns)

    for name in models:
        vals = [pivot.loc[name, a] for a in angles_deg]
        label = DISPLAY.get(name, name)
        color = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        lw = 0.8 if name in TRADITIONAL else 1.2
        ls = "--" if name in TRADITIONAL else "-"
        ax.plot(angles_deg, vals, ls, color=color, marker=marker,
                linewidth=lw, label=label, markersize=3.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.set_ylabel(r"Median leakage $L_k$")
    ax.set_xticks(angles_deg)
    ax.grid(True, alpha=0.3, which="both")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=5.5,
              frameon=True, fancybox=False, edgecolor="0.7",
              bbox_to_anchor=(0.5, 0.06))

    fig.tight_layout(rect=[0, 0.19, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_table(pivot):
    angles_all = sorted(pivot.columns)
    table_angles = angles_all  # all angles

    # sort models by median L_k at 0 deg (ascending = best first)
    order = pivot.sort_values(0.0).index.tolist()

    lines = []
    lines.append(r"\begin{tabular}{l" + "c" * len(table_angles) + "c}")
    lines.append(r"\toprule")

    header = "Model"
    for a in table_angles:
        header += rf" & ${int(a)}^\circ$"
    header += r" & Range \\"
    lines.append(header)
    lines.append(r"\midrule")

    for i, name in enumerate(order):
        # Thin line between traditional codecs and NICs
        if i > 0 and order[i - 1] in TRADITIONAL_TABLE and name not in TRADITIONAL_TABLE:
            lines.append(r"\midrule")
        row_vals = {a: pivot.loc[name, a] for a in table_angles}
        all_vals = [pivot.loc[name, a] for a in angles_all]
        rng = max(all_vals) - min(all_vals)
        worst = max(row_vals, key=row_vals.get)

        cells = DISPLAY.get(name, name)
        for a in table_angles:
            v = row_vals[a]
            fmt = f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
            if a == worst:
                fmt = r"\textbf{" + fmt + "}"
            cells += f" & {fmt}"
        cells += f" & {rng:.3f}" if rng >= 0.001 else f" & {rng:.1e}"
        cells += r" \\"
        lines.append(cells)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_table_twohalves(pivot):
    """Full-width table split into two halves (side by side)."""
    angles_all = sorted(pivot.columns)
    n = len(angles_all)
    mid = (n + 1) // 2  # 7 angles -> left 0,15,30,45; right 60,75,90 + Range
    left_angles = angles_all[:mid]   # 0, 15, 30, 45
    right_angles = angles_all[mid:]  # 60, 75, 90

    order = pivot.sort_values(0.0).index.tolist()

    def row_cells(name, angles_part, row_vals, worst, rng, include_range=False):
        cells = DISPLAY.get(name, name)
        for a in angles_part:
            v = row_vals[a]
            fmt = f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
            if a == worst:
                fmt = r"\textbf{" + fmt + "}"
            cells += f" & {fmt}"
        if include_range:
            cells += f" & {rng:.3f}" if rng >= 0.001 else f" & {rng:.1e}"
        cells += r" \\"
        return cells

    left_lines = []
    left_lines.append(r"\begin{tabular}{l" + "c" * len(left_angles) + "}")
    left_lines.append(r"\toprule")
    left_lines.append("Model" + "".join(rf" & ${int(a)}^\circ$" for a in left_angles) + r" \\")
    left_lines.append(r"\midrule")

    right_lines = []
    right_lines.append(r"\begin{tabular}{l" + "c" * len(right_angles) + "c}")
    right_lines.append(r"\toprule")
    right_lines.append("Model" + "".join(rf" & ${int(a)}^\circ$" for a in right_angles) + r" & Range \\")
    right_lines.append(r"\midrule")

    for i, name in enumerate(order):
        if i > 0 and order[i - 1] in TRADITIONAL_TABLE and name not in TRADITIONAL_TABLE:
            left_lines.append(r"\midrule")
            right_lines.append(r"\midrule")
        row_vals = {a: pivot.loc[name, a] for a in angles_all}
        all_vals = [pivot.loc[name, a] for a in angles_all]
        rng = max(all_vals) - min(all_vals)
        worst = max(row_vals, key=row_vals.get)
        left_lines.append(row_cells(name, left_angles, row_vals, worst, rng, include_range=False))
        right_lines.append(row_cells(name, right_angles, row_vals, worst, rng, include_range=True))

    left_lines.append(r"\bottomrule")
    left_lines.append(r"\end{tabular}")
    right_lines.append(r"\bottomrule")
    right_lines.append(r"\end{tabular}")

    out = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Median leakage $\tilde{L}(\theta)$ per codec at each rotation angle $\theta$ (1024$\times$1024, highest quality). Bold: angle with maximum leakage for that row; Range: max minus min over angles (anisotropy).}",
        r"\label{tab:directional}",
        r"\begin{minipage}[t]{0.48\textwidth}",
        r"\centering",
        "\n".join(left_lines),
        r"\end{minipage}",
        r"\hfill",
        r"\begin{minipage}[t]{0.48\textwidth}",
        r"\centering",
        "\n".join(right_lines),
        r"\end{minipage}",
        r"\end{table*}",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Paper figures for directional leakage.")
    ap.add_argument("--size", type=int, default=512, help="Patch size (default: 512)")
    ap.add_argument("--quality", type=int, default=6, help="Quality level (default: 6)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df, pivot = load_data(size_filter=args.size, quality_filter=args.quality)

    fig = make_figure(pivot)
    fig_path = OUT / "fig_directional_leakage.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {fig_path}")

    # also save PNG for quick preview
    fig = make_figure(pivot)
    fig.savefig(OUT / "fig_directional_leakage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tabular_tex = make_table(pivot)
    full_tex = (
        r"\begin{table*}[t]"
        "\n"
        r"\centering"
        "\n"
        r"\caption{Median leakage $\tilde{L}(\theta)$ per codec at each rotation angle $\theta$ (1024$\times$1024, highest quality). Bold: angle with maximum leakage for that row; Range: max minus min over angles (anisotropy).}"
        "\n"
        r"\label{tab:directional}"
        "\n"
        r"\resizebox{\textwidth}{!}{%"
        "\n"
        + tabular_tex
        + "\n"
        r"}"
        "\n"
        r"\end{table*}"
    )
    tex_path = OUT / "table_directional.tex"
    tex_path.write_text(full_tex)
    print(f"Table  -> {tex_path}")
    print()
    print(full_tex)


if __name__ == "__main__":
    main()
