#!/usr/bin/env python3
"""Generate compact publication figures for directional frequency leakage.

Outputs:
  results/directional/paper/fig_directional_leakage.pdf   (1 figure, 2 subplots)
  results/directional/paper/table_directional.tex          (LaTeX tabular)
"""

import math
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
SKIP = {"identity"}

# Display names for the paper
DISPLAY = {
    "jpeg": "JPEG",
    "webp": "WebP",
    "jpegxl": "JPEG XL",
    "ftic": "FTIC",
    "tcm": "TCM",
    "cheng2020-anchor": "Cheng2020-Anchor",
    "cheng2020-attn": "Cheng2020-Att",
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

def load_data():
    df = pd.read_csv(CSV)
    df = df[~df["model"].isin(SKIP)]
    pivot = df.pivot_table(index="model", columns="angle_deg", values="L_k")
    return df, pivot


def mirror_angles(angles, values):
    """Mirror 0-90 range to 0-180 for polar symmetry."""
    a = list(angles) + [180 - a for a in reversed(angles) if a not in (0, 180)]
    v = list(values) + [v for v in reversed(values) if True][
        1 if angles[0] == 0 else 0:
    ]
    # clean: build explicitly
    a_full, v_full = [], []
    for ang, val in zip(angles, values):
        a_full.append(ang)
        v_full.append(val)
    for ang, val in zip(reversed(angles), reversed(values)):
        mir = 180 - ang
        if mir not in a_full:
            a_full.append(mir)
            v_full.append(val)
    return a_full, v_full


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(pivot):
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

    fig = plt.figure(figsize=(3.5, 5.2))

    # --- (a) Polar plot ---
    ax_pol = fig.add_subplot(2, 1, 1, projection="polar")

    models = list(pivot.index)
    angles_deg = sorted(pivot.columns)

    for name in models:
        vals = [pivot.loc[name, a] for a in angles_deg]
        a_full, v_full = mirror_angles(angles_deg, vals)
        theta = [math.radians(a) for a in a_full]
        # close the curve
        theta.append(theta[0])
        v_full.append(v_full[0])

        label = DISPLAY.get(name, name)
        color = COLORS.get(name, "gray")
        lw = 0.8 if name in TRADITIONAL else 1.2
        ls = "--" if name in TRADITIONAL else "-"
        ax_pol.plot(theta, v_full, ls, color=color, linewidth=lw, label=label)

    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_rlabel_position(225)
    ax_pol.set_rscale("log")
    ax_pol.set_rlim(1e-4, 1.5)
    ax_pol.set_title("(a) Directional leakage (polar)", fontsize=8,
                     pad=12)

    # --- (b) Line plot ---
    ax_lin = fig.add_subplot(2, 1, 2)

    for name in models:
        vals = [pivot.loc[name, a] for a in angles_deg]
        label = DISPLAY.get(name, name)
        color = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        lw = 0.8 if name in TRADITIONAL else 1.2
        ls = "--" if name in TRADITIONAL else "-"
        ax_lin.plot(angles_deg, vals, ls, color=color, marker=marker,
                    linewidth=lw, label=label, markersize=3.5)

    ax_lin.set_yscale("log")
    ax_lin.set_xlabel(r"Angle $\theta$ (deg)")
    ax_lin.set_ylabel(r"Median leakage $L_k$")
    ax_lin.set_xticks(angles_deg)
    ax_lin.grid(True, alpha=0.3, which="both")
    ax_lin.set_title("(b) Leakage vs. angle", fontsize=8)

    # single legend below the figure
    handles, labels = ax_lin.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_table(pivot):
    angles_all = sorted(pivot.columns)
    # 3 representative angles + all-angle range
    table_angles = [a for a in [0.0, 45.0, 90.0] if a in angles_all]

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

    for name in order:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, pivot = load_data()

    fig = make_figure(pivot)
    fig_path = OUT / "fig_directional_leakage.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure -> {fig_path}")

    # also save PNG for quick preview
    fig = make_figure(pivot)
    fig.savefig(OUT / "fig_directional_leakage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tex = make_table(pivot)
    tex_path = OUT / "table_directional.tex"
    tex_path.write_text(tex)
    print(f"Table  -> {tex_path}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
