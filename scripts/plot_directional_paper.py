#!/usr/bin/env python3
"""Generate directional leakage figure for paper (Fig. S1)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results/directional/directional_leakage.csv"
OUT = ROOT / "results/directional/paper"

TRADITIONAL = {"jpeg", "webp", "jpegxl"}
SKIP = {"identity"}

DISPLAY = {
    "bmshj2018-factorized": "BMSHJ2018-Factorized",
    "bmshj2018-hyperprior": "BMSHJ2018-Hyperprior",
    "mbt2018-mean": "MBT2018-Mean",
    "mbt2018": "MBT2018",
    "cheng2020-anchor": "Cheng2020-Anchor",
    "cheng2020-attn": "Cheng2020-Attention",
    "tcm": "TCM",
    "ftic": "FTIC",
    "jpeg": "JPEG",
    "jpegxl": "JPEG XL",
    "webp": "WebP",
}
COLORS = {
    "cheng2020-anchor": "#7E489E",
    "cheng2020-attn": "#E883B6",
    "mbt2018-mean": "#717171",
    "mbt2018": "#A0BBD4",
    "bmshj2018-hyperprior": "#B8B028",
    "bmshj2018-factorized": "#7B4B3D",
    "ftic": "#D82F30",
    "tcm": "#F0801F",
    "jpeg": "#287BB8",
    "webp": "#429A38",
    "jpegxl": "#3AC1CF",
}
MARKERS = {
    "cheng2020-anchor": "s",
    "cheng2020-attn": "D",
    "mbt2018-mean": "o",
    "mbt2018": "P",
    "bmshj2018-hyperprior": "p",
    "bmshj2018-factorized": "^",
    "ftic": "h",
    "tcm": "X",
    "jpeg": "*",
    "webp": "v",
    "jpegxl": "d",
}


def load_data(size_filter=None, quality_filter=None):
    """Load CSV and build pivot. For classical codecs, prefer matched_bpp==1 when present."""
    df = pd.read_csv(CSV)
    df = df[~df["model"].isin(SKIP)]
    if "matched_bpp" not in df.columns:
        df["matched_bpp"] = 0
    if size_filter is not None:
        size_str = f"{size_filter}x{size_filter}"
        df = df[df["size"] == size_str]
    if "matched_bpp" in df.columns:
        for m in TRADITIONAL:
            sub = df[df["model"] == m]
            if sub["matched_bpp"].fillna(0).astype(int).eq(1).any():
                df = df[~((df["model"] == m) & (df["matched_bpp"].fillna(0).astype(int) != 1))]
    if quality_filter is not None:
        df = df[(df["q"] == quality_filter) | (df["q"].isna() & (df["p"] == 128))]
    pivot = df.pivot_table(index="model", columns="angle_deg", values="L_k")
    return df, pivot


def make_figure(pivot):
    """L_k vs angle. Style matches plot_leakage_vs_bpp."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    models = list(pivot.index)
    angles_deg = sorted(pivot.columns)

    for name in models:
        vals = [pivot.loc[name, a] for a in angles_deg]
        color = COLORS.get(name, "gray")
        marker = MARKERS.get(name, "o")
        ls = "--" if name in TRADITIONAL else "-"
        lw = 0.8 if ls == "--" else 1.0
        ax.plot(
            angles_deg, vals, ls,
            color=color, marker=marker, markersize=3.5,
            markeredgecolor="black", markeredgewidth=0.35,
            linewidth=lw, alpha=0.9, zorder=5,
            label=DISPLAY.get(name, name),
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Angle $\theta$ (deg)", fontsize=9.5)
    ax.set_ylabel(r"Median $L_k$", fontsize=9.5)
    ax.set_xticks(angles_deg)
    ax.tick_params(labelsize=9.5, length=2, width=0.4)
    ax.grid(True, ls=":", lw=0.3, alpha=0.5, which="both")
    ax.set_ylim(5e-5, 1.2)

    handles, labels = ax.get_legend_handles_labels()
    hl_dict = dict(zip(labels, handles))
    col1_keys = ["jpegxl", "jpeg", "webp", "ftic", "tcm", "mbt2018"]
    col2_keys = ["mbt2018-mean", "bmshj2018-hyperprior", "bmshj2018-factorized",
                 "cheng2020-attn", "cheng2020-anchor"]
    interleaved_keys = col1_keys + col2_keys
    num_handles = []
    num_labels = []
    for k in interleaved_keys:
        disp = DISPLAY.get(k, k)
        if disp in hl_dict:
            num_handles.append(hl_dict[disp])
            num_labels.append(disp)

    ax.legend(
        num_handles, num_labels,
        fontsize=5.5, ncol=2, loc="lower center",
        frameon=True, framealpha=0.9, edgecolor="0.7",
        handlelength=1.5, handletextpad=0.4, columnspacing=0.8,
        borderpad=0.4, labelspacing=0.2,
    )

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.15, top=0.92)
    return fig


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Directional leakage figure (Fig. S1).")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    _, pivot = load_data(size_filter=args.size, quality_filter=args.quality)

    fig = make_figure(pivot)
    fig.savefig(OUT / "fig_directional_leakage.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {OUT / 'fig_directional_leakage.pdf'}")


if __name__ == "__main__":
    main()
