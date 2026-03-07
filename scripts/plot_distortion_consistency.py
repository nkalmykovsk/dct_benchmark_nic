#!/usr/bin/env python3
"""Generate Fig. S2: spectral leakage coupling L̃ consistency across distortion types.

Two panels: Cheng2020-Anchor (high leakage) and MBT2018 (moderate leakage).
Each panel shows mean PSNR vs mean L̃ per distortion level, ±1σ over Kodak images.

Paper settings:
    24 Kodak images (768×512), q=6, DCT size=512, N_b=512 radial bins
    Distortions: codec clean, codec+Gaussian, Gaussian-only, bit-depth quant, JPEG recomp

Usage:
    python scripts/plot_distortion_consistency.py
    python scripts/plot_distortion_consistency.py --csv results/kodak_eval/kodak_per_image.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

DIST_ORDER = ["codec_clean", "codec+gauss", "gauss_only", "quantization", "jpeg_recomp"]
DIST_LABELS = {
    "codec_clean": "Clean codec",
    "codec+gauss": "Codec + Gauss",
    "gauss_only": "Gaussian only",
    "quantization": "Quantization",
    "jpeg_recomp": "JPEG recomp.",
}
DIST_COLORS = {
    "codec_clean": "#2ca02c",
    "codec+gauss": "#1f77b4",
    "gauss_only": "#ff7f0e",
    "quantization": "#9467bd",
    "jpeg_recomp": "#d62728",
}
DIST_MARKERS = {
    "codec_clean": "s",
    "codec+gauss": "o",
    "gauss_only": "^",
    "quantization": "D",
    "jpeg_recomp": "P",
}

LE_COL = None


def _detect_le_col(df: pd.DataFrame) -> str:
    """Detect column name for spectral leakage coupling."""
    for c in ("Le", "L_tilde"):
        if c in df.columns:
            return c
    raise KeyError("No Le or L_tilde column found in CSV")


def plot_model_panel(ax, df_model: pd.DataFrame, model_label: str):
    """Plot one model's PSNR vs L̃ scatter with error bars."""
    for dist in DIST_ORDER:
        sub = df_model[df_model["distortion"] == dist]
        if sub.empty:
            continue
        grp = sub.groupby("param").agg(
            pm=("psnr", "mean"), ps=("psnr", "std"),
            lm=(LE_COL, "mean"), ls=(LE_COL, "std"),
        ).reset_index().sort_values("pm")

        ax.errorbar(
            grp["pm"], grp["lm"],
            xerr=grp["ps"].fillna(0), yerr=grp["ls"].fillna(0),
            fmt=DIST_MARKERS[dist], ms=5, color=DIST_COLORS[dist],
            markeredgecolor="black", markeredgewidth=0.3,
            capsize=1.5, capthick=0.4, elinewidth=0.4,
            label=DIST_LABELS[dist], zorder=5, alpha=0.9,
        )

    rs, _ = spearmanr(df_model["psnr"], df_model[LE_COL])
    ax.set_xlabel("PSNR (dB)", fontsize=7)
    ax.set_ylabel(r"$\tilde{\mathcal{L}}(X, \hat{X})$", fontsize=7)
    ax.set_title(
        f"{model_label}\nSpearman " + r"$\rho_s$" + f" = {rs:.3f}",
        fontsize=6.5,
    )
    ax.tick_params(labelsize=5.5, length=2, width=0.4)
    ax.grid(True, ls=":", lw=0.3, alpha=0.5)
    ax.set_ylim(-0.01, None)
    ax.set_box_aspect(1)


def main():
    global LE_COL

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("results/kodak_eval/kodak_per_image.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    LE_COL = _detect_le_col(df)

    models = [
        ("cheng2020-anchor", "Cheng2020-Anchor"),
        ("mbt2018", "MBT2018"),
    ]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "mathtext.fontset": "cm",
        "axes.labelsize": 11, "axes.titlesize": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.8), sharey=False)

    for ax, (mname, mlabel) in zip(axes, models):
        df_m = df[df["model"] == mname]
        if df_m.empty:
            ax.text(0.5, 0.5, f"No data for {mname}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        plot_model_panel(ax, df_m, mlabel)

    axes[1].legend(
        fontsize=5.5, loc="upper right", framealpha=0.9,
        handlelength=1.2, borderpad=0.3, handletextpad=0.3,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.92)

    args.out.mkdir(parents=True, exist_ok=True)
    out_pdf = args.out / "figS2_distortion_consistency.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
