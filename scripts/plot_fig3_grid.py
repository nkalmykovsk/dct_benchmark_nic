#!/usr/bin/env python3
"""Generate Fig. 3: DCT-basis response grid (2 rows × 11 codecs).

Two modes:
  --from-csv  (default) reads all_metrics_summary.csv + pre-saved images
  --live      runs each codec on-the-fly (requires GPU, ~10 min)

Paper settings:
    Row 1: low quality  (q=1 or p=64)
    Row 2: high quality (q=6 or p=128)
    DCT size: 256×256

Usage:
    python scripts/plot_fig3_grid.py                           # from CSV + images
    python scripts/plot_fig3_grid.py --live --device cuda       # generate live
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CODEC_SPEC = [
    ("JPEG",                    "jpeg",                 "q", [1, 6]),
    ("JPEG XL",                 "jpegxl",               "q", [1, 6]),
    ("WebP",                    "webp",                 "q", [1, 6]),
    ("BMSHJ2018-\nFactorized",  "bmshj2018-factorized", "q", [1, 6]),
    ("BMSHJ2018-\nHyperprior",  "bmshj2018-hyperprior", "q", [1, 6]),
    ("MBT2018-\nMean",          "mbt2018-mean",         "q", [1, 6]),
    ("MBT2018",                 "mbt2018",              "q", [1, 6]),
    ("Cheng2020-\nAnchor",      "cheng2020-anchor",     "q", [1, 6]),
    ("Cheng2020-\nAttention",   "cheng2020-attn",       "q", [1, 6]),
    ("TCM",                     "tcm",                  "p", [64, 128]),
    ("FTIC",                    "ftic",                 "q", [1, 6]),
]

SIZE = 256
N_TRAD = 3


def _fmt_lk(v: float) -> str:
    return f"{v:.0e}".replace("e-0", "e-") if v < 0.001 else f"{v:.3f}"


def _lk_color(lk: float) -> str:
    if lk < 0.1:
        return "#2D6A2E"
    if lk < 0.3:
        return "#8B7500"
    if lk < 0.6:
        return "#B45309"
    return "#A91E1E"


def generate_live(model_name: str, quality: int, device, p: int = 128):
    """Run codec and return (image_rgb, metrics_dict)."""
    import torch
    from dct_nic import evaluate_codec, load_model

    base = str(Path(__file__).resolve().parent.parent / "third_party")
    if model_name == "tcm":
        model = load_model(model_name, quality, device, p=p, base_dir=base)
    else:
        model = load_model(model_name, quality, device, base_dir=base)
    result = evaluate_codec(model, size=SIZE, device=device, model_name=model_name)
    recon = result["recon"]
    recon_disp = np.clip(
        (recon - recon.min()) / (recon.max() - recon.min() + 1e-9), 0, 1
    )
    metrics = {
        "L_k": result["L_k"],
        "L_low": result["L_low"],
        "L_high": result["L_high"],
        "bpp": result["bpp"],
    }
    del model
    torch.cuda.empty_cache()
    return recon_disp, metrics


def load_from_csv(results_dir: Path, model_name: str, param_type: str, level: int):
    """Load pre-saved image and metrics from results directory."""
    import matplotlib.image as mpimg
    import pandas as pd

    csv = results_dir / "all_metrics_summary.csv"
    df = pd.read_csv(csv)
    d256 = df[df["Size"] == "256x256"]

    if param_type == "p":
        mask = (d256["Model"] == model_name) & (d256["p"] == float(level))
    else:
        mask = (d256["Model"] == model_name) & (d256["q"] == float(level))

    rows = d256[mask]
    metrics = {}
    if not rows.empty:
        r = rows.iloc[0]
        metrics = {"L_k": r["L_k"], "L_low": r["L_low"], "L_high": r["L_high"], "bpp": r["bpp"]}

    subdir = f"p_{level}" if param_type == "p" else f"q_{level}"
    img_path = results_dir / model_name / "256" / subdir / "decompressed_dct_rgb.png"
    img = None
    if img_path.exists():
        img = mpimg.imread(str(img_path))
    return img, metrics


def build_figure(data: dict, out_dir: Path):
    """Compose the 2×11 grid figure."""
    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "mathtext.fontset": "stix",
    })

    n_models = len(CODEC_SPEC)
    fig = plt.figure(figsize=(7.16, 2.9), dpi=300)
    gs = GridSpec(
        2, n_models, figure=fig,
        hspace=0.12, wspace=0.04,
        left=0.04, right=0.99, top=0.88, bottom=0.08,
    )

    axes_top = []
    for col, (label, model_dir, param_type, levels) in enumerate(CODEC_SPEC):
        for row, level in enumerate(levels):
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                axes_top.append(ax)

            key = (model_dir, level)
            img, metrics = data.get(key, (None, {}))

            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
                sp.set_color("#AAAAAA")

            if row == 0:
                ax.set_title(label, fontsize=5.5, fontweight="bold",
                             pad=2, linespacing=0.85)
            if col == 0:
                qlbl = f"$p={level}$" if param_type == "p" else f"$q={level}$"
                ax.set_ylabel(qlbl, fontsize=7, rotation=0,
                              labelpad=10, va="center", fontweight="bold")

            if metrics:
                lk = metrics["L_k"]
                color = _lk_color(lk)
                ax.text(0.5, -0.06, f"$L_k$={_fmt_lk(lk)}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=4.5, fontweight="bold", color=color, clip_on=False)
                ax.text(0.5, -0.19,
                        f"(low={metrics['L_low']:.2f}, high={metrics['L_high']:.2f})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=3.3, color="#000000", clip_on=False)
                if metrics.get("bpp") is not None:
                    ax.text(0.5, -0.29, f"{metrics['bpp']:.2f} bpp",
                            transform=ax.transAxes, ha="center", va="top",
                            fontsize=3.3, fontweight="bold", color="#000000", clip_on=False)

    # Separator between traditional and neural codecs
    x_sep = (axes_top[N_TRAD - 1].get_position().x1 +
             axes_top[N_TRAD].get_position().x0) / 2
    fig.add_artist(plt.Line2D(
        [x_sep, x_sep], [0.10, 0.94],
        transform=fig.transFigure,
        color="#999999", linewidth=0.5, linestyle="--",
    ))
    fig.text(0.18, 0.91, "Traditional codecs", ha="center", fontsize=5.5,
             fontstyle="italic", color="#666666")
    fig.text(0.63, 0.91, "Neural Image Compression", ha="center", fontsize=5.5,
             fontstyle="italic", color="#666666")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"fig3_dct_response_grid.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.0)
        print(f"Saved: {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true",
                        help="Generate reconstructions on-the-fly (GPU required)")
    parser.add_argument("--results", type=Path, default="results",
                        help="Results directory with pre-computed images + CSV")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default="paper/figures")
    args = parser.parse_args()

    data = {}

    for label, model_dir, param_type, levels in CODEC_SPEC:
        for level in levels:
            key = (model_dir, level)
            if args.live:
                import torch
                device = torch.device(args.device if torch.cuda.is_available() else "cpu")
                p = level if param_type == "p" else 128
                q = level if param_type == "q" else 6
                print(f"  {model_dir} {param_type}={level}", end=" ... ", flush=True)
                try:
                    img, metrics = generate_live(model_dir, q, device, p=p)
                    data[key] = (img, metrics)
                    print(f"L_k={metrics['L_k']:.4f}")
                except Exception as e:
                    print(f"FAILED: {e}")
                    data[key] = (None, {})
            else:
                img, metrics = load_from_csv(Path(args.results), model_dir, param_type, level)
                data[key] = (img, metrics)

    build_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
