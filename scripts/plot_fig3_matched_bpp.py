#!/usr/bin/env python3
"""Matched-bitrate DCT-basis response row (Reviewer R1.3).

Fig. 3 shows codecs at their native quality, hence different bitrates. This
renders ONE row of 256x256 DCT-basis reconstructions where every codec is set to
the configuration whose bitrate is closest to --target-bpp, selected from the
rate-leakage sweep (results/leakage_vs_bpp/scan_leakage_vs_bpp_256.csv). The
high-frequency gap between learned and classical codecs persists at matched rate.

Usage:
    python3 scripts/plot_fig3_matched_bpp.py --target-bpp 1.0 --device cuda
    python3 scripts/plot_fig3_matched_bpp.py --smoke   # print selected settings only
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent
if not (_REPO / "utils" / "loaders.py").exists():
    _REPO = _REPO.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SIZE = 256
N_TRAD = 3
# (display label, model name) in the paper's column order
CODECS = [
    ("JPEG", "jpeg"), ("JPEG XL", "jpegxl"), ("WebP", "webp"),
    ("BMSHJ2018-\nFactorized", "bmshj2018-factorized"),
    ("BMSHJ2018-\nHyperprior", "bmshj2018-hyperprior"),
    ("MBT2018-\nMean", "mbt2018-mean"), ("MBT2018", "mbt2018"),
    ("Cheng2020-\nAnchor", "cheng2020-anchor"),
    ("Cheng2020-\nAttention", "cheng2020-attn"),
    ("TCM", "tcm"), ("FTIC", "ftic"),
]


def _fmt_lk(v):
    return f"{v:.0e}".replace("e-0", "e-") if v < 0.001 else f"{v:.3f}"


def _lk_color(lk):
    return "#2D6A2E" if lk < 0.1 else "#8B7500" if lk < 0.3 else "#B45309" if lk < 0.6 else "#A91E1E"


def _resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else _REPO / p


def select_settings(scan_csv, target):
    """For each codec pick the sweep row whose bpp is closest to `target`."""
    rows = defaultdict(list)
    with open(scan_csv) as f:
        for r in csv.DictReader(f):
            try:
                rows[r["model"]].append((r["setting"], float(r["bpp"]), float(r["L_k"])))
            except (KeyError, ValueError):
                pass
    chosen = {}
    for _, model in CODECS:
        if rows.get(model):
            chosen[model] = min(rows[model], key=lambda x: abs(x[1] - target))
    return chosen


def load_for_setting(model, setting, device, base):
    """Build the codec for a sweep `setting` string (override=V / q=N / p=N)."""
    try:
        from dct_nic import load_model         # public repo package
    except ImportError:
        from utils.loaders import load_model   # server working copy
    if setting.startswith("override="):
        v = setting.split("=", 1)[1]
        return load_model(model, 6, device, base_dir=base,
                          classical_overrides={model: v})
    if setting.startswith("p="):
        return load_model("tcm", 6, device, p=int(float(setting.split("=", 1)[1])), base_dir=base)
    # q=N
    return load_model(model, int(float(setting.split("=", 1)[1])), device, base_dir=base)


def _prepare_display(recon):
    """Grayscale [0,1] tile for crisp imshow (no RGB fringing on DCT basis)."""
    if recon.ndim == 3:
        recon = recon.mean(axis=2)
    lo, hi = float(recon.min()), float(recon.max())
    return np.clip((recon - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def evaluate_codec(codec, size, device, model_name=""):
    """Single-pass DCT-basis eval. Uses the dct_nic helper if present, else the utils API."""
    try:
        from dct_nic import evaluate_codec as _eval   # public repo package
        return _eval(codec, size=size, device=device, model_name=model_name)
    except ImportError:
        pass
    from utils.functions import evaluate_frequency_response, compute_band_leakage
    _, x_hat, metrics = evaluate_frequency_response(
        codec, size=size, device=str(device),
        num_runs=1, show_plots=False, show_metric_plots=False, verbose=False,
    )
    leakage = metrics["leakage"]
    bands = compute_band_leakage(leakage)
    summary = metrics.get("summary") or {}
    return {
        "recon": x_hat,
        "L_k": summary.get("L_k", float(np.median(leakage))),
        "bpp": metrics.get("bpp"),
        "L_low": bands["L_low"],
        "L_high": bands["L_high"],
    }


def render(panels, target, out_dir, dpi=600):
    """panels: list of (label, model, img_or_None, metrics)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix"})
    n = len(panels)
    fig = plt.figure(figsize=(7.16, 1.8), dpi=dpi)
    gs = GridSpec(1, n, figure=fig, wspace=0.04, left=0.02, right=0.99, top=0.84, bottom=0.08)
    tops = []
    for col, (label, _model, img, m) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col]); tops.append(ax)
        if img is not None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color("#AAAAAA")
        ax.set_title(label, fontsize=5.5, fontweight="bold", pad=2, linespacing=0.85)
        if m:
            ax.text(0.5, -0.06, f"$L_k$={_fmt_lk(m['L_k'])}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=4.5, fontweight="bold",
                    color=_lk_color(m["L_k"]), clip_on=False)
            if m.get("L_low") is not None:
                ax.text(0.5, -0.19, f"(low={m['L_low']:.2f}, high={m['L_high']:.2f})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=3.3, color="#000000", clip_on=False)
            if m.get("bpp") is not None:
                ax.text(0.5, -0.30, f"{m['bpp']:.2f} bpp", transform=ax.transAxes,
                        ha="center", va="top", fontsize=3.3, fontweight="bold",
                        color="#000000", clip_on=False)
    # separator + group labels — line spans title..bpp text; labels sit above titles
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _panel_bbox(ax):
        extra = [t for t in ax.texts if not t.get_clip_on()]
        bb = ax.get_tightbbox(renderer, call_axes_locator=True,
                              bbox_extra_artists=extra or None)
        return bb.transformed(fig.transFigure.inverted())

    y_top = max(_panel_bbox(ax).y1 for ax in tops)
    y_bot = min(_panel_bbox(ax).y0 for ax in tops)

    pos_l = tops[N_TRAD - 1].get_position()
    pos_r = tops[N_TRAD].get_position()
    x = (pos_l.x1 + pos_r.x0) / 2
    fig.add_artist(plt.Line2D([x, x], [y_bot, y_top], transform=fig.transFigure,
                              color="#999999", lw=0.5, ls="--", clip_on=False))
    trad_c = (tops[0].get_position().x0 + tops[N_TRAD - 1].get_position().x1) / 2
    neur_c = (tops[N_TRAD].get_position().x0 + tops[-1].get_position().x1) / 2
    y_group = y_top + 0.012
    fig.text(trad_c, y_group, "Traditional codecs", ha="center", va="bottom", fontsize=5.5,
             fontstyle="italic", color="#666666")
    fig.text(neur_c, y_group, "Neural Image Compression", ha="center", va="bottom", fontsize=5.5,
             fontstyle="italic", color="#666666")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"fig3_matched_bpp_{target:.1f}"
    fig.savefig(f"{stem}.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.01,
                facecolor="white", edgecolor="none")
    print(f"Saved: {stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.01,
                facecolor="white", edgecolor="none", pil_kwargs={"compress_level": 1})
    print(f"Saved: {stem}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-bpp", type=float, default=1.0)
    ap.add_argument("--scan-csv", default="results/leakage_vs_bpp/scan_leakage_vs_bpp_256.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/fig3_matched")
    ap.add_argument("--dpi", type=int, default=600, help="export DPI (600 = print-ready panels)")
    ap.add_argument("--smoke", action="store_true", help="print selected settings, no codec runs")
    args = ap.parse_args()

    scan_csv = _resolve(args.scan_csv)
    if not scan_csv.exists():
        raise FileNotFoundError(f"Scan CSV not found: {scan_csv}")

    sel = select_settings(scan_csv, args.target_bpp)
    print(f"Settings nearest to {args.target_bpp} bpp:")
    for _, model in CODECS:
        s = sel.get(model)
        print(f"  {model:22s} {s[0]:14s} bpp_sweep={s[1]:.3f}" if s else f"  {model:22s} (no data)")
    if args.smoke:
        return

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = str(_REPO / "third_party")

    out_dir = _resolve(args.out)
    panels, summary = [], []
    for label, model in CODECS:
        s = sel.get(model)
        if s is None:
            panels.append((label, model, None, {})); continue
        try:
            codec = load_for_setting(model, s[0], device, base)
            res = evaluate_codec(codec, size=SIZE, device=device, model_name=model)
            disp = _prepare_display(res["recon"])
            m = {"L_k": res["L_k"], "L_low": res["L_low"], "L_high": res["L_high"], "bpp": res["bpp"]}
            panels.append((label, model, disp, m))
            summary.append([model, s[0], f"{res['bpp']:.4f}" if res["bpp"] else "",
                            f"{res['L_k']:.6f}", f"{res['L_low']:.6f}", f"{res['L_high']:.6f}"])
            print(f"  {model:22s} {s[0]:14s} -> bpp={res['bpp']:.3f}  L_k={res['L_k']:.4f}")
        except Exception as e:
            print(f"  {model:22s} FAILED: {e}")
            panels.append((label, model, None, {}))
        if model not in ("jpeg", "webp", "jpegxl"):
            torch.cuda.empty_cache()

    render(panels, args.target_bpp, out_dir, dpi=args.dpi)
    with open(out_dir / "matched_bpp_settings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "setting", "bpp", "L_k", "L_low", "L_high"])
        w.writerows(summary)
    print(f"-> saved fig + matched_bpp_settings.csv in {out_dir}")


if __name__ == "__main__":
    main()
