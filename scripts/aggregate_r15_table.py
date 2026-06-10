#!/usr/bin/env python3
"""Aggregate Table I across datasets for the generality check (Reviewer R1.5).

Reads the per-image CSVs from run_kodak_eval.py for several datasets, pools the
clean-codec rows, and prints the combined codec table (mean +/- std over ALL
images of all datasets) + NIC-only Spearman + a ready-to-paste LaTeX body with
the best value per column bold-faced.

L_k (synthetic DCT-basis leakage) is NOT in the per-image CSV; it is computed
once per codec on the basis. NIC values are dataset-independent (q=6); classical
values depend on the matched-bpp quality, so they are averaged over the datasets
actually present. The constants below come from the run logs ("Median L_k = ...").

Usage:
    python3 scripts/aggregate_r15_table.py \
        results/kodak_eval/kodak_per_image.csv \
        results/kodak_eval/clic_per_image.csv \
        results/kodak_eval/div2k_per_image.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORDER = ["bmshj2018-factorized", "cheng2020-attn", "cheng2020-anchor",
         "mbt2018", "bmshj2018-hyperprior", "mbt2018-mean", "ftic", "tcm",
         "jpeg", "jpegxl", "webp"]
NICS = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean", "mbt2018",
        "cheng2020-anchor", "cheng2020-attn", "tcm", "ftic"]
CLASSICAL = ["jpeg", "jpegxl", "webp"]
LABEL = {
    "bmshj2018-factorized": "BMSHJ2018-Factorized", "bmshj2018-hyperprior": "BMSHJ2018-Hyperprior",
    "mbt2018-mean": "MBT2018-Mean", "mbt2018": "MBT2018",
    "cheng2020-anchor": "Cheng2020-Anchor", "cheng2020-attn": "Cheng2020-Attention",
    "tcm": "TCM", "ftic": "FTIC", "jpeg": "JPEG", "jpegxl": "JPEG~XL", "webp": "WebP",
}

# Synthetic-basis median L_k from the run logs.
LK_NIC = {"bmshj2018-factorized": 0.4551, "cheng2020-attn": 0.2963,
          "cheng2020-anchor": 0.7294, "mbt2018": 0.1150,
          "bmshj2018-hyperprior": 0.1461, "mbt2018-mean": 0.1157,
          "ftic": 0.0046, "tcm": 0.0019}
LK_CLASSICAL = {  # per dataset (matched bpp -> quality -> L_k)
    "jpeg":   {"kodak": 0.0223, "clic": 0.0159, "div2k": 0.0441},
    "jpegxl": {"kodak": 0.0041, "clic": 0.0031, "div2k": 0.0065},
    "webp":   {"kodak": 0.0032, "clic": 0.0018, "div2k": 0.0068},
}


def lk_value(codec, datasets):
    if codec in LK_NIC:
        return LK_NIC[codec]
    vals = [LK_CLASSICAL[codec][d] for d in datasets if d in LK_CLASSICAL.get(codec, {})]
    return float(np.mean(vals)) if vals else float("nan")


def f_small(v):
    """Adaptive formatting: >=0.1 -> 3 dp; else 4 dp."""
    return f"{v:.3f}" if abs(v) >= 0.1 else f"{v:.4f}"


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    frames = []
    for p in sys.argv[1:]:
        df = pd.read_csv(p)
        df = df[df["distortion"] == "codec_clean"].copy()
        if "dataset" not in df.columns:
            df["dataset"] = Path(p).stem.replace("_per_image", "")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    datasets = sorted(data["dataset"].unique())
    print(f"datasets: {datasets}   pooled clean rows: {len(data)}\n")

    metrics = ["bpp", "psnr", "ms_ssim", "lpips", "L_tilde", "ratio"]
    rows = {}
    for codec in ORDER:
        sub = data[data["model"] == codec]
        if sub.empty:
            continue
        rec = {"n": len(sub), "L_k": lk_value(codec, datasets)}
        for m in metrics:
            rec[m] = (float(sub[m].mean()), float(sub[m].std())) if m in sub else (np.nan, np.nan)
        rows[codec] = rec

    # plain summary
    hdr = (f"{'codec':22s} {'n':>3s} {'L_k':>7s} {'bpp':>5s} {'PSNR':>6s} "
           f"{'MSSSIM':>7s} {'LPIPS':>6s} {'L_t':>7s} {'ratio':>7s}")
    print(hdr); print("-" * len(hdr))
    for c in ORDER:
        if c not in rows: continue
        r = rows[c]
        print(f"{LABEL[c].replace('~',' '):22s} {r['n']:>3d} {r['L_k']:>7.4f} "
              f"{r['bpp'][0]:>5.2f} {r['psnr'][0]:>6.2f} {r['ms_ssim'][0]:>7.3f} "
              f"{r['lpips'][0]:>6.3f} {r['L_tilde'][0]:>7.4f} {r['ratio'][0]:>7.4f}")

    # NIC-only Spearman (pooled per-codec means)
    nic = [c for c in NICS if c in rows]
    lt = [rows[c]["L_tilde"][0] for c in nic]
    print("\nNIC-only Spearman (8 codecs, pooled means):")
    sp = {}
    for m in ("psnr", "ms_ssim", "lpips"):
        sp[m] = spearmanr(lt, [rows[c][m][0] for c in nic]).statistic
        print(f"  rho_s(L_tilde, {m}) = {sp[m]:+.2f}")

    # best per column (for bold): min for L_k/lpips/L_tilde/ratio, max for psnr/ms_ssim
    present = [c for c in ORDER if c in rows]
    best = {}
    best["L_k"] = min(present, key=lambda c: rows[c]["L_k"])
    best["psnr"] = max(present, key=lambda c: rows[c]["psnr"][0])
    best["ms_ssim"] = max(present, key=lambda c: rows[c]["ms_ssim"][0])
    for m in ("lpips", "L_tilde", "ratio"):
        best[m] = min(present, key=lambda c: rows[c][m][0])

    def cell(c, key, text):
        return f"\\mathbf{{{text}}}" if best.get(key) == c else text

    print("\n% --- LaTeX rows (mean +/- std over all datasets; best per column bold) ---")
    for c in ORDER:
        if c not in rows: continue
        r = rows[c]
        lk = cell(c, "L_k", f"{r['L_k']:.4f}" if r['L_k'] < 0.1 else f"{r['L_k']:.3f}")
        psnr = cell(c, "psnr", f"{r['psnr'][0]:.2f} \\pm {r['psnr'][1]:.2f}")
        mss = cell(c, "ms_ssim", f"{r['ms_ssim'][0]:.3f} \\pm {r['ms_ssim'][1]:.3f}")
        lp = cell(c, "lpips", f"{r['lpips'][0]:.3f} \\pm {r['lpips'][1]:.3f}")
        lt_ = cell(c, "L_tilde", f"{f_small(r['L_tilde'][0])} \\pm {f_small(r['L_tilde'][1])}")
        rt = cell(c, "ratio", f_small(r["ratio"][0]))
        print(f"{LABEL[c]:20s} & ${lk}$ & ${r['bpp'][0]:.2f} \\pm {r['bpp'][1]:.2f}$ "
              f"& ${psnr}$ & ${mss}$ & ${lp}$ & ${lt_}$ & ${rt}$ \\\\")

    print("\n% Spearman footnote:")
    print(f"% rho_s(L,PSNR)={sp['psnr']:+.2f}, rho_s(L,MS-SSIM)={sp['ms_ssim']:+.2f}, "
          f"rho_s(L,LPIPS)={sp['lpips']:+.2f}")


if __name__ == "__main__":
    main()
