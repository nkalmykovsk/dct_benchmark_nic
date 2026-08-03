#!/usr/bin/env python3
"""S20 (CPU): redesigned excess-leakage figure (main-paper Fig. 9).

Changes vs the s14 version: the six high-leakage NIC curves are collapsed
into a min-max band with a median line (they carry one message and tangled
as individual lines), legend moves inside the axes, per-codec bpp values
move to the caption as a range. Data identical: matched Fig-3 configs,
floor = pointwise minimum across all eleven in-band codecs, window-9
smoothing applied to each profile before band/median construction.

Usage: s20_excess_v2.py [PROFILES_DIR] [OUT_DIR]
Defaults: /root/dct_benchmark_nic/results/profiles and
/root/dct_benchmark_nic/results/analysis_s4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/root/dct_benchmark_nic/results/profiles")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/root/dct_benchmark_nic/results/analysis_s4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICS6 = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean",
         "mbt2018", "cheng2020-anchor", "cheng2020-attn"]
CLS_OVR = {"jpeg": "jpeg_ovr11_n256.npz", "webp": "webp_ovr0_n256.npz",
           "jpegxl": "jpegxl_ovr6.0_n256.npz"}

prof_sum = pd.read_csv(PROF / "profiles_summary.csv")
profs, bpps = {}, {}
for name, fn in CLS_OVR.items():
    z = np.load(PROF / fn)
    profs[name], bpps[name] = z["leakage"], float(z["bpp"])
for mname in NICS6 + ["ftic"]:
    sub = prof_sum[(prof_sum.model == mname)
                   & (prof_sum["size"] == 256)].dropna(subset=["bpp"])
    r = sub.iloc[(sub["bpp"] - 1.0).abs().argsort().iloc[0]]
    profs[mname] = np.load(PROF / f"{mname}_q{int(r.q)}_n256.npz")["leakage"]
    bpps[mname] = float(r.bpp)
z64 = np.load(PROF / "tcm-p64_q64_n256.npz")
profs["tcm"], bpps["tcm"] = z64["leakage"], float(
    prof_sum[(prof_sum.model == "tcm-p64")
             & (prof_sum["size"] == 256)]["bpp"].iloc[0])

in_band = [m for m in profs if 0.5 <= bpps[m] <= 1.5]
assert len(in_band) == 11, in_band
floor = np.min(np.stack([profs[m] for m in in_band]), axis=0)
print(f"bpp range: {min(bpps.values()):.2f}-{max(bpps.values()):.2f}")

sm9 = lambda v: np.convolve(v, np.ones(9) / 9, "same")
exc = {m: sm9(profs[m] - floor) for m in profs}
band = np.stack([exc[m] for m in NICS6])

fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
ax.fill_between(np.arange(256), band.min(axis=0), band.max(axis=0),
                color="#4c72b0", alpha=0.15, lw=0,
                label="BMSHJ/MBT/Cheng (6 models, min–max)")
for m in NICS6:
    ax.plot(exc[m], color="#4c72b0", lw=0.7, alpha=0.38, zorder=2)
ax.plot(np.median(band, axis=0), color="#4c72b0", lw=1.9, zorder=3,
        label="BMSHJ/MBT/Cheng median")
ax.plot(exc["tcm"], color="#e377c2", lw=1.5, label="TCM (p64)")
ax.plot(exc["ftic"], color="#7f7f7f", lw=1.5, label="FTIC")
ax.plot(exc["jpeg"], color="#bcbd22", lw=1.4, ls="--", label="JPEG")
ax.plot(exc["webp"], color="#17becf", lw=1.4, ls="--", label="WebP")
ax.plot(exc["jpegxl"], color="k", lw=1.4, ls="--", label="JPEG XL")
ax.set_xlabel("frequency index $k$", fontsize=10)
ax.set_ylabel(r"excess leakage $\Delta L_k = L_k - F_k$", fontsize=10)
ax.set_xlim(0, 255)
ax.tick_params(labelsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_excess_leakage_real.{ext}", dpi=170)
plt.close(fig)
print("S20_DONE")
