#!/usr/bin/env python3
"""S22 (CPU): redesigned single-frequency figure (main-paper Fig. 6).

Panel (a): X2 single-frequency reconstruction error vs frequency k (six
representative codecs, clamped, smoothed) -- the direct probe response.
Panel (b): the X1/X2 error ratio vs k, same codecs, reference line at 1 --
makes orientation anisotropy explicit and frequency-resolved (above 1 =
worse on vertical gratings X1; below 1 = worse on diagonal X2). This
replaces the previous two-stacked-panels layout, where the anisotropy (the
gap between panels) was illegible.

Usage: s22_singlefreq_v2.py [SINGLEFREQ_DIR] [OUT_DIR]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/root/dct_benchmark_nic/results/singlefreq")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "/root/dct_benchmark_nic/results/analysis_s7")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRETTY = {"cheng2020-anchor": "Cheng2020-Anchor",
          "cheng2020-attn": "Cheng2020-Attn.",
          "bmshj2018-factorized": "BMSHJ2018-Fact.",
          "jpeg": "JPEG", "jpegxl": "JPEG XL", "tcm": "TCM (p64)"}
SHOW = [("cheng2020-anchor", "#9467bd"), ("cheng2020-attn", "#8c564b"),
        ("bmshj2018-factorized", "#1f77b4"), ("jpeg", "#bcbd22"),
        ("jpegxl", "#000000"), ("tcm", "#e377c2")]

sfz = np.load(SF / "singlefreq_profiles.npz")
sf64 = np.load(SF / "singlefreq_tcm64_full.npz")


def curve(mn, stim):
    key = f"tcm_{stim}_frob2" if mn == "tcm" else f"{mn}_{stim}_frob2"
    src = sf64 if mn == "tcm" else sfz
    return src[key]


def sm(v, w=7):
    return np.convolve(v, np.ones(w) / w, "same")


fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.25, 1.0]})
n = 256
kk = np.arange(n)
k0 = 2  # first indices have ~zero error on both stimuli -> noisy ratio
for mn, c in SHOW:
    x2 = sm(curve(mn, "x2"))
    x1 = sm(curve(mn, "x1"))
    axes[0].semilogy(kk, x2 + 1e-6, lw=1.3, color=c, label=PRETTY[mn])
    axes[1].semilogy(kk[k0:], np.clip(x2[k0:] / (x1[k0:] + 1e-6), 3e-2, 3e1),
                     lw=1.3, color=c)

axes[0].set_ylabel(r"$\|\mathbf{X}_k^{(2)}-\hat{\mathbf{X}}\|_F^2$",
                   fontsize=10)
axes[0].legend(fontsize=8, ncol=3, loc="lower right")
axes[0].set_title(r"single-frequency error (diagonal stimulus $\mathbf{X}^{(2)}$)",
                  fontsize=9.5)

axes[1].axhline(1.0, color="0.4", lw=1.0, ls="--")
axes[1].set_yscale("log")
axes[1].set_ylim(3e-2, 3e1)
axes[1].set_ylabel(r"$\|\mathbf{X}^{(2)}{-}\hat{\mathbf{X}}\|_F^2 / "
                   r"\|\mathbf{X}^{(1)}{-}\hat{\mathbf{X}}\|_F^2$",
                   fontsize=9.5)
axes[1].set_xlabel("frequency index $k$", fontsize=10)
axes[1].set_title(r"orientation anisotropy  ($>$1: worse on diagonal "
                  r"$\mathbf{X}^{(2)}$;  $<$1: worse on vertical "
                  r"$\mathbf{X}^{(1)}$)", fontsize=9.5)
axes[1].set_xlim(0, n - 1)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_singlefreq_sweep.{ext}", dpi=170)
plt.close(fig)

# print the anisotropy spread for the caption/text
print("[X1/X2 median-error ratio, all codecs]", flush=True)
for mn in list(PRETTY) + ["mbt2018", "mbt2018-mean",
                          "bmshj2018-hyperprior", "webp", "ftic"]:
    try:
        x2 = curve(mn, "x2"); x1 = curve(mn, "x1")
        print(f"  {mn:22s} {np.median(x1)/np.median(x2):.2f}", flush=True)
    except Exception:
        pass
print("S22_DONE", flush=True)
