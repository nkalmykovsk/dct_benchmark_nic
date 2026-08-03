#!/usr/bin/env python3
"""S25 (CPU): rate-leakage figure (main-paper Fig. 5) regenerated from
profiles_summary.csv with the HiFiC anchor added (three fixed operating
points, 256 basis). Matches the original figure's content and styling;
HiFiC drawn as a solid black starred line.

Usage: s25_fig5_hific.py [RESULTS_ROOT] [OUT_DIR]
RESULTS_ROOT must contain profiles/profiles_summary.csv and
hific/hific_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/root/dct_benchmark_nic/results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "analysis_s7"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = [
    # (model, label, color, marker, dashed)
    ("jpegxl", "JPEG XL", "#4fc3c8", "d", True),
    ("jpeg", "JPEG", "#2a4d8f", "*", True),
    ("webp", "WebP", "#3f8f3f", "v", True),
    ("ftic", "FTIC", "#c23b3b", "H", False),
    ("tcm", "TCM", "#e8842c", "X", False),
    ("mbt2018", "MBT2018", "#a8c4e0", "P", False),
    ("mbt2018-mean", "MBT2018-Mean", "#7f7f7f", "o", False),
    ("bmshj2018-hyperprior", "BMSHJ2018-Hyperprior", "#b5a642", "p", False),
    ("bmshj2018-factorized", "BMSHJ2018-Factorized", "#6b4226", "^", False),
    ("cheng2020-attn", "Cheng2020-Attention", "#e07fbf", "D", False),
    ("cheng2020-anchor", "Cheng2020-Anchor", "#7b3294", "s", False),
]

df = pd.read_csv(ROOT / "profiles/profiles_summary.csv")
df = df[df["size"] == 256].dropna(subset=["bpp"])
hific = json.load(open(ROOT / "hific/hific_summary.json"))

fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
for model, label, color, marker, dashed in STYLE:
    if model == "tcm":
        sub = pd.concat([df[df.model == "tcm-p64"], df[df.model == "tcm-p128"]])
    else:
        sub = df[df.model == model]
    if sub.empty:
        continue
    sub = sub.sort_values("bpp")
    ax.semilogy(sub["bpp"], sub["L_k"].clip(lower=1e-5),
                marker=marker, ms=6, lw=1.3, color=color,
                ls="--" if dashed else "-", label=label,
                markeredgecolor="k", markeredgewidth=0.4)

# HiFiC anchor: three fixed operating points
hx = [hific[k]["basis_bpp"] for k in ("low", "med", "hi")]
hy = [hific[k]["L_k"] for k in ("low", "med", "hi")]
ax.semilogy(hx, hy, marker="*", ms=11, lw=1.6, color="k", ls="-",
            label="HiFiC (GAN)", markeredgecolor="k", zorder=5)

ax.set_xlabel("bpp", fontsize=11)
ax.set_ylabel(r"Median $L_k$", fontsize=11)
ax.set_xlim(0, 8)
ax.tick_params(labelsize=9)
ax.grid(alpha=0.3)
ax.legend(
    fontsize=7.5,
    ncol=1,
    loc="lower right",
    framealpha=0.9,
    borderpad=0.45,
    labelspacing=0.25,
    handlelength=1.8,
)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_leakage_vs_bpp_256.{ext}", dpi=170)
plt.close(fig)
print("hific points (bpp, L_k):", list(zip(np.round(hx, 3), np.round(hy, 3))))
print("S25_DONE")
