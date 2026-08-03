#!/usr/bin/env python3
"""S23b (CPU): instability figure for all released Anchor checkpoints.

Shows unclamped single-frequency error over k for q=1,...,6. The q=1,2,3
checkpoints use N=128, while q=4,5,6 use N=192; this is reported as a
checkpoint-family comparison rather than an isolated architecture effect.

Usage: s23b_instability_fig.py [SINGLEFREQ_DIR] [OUT_DIR]
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

iz = np.load(SF / "anchor_instability_allq.npz")

fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
curves = (
    (1, "#9ecae1", 128), (2, "#6baed6", 128), (3, "#3182bd", 128),
    (4, "#fdae6b", 192), (5, "#f16913", 192), (6, "#d7301f", 192),
)
for q, color, channels in curves:
    ax.semilogy(
        iz[f"q{q}_s1.0"] + 1e-6,
        lw=0.9,
        color=color,
        label=rf"$q={q}$ ($N={channels}$)",
    )
ax.axhline(1.0, color="k", ls=":", lw=1.0)
ax.set_xlabel("frequency index $k$", fontsize=10)
ax.set_ylabel(r"$\|\mathbf{X}_k^{(2)}-\hat{\mathbf{X}}\|_F^2$ (unclamped)",
              fontsize=10)
ax.set_xlim(0, 255)
ax.set_ylim(1e-5, 3e12)
ax.legend(fontsize=7.5, loc="upper right", ncol=2)
ax.tick_params(labelsize=9)
ax.grid(alpha=0.25)
ax.text(0.985, 0.06,
        r"dotted line: input energy $\|\mathbf{X}\|_F^2{=}1$",
        transform=ax.transAxes, fontsize=8, ha="right", color="0.35")

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_anchor_instability.{ext}", dpi=170)
plt.close(fig)
print("S23B_DONE")
