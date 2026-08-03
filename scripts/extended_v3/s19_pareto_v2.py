#!/usr/bin/env python3
"""S19 (CPU): redesigned joint-FT Pareto figure (main-paper Fig. 12).

Changes vs s12 version: no checkpoint-to-checkpoint connecting lines (they
tangled and ran off the clipped left edge), open-circle baselines at
dPSNR=0, one curved arrow per model from baseline to the starred selected
operating point, marker-semantics legend, and the endpoint that exceeds the
PSNR limit flagged in the color legend. Off-range transient checkpoints are omitted (count
printed for the caption).

Usage: s19_pareto_v2.py [RESULTS_ROOT] [OUT_DIR]
Defaults: /root/dct_benchmark_nic/results and RESULTS_ROOT/ft_tables_tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/root/dct_benchmark_nic/results")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "ft_tables_tex"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

PRETTY = {"bmshj2018-factorized": "BMSHJ2018-Factorized",
          "bmshj2018-hyperprior": "BMSHJ2018-Hyperprior",
          "mbt2018-mean": "MBT2018-Mean", "mbt2018": "MBT2018",
          "cheng2020-anchor": "Cheng2020-Anchor",
          "cheng2020-attn": "Cheng2020-Attention",
          "tcm": "TCM", "ftic": "FTIC"}
colors = dict(zip(PRETTY, plt.cm.tab10.colors))

t1t = pd.read_csv(ROOT / "ft_joint_ckpt/trajectories.csv")
t1t["lam"] = np.where(t1t.config == "joint", 0.03, 0.0)
t2t = pd.read_csv(ROOT / "ft_joint_ckpt_lam01/trajectories_lam01.csv")
lam_col = []
for mn, g in t2t.groupby("model", sort=False):
    n0 = (g.step == 0).cumsum()
    lam_col.extend(np.where(n0 <= 1, 0.01, 0.003)[: len(g)])
t2t["lam"] = lam_col
traj = pd.concat([t1t, t2t], ignore_index=True)
sel = pd.read_csv(ROOT / "ft_tables_tex/ft_joint_final.csv")

XMIN, YMIN = -1.13, 2e-5
fig, ax = plt.subplots(figsize=(6.6, 4.0))
n_omit = 0
for _, r in sel.iterrows():
    mn = r.model
    lam = r.lam
    g = traj[(traj.model == mn) & (traj.lam == lam)].sort_values("step")
    if g.empty:
        continue
    base = g[g.step == 0].iloc[0]
    dx = (g.kod_psnr - base.kod_psnr).to_numpy()
    y = g.L_basis.clip(lower=YMIN).to_numpy()
    steps = g.step.to_numpy()
    inside = dx >= XMIN
    n_omit += int((~inside).sum())
    mid = (steps > 0) & inside
    label = PRETTY[mn]
    ax.scatter(dx[mid], y[mid], s=6 + 22 * steps[mid] / steps.max(),
               color=colors[mn], alpha=0.4, edgecolors="none", zorder=3,
               label=label)
    # baseline: open circle at dPSNR = 0
    ax.scatter([0], [max(base.L_basis, YMIN)], s=34, facecolors="white",
               edgecolors=colors[mn], linewidths=1.2, zorder=4)
    pt = g[g.step == r.step].iloc[0]
    sx, sy = pt.kod_psnr - base.kod_psnr, max(pt.L_basis, YMIN)
    ax.add_patch(FancyArrowPatch(
        (0, max(base.L_basis, YMIN)), (sx, sy),
        connectionstyle="arc3,rad=0.18", arrowstyle="-|>",
        mutation_scale=11, lw=1.3, color=colors[mn], alpha=0.8,
        shrinkA=5, shrinkB=9, zorder=4))
    ax.scatter([sx], [sy], marker="*", s=250, color=colors[mn],
               edgecolors="k", linewidths=0.6, zorder=5)

ax.axvline(0, color="k", lw=0.6, ls=":")
ax.set_xlim(XMIN, 0.2)
ax.set_ylim(YMIN, 1.6)
ax.set_yscale("log")
ax.set_xlabel(r"$\Delta$PSNR (dB)")
ax.set_ylabel(r"median leakage $L_k$")
ax.text(0.035, 0.32, "baseline", fontsize=7, color="0.25",
        ha="left", va="center")
leg1 = ax.legend(fontsize=6, ncol=2, loc="upper left", framealpha=0.9)
ax.add_artist(leg1)
sem = [Line2D([], [], marker="o", mfc="white", mec="0.2", ls="", ms=7,
              label="baseline"),
       Line2D([], [], marker="o", color="0.5", ls="", ms=4, alpha=0.6,
              label="checkpoints"),
       Line2D([], [], marker="*", color="0.3", mec="k", ls="", ms=11,
              label="selected checkpoint")]
ax.legend(handles=sem, fontsize=6, loc="lower left", framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_ft_pareto.pdf")
fig.savefig(OUT / "fig_ft_pareto.png", dpi=160)
plt.close(fig)
print(f"omitted transient checkpoints beyond {XMIN} dB: {n_omit}")
print("S19_DONE")
