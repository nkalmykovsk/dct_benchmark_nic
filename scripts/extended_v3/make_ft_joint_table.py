#!/usr/bin/env python3
"""S9 (CPU): final joint fine-tuning table from S1c/S1d trajectories.

Operating-point selection uses ONLY the disjoint DIV2K-val monitor:
  step* = argmax_{step} leakage reduction s.t. mon_psnr(step) >= mon_psnr(0) - 0.10 dB
Kodak-24 metrics at step* are then reported (no selection on the test set).
For the 4 architectures with lambda sweep (S1d: 0.01/0.003), lambda is also
selected on the monitor by the same rule (largest leakage reduction among
admissible (lambda, step) pairs).

Output: results/ft_tables_tex/table_ft_joint_final.tex + CSV + Pareto figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/dct_benchmark_nic/results")
OUT = ROOT / "ft_tables_tex"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRETTY = {"bmshj2018-factorized": "BMSHJ2018-Fact.",
          "bmshj2018-hyperprior": "BMSHJ2018-Hyp.",
          "mbt2018-mean": "MBT2018-Mean", "mbt2018": "MBT2018",
          "cheng2020-anchor": "Cheng2020-Anch.",
          "cheng2020-attn": "Cheng2020-Att.",
          "tcm": "TCM", "ftic": "FTIC"}
GATE = 0.10

t1 = pd.read_csv(ROOT / "ft_joint_ckpt/trajectories.csv")
t1["lam"] = np.where(t1.config == "joint", 0.03, 0.0)
frames = [t1]
p2 = ROOT / "ft_joint_ckpt_lam01/trajectories_lam01.csv"
if p2.exists():
    t2 = pd.read_csv(p2)
    # config column is 'joint' for both lams there; recover lam by run order:
    # script ran (0.01, guards) then (0.003, guards) per model, both cfg='joint'
    # -> disambiguate by cumulative step resets
    lam_seq = []
    cur = 0.01
    prev_step = -1
    for _, r in t2.iterrows():
        if r.step < prev_step or (r.step == 0 and prev_step >= 0):
            cur = 0.003 if cur == 0.01 else 0.01
        lam_seq.append(cur)
        prev_step = r.step
    # simpler: alternate per (model, block); recompute per model
    lam_col = []
    for m, g in t2.groupby("model", sort=False):
        n0 = (g.step == 0).cumsum()
        lam_col.extend(np.where(n0 <= 1, 0.01, 0.003)[: len(g)])
    t2 = t2.sort_index()
    t2["lam"] = lam_col
    frames.append(t2)
traj = pd.concat(frames, ignore_index=True)


def select(g):
    g = g.sort_values("step")
    base = g[g.step == 0].iloc[0]
    ok = g[g.mon_psnr >= base.mon_psnr - GATE]
    best = ok.sort_values("L_basis").iloc[0] if len(ok) else base
    return base, best


rows = []
for m in PRETTY:
    cand = []
    for lam, g in traj[(traj.model == m) & (traj.lam > 0)].groupby("lam"):
        base, best = select(g)
        cand.append((lam, base, best))
    if not cand:
        continue
    lam, base, best = min(cand, key=lambda c: c[2].L_basis)
    # mse-only control at ITS admissible best step
    ctrl = traj[(traj.model == m) & (traj.lam == 0.0)]
    cbase, cbest = select(ctrl) if len(ctrl) else (None, None)
    rows.append({
        "model": m, "lam": lam, "step": int(best.step),
        "L0": base.L_basis, "L1": best.L_basis,
        "dPSNR": best.kod_psnr - base.kod_psnr,
        "dSSIM": best.kod_ms_ssim - base.kod_ms_ssim,
        "dLPIPS": best.kod_lpips - base.kod_lpips,
        "dHF": best.kod_hf_psnr - base.kod_hf_psnr,
        "ctrl_step": int(cbest.step) if cbest is not None else -1,
        "ctrl_L1": cbest.L_basis if cbest is not None else np.nan,
        "ctrl_dPSNR": (cbest.kod_psnr - cbase.kod_psnr) if cbest is not None else np.nan,
        "ctrl_dLPIPS": (cbest.kod_lpips - cbase.kod_lpips) if cbest is not None else np.nan,
    })
df = pd.DataFrame(rows)
df.to_csv(OUT / "ft_joint_final.csv", index=False)
print(df.round(4).to_string())

lines = [r"\begin{tabular}{lccrrrrcc}", r"\toprule",
         r"Codec & $\lambda_{\mathrm{freq}}$ & step$^{*}$ & "
         r"$\bar L$ bef$\to$aft & $\Delta$PSNR & $\Delta$HF-PSNR & $\Delta$LPIPS & "
         r"$\bar L^{\mathrm{ctrl}}$ & $\Delta$PSNR$^{\mathrm{ctrl}}$ \\",
         r"\midrule"]
for _, r in df.iterrows():
    lines.append(
        f"{PRETTY[r.model]} & {r.lam:g} & {r.step} & "
        f"{r.L0:.3f}$\\to${r.L1:.3f} & {r.dPSNR:+.2f} & {r.dHF:+.2f} & "
        f"{r.dLPIPS:+.4f} & {r.ctrl_L1:.3f} & {r.ctrl_dPSNR:+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_ft_joint_final.tex").write_text("\n".join(lines))

# Pareto figure: leakage vs kodak dPSNR trajectories
fig, ax = plt.subplots(figsize=(6.4, 4.2))
colors = dict(zip(PRETTY, plt.cm.tab10.colors))
for m in PRETTY:
    g = traj[(traj.model == m) & (traj.lam == 0.03)].sort_values("step")
    if g.empty:
        continue
    base = g[g.step == 0].iloc[0]
    ax.plot(g.kod_psnr - base.kod_psnr, g.L_basis, "-o", ms=2.5, lw=1,
            color=colors[m], label=m)
ax.axvline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel(r"$\Delta$PSNR on Kodak-24 (dB)")
ax.set_ylabel(r"median $L_k$ (256 basis)")
ax.set_yscale("log")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_ft_pareto.png", dpi=150)
print("S9_DONE")
