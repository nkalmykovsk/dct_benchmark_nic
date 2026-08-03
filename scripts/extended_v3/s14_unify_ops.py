#!/usr/bin/env python3
"""S14: unify classical operating points with Fig 3 overrides (jpeg@11,
webp@0, jpegxl@6.0) for all matched-basis analyses; regenerate readable
figures; recompute dependent numbers.

Outputs: profiles npz for overrides, new fig_excess_leakage_real.pdf,
fig_singlefreq_sweep.pdf (unchanged data, readable), fig_anchor_instability.pdf,
fig_median_leakage_q.pdf (Fig 4a replacement), fig_spatial_maps.pdf (unified),
table_ft_joint_v2.tex (4-decimal L), stats to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
from scipy.fft import dct
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model, evaluate_codec
from dct_nic.metrics import build_dct_basis

ROOT = Path("/root/dct_benchmark_nic")
PROF, S4, S7, SF, NC = (ROOT / "results/profiles", ROOT / "results/analysis_s4",
                        ROOT / "results/analysis_s7", ROOT / "results/singlefreq",
                        ROOT / "results/natural_cache")
FT = ROOT / "results/ft_tables_tex"
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
BLOCK = 32

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NICS6 = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean",
         "mbt2018", "cheng2020-anchor", "cheng2020-attn"]
CLS_OVR = {"jpeg": 11, "webp": 0, "jpegxl": 6.0}
PRETTY = {"bmshj2018-factorized": "BMSHJ2018-Fact.", "bmshj2018-hyperprior": "BMSHJ2018-Hyp.",
          "mbt2018-mean": "MBT2018-Mean", "mbt2018": "MBT2018",
          "cheng2020-anchor": "Cheng2020-Anchor", "cheng2020-attn": "Cheng2020-Attn.",
          "tcm": "TCM (p64)", "ftic": "FTIC",
          "jpeg": "JPEG", "webp": "WebP", "jpegxl": "JPEG XL"}

# ---------- (a) classical override profiles at 256 ----------
prof_sum = pd.read_csv(PROF / "profiles_summary.csv")
profs, bpps = {}, {}
for name, ovr in CLS_OVR.items():
    m = load_model(name, 6, DEV, classical_overrides={name: ovr})
    res = evaluate_codec(m, size=256, device=DEV, model_name=name)
    profs[name], bpps[name] = res["leakage"], float(res["bpp"])
    np.savez_compressed(PROF / f"{name}_ovr{ovr}_n256.npz",
                        leakage=res["leakage"], odr=res["odr"],
                        centroid_shift=res["centroid_shift"],
                        spread=res["spread"], entropy=res["entropy"],
                        bpp=res["bpp"], R=res["R"])
    print(f"[ovr] {name}@{ovr}: bpp={res['bpp']:.2f} L_med={res['L_k']:.4f}",
          flush=True)

for mname in NICS6 + ["ftic"]:
    sub = prof_sum[(prof_sum.model == mname) & (prof_sum["size"] == 256)].dropna(subset=["bpp"])
    r = sub.iloc[(sub["bpp"] - 1.0).abs().argsort().iloc[0]]
    profs[mname] = np.load(PROF / f"{mname}_q{int(r.q)}_n256.npz")["leakage"]
    bpps[mname] = float(r.bpp)
z64 = np.load(PROF / "tcm-p64_q64_n256.npz")
profs["tcm"], bpps["tcm"] = z64["leakage"], float(
    prof_sum[(prof_sum.model == "tcm-p64") & (prof_sum["size"] == 256)]["bpp"].iloc[0])

in_band = [m for m in profs if 0.5 <= bpps[m] <= 1.5]
print("[band] all in-band:", len(in_band) == 11, {m: round(bpps[m], 2) for m in profs},
      flush=True)
floor = np.min(np.stack([profs[m] for m in in_band]), axis=0)
import collections
stack = np.stack([profs[m] for m in in_band])
comp = collections.Counter([in_band[i] for i in np.argmin(stack, axis=0)])
print("[floor composition]", dict(comp), flush=True)
hi = slice(128, 256)
for m in profs:
    exc = profs[m] - floor
    print(f"  {m:22s} mean_excess_upper={exc[hi].mean():.3f}", flush=True)

# excess figure (readable: legend below, distinct colors)
order = NICS6 + ["tcm", "ftic", "jpeg", "webp", "jpegxl"]
colors = {"bmshj2018-factorized": "#1f77b4", "bmshj2018-hyperprior": "#ff7f0e",
          "mbt2018-mean": "#2ca02c", "mbt2018": "#d62728",
          "cheng2020-anchor": "#9467bd", "cheng2020-attn": "#8c564b",
          "tcm": "#e377c2", "ftic": "#7f7f7f",
          "jpeg": "#bcbd22", "webp": "#17becf", "jpegxl": "#000000"}
sm9 = lambda v: np.convolve(v, np.ones(9) / 9, "same")
fig, ax = plt.subplots(figsize=(7.0, 4.2))
for m in order:
    ls = "--" if m in CLS_OVR else "-"
    ax.plot(sm9(profs[m] - floor), ls=ls, lw=1.3, color=colors[m],
            label=f"{PRETTY[m]} ({bpps[m]:.2f})")
ax.set_xlabel("frequency index $k$", fontsize=10)
ax.set_ylabel(r"excess leakage $\Delta L_k$", fontsize=10)
ax.tick_params(labelsize=9)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          title="codec (basis bpp)", title_fontsize=7.5)
fig.tight_layout()
fig.savefig(S4 / "fig_excess_leakage_real.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- pooled metric Spearman on unified in-band set ----------
pool = {k: [] for k in ("L", "D", "S", "H")}
for m in in_band:
    if m in CLS_OVR:
        z = np.load(PROF / f"{m}_ovr{CLS_OVR[m]}_n256.npz")
    elif m == "tcm":
        z = z64
    else:
        sub = prof_sum[(prof_sum.model == m) & (prof_sum["size"] == 256)].dropna(subset=["bpp"])
        r = sub.iloc[(sub["bpp"] - 1.0).abs().argsort().iloc[0]]
        z = np.load(PROF / f"{m}_q{int(r.q)}_n256.npz")
    pool["L"].append(z["leakage"]); pool["D"].append(np.abs(z["centroid_shift"]))
    pool["S"].append(z["spread"]); pool["H"].append(z["entropy"])
pool = {k: np.concatenate(v) for k, v in pool.items()}
print("[pooled Spearman unified]",
      {f"L~{b}": round(float(spearmanr(pool['L'], pool[b])[0]), 3)
       for b in ("D", "S", "H")}, "n=", len(pool["L"]), flush=True)

# ---------- CV maps at unified points ----------
Dm = build_dct_basis(256)
maps, cvs, means = {}, {}, {}
def block_map(rec):
    nb = 256 // BLOCK
    lam = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            a = Dm[i*BLOCK:(i+1)*BLOCK, j*BLOCK:(j+1)*BLOCK]
            b_ = rec[i*BLOCK:(i+1)*BLOCK, j*BLOCK:(j+1)*BLOCK]
            Ca = dct(a, axis=0, norm="ortho"); Cb = dct(b_, axis=0, norm="ortho")
            lam[i, j] = np.sum((Ca - Cb) ** 2) / (np.sum(Ca ** 2) + 1e-12)
    return np.clip(lam, 0, 1)

for m in order:
    if m in CLS_OVR:
        mod = load_model(m, 6, DEV, classical_overrides={m: CLS_OVR[m]})
    elif m == "tcm":
        mod = load_model("tcm", 1, DEV, p=64, base_dir=BASE)
        mod.eval()
    else:
        sub = prof_sum[(prof_sum.model == m) & (prof_sum["size"] == 256)].dropna(subset=["bpp"])
        rq = int(sub.iloc[(sub["bpp"] - 1.0).abs().argsort().iloc[0]]["q"])
        mod = load_model(m, rq, DEV, base_dir=BASE)
        mod.eval()
    res = evaluate_codec(mod, size=256, device=DEV, model_name=m)
    maps[m] = block_map(res["recon"].mean(axis=2))
    cvs[m] = float(maps[m].std() / (maps[m].mean() + 1e-12))
    means[m] = float(maps[m].mean())
    if hasattr(mod, "parameters"):
        del mod
        torch.cuda.empty_cache()
json.dump({"cv": cvs, "mean": means}, open(NC / "basis_cv_unified.json", "w"), indent=1)
print("[CV unified]", {k: round(v, 2) for k, v in cvs.items()}, flush=True)

show = ["cheng2020-anchor", "mbt2018", "bmshj2018-hyperprior", "jpegxl"]
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1))
for ax, mn in zip(axes, show):
    im = ax.imshow(np.clip(maps[mn], 1e-3, 1), cmap="magma",
                   norm=LogNorm(vmin=1e-3, vmax=1.0))
    ax.set_title(f"{PRETTY[mn]}\n"
                 + rf"$\bar\Lambda$={means[mn]:.3f}, CV={cvs[mn]:.2f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
cb.set_label(r"block leakage $\Lambda^{(m)}$ (log)", fontsize=9)
fig.savefig(S7 / "fig_spatial_maps.pdf", bbox_inches="tight")
plt.close(fig)

# ---------- readable fig 6 / fig 7 ----------
sfz = np.load(SF / "singlefreq_profiles.npz")
sf64 = np.load(SF / "singlefreq_tcm64_full.npz")
sm5 = lambda v: np.convolve(v, np.ones(5) / 5, "same")
show6 = [("cheng2020-anchor", "#9467bd"), ("cheng2020-attn", "#8c564b"),
         ("bmshj2018-factorized", "#1f77b4"), ("jpeg", "#bcbd22"),
         ("jpegxl", "#000000")]
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
for mn, c in show6:
    axes[0].semilogy(sm5(sfz[f"{mn}_x2_frob2"]) + 1e-6, lw=1.3, color=c,
                     label=PRETTY[mn])
    axes[1].semilogy(sm5(sfz[f"{mn}_x1_frob2"]) + 1e-6, lw=1.3, color=c)
axes[0].semilogy(sm5(sf64["tcm_x2_frob2"]) + 1e-6, lw=1.3, color="#e377c2",
                 label="TCM (p64)")
axes[1].semilogy(sm5(sf64["tcm_x1_frob2"]) + 1e-6, lw=1.3, color="#e377c2")
axes[0].set_ylabel(r"$\|\mathbf{X}_k^{(2)}-\hat{\mathbf{X}}\|_F^2$", fontsize=10)
axes[1].set_ylabel(r"$\|\mathbf{X}_k^{(1)}-\hat{\mathbf{X}}\|_F^2$", fontsize=10)
axes[1].set_xlabel("frequency index $k$", fontsize=10)
axes[0].legend(fontsize=8.5, ncol=3)
for ax in axes:
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(S7 / "fig_singlefreq_sweep.pdf")
plt.close(fig)

iz = np.load(SF / "anchor_instability.npz")
fig, ax = plt.subplots(figsize=(7.2, 3.1))
for key, c, lab in (("q4_s1.0", "#1f77b4", "$q=4$"), ("q5_s1.0", "#ff7f0e", "$q=5$"),
                    ("q6_s1.0", "#d62728", "$q=6$")):
    ax.semilogy(iz[key] + 1e-6, lw=1.0, color=c, label=lab)
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel("frequency index $k$", fontsize=10)
ax.set_ylabel(r"$\|\mathbf{X}_k^{(2)}-\hat{\mathbf{X}}\|_F^2$", fontsize=10)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(S7 / "fig_anchor_instability.pdf")
plt.close(fig)

# ---------- Fig 4a replacement: median leakage vs q across sizes ----------
ref = pd.read_csv(ROOT / "results/all_metrics_summary.csv")
ref["size_n"] = ref["Size"].str.split("x").str[0].astype(int)
fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True)
for ax, model, title in ((axes[0], "cheng2020-anchor", "Cheng2020-Anchor"),
                         (axes[1], "cheng2020-attn", "Cheng2020-Attention")):
    for sz, c in zip((64, 128, 256, 512, 1024),
                     plt.cm.viridis(np.linspace(0.05, 0.85, 5))):
        sub = ref[(ref.Model == model) & (ref.size_n == sz)].sort_values("q")
        if sub.empty:
            continue
        ax.plot(sub.q, sub.L_k, "-o", ms=3, lw=1.2, color=c, label=f"$n={sz}$")
    ax.set_ylabel(r"median $L_k$", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
axes[1].set_xlabel("quality $q$", fontsize=9)
axes[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(S4 / "fig_median_leakage_q.pdf")
plt.close(fig)

# ---------- Table III (joint) with 4-decimal L ----------
fj = pd.read_csv(FT / "ft_joint_final.csv")
lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Codec & $\lambda_{\mathrm{freq}}$ & step$^{*}$ & "
         r"$\bar L$: bef$\,\to\,$aft & $\Delta$PSNR & $\Delta$LPIPS & "
         r"MSE-only: $\bar L$ ($\Delta$PSNR) \\", r"\midrule"]
for _, r in fj.iterrows():
    if r.step == 0:
        lines.append(f"{PRETTY[r.model].replace(' (p64)','')} & \\multicolumn{{2}}{{c}}{{---}} & "
                     f"{r.L0:.4f} (no adm.\\ point) & --- & --- & "
                     f"{r.ctrl_L1:.4f} ({r.ctrl_dPSNR:+.2f}) \\\\")
    else:
        lines.append(f"{PRETTY[r.model].replace(' (p64)','')} & {r.lam:g} & {int(r.step)} & "
                     f"{r.L0:.4f}$\\,\\to\\,${r.L1:.4f} & {r.dPSNR:+.2f} & "
                     f"{r.dLPIPS:+.4f} & {r.ctrl_L1:.4f} ({r.ctrl_dPSNR:+.2f}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(FT / "table_ft_joint_v2.tex").write_text("\n".join(lines))
# verified gap range
adm = fj[fj.step > 0]
gaps = adm.ctrl_L1 / adm.L1
print("[gap range]", round(gaps.min(), 1), "-", round(gaps.max(), 1), flush=True)
print("S14_DONE", flush=True)
