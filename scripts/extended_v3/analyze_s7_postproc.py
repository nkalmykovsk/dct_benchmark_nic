#!/usr/bin/env python3
"""S7 (CPU): post-processing of S3/S5 caches.

A) Natural-image coupling: recompute L_tilde from cached radial profiles,
   VERIFY against published kodak_eval CSVs, then compute CSF-weighted
   variants and NIC-only Spearman correlations with PSNR/LPIPS.
B) Block-wise analysis: content x leakage predictor vs measured block
   artifact energy; compare against content-only baseline predictors.
   + figure of basis-domain spatial leakage maps.
C) Single-frequency post-processing: full-basis vs single-freq consistency
   (rank correlations), X1/X2 anisotropy table, instability profile figure,
   sweep figure for the paper.

Outputs -> results/analysis_s7/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/root/dct_benchmark_nic")
NC = ROOT / "results/natural_cache"
SF = ROOT / "results/singlefreq"
PROF = ROOT / "results/profiles"
OUT = ROOT / "results/analysis_s7"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NICS = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean",
        "mbt2018", "cheng2020-anchor", "cheng2020-attn", "tcm", "ftic"]
CLS = ["jpeg", "webp", "jpegxl"]
ALL = NICS + CLS
DATASETS = ["kodak", "clic", "div2k"]

nat = pd.read_csv(NC / "natural_metrics.csv")


def leak_profile_512(model, ds):
    """L_k profile at the Table-I operating point (512 basis)."""
    if model == "tcm":
        return np.load(PROF / "tcm-p128_q128_n512.npz")["leakage"]
    if model in NICS:
        return np.load(PROF / f"{model}_q6_n512.npz")["leakage"]
    q = int(nat[(nat["model"] == model) & (nat["dataset"] == ds)]["q"].iloc[0])
    return np.load(PROF / f"{model}_q{q}_n512.npz")["leakage"]


def csf_ms(f_cpd):
    return np.maximum(2.6 * (0.0192 + 0.114 * f_cpd)
                      * np.exp(-(0.114 * f_cpd) ** 1.1), 0)


# ---------------- A) coupling + CSF ----------------
rows = []
for model in ALL:
    for ds in DATASETS:
        f = NC / f"radial_{model}_{ds}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        S, D = z["S"], z["D"]                       # (n_img, 512)
        n_bins = S.shape[1]
        rho = D / (S + D + 1e-12)
        Lk = np.clip(leak_profile_512(model, ds), 0, 1)
        centers = (np.arange(n_bins) + 0.5) / n_bins
        L_r = np.interp(centers, np.linspace(0, 1, len(Lk)), Lk)
        # CSF weight on the radial axis (32 px/deg default)
        for ppd in (32,):
            f_cpd = centers * 0.5 * ppd     # bin center in cycles/pixel * ppd
            w_csf = csf_ms(f_cpd)
            w_csf = w_csf / w_csf.mean()
            # content weight per image: S(f)/max S(f)
            w_img = w_csf[None, :] * (S / (S.max(axis=1, keepdims=True) + 1e-12))
            L_t = (rho * L_r[None, :]).mean(axis=1)
            rho_b = rho.mean(axis=1)
            L_csf = (rho * w_csf[None, :] * L_r[None, :]).mean(axis=1)
            L_csf_img = (rho * w_img * L_r[None, :]).mean(axis=1)
            for i, img in enumerate(z["images"]):
                rows.append({"model": model, "dataset": ds, "image": str(img),
                             "ppd": ppd,
                             "L_tilde": float(L_t[i]), "rho_bar": float(rho_b[i]),
                             "L_csf": float(L_csf[i]),
                             "L_csf_img": float(L_csf_img[i])})
cp = pd.DataFrame(rows)
cp.to_csv(OUT / "coupling_per_image.csv", index=False)

# verification vs published per-image L_tilde (kodak file has column L_tilde)
print("[VERIFY coupling vs published]")
ok_all = True
for ds in DATASETS:
    ref = pd.read_csv(ROOT / f"results/kodak_eval/{ds}_per_image.csv")
    lt_col = "L_tilde" if "L_tilde" in ref.columns else ("Le" if "Le" in ref.columns else None)
    if lt_col is None:
        print(f"  {ds}: no L_tilde column ({list(ref.columns)[:8]})")
        continue
    if "distortion" in ref.columns and (ref["distortion"] == "clean").any():
        ref = ref[ref["distortion"] == "clean"]
    a = cp[cp["dataset"] == ds].groupby("model")["L_tilde"].mean()
    b = ref.groupby("model")[lt_col].mean()
    for m in a.index:
        if m in b.index:
            d = abs(a[m] - b[m])
            status = "OK " if d < 0.02 else "DIFF"
            if d >= 0.02:
                ok_all = False
            print(f"  {status} {ds}/{m}: mine={a[m]:.4f} ref={b[m]:.4f}")

# NIC-only Spearman with quality metrics (pooled 115 images)
met = nat.groupby("model")[["psnr", "ms_ssim", "lpips"]].mean()
agg = cp.groupby("model")[["L_tilde", "L_csf", "L_csf_img", "rho_bar"]].mean()
tbl = met.join(agg)
tbl["ratio"] = tbl["L_tilde"] / tbl["rho_bar"]
tbl.to_csv(OUT / "coupling_table.csv")
nic_tbl = tbl.loc[[m for m in NICS if m in tbl.index]]
print("\n[NIC-only Spearman, 8 models]")
res_sp = {}
for col in ("L_tilde", "L_csf", "L_csf_img"):
    for target in ("psnr", "ms_ssim", "lpips"):
        r = spearmanr(nic_tbl[col], nic_tbl[target])[0]
        res_sp[f"{col}~{target}"] = round(float(r), 3)
print(json.dumps(res_sp, indent=1))
print("\n[coupling table]\n", tbl.round(4).to_string())

# ---------------- B) block-wise predictor ----------------
print("\n[block-wise leakage x content predictor]")
block_rows = []
for model in ALL:
    Lk = np.clip(leak_profile_512(model, "kodak"), 0, 1)
    n_bands = 16
    band_edges = np.linspace(0, 1, n_bands + 1)
    Lb = np.array([np.interp((band_edges[i] + band_edges[i + 1]) / 2,
                             np.linspace(0, 1, len(Lk)), Lk)
                   for i in range(n_bands)])
    preds, arts, base_hf, base_tot = [], [], [], []
    for ds in DATASETS:
        f = NC / f"blocks_{model}_{ds}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        n_img = len(z["images"])
        for i in range(n_img):
            E = z[f"b{i}_energy"]        # (nh, nw, 16) original spectral energy
            art = z[f"b{i}_mse"]         # measured block MSE
            P = (E * Lb[None, None, :]).sum(axis=2)
            preds.append(P.ravel()); arts.append(art.ravel())
            base_hf.append(E[:, :, int(0.66 * n_bands):].sum(axis=2).ravel())
            base_tot.append(E.sum(axis=2).ravel())
    P = np.concatenate(preds); A = np.concatenate(arts)
    BH = np.concatenate(base_hf); BT = np.concatenate(base_tot)
    r_leak = spearmanr(P, A)[0]
    r_hf = spearmanr(BH, A)[0]
    r_tot = spearmanr(BT, A)[0]
    block_rows.append({"model": model, "sp_leakXcontent": round(float(r_leak), 3),
                       "sp_hf_energy": round(float(r_hf), 3),
                       "sp_total_energy": round(float(r_tot), 3),
                       "n_blocks": len(P)})
    print(f"  {model}: leakXcontent={r_leak:.3f}  hf-only={r_hf:.3f} "
          f"total-only={r_tot:.3f}  (n={len(P)})")
pd.DataFrame(block_rows).to_csv(OUT / "block_predictor.csv", index=False)

# spatial maps figure
try:
    maps = np.load(NC / "basis_spatial_maps.npz")
    cvs = json.load(open(NC / "basis_cv.json"))
    show = ["jpegxl", "cheng2020-anchor", "tcm-p128", "ftic"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, m in zip(axes, show):
        im = ax.imshow(maps[m], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{m}\nCV={cvs[m]:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "fig_spatial_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("maps figure failed:", e)

# ---------------- C) single-freq post-processing ----------------
print("\n[full-basis vs single-freq consistency, matched ~1bpp]")
sfz = np.load(SF / "singlefreq_profiles.npz")
sfsum = pd.read_csv(SF / "singlefreq_summary.csv")
cons_rows = []
prof_sum = pd.read_csv(PROF / "profiles_summary.csv")
for model in ALL:
    key = "tcm-p128" if model == "tcm" else model
    qrow = sfsum[(sfsum["model"] == model) & (sfsum["stim"] == "x2")]
    if qrow.empty:
        continue
    q = int(qrow["q"].iloc[0])
    try:
        Lfull = np.load(PROF / f"{key}_q{q}_n256.npz")["leakage"]
    except FileNotFoundError:
        continue
    for stimname in ("x2", "x1"):
        e2 = sfz[f"{model}_{stimname}_frob2"]
        lam = 1.0 - sfz[f"{model}_{stimname}_ret"] ** 2 / (
            sfz[f"{model}_{stimname}_ret"] ** 2 + sfz[f"{model}_{stimname}_offe"] + 1e-15)
        rs_e = spearmanr(Lfull, e2)[0]
        rs_l = spearmanr(Lfull, lam)[0]
        cons_rows.append({"model": model, "stim": stimname, "q": q,
                          "sp_Lfull_vs_e2": round(float(rs_e), 3),
                          "sp_Lfull_vs_lambda": round(float(rs_l), 3),
                          "med_e2": float(np.median(e2))})
        print(f"  {model}/{stimname}: rs(L_full, e2)={rs_e:.3f} "
              f"rs(L_full, lambda)={rs_l:.3f}")
cons = pd.DataFrame(cons_rows)
cons.to_csv(OUT / "consistency.csv", index=False)

# anisotropy table
ani = sfsum.pivot_table(index="model", columns="stim", values="frob2_med")
ani["anisotropy_x2_over_x1"] = ani["x2"] / (ani["x1"] + 1e-12)
ani.to_csv(OUT / "anisotropy.csv")
print("\n[anisotropy x2/x1]\n", ani.round(4).to_string())

# sweep figure for the paper
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
show = ["cheng2020-anchor", "cheng2020-attn", "bmshj2018-factorized",
        "tcm", "jpeg", "jpegxl"]
for m in show:
    if f"{m}_x2_frob2" not in sfz:
        continue
    axes[0].semilogy(np.convolve(sfz[f"{m}_x2_frob2"], np.ones(5) / 5, "same")
                     + 1e-6, lw=1.1, label=m)
    axes[1].semilogy(np.convolve(sfz[f"{m}_x1_frob2"], np.ones(5) / 5, "same")
                     + 1e-6, lw=1.1, label=m)
axes[0].set_ylabel(r"$\|X_k^{(2)}-\hat X\|_F^2$")
axes[1].set_ylabel(r"$\|X_k^{(1)}-\hat X\|_F^2$")
axes[1].set_xlabel("frequency index $k$")
for ax in axes:
    ax.legend(fontsize=6, ncol=3); ax.grid(alpha=0.3)
fig.suptitle("Single-frequency sweeps at matched ~1.0 bpp (n=256, clamped)",
             fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_singlefreq_sweep.png", dpi=150)
plt.close(fig)

# instability figure
iz = np.load(SF / "anchor_instability.npz")
fig, ax = plt.subplots(figsize=(7.2, 3.2))
for key, c in zip(["q4_s1.0", "q5_s1.0", "q6_s1.0"], ["C0", "C1", "C3"]):
    ax.semilogy(iz[key] + 1e-6, lw=0.9, color=c, label=f"anchor {key}")
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel("frequency index $k$")
ax.set_ylabel(r"$\|X_k^{(2)}-\hat X\|_F^2$ (unclamped)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_anchor_instability.png", dpi=150)
plt.close(fig)

print("S7_DONE")
