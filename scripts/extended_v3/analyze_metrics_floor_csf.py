#!/usr/bin/env python3
"""S4 (CPU): offline analyses from S2 profile sweep.

1. Metric inter-correlations at matched ~1.0 bpp on 256 basis (real Table
   new:tab:metric_corr) + numeric check of professor's ordering inequalities
   + exact identity check ODR_k = tanh(L_k / (2(1-L_k))).
2. Parallel-axis decomposition check: second moment about k = spread^2+shift^2.
3. Empirical excess-leakage figure at matched bpp (replaces water-filling
   corollary) + idealized DCT transform-coder floor via uniform quantization.
4. CSF-weighted mean leakage L_bar^CSF (Mannos-Sakrison CSF) for all codecs
   at q6/512 with robustness across pixels-per-degree in {16, 32, 64}.

Outputs -> results/analysis_s4/
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dctn
from scipy.stats import spearmanr

PROF = Path("/root/dct_benchmark_nic/results/profiles")
OUT = Path("/root/dct_benchmark_nic/results/analysis_s4")
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = pd.read_csv(PROF / "profiles_summary.csv")

NIC = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean",
       "mbt2018", "cheng2020-anchor", "cheng2020-attn"]
CLS = ["jpeg", "webp", "jpegxl"]
ALL11 = NIC + ["tcm-p128", "ftic"] + CLS


def nearest_bpp_config(model, size, target=1.0):
    sub = summary[(summary["model"] == model) & (summary["size"] == size)].copy()
    sub = sub.dropna(subset=["bpp"])
    if sub.empty:
        return None
    sub["d"] = (sub["bpp"] - target).abs()
    return sub.sort_values("d").iloc[0]


def load_prof(model, q, size):
    return np.load(PROF / f"{model}_q{q}_n{size}.npz")


# ---------------- 1) metric correlations at matched bpp ----------------
pool = {m: [] for m in ["L", "ODR", "absD", "S", "H"]}
per_codec = {}
matched = {}
for m in ALL11:
    row = nearest_bpp_config(m, 256, 1.0)
    if row is None:
        continue
    matched[m] = {"q": int(row["q"]), "bpp": float(row["bpp"])}
    z = load_prof(m, int(row["q"]), 256)
    L, O = z["leakage"], z["odr"]
    D, S, H = np.abs(z["centroid_shift"]), z["spread"], z["entropy"]
    per_codec[m] = dict(L=L, O=O, D=D, S=S, H=H, R=z["R"])
    pool["L"].append(L); pool["ODR"].append(O); pool["absD"].append(D)
    pool["S"].append(S); pool["H"].append(H)
pool = {k: np.concatenate(v) for k, v in pool.items()}
print("matched configs:", json.dumps(matched, indent=0), flush=True)

names = ["L", "ODR", "absD", "S", "H"]
corr = np.zeros((5, 5))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        corr[i, j] = spearmanr(pool[a], pool[b])[0]
corr_df = pd.DataFrame(corr, index=names, columns=names)
corr_df.to_csv(OUT / "metric_spearman.csv")
print("\nPooled Spearman (n=%d):\n" % len(pool["L"]), corr_df.round(3), flush=True)

# ---------------- identity + inequality checks ----------------
L, O = pool["L"], pool["ODR"]
odr_pred = np.tanh(np.clip(L, 0, 1 - 1e-12) / (2 * (1 - np.clip(L, 0, 1 - 1e-12))))
ident_err = np.nanmax(np.abs(O - odr_pred))
print(f"\n[identity] max |ODR - tanh(L/(2(1-L)))| = {ident_err:.2e}", flush=True)

viol = {}
n = 256
viol["H >= S/log n"] = float(np.mean(pool["H"] < pool["S"] / np.log(n)))
viol["absD <= S"] = float(np.mean(pool["absD"] > pool["S"]))
viol["ODR >= L/(1+L)"] = float(np.mean(pool["ODR"] < pool["L"] / (1 + pool["L"])))
print("[ordering violations, fraction of pooled k]:", viol, flush=True)

# parallel-axis identity on a real R (exact math check on one codec)
z = per_codec["cheng2020-anchor"]
R = z["R"]
idx = np.arange(R.shape[0], dtype=float)
mu = R.T @ idx
m2_about_k = np.array([np.sum((idx - k) ** 2 * R[:, k]) for k in range(R.shape[0])])
var = np.array([np.sum((idx - mu[k]) ** 2 * R[:, k]) for k in range(R.shape[0])])
pax_err = np.max(np.abs(m2_about_k - (var + (mu - idx) ** 2)))
print(f"[parallel-axis] max |M2_k - (var + shift^2)| = {pax_err:.2e}", flush=True)

# ---------------- 3) excess leakage + ideal transform-coder floor ----------
def ideal_dct_coder_profile(nn=256, target_bpp=1.0):
    """Uniform deadzone quantizer on the 2-D DCT of the normalized basis image;
    bpp from empirical entropy of quantized symbols; leakage via repo metric."""
    from scipy.fft import dct as dct1
    D = dct1(np.eye(nn), axis=1, norm="ortho")
    rgbmin, rgbmax = D.min(), D.max()
    img = (D - rgbmin) / (rgbmax - rgbmin + 1e-9)
    Y = dctn(img, norm="ortho")

    def roundtrip(step):
        q = np.round(Y / step)
        vals, counts = np.unique(q, return_counts=True)
        p = counts / counts.sum()
        bits = -(p * np.log2(p)).sum() * q.size
        rec = q * step
        from scipy.fft import idctn
        return idctn(rec, norm="ortho"), bits / (nn * nn)

    lo, hi = 1e-5, 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        _, b = roundtrip(mid)
        if b > target_bpp:
            lo = mid
        else:
            hi = mid
    rec, bpp = roundtrip(np.sqrt(lo * hi))
    rec_dn = rgbmin + (rgbmax - rgbmin) * rec
    from scipy.fft import dct as dct1b
    Cc = dct1b(rec_dn, axis=0, norm="ortho")
    Pw = Cc ** 2
    Rm = Pw / (Pw.sum(axis=0, keepdims=True) + 1e-12)
    return 1.0 - np.diag(Rm), bpp

ideal_L, ideal_bpp = ideal_dct_coder_profile(256, 1.0)
floor_emp = np.min(np.stack([per_codec[m]["L"] for m in per_codec]), axis=0)

fig, ax = plt.subplots(figsize=(7, 4))
order = [m for m in ALL11 if m in per_codec]
for m in order:
    ls = "--" if m in CLS else "-"
    lw = 1.6 if m in ("tcm-p128", "ftic") else 1.1
    ax.plot(np.convolve(per_codec[m]["L"] - floor_emp, np.ones(9) / 9, "same"),
            ls=ls, lw=lw, label=f"{m} ({matched[m]['bpp']:.2f}bpp)")
ax.set_xlabel("frequency index k"); ax.set_ylabel(r"excess leakage $\Delta L_k$")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
ax.set_title("Excess over empirical floor at matched ~1.0 bpp (256)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_excess_leakage_real.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for m in order:
    ls = "--" if m in CLS else "-"
    ax.semilogy(np.convolve(per_codec[m]["L"], np.ones(9) / 9, "same") + 1e-6,
                ls=ls, lw=1.1, label=m)
ax.semilogy(ideal_L + 1e-6, "k-", lw=2.2,
            label=f"ideal DCT coder ({ideal_bpp:.2f}bpp)")
ax.semilogy(floor_emp + 1e-6, "k:", lw=2.0, label="empirical min envelope")
ax.set_xlabel("frequency index k"); ax.set_ylabel(r"$L_k$")
ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_floor_comparison.png", dpi=150)
plt.close(fig)
print(f"[floor] ideal coder bpp={ideal_bpp:.3f}, median ideal L={np.median(ideal_L):.4f}",
      flush=True)

# ---------------- 4) CSF-weighted leakage (Mannos-Sakrison) ----------------
def csf_ms(f_cpd):
    a = 2.6 * (0.0192 + 0.114 * f_cpd) * np.exp(-(0.114 * f_cpd) ** 1.1)
    return np.maximum(a, 0)

rows_csf = []
for ppd in (16, 32, 64):
    for m in ALL11:
        row = nearest_bpp_config(m, 512, 0.9)
        if m in NIC or m == "ftic":
            qq = 6
        elif m.startswith("tcm"):
            qq = 128
        else:
            qq = int(row["q"]) if row is not None else 4
        try:
            z = load_prof(m, qq, 512)
        except FileNotFoundError:
            continue
        Lk = np.clip(z["leakage"], 0, 1)
        nn = len(Lk)
        f_cyc_px = np.arange(nn) / (2.0 * nn)
        w = csf_ms(f_cyc_px * ppd)
        w = w / w.sum() * nn
        rows_csf.append({"ppd": ppd, "model": m, "q": qq,
                         "bpp": float(z["bpp"]),
                         "L_mean": float(Lk.mean()),
                         "L_csf": float((w * Lk).mean())})
csf_df = pd.DataFrame(rows_csf)
csf_df.to_csv(OUT / "csf_leakage.csv", index=False)
for ppd in (16, 32, 64):
    sub = csf_df[csf_df["ppd"] == ppd].sort_values("L_csf")
    rank_plain = csf_df[csf_df["ppd"] == ppd].sort_values("L_mean")["model"].tolist()
    rank_csf = sub["model"].tolist()
    rho = spearmanr(
        [rank_plain.index(m) for m in ALL11 if m in rank_plain],
        [rank_csf.index(m) for m in ALL11 if m in rank_csf])[0]
    print(f"[CSF ppd={ppd}] ranking vs unweighted Spearman={rho:.3f}; "
          f"order: {rank_csf}", flush=True)

with open(OUT / "s4_summary.json", "w") as f:
    json.dump({"matched": matched, "identity_err": float(ident_err),
               "violations": viol, "parallel_axis_err": float(pax_err),
               "ideal_bpp": float(ideal_bpp)}, f, indent=2)
print("S4_DONE", flush=True)
