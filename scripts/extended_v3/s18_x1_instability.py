#!/usr/bin/env python3
"""S18: X1 (striped) single-frequency instability sweep + redesigned figures.

(a) Unclamped sweep of X1 = n^{-1/2} d_k 1^T (horizontal gratings) for
    cheng2020-anchor q in {4,5,6}, n=256, centered embedding 0.5 + s*X with
    peak amplitude a in {0.225, 0.45} (s = a*n/sqrt(2)); amplitude-matched
    to the X2 instability sweep of run_singlefreq_full.py (s_rel 0.5/1.0).
    Records e2 = ||X - rec||_F^2 per k (||X||_F = 1), blow-up sets e2 > 10.
(b) Main figure fig_anchor_artifact_x1.pdf: stimulus/reconstruction pairs at
    the worst diverging k vs its nearest stable neighbor (q=6, a=0.45).
(c) Supplementary gallery fig_x1_gallery.pdf: stim/recon pairs at 6 values
    of k (3 diverging, 3 stable), same protocol.
(d) fig_x1_instability_curve.pdf: unclamped e2 vs k for q in {4,5,6}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model
from dct_nic.metrics import build_dct_basis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/dct_benchmark_nic")
OUT = ROOT / "results/singlefreq"
FIGDIR = ROOT / "results/analysis_s7"
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
N = 256

D = build_dct_basis(N)


def x1_stim(k, amp):
    X = np.outer(D[:, k], np.ones(N) / np.sqrt(N))   # ||X||_F = 1
    s = amp * N / np.sqrt(2.0)                       # peak |s*X| ~= amp
    img = 0.5 + s * X
    img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    return X, s, img, img3


def forward(model, img3, clamp):
    x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        xh = model(x)["x_hat"]
    if clamp:
        xh = xh.clamp(0, 1)
    return xh.squeeze(0).permute(1, 2, 0).cpu().numpy()


def sweep(model, amp):
    errs = np.zeros(N)
    for k in range(N):
        X, s, _, img3 = x1_stim(k, amp)
        rec = (forward(model, img3, clamp=False).mean(axis=2) - 0.5) / s
        errs[k] = float(np.sum((X - rec) ** 2))
    return errs


# ---------- (a) sweep ----------
prof = {}
rows = []
for q in (4, 5, 6):
    model = load_model("cheng2020-anchor", q, DEV, base_dir=BASE)
    model.eval()
    for amp in (0.225, 0.45):
        errs = sweep(model, amp)
        prof[f"q{q}_a{amp}"] = errs
        blow = np.where(errs > 10)[0]
        rows.append({"q": q, "amp": amp, "n_blowup": int(len(blow)),
                     "blow_ks": blow.tolist(),
                     "worst_k": int(np.argmax(errs)),
                     "worst_e2": float(errs.max()),
                     "e2_k70": float(errs[70])})
        print(f"[x1 q={q} amp={amp}] blowups(e2>10): {len(blow)} "
              f"worst k={int(np.argmax(errs))} e2={errs.max():.3g} "
              f"e2(k=70)={errs[70]:.3g}", flush=True)
    if q != 6:
        del model
        torch.cuda.empty_cache()

np.savez_compressed(OUT / "x1_instability.npz", **prof)
with open(OUT / "x1_instability.json", "w") as f:
    json.dump(rows, f, indent=2)

# ---------- figure helpers (q=6 model still loaded) ----------
AMP = 0.45
errs6 = prof[f"q6_a{AMP}"]
blow6 = np.where(errs6 > 10)[0]
stable6 = np.where(errs6 < 0.05)[0]
kbad = int(np.argmax(errs6))
kgood = int(stable6[np.argmin(np.abs(stable6 - kbad))]) if len(stable6) else 40
print(f"figure choice: kbad={kbad} (e2={errs6[kbad]:.3g}), "
      f"kgood={kgood} (e2={errs6[kgood]:.3g})", flush=True)


def render(k):
    _, s, img, img3 = x1_stim(k, AMP)
    rec = forward(model, img3, clamp=True)
    return img, rec


# ---------- (b) main figure ----------
img_b, rec_b = render(kbad)
img_g, rec_g = render(kgood)
fig, axes = plt.subplots(1, 4, figsize=(10.8, 2.95), constrained_layout=True)
panels = [
    (img_b, rf"stimulus $\mathbf{{X}}^{{(1)}}_{{{kbad}}}$", "gray"),
    (rec_b, rf"reconstruction, $k{{=}}{kbad}$"
            "\n" rf"$\|\mathbf{{X}}-\hat{{\mathbf{{X}}}}\|_F^2 = "
            rf"{errs6[kbad]:.3g}$ (unclamped)", None),
    (img_g, rf"stimulus $\mathbf{{X}}^{{(1)}}_{{{kgood}}}$", "gray"),
    (rec_g, rf"reconstruction, $k{{=}}{kgood}$"
            "\n" rf"$\|\mathbf{{X}}-\hat{{\mathbf{{X}}}}\|_F^2 = "
            rf"{errs6[kgood]:.3g}$", None),
]
for ax, (im, title, cmap) in zip(axes, panels):
    if cmap:
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(im, 0, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.savefig(FIGDIR / "fig_anchor_artifact_x1.pdf")
plt.close(fig)

# ---------- (c) supplementary gallery ----------
bad_ks = []
if len(blow6):
    for cand in (int(blow6[0]), int(blow6[len(blow6) // 2]), kbad):
        if cand not in bad_ks:
            bad_ks.append(cand)
    while len(bad_ks) < 3 and len(blow6) > len(bad_ks):
        for c in blow6:
            if int(c) not in bad_ks:
                bad_ks.append(int(c))
                break
good_ks = []
for b in bad_ks:
    cand = stable6[np.argsort(np.abs(stable6 - b))]
    for c in cand:
        if int(c) not in good_ks and int(c) not in bad_ks:
            good_ks.append(int(c))
            break
gal = sorted(set(bad_ks + good_ks))
print(f"gallery ks: {gal}", flush=True)

fig, axes = plt.subplots(2, len(gal), figsize=(1.95 * len(gal), 4.3),
                         constrained_layout=True)
for j, k in enumerate(gal):
    img, rec = render(k)
    diverged = errs6[k] > 10
    col = "firebrick" if diverged else "black"
    axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(rf"$k={k}$", fontsize=10, color=col)
    axes[1, j].imshow(np.clip(rec, 0, 1))
    axes[1, j].set_title(rf"$e^2={errs6[k]:.3g}$", fontsize=9, color=col)
    for ax in (axes[0, j], axes[1, j]):
        ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel(r"stimulus $\mathbf{X}^{(1)}_k$", fontsize=10)
axes[1, 0].set_ylabel("reconstruction", fontsize=10)
fig.savefig(FIGDIR / "fig_x1_gallery.pdf")
plt.close(fig)

# ---------- (d) e2 vs k curve ----------
fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
for q, c in ((4, "tab:blue"), (5, "tab:orange"), (6, "tab:red")):
    ax.semilogy(np.arange(N), np.maximum(prof[f"q{q}_a0.45"], 1e-4),
                lw=0.9, color=c, label=rf"$q={q}$")
ax.axhline(1.0, color="k", ls=":", lw=0.8)
ax.set_xlabel(r"frequency index $k$")
ax.set_ylabel(r"unclamped $\|\mathbf{X}-\hat{\mathbf{X}}\|_F^2$")
ax.legend(fontsize=8, ncol=3)
fig.savefig(FIGDIR / "fig_x1_instability_curve.pdf")
plt.close(fig)

print("S18_DONE", flush=True)
