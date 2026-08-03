#!/usr/bin/env python3
"""S18b: final X1 instability figures with visually legible frequency choices.

Main figure: adjacent pair k=61 (stable) vs k=62 (diverging), matched in
peak and RMS, with stripe period ~8.3 px (visible in print).
Gallery: k in {61, 62, 63, 70, 96, 133, 169, 253} — stable/diverging mix,
includes the near-Nyquist worst case 253.
Saves PDF + PNG (for visual inspection) for both figures.
"""
from __future__ import annotations

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
FIGDIR = ROOT / "results/analysis_s7"
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
N = 256
AMP = 0.45

D = build_dct_basis(N)
model = load_model("cheng2020-anchor", 6, DEV, base_dir=BASE)
model.eval()


def probe(k):
    X = np.outer(D[:, k], np.ones(N) / np.sqrt(N))
    s = AMP * N / np.sqrt(2.0)
    img = 0.5 + s * X
    img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
    x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        xh = model(x)["x_hat"]
    rec3 = xh.squeeze(0).permute(1, 2, 0).cpu().numpy()
    e2 = float(np.sum(((rec3.mean(axis=2) - 0.5) / s - X) ** 2))
    return img, np.clip(rec3, 0, 1), e2


cache = {k: probe(k) for k in (61, 62, 63, 70, 96, 133, 169, 253)}
for k, (_, _, e2) in cache.items():
    print(f"k={k:3d} e2={e2:.4g}", flush=True)


def fmt_e2(e2):
    if e2 >= 100:
        exp = int(np.floor(np.log10(e2)))
        return rf"{e2 / 10**exp:.1f}\times 10^{{{exp}}}"
    return f"{e2:.3f}"


# ---------- main figure: stimulus 62 | recon 62 | recon 61 ----------
img61, rec61, e61 = cache[61]
img62, rec62, e62 = cache[62]
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.05), constrained_layout=True)
panels = [
    (img62, "stimulus $\\mathbf{X}^{(1)}_{62}$\n"
            r"(same peak and RMS as $\mathbf{X}^{(1)}_{61}$)", "gray"),
    (rec62, "reconstruction, $k{=}62$"
            "\n" rf"$\|\mathbf{{X}}-\hat{{\mathbf{{X}}}}\|_F^2 = {fmt_e2(e62)}$"
            " (unclamped)", None),
    (rec61, "reconstruction, $k{=}61$"
            "\n" rf"$\|\mathbf{{X}}-\hat{{\mathbf{{X}}}}\|_F^2 = {fmt_e2(e61)}$",
     None),
]
for ax, (im, title, cmap) in zip(axes, panels):
    if cmap:
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(im)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_anchor_artifact_x1.{ext}", dpi=180)
plt.close(fig)

# ---------- gallery ----------
gal = [61, 62, 63, 70, 96, 133, 169, 253]
fig, axes = plt.subplots(2, len(gal), figsize=(1.72 * len(gal), 4.0),
                         constrained_layout=True)
for j, k in enumerate(gal):
    img, rec, e2 = cache[k]
    diverged = e2 > 10
    col = "firebrick" if diverged else "black"
    axes[0, j].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(rf"$k={k}$", fontsize=10, color=col)
    axes[1, j].imshow(rec)
    axes[1, j].set_title(rf"$e^2={fmt_e2(e2)}$", fontsize=8.5, color=col)
    for ax in (axes[0, j], axes[1, j]):
        ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_ylabel(r"stimulus $\mathbf{X}^{(1)}_k$", fontsize=10)
axes[1, 0].set_ylabel("reconstruction", fontsize=10)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_x1_gallery.{ext}", dpi=180)
plt.close(fig)

print("S18B_DONE", flush=True)
