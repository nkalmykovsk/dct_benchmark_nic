#!/usr/bin/env python3
"""Pilot 2: (a) full linear transfer-operator predictor of natural-image MSE;
(b) clamped vs unclamped single-frequency sweep (anchor instability).

Predictor: measure T[i,k] = DCT(d_hat_k)_i on the full basis (signed, denorm),
model the codec as linear in the column-DCT domain: C_hat ~= T @ C.
Compare with diagonal-only predictors from pilot 1.

Outputs -> /root/dct_benchmark_nic/results/pilot2/
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.fft import dct, dctn
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model
from dct_nic.metrics import build_dct_basis, build_dct_basis_rgb

OUT = Path("/root/dct_benchmark_nic/results/pilot2")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class JPEGCodec:
    def __init__(self, quality=85):
        self.q = quality

    def __call__(self, x):
        img = (x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=self.q)
        buf.seek(0)
        rec = np.asarray(Image.open(buf).convert("RGB")).astype("float32") / 255.0
        return {"x_hat": torch.from_numpy(rec).permute(2, 0, 1).unsqueeze(0)}


def forward(model, img01, clamp=False):
    x = torch.from_numpy(img01).float().permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = model(x)
    xh = out["x_hat"]
    if clamp:
        xh = xh.clamp(0.0, 1.0)
    return xh.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def measure_T(model, n, clamp):
    """Transfer operator T from the full-basis roundtrip (column-DCT domain)."""
    basis_rgb, vmin, vmax = build_dct_basis_rgb(n)
    rec = forward(model, basis_rgb, clamp=clamp)
    rec_gray = vmin + (vmax - vmin) * rec.mean(axis=2)
    return dct(rec_gray.astype(np.float64), axis=0, norm="ortho")  # (n, n) = C_hat, input C = I


def main():
    n = 512
    models = {}
    for name in ["cheng2020-anchor", "cheng2020-attn", "bmshj2018-factorized"]:
        models[name] = load_model(name, 6, DEV,
                                  base_dir="/root/dct_benchmark_nic/third_party")
        models[name].eval()
    models["jpeg-q85"] = JPEGCodec(85)

    kodak = sorted(Path("/root/dct_benchmark_nic/data/kodak").glob("*.png"))[:6]
    rows = []
    for name, m in models.items():
        for clamp in (False, True):
            T = measure_T(m, n, clamp)
            diagT = np.diag(T).copy()
            for p in kodak:
                im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
                h, w = im.shape[:2]
                im = im[(h - n) // 2:(h + n) // 2, (w - n) // 2:(w + n) // 2]
                rec = forward(m, im, clamp=clamp)
                X = im.mean(axis=2).astype(np.float64)
                Xh = rec.mean(axis=2).astype(np.float64)
                C = dct(X, axis=0, norm="ortho")
                Ch = dct(Xh, axis=0, norm="ortho")
                delta = np.sum((C - Ch) ** 2, axis=1)
                mse = float(delta.sum())
                # full-operator prediction
                Cp_full = T @ C
                d_full = np.sum((C - Cp_full) ** 2, axis=1)
                # diagonal(amplitude)-only prediction
                Cp_diag = diagT[:, None] * C
                d_diag = np.sum((C - Cp_diag) ** 2, axis=1)
                rows.append({
                    "model": name, "clamp": clamp, "img": p.name, "mse": mse,
                    "pred_full": float(d_full.sum()),
                    "pred_diag": float(d_diag.sum()),
                    "pearson_full": float(pearsonr(delta, d_full)[0]),
                    "pearson_diag": float(pearsonr(delta, d_diag)[0]),
                    "spearman_full": float(spearmanr(delta, d_full)[0]),
                })
            sub = [r for r in rows if r["model"] == name and r["clamp"] == clamp]
            print(f"[T] {name} clamp={clamp}: "
                  f"ratio_full={np.mean([r['pred_full']/r['mse'] for r in sub]):.2f} "
                  f"ratio_diag={np.mean([r['pred_diag']/r['mse'] for r in sub]):.2f} "
                  f"pearson_full={np.mean([r['pearson_full'] for r in sub]):.3f} "
                  f"pearson_diag={np.mean([r['pearson_diag'] for r in sub]):.3f}",
                  flush=True)

    with open(OUT / "transfer_predictor.json", "w") as f:
        json.dump(rows, f, indent=2)

    # scatter figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    colors = dict(zip(models, plt.cm.tab10.colors))
    for ax, tag in zip(axes, ("pred_full", "pred_diag")):
        for name in models:
            xs = [r["mse"] for r in rows if r["model"] == name and not r["clamp"]]
            ys = [r[tag] for r in rows if r["model"] == name and not r["clamp"]]
            ax.loglog(xs, ys, "o", ms=4, color=colors[name], label=name)
        lo = min(min(r["mse"] for r in rows), 1e-1)
        hi = max(r["mse"] for r in rows) * 3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("actual MSE")
        ax.set_ylabel("predicted MSE")
        ax.set_title("full operator T" if tag == "pred_full" else "diag(T) only",
                     fontsize=9)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "pilot2_transfer.png", dpi=150)
    plt.close(fig)

    # (b) clamped vs unclamped single-freq sweep for anchor + attn, n=256
    n_c = 256
    D = build_dct_basis(n_c)
    s = 0.225 * n_c
    prof = {}
    for name in ["cheng2020-anchor", "cheng2020-attn"]:
        m = models[name]
        for clamp in (False, True):
            errs = []
            for k in range(n_c):
                X = np.outer(D[:, k], D[:, k])
                img = 0.5 + s * X
                img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
                rec = (forward(m, img3, clamp=clamp).mean(axis=2) - 0.5) / s
                errs.append(float(np.sum((X - rec) ** 2)))
            prof[f"{name}_clamp{int(clamp)}"] = errs
            print(f"[sweep] {name} clamp={clamp}: median={np.median(errs):.4f} "
                  f"max={np.max(errs):.3g}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    styles = {"cheng2020-anchor_clamp0": ("C0", ":"), "cheng2020-anchor_clamp1": ("C0", "-"),
              "cheng2020-attn_clamp0": ("C1", ":"), "cheng2020-attn_clamp1": ("C1", "-")}
    for key, errs in prof.items():
        c, ls = styles[key]
        ax.semilogy(errs, color=c, ls=ls, lw=1, label=key)
    ax.set_xlabel("frequency index k")
    ax.set_ylabel(r"$\|X_k^{(2)}-\hat X\|_F^2$")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "pilot2_sweep_clamp.png", dpi=150)
    plt.close(fig)

    np.savez(OUT / "sweep_clamp_profiles.npz", **{k: np.array(v) for k, v in prof.items()})
    print("PILOT2_DONE", flush=True)


if __name__ == "__main__":
    main()
