#!/usr/bin/env python3
"""S5: single-frequency benchmark, full version for the paper.

(a) 11 codecs at matched ~1.0 bpp, n=256: sweep k = 0..255 for both stimuli
    X1 = d_k 1^T (scaled) and X2 = d_k d_k^T (scaled), centered normalization.
    Metrics per k (recon clamped to [0,1] before de-normalization):
      frob2   : ||X - X_hat||_F^2 / ||X||_F^2
      ret2d   : 2-D DCT amplitude retention at the correct bin
                (X1 -> (k,0), X2 -> (k,k)), target coeff normalized to 1
      leak2d  : 1 - retained-bin energy fraction of total error-free... see code
    Parseval identity check: frob2 == (1-ret)^2 + off-bin energy (exact).
(b) cheng2020-anchor instability: unclamped sweeps for q in {4,5,6}, n=256,
    contrast scales s_rel in {0.5, 1.0}; record blow-up set {k: err > 10};
    save worst-k reconstruction images for a figure.

Outputs -> results/singlefreq/
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model
from dct_nic.metrics import build_dct_basis

ROOT = Path("/root/dct_benchmark_nic")
OUT = ROOT / "results/singlefreq"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
N = 256

MODELS = [("bmshj2018-factorized", "cai", 6), ("bmshj2018-hyperprior", "cai", 6),
          ("mbt2018-mean", "cai", 6), ("mbt2018", "cai", 6),
          ("cheng2020-anchor", "cai", 6), ("cheng2020-attn", "cai", 6),
          ("tcm", "tcm", 128), ("ftic", "ftic", 6),
          ("jpeg", "cls", None), ("webp", "cls", None), ("jpegxl", "cls", None)]


def get_model(name, kind, q):
    if kind == "cai":
        m = load_model(name, q, DEV, base_dir=BASE)
        m.eval()
        return m
    if kind == "tcm":
        m = load_model("tcm", 1, DEV, p=q, base_dir=BASE)
        m.eval()
        return m
    if kind == "ftic":
        m = load_model("ftic", q, DEV, base_dir=BASE)
        m.eval()
        return m
    return load_model(name, q, DEV)


def matched_quality(name, size=256, target=1.0):
    df = pd.read_csv(ROOT / "results/profiles/profiles_summary.csv")
    key = "tcm-p128" if name == "tcm" else name
    sub = df[(df["model"] == key) & (df["size"] == size)].dropna(subset=["bpp"])
    if sub.empty:
        return None
    return int(sub.iloc[(sub["bpp"] - target).abs().argsort().iloc[0]]["q"])


def forward(model, img3, clamp=True):
    x = torch.from_numpy(img3).float().permute(2, 0, 1).unsqueeze(0)
    if hasattr(model, "parameters"):
        x = x.to(DEV)
        with torch.no_grad():
            out = model(x)
    else:
        out = model(x)
    xh = out["x_hat"]
    if clamp:
        xh = xh.clamp(0, 1)
    return xh.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def stim(D, k, which, n=N):
    if which == "x2":
        X = np.outer(D[:, k], D[:, k])
        s = 0.225 * n
    else:
        X = np.outer(D[:, k], np.ones(n) / np.sqrt(n))  # unit Frobenius norm
        s = 0.45 * np.sqrt(n)
    return X, s


def sweep(model, D, which, clamp=True):
    frob2, ret, offe = [], [], []
    for k in range(N):
        X, s = stim(D, k, which)
        img = 0.5 + s * X
        img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
        rec = (forward(model, img3, clamp=clamp).mean(axis=2) - 0.5) / s
        e2 = float(np.sum((X - rec) ** 2))          # ||X||_F = 1 both stimuli
        G = dctn(rec, norm="ortho")
        if which == "x2":
            tgt = G[k, k]
            off = float(np.sum(G ** 2) - G[k, k] ** 2)
        else:
            tgt = G[k, 0]
            off = float(np.sum(G ** 2) - G[k, 0] ** 2)
        frob2.append(e2)
        ret.append(float(tgt))
        offe.append(off)
    return np.array(frob2), np.array(ret), np.array(offe)


def main():
    D = build_dct_basis(N)
    all_prof = {}
    rows = []
    for name, kind, qdef in MODELS:
        q = matched_quality(name) if kind in ("cai", "cls") else qdef
        if q is None:
            q = qdef or 4
        try:
            model = get_model(name, kind, q)
        except Exception:
            traceback.print_exc()
            continue
        t0 = time.time()
        for which in ("x2", "x1"):
            f2, ret, offe = sweep(model, D, which, clamp=True)
            all_prof[f"{name}_{which}_frob2"] = f2
            all_prof[f"{name}_{which}_ret"] = ret
            all_prof[f"{name}_{which}_offe"] = offe
            # Parseval check: e2 = (1-ret)^2 + off-bin error energy.
            # off-bin error energy = total error energy - (1-ret)^2 at bin:
            # here verify e2 - [(1-ret)^2 + offe] ~ 0 (X has single unit bin)
            resid = np.max(np.abs(f2 - ((1 - ret) ** 2 + offe)))
            rows.append({"model": name, "stim": which, "q": q,
                         "frob2_med": float(np.median(f2)),
                         "frob2_max": float(np.max(f2)),
                         "ret_med": float(np.median(ret)),
                         "parseval_resid": float(resid)})
            print(f"[{name}/{which}] q={q} med_frob2={np.median(f2):.4f} "
                  f"max={np.max(f2):.3g} parseval_resid={resid:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if hasattr(model, "parameters"):
            del model
            torch.cuda.empty_cache()

    np.savez_compressed(OUT / "singlefreq_profiles.npz", **all_prof)
    pd.DataFrame(rows).to_csv(OUT / "singlefreq_summary.csv", index=False)

    # ---------- (b) anchor instability ----------
    inst_rows = []
    inst_prof = {}
    for q in (4, 5, 6):
        model = get_model("cheng2020-anchor", "cai", q)
        for s_rel in (0.5, 1.0):
            errs = []
            worst = (-1, -1.0)
            for k in range(N):
                X, s = stim(D, k, "x2")
                s = s * s_rel
                img = 0.5 + s * X
                img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
                rec_img = forward(model, img3, clamp=False)
                rec = (rec_img.mean(axis=2) - 0.5) / s
                e2 = float(np.sum((X - rec) ** 2))
                errs.append(e2)
                if e2 > worst[1]:
                    worst = (k, e2)
                    worst_img = rec_img
            errs = np.array(errs)
            blow = np.where(errs > 10)[0]
            inst_prof[f"q{q}_s{s_rel}"] = errs
            inst_rows.append({"q": q, "s_rel": s_rel,
                              "n_blowup": int(len(blow)),
                              "blow_ks": blow.tolist()[:40],
                              "worst_k": worst[0], "worst_e2": worst[1]})
            print(f"[anchor q={q} s={s_rel}] blowups(e2>10): {len(blow)} "
                  f"worst k={worst[0]} e2={worst[1]:.3g}", flush=True)
            if len(blow) and s_rel == 1.0:
                im = np.clip(worst_img, 0, 1)
                Image.fromarray((im * 255).astype(np.uint8)).save(
                    OUT / f"anchor_q{q}_worstk{worst[0]}.png")
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(OUT / "anchor_instability.npz", **inst_prof)
    with open(OUT / "anchor_instability.json", "w") as f:
        json.dump(inst_rows, f, indent=2)
    print("S5_DONE", flush=True)


if __name__ == "__main__":
    main()
