#!/usr/bin/env python3
"""S23: complete the Cheng2020-Anchor instability sweep across ALL quality
levels q=1..6 (previously only 4,5,6 were run). Unclamped X2 single-frequency
stimuli, n=256, full probe contrast (s_rel=1.0) and half (0.5). Records the
per-k squared error curve, blow-up count (e2>10), and worst case per q.

Outputs -> results/singlefreq/anchor_instability_allq.{npz,json}
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

ROOT = Path("/root/dct_benchmark_nic")
OUT = ROOT / "results/singlefreq"
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
N = 256
D = build_dct_basis(N)


def sweep(model, s_rel):
    errs = np.zeros(N)
    for k in range(N):
        X = np.outer(D[:, k], D[:, k])
        s = 0.225 * N * s_rel
        img = 0.5 + s * X
        img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
        x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).to(DEV)
        with torch.no_grad():
            rec = model(x)["x_hat"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        rec = (rec.mean(axis=2) - 0.5) / s
        errs[k] = float(np.sum((X - rec) ** 2))
    return errs


prof, rows = {}, []
for q in (1, 2, 3, 4, 5, 6):
    model = load_model("cheng2020-anchor", q, DEV, base_dir=BASE)
    model.eval()
    for s_rel in (1.0, 0.5):
        errs = sweep(model, s_rel)
        prof[f"q{q}_s{s_rel}"] = errs
        blow = np.where(errs > 10)[0]
        rows.append({"q": q, "s_rel": s_rel, "n_blowup": int(len(blow)),
                     "worst_k": int(np.argmax(errs)),
                     "worst_e2": float(errs.max()),
                     "median_e2": float(np.median(errs)),
                     "blow_ks": blow.tolist()[:60]})
        print(f"[q={q} s={s_rel}] blowups(e2>10)={len(blow):3d} "
              f"worst_k={int(np.argmax(errs))} worst_e2={errs.max():.2e} "
              f"median={np.median(errs):.3f}", flush=True)
    del model
    torch.cuda.empty_cache()

np.savez_compressed(OUT / "anchor_instability_allq.npz", **prof)
with open(OUT / "anchor_instability_allq.json", "w") as f:
    json.dump(rows, f, indent=2)
print("S23_DONE", flush=True)
