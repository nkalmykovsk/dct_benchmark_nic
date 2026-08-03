#!/usr/bin/env python3
"""S2: full per-frequency profile sweep for all codecs.

For every codec/quality:
  n=256: leakage/odr/centroid/spread/entropy per-k + R matrix + amplitude diag
  n=512: same (no R) for q=6 NICs and all classical qualities (CSF task input)

Saves results/profiles/<model>_q<q>_n<size>.npz + profiles_summary.csv.
Ends with a verification block against results/all_metrics_summary.csv.
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
from scipy.fft import dct

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model, evaluate_codec

OUT = Path("/root/dct_benchmark_nic/results/profiles")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")

COMPRESSAI = ["bmshj2018-factorized", "bmshj2018-hyperprior",
              "mbt2018-mean", "mbt2018",
              "cheng2020-anchor", "cheng2020-attn"]
CLASSICAL = ["jpeg", "webp", "jpegxl"]
BASE = "/root/dct_benchmark_nic/third_party"


def eval_and_save(model, name, q, size, save_R=False):
    t0 = time.time()
    res = evaluate_codec(model, size=size, device=DEV, model_name=name)
    recon_gray = res["recon"].mean(axis=2)
    g = np.diag(dct(recon_gray.astype(np.float64), axis=0, norm="ortho"))
    payload = {
        "leakage": res["leakage"], "odr": res["odr"],
        "centroid_shift": res["centroid_shift"], "spread": res["spread"],
        "entropy": res["entropy"], "amp_gain": g,
        "bpp": res["bpp"] if res["bpp"] is not None else np.nan,
    }
    if save_R:
        payload["R"] = res["R"]
    np.savez_compressed(OUT / f"{name}_q{q}_n{size}.npz", **payload)
    row = {"model": name, "q": q, "size": size,
           "bpp": payload["bpp"],
           "L_k": res["L_k"], "L_low": res["L_low"], "L_high": res["L_high"],
           "ODR_k": res["ODR_k"], "|Dc_k|": res["|Δc_k|"],
           "s_k": res["s_k"], "H_k": res["H_k"],
           "sec": round(time.time() - t0, 1)}
    print(f"  {name:22s} q={q} n={size}: L_k={res['L_k']:.4f} "
          f"bpp={payload['bpp']:.3f} ({row['sec']}s)", flush=True)
    return row


def main():
    rows = []

    # --- CompressAI: q=1..6, n=256 (with R) and n=512 ---
    for name in COMPRESSAI:
        for q in range(1, 7):
            try:
                m = load_model(name, q, DEV, base_dir=BASE)
                m.eval()
                rows.append(eval_and_save(m, name, q, 256, save_R=True))
                rows.append(eval_and_save(m, name, q, 512))
                del m
                torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()

    # --- FTIC: q=1..6, n>=256 ---
    for q in range(1, 7):
        try:
            m = load_model("ftic", q, DEV, base_dir=BASE)
            m.eval()
            rows.append(eval_and_save(m, "ftic", q, 256, save_R=True))
            rows.append(eval_and_save(m, "ftic", q, 512))
            del m
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()

    # --- TCM: p=64 (lambda 0.0025), p=128 (lambda 0.05) ---
    for p, tag in ((64, "tcm-p64"), (128, "tcm-p128")):
        try:
            m = load_model("tcm", 1, DEV, p=p, base_dir=BASE)
            m.eval()
            rows.append(eval_and_save(m, tag, p, 256, save_R=True))
            rows.append(eval_and_save(m, tag, p, 512))
            del m
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()

    # --- Classical: q levels 1..6 at both sizes ---
    for name in CLASSICAL:
        for q in range(1, 7):
            try:
                m = load_model(name, q, DEV)
                rows.append(eval_and_save(m, name, q, 256, save_R=True))
                rows.append(eval_and_save(m, name, q, 512))
            except Exception:
                traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "profiles_summary.csv", index=False)
    print(f"\nSaved {len(df)} configs -> profiles_summary.csv", flush=True)

    # ---------------- verification vs published repo results ----------------
    ref = pd.read_csv("/root/dct_benchmark_nic/results/all_metrics_summary.csv")
    ref["size_n"] = ref["Size"].str.split("x").str[0].astype(int)
    checks, fails = 0, []
    for _, r in df.iterrows():
        if r["model"].startswith("tcm"):
            sub = ref[(ref["Model"].str.startswith("tcm")) &
                      (ref["size_n"] == r["size"]) & (ref["p"] == r["q"])]
        else:
            sub = ref[(ref["Model"] == r["model"]) &
                      (ref["size_n"] == r["size"]) & (ref["q"] == r["q"])]
        if len(sub) != 1:
            continue
        checks += 1
        dL = abs(float(sub["L_k"].iloc[0]) - r["L_k"])
        if dL > 0.02:
            fails.append((r["model"], r["q"], r["size"],
                          float(sub["L_k"].iloc[0]), r["L_k"]))
    print(f"[VERIFY] compared {checks} configs with published CSV; "
          f"{len(fails)} mismatches (|dL_k|>0.02)", flush=True)
    for f in fails:
        print("   MISMATCH", f, flush=True)
    with open(OUT / "verify_report.json", "w") as fp:
        json.dump({"checks": checks, "fails": fails}, fp, indent=2, default=str)
    print("S2_DONE", flush=True)


if __name__ == "__main__":
    main()
