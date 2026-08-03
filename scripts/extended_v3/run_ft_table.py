#!/usr/bin/env python3
"""S1: systematic leakage-as-loss fine-tuning across codecs and module configs.

Replaces the fabricated Tables new:tab:finetune / new:tab:finetune_generalise
with measured numbers.

Protocol (faithful to paper Sec. IV-B / run_finetune.py):
  input      : single DCT basis image (128x128 for CompressAI, 256 for TCM/FTIC)
  loss       : band-weighted leakage + low-band guards (repo recipe)
               + lam_rate * bpp when the encoder side is trainable
  optimizer  : Adam lr=1e-4, 1000 steps, grad-clip 1.0
  configs    : dec (g_s), enc (g_a), dec_enc, full (all params + aux loss)
  seeds      : dec -> 3 seeds, others -> 1
Evaluation per run:
  leakage on the training-size basis and on a fresh 512 basis (generalisation)
  Kodak-24 natural images: PSNR / MS-SSIM / LPIPS / bpp before vs after
Sanity checks:
  dec config must leave basis bpp and Kodak bpp unchanged (|d| < 1e-4 bpp).

Usage: python3 run_ft_table.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model, evaluate_codec
from dct_nic.metrics import build_dct_basis_rgb

OUT = Path("/root/dct_benchmark_nic/results/ft_table")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")
BASE = "/root/dct_benchmark_nic/third_party"
KODAK = sorted(Path("/root/dct_benchmark_nic/data/kodak").glob("*.png"))

CONFIGS = {
    "dec":     {"prefixes": ("g_s.",), "lam_rate": 0.0},
    "enc":     {"prefixes": ("g_a.",), "lam_rate": 0.01},
    "dec_enc": {"prefixes": ("g_s.", "g_a."), "lam_rate": 0.0},
    "full":    {"prefixes": None, "lam_rate": 0.01},
}


# ----------------------------- helpers ------------------------------------

def build_dct_matrix(n: int, device) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    m = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    C = torch.cos(math.pi * k * (m + 0.5) / n)
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def bpp_from_likelihoods(out, num_pixels: int):
    return sum((-torch.log2(lk.clamp(min=1e-9))).sum()
               for lk in out["likelihoods"].values()) / num_pixels


def load_kodak_tensors(n_images=None):
    paths = KODAK[:n_images] if n_images else KODAK
    tensors = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        ch, cw = (h // 64) * 64, (w // 64) * 64
        t, l = (h - ch) // 2, (w - cw) // 2
        img = img[t:t + ch, l:l + cw]
        tensors.append(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0))
    return paths, tensors


_LPIPS = None
def lpips_fn():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    return _LPIPS


@torch.no_grad()
def eval_kodak(model, tensors) -> dict:
    from pytorch_msssim import ms_ssim
    model.eval()
    psnrs, ssims, lps, bpps = [], [], [], []
    for x in tensors:
        x = x.to(DEV)
        out = model(x)
        xh = out["x_hat"].clamp(0, 1)
        mse = F.mse_loss(xh, x).item()
        psnrs.append(-10.0 * math.log10(max(mse, 1e-12)))
        ssims.append(float(ms_ssim(xh, x, data_range=1.0)))
        lps.append(float(lpips_fn()(xh, x, normalize=True)))
        bpps.append(float(bpp_from_likelihoods(out, x.shape[2] * x.shape[3])))
    return {"psnr": float(np.mean(psnrs)), "ms_ssim": float(np.mean(ssims)),
            "lpips": float(np.mean(lps)), "bpp": float(np.mean(bpps))}


def eval_leakage(model, name, size) -> dict:
    model.eval()
    res = evaluate_codec(model, size=size, device=DEV, model_name=name)
    return {"L_med": res["L_k"], "L_low": res["L_low"], "L_high": res["L_high"],
            "bpp": res["bpp"], "leak_profile": res["leakage"]}


# ----------------------------- fine-tuning --------------------------------

def finetune_one(name, quality, config, seed, steps, n_train, tcm_p=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = CONFIGS[config]

    model = load_model(name, quality, DEV, p=tcm_p, base_dir=BASE)

    # trainable module selection
    if cfg["prefixes"] is None:
        for p_ in model.parameters():
            p_.requires_grad = True
    else:
        for pname, p_ in model.named_parameters():
            p_.requires_grad = pname.startswith(cfg["prefixes"])
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    n_trainable = sum(p_.numel() for p_ in trainable)
    if not trainable:
        raise RuntimeError(f"{name}/{config}: no trainable params")

    basis_rgb, vmin, vmax = build_dct_basis_rgb(n_train)
    x_in = torch.from_numpy(basis_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEV)
    C = build_dct_matrix(n_train, DEV)
    D_target = C @ torch.eye(n_train, device=DEV)

    # baseline diag for low-band guard
    model.eval()
    with torch.no_grad():
        res0 = evaluate_codec(model, size=n_train, device=DEV, model_name=name)
    low_n = max(1, int(0.33 * n_train))
    idx_low = torch.arange(low_n, device=DEV)
    base_diag_low = torch.tensor(np.diag(res0["R"])[:low_n], device=DEV,
                                 dtype=torch.float32)

    # frequency weights (repo recipe)
    freq = torch.arange(n_train, device=DEV, dtype=torch.float32) / max(1.0, n_train - 1)
    w = freq.pow(2.0) + 0.2
    w[:low_n] = 0.0
    w = w / (w.sum() + 1e-8)

    opt = torch.optim.Adam(trainable, lr=1e-4)
    aux_opt = None
    if config == "full" and hasattr(model, "aux_loss"):
        aux_params = [p_ for pname, p_ in model.named_parameters()
                      if pname.endswith("quantiles")]
        if aux_params:
            aux_opt = torch.optim.Adam(aux_params, lr=1e-3)

    model.train()
    hist = []
    num_px = n_train * n_train
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(x_in)
        x_hat = out["x_hat"]
        x_hat_denorm = (vmin + (vmax - vmin) * x_hat).mean(dim=1).squeeze(0)

        power = (C @ x_hat_denorm.float()).pow(2)
        R = power / (power.sum(dim=0, keepdim=True) + 1e-12)
        idx = torch.arange(n_train, device=DEV)
        diag_R = R[idx, idx]
        leak_vec = 1.0 - diag_R

        loss = (w * leak_vec).sum()
        loss = loss + 0.12 * torch.relu(
            base_diag_low - diag_R.index_select(0, idx_low) - 0.02).mean()
        loss = loss + 2e-2 * F.mse_loss(
            x_hat_denorm.to(D_target.dtype).index_select(1, idx_low),
            D_target.index_select(1, idx_low))
        if cfg["lam_rate"] > 0 and "likelihoods" in out:
            loss = loss + cfg["lam_rate"] * bpp_from_likelihoods(out, num_px)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if aux_opt is not None:
            aux_opt.zero_grad()
            aux = model.aux_loss()
            aux.backward()
            aux_opt.step()

        if step % 10 == 0 or step == 1:
            hist.append((step, float(loss.item()),
                         float(leak_vec.detach().median().item())))
    return model, res0, hist, n_trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    if args.smoke:
        jobs = [("bmshj2018-factorized", 6, 128)]
        configs = ["dec", "enc"]
        seeds = {"dec": [0], "enc": [0]}
        steps = 60
        kodak_n = 3
    else:
        jobs = [(m, 6, 128) for m in
                ["bmshj2018-factorized", "bmshj2018-hyperprior",
                 "mbt2018-mean", "mbt2018",
                 "cheng2020-anchor", "cheng2020-attn"]]
        jobs += [("tcm", 1, 256), ("ftic", 6, 256)]
        configs = list(CONFIGS)
        seeds = {"dec": [0, 1, 2], "enc": [0], "dec_enc": [0], "full": [0]}
        steps = args.steps
        kodak_n = None

    paths, kodak = load_kodak_tensors(kodak_n)
    print(f"kodak images: {len(kodak)}", flush=True)

    rows = []
    histories = {}
    for name, quality, n_train in jobs:
        tag0 = f"{name}-q{quality}"
        # ---- baseline (once per codec) ----
        m0 = load_model(name, quality, DEV, base_dir=BASE)
        m0.eval()
        base_train = eval_leakage(m0, name, n_train)
        base_gen = eval_leakage(m0, name, 512)
        base_kodak = eval_kodak(m0, kodak)
        del m0
        torch.cuda.empty_cache()
        print(f"[{tag0}] baseline: L{n_train}={base_train['L_med']:.4f} "
              f"L512={base_gen['L_med']:.4f} "
              f"kodak psnr={base_kodak['psnr']:.2f} lpips={base_kodak['lpips']:.4f} "
              f"bpp={base_kodak['bpp']:.4f}", flush=True)

        run_configs = configs
        if name in ("tcm", "ftic") and not args.smoke:
            run_configs = ["dec", "full"]      # saturation check only

        for config in run_configs:
            for seed in seeds[config]:
                t0 = time.time()
                try:
                    model, res0, hist, n_tr = finetune_one(
                        name, quality, config, seed, steps, n_train)
                    after_train = eval_leakage(model, name, n_train)
                    after_gen = eval_leakage(model, name, 512)
                    after_kodak = eval_kodak(model, kodak)
                    row = {
                        "model": name, "config": config, "seed": seed,
                        "n_train": n_train, "steps": steps,
                        "n_trainable": n_tr,
                        "L_before": base_train["L_med"],
                        "L_after": after_train["L_med"],
                        "L_high_before": base_train["L_high"],
                        "L_high_after": after_train["L_high"],
                        "L512_before": base_gen["L_med"],
                        "L512_after": after_gen["L_med"],
                        "bpp_basis_before": base_train["bpp"],
                        "bpp_basis_after": after_train["bpp"],
                        "psnr_before": base_kodak["psnr"],
                        "psnr_after": after_kodak["psnr"],
                        "msssim_before": base_kodak["ms_ssim"],
                        "msssim_after": after_kodak["ms_ssim"],
                        "lpips_before": base_kodak["lpips"],
                        "lpips_after": after_kodak["lpips"],
                        "bpp_kodak_before": base_kodak["bpp"],
                        "bpp_kodak_after": after_kodak["bpp"],
                        "sec": round(time.time() - t0, 1),
                    }
                    # sanity: dec must not change rate
                    if config == "dec":
                        db = abs(row["bpp_kodak_after"] - row["bpp_kodak_before"])
                        row["rate_frozen_ok"] = bool(db < 1e-4)
                        if db >= 1e-4:
                            print(f"  !! RATE DRIFT in dec config: d_bpp={db:.6f}",
                                  flush=True)
                    rows.append(row)
                    histories[f"{tag0}_{config}_s{seed}"] = np.array(hist)
                    print(f"[{tag0}/{config}/s{seed}] "
                          f"L:{row['L_before']:.4f}->{row['L_after']:.4f} "
                          f"L512:{row['L512_before']:.4f}->{row['L512_after']:.4f} "
                          f"psnr:{row['psnr_before']:.2f}->{row['psnr_after']:.2f} "
                          f"lpips:{row['lpips_before']:.4f}->{row['lpips_after']:.4f} "
                          f"bpp:{row['bpp_kodak_before']:.4f}->{row['bpp_kodak_after']:.4f} "
                          f"({row['sec']}s)", flush=True)
                    del model
                    torch.cuda.empty_cache()
                except Exception:
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                # checkpoint results incrementally
                pd.DataFrame(rows).to_csv(OUT / "ft_table.csv", index=False)
                np.savez_compressed(OUT / "ft_histories.npz", **histories)

    pd.DataFrame(rows).to_csv(OUT / "ft_table.csv", index=False)
    np.savez_compressed(OUT / "ft_histories.npz", **histories)
    print("S1_DONE", flush=True)


if __name__ == "__main__":
    main()
