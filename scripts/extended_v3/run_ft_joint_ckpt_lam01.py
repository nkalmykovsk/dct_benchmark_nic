#!/usr/bin/env python3
"""S1c: joint fine-tuning with checkpoint trajectories (Pareto analysis).

Same protocol as S1b (decoder-only, DIV2K patches MSE + lambda*leakage on a
256 basis, rate frozen), but every 250 steps we evaluate:
  - median leakage on the 256 basis (train-domain progress)
  - PSNR/LPIPS on a DISJOINT DIV2K-val monitor set (8 images, excluded from
    both the training pool and the Table-I DIV2K-50 subset)
  - PSNR/MS-SSIM/LPIPS/HF-PSNR on Kodak-24 (reporting set)
Post-hoc, an operating point can be chosen on the monitor set only
(e.g. max leakage reduction s.t. monitor dPSNR >= -0.1 dB) and reported on
Kodak — no selection on the test set.

Outputs -> results/ft_joint_ckpt/trajectories_lam01.csv
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as F
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model, evaluate_codec
from dct_nic.metrics import build_dct_basis_rgb

OUT = Path("/root/dct_benchmark_nic/results/ft_joint_ckpt_lam01")
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda")
BASE = "/root/dct_benchmark_nic/third_party"
KODAK = sorted(Path("/root/dct_benchmark_nic/data/kodak").glob("*.png"))
DIV2K_ALL = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                   for p in Path("/root/dct_benchmark_nic/data/div2k").glob(ext))

# Table-I subset indices (linspace 50) — excluded from monitor candidates
_tab1_idx = set(np.linspace(0, len(DIV2K_ALL) - 1, 50).round().astype(int).tolist())
MONITOR_PATHS = [p for i, p in enumerate(DIV2K_ALL) if i not in _tab1_idx][3:100:12][:8]
TRAIN_PATHS = [p for p in DIV2K_ALL if p not in MONITOR_PATHS]


def build_dct_matrix(n, device):
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    m = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    C = torch.cos(math.pi * k * (m + 0.5) / n)
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def leakage_diag(x_hat_denorm, C):
    power = (C @ x_hat_denorm.float()).pow(2)
    R = power / (power.sum(dim=0, keepdim=True) + 1e-12)
    idx = torch.arange(x_hat_denorm.shape[0], device=x_hat_denorm.device)
    return R[idx, idx]


_LPIPS = None
def lpips_fn():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    return _LPIPS


def load_tensor(path, max_side=768, mult=64):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    ch = (min(h, max_side) // mult) * mult
    cw = (min(w, max_side) // mult) * mult
    t, l = (h - ch) // 2, (w - cw) // 2
    return torch.from_numpy(img[t:t + ch, l:l + cw]).permute(2, 0, 1).unsqueeze(0)


def hf_psnr_np(a, b, frac=1.0 / 3.0):
    A = dctn(a.mean(axis=2), norm="ortho"); B = dctn(b.mean(axis=2), norm="ortho")
    h, w = A.shape
    r = np.sqrt(np.arange(h)[:, None] ** 2 + np.arange(w)[None, :] ** 2)
    mask = r >= (1.0 - frac) * r.max()
    return 10.0 * np.log10(1.0 / (float(np.mean((A[mask] - B[mask]) ** 2)) + 1e-12))


@torch.no_grad()
def eval_set(model, tensors, with_extras=False):
    from pytorch_msssim import ms_ssim
    model.eval()
    ps, lp, ss, hf = [], [], [], []
    for x in tensors:
        x = x.to(DEV)
        xh = model(x)["x_hat"].clamp(0, 1)
        mse = F.mse_loss(xh, x).item()
        ps.append(-10.0 * math.log10(max(mse, 1e-12)))
        lp.append(float(lpips_fn()(xh, x, normalize=True)))
        if with_extras:
            ss.append(float(ms_ssim(xh, x, data_range=1.0)))
            a = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
            b = xh.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hf.append(hf_psnr_np(a, b))
    out = {"psnr": float(np.mean(ps)), "lpips": float(np.mean(lp))}
    if with_extras:
        out["ms_ssim"] = float(np.mean(ss))
        out["hf_psnr"] = float(np.mean(hf))
    return out


class PatchPool:
    def __init__(self, paths, patch, max_images, rng):
        self.patch, self.rng, self.images = patch, rng, []
        for p in paths[:max_images]:
            arr = np.asarray(Image.open(p).convert("RGB"))
            if min(arr.shape[:2]) >= patch:
                self.images.append(arr)

    def batch(self, bs):
        out = []
        for _ in range(bs):
            im = self.images[self.rng.integers(len(self.images))]
            h, w, _ = im.shape
            t = self.rng.integers(0, h - self.patch + 1)
            l = self.rng.integers(0, w - self.patch + 1)
            out.append(im[t:t + self.patch, l:l + self.patch])
        x = np.stack(out).astype(np.float32) / 255.0
        return torch.from_numpy(x).permute(0, 3, 1, 2).to(DEV)


def run_one(name, quality, lam, guards, steps, ckpt_every, pool,
            kodak, monitor, rows):
    torch.manual_seed(0); np.random.seed(0)
    model = load_model(name, quality, DEV, p=128, base_dir=BASE)
    for pname, p_ in model.named_parameters():
        p_.requires_grad = pname.startswith("g_s.")
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]

    n = 256
    basis_rgb, vmin, vmax = build_dct_basis_rgb(n)
    x_basis = torch.from_numpy(basis_rgb).permute(2, 0, 1).unsqueeze(0).to(DEV)
    C = build_dct_matrix(n, DEV)
    D_target = C @ torch.eye(n, device=DEV)
    model.eval()
    with torch.no_grad():
        xb = model(x_basis)["x_hat"]
    base_diag = leakage_diag((vmin + (vmax - vmin) * xb).mean(1).squeeze(0), C).detach()
    low_n = max(1, int(0.33 * n))
    idx_low = torch.arange(low_n, device=DEV)
    base_low = base_diag[:low_n]
    freq = torch.arange(n, device=DEV, dtype=torch.float32) / (n - 1.0)
    w = freq.pow(3.0) + 0.2
    w[:low_n] = 0.0
    w = w / (w.sum() + 1e-8)

    cfg = "joint" if lam > 0 else "mse_only"
    # step-0 record
    k0 = eval_set(model, kodak, with_extras=True)
    m0 = eval_set(model, monitor)
    L0 = float((1.0 - base_diag).median())
    rows.append({"model": name, "config": cfg, "step": 0, "L_basis": L0,
                 **{f"mon_{k}": v for k, v in m0.items()},
                 **{f"kod_{k}": v for k, v in k0.items()}})

    opt = torch.optim.Adam(trainable, lr=1e-4)
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        x_nat = pool.batch(4)
        loss = F.mse_loss(model(x_nat)["x_hat"], x_nat)
        if lam > 0:
            xd = (vmin + (vmax - vmin) * model(x_basis)["x_hat"]).mean(1).squeeze(0)
            diag_R = leakage_diag(xd, C)
            loss = loss + lam * (w * (1.0 - diag_R)).sum()
            if guards:
                loss = loss + 0.12 * torch.relu(
                    base_low - diag_R.index_select(0, idx_low) - 0.02).mean()
                loss = loss + 2e-2 * F.mse_loss(
                    xd.to(D_target.dtype).index_select(1, idx_low),
                    D_target.index_select(1, idx_low))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % ckpt_every == 0:
            model.eval()
            with torch.no_grad():
                xb = model(x_basis)["x_hat"]
            dg = leakage_diag((vmin + (vmax - vmin) * xb).mean(1).squeeze(0), C)
            kk = eval_set(model, kodak, with_extras=True)
            mm = eval_set(model, monitor)
            rows.append({"model": name, "config": cfg, "step": step,
                         "L_basis": float((1.0 - dg).median()),
                         **{f"mon_{k}": v for k, v in mm.items()},
                         **{f"kod_{k}": v for k, v in kk.items()}})
            print(f"  [{name}/{cfg}] step {step}: L={rows[-1]['L_basis']:.4f} "
                  f"monPSNR={mm['psnr']:.2f} kodPSNR={kk['psnr']:.2f} "
                  f"kodLPIPS={kk['lpips']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=250)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    pool = PatchPool(TRAIN_PATHS, 256, 400, rng)
    kodak = [load_tensor(p) for p in KODAK]
    monitor = [load_tensor(p) for p in MONITOR_PATHS]
    print(f"pool={len(pool.images)} kodak={len(kodak)} monitor={len(monitor)}",
          flush=True)
    print("monitor:", [p.stem for p in MONITOR_PATHS], flush=True)

    rows = []
    jobs = [("bmshj2018-factorized", 6), ("bmshj2018-hyperprior", 6),
            ("mbt2018-mean", 6), ("mbt2018", 6)]
    for name, q in jobs:
        for lam, guards in ((0.01, True), (0.003, True)):
            t0 = time.time()
            try:
                run_one(name, q, lam, guards, args.steps, args.ckpt_every,
                        pool, kodak, monitor, rows)
            except Exception:
                traceback.print_exc()
                torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(OUT / "trajectories_lam01.csv", index=False)
            print(f"[{name}/lam={lam}] {time.time()-t0:.0f}s", flush=True)
    print("S1C_DONE", flush=True)


if __name__ == "__main__":
    main()
