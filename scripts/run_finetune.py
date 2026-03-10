#!/usr/bin/env python3
"""Leakage-as-loss single-image fine-tuning demo (Section III-B / Fig. 2c / Fig. S3).

Fine-tunes the decoder of a NIC model on the DCT basis input using leakage L_k
as the training objective, without penalizing bitrate.

Paper settings:
    Fig. 2c  (main text) : model=cheng2020-anchor, size=128,  quality=6, steps=900
    Fig. S3  (supp.)     : model=cheng2020-anchor, size=1024, quality=6, steps=900
    Optimizer: Adam, lr=1e-4, decoder-only (g_s)
    Result: median L_k: 0.291 → 0.090 (-69%) at 128×128

Usage:
    python scripts/run_finetune.py                            # defaults
    python scripts/run_finetune.py --size 1024 --steps 900   # large-scale
    python scripts/run_finetune.py --model mbt2018 --size 256
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dct_nic import load_model, evaluate_codec


def build_dct_matrix(n: int, device) -> torch.Tensor:
    """n×n DCT-II matrix (same normalization as scipy.fft.dct, norm='ortho')."""
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    m = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    C = torch.cos(math.pi * k * (m + 0.5) / n)
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C  # (n, n)


def compute_leakage_loss(x_hat_denorm: torch.Tensor, eps: float = 1e-12):
    """Differentiable leakage loss from decoder output (grayscale 2-D).

    Args:
        x_hat_denorm: (n, n) tensor in original DCT range.
    Returns:
        (loss_scalar, diag_R, R)
    """
    n = x_hat_denorm.shape[0]
    device = x_hat_denorm.device
    C = build_dct_matrix(n, device)
    power = (C @ x_hat_denorm.float()).pow(2)
    denom = power.sum(dim=0, keepdim=True) + eps
    R = power / denom
    idx = torch.arange(n, device=device)
    diag_R = R[idx, idx]
    return (1.0 - diag_R).mean(), diag_R, R


def finetune(
    model,
    size: int,
    steps: int,
    lr: float,
    device: torch.device,
    model_name: str = "",
    low_band_frac: float = 0.33,
    low_reg_weight: float = 0.12,
    low_mse_weight: float = 2e-2,
    verbose: bool = True,
) -> dict:
    """Decoder-only fine-tuning with leakage loss + low-band guard.

    Returns dict with keys: leak_before, leak_after, history.
    """
    # Freeze encoder; unfreeze decoder (g_s)
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("g_s.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Fallback: train all parameters
        for p in model.parameters():
            p.requires_grad = True
        trainable = list(model.parameters())

    model = model.to(device).train()

    # DCT basis input
    from dct_nic.metrics import build_dct_basis_rgb
    basis_rgb, vmin, vmax = build_dct_basis_rgb(size)
    x_in = torch.from_numpy(basis_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # DCT target (original basis in float)
    C = build_dct_matrix(size, device)
    D_target = C @ torch.eye(size, device=device)  # (n, n)

    # Baseline leakage
    model.eval()
    with torch.no_grad():
        res_before = evaluate_codec(
            model, size=size, device=device,
            model_name=model_name, num_runs=1,
        )
    model.train()
    leak_before = res_before["leakage"].copy()
    if verbose:
        print(f"Before fine-tuning: median L_k = {res_before['L_k']:.4f}")

    n = size
    low_n = max(1, int(low_band_frac * n))
    idx_low = torch.arange(low_n, device=device)
    base_diag_low = torch.tensor(np.diag(res_before["R"])[:low_n], device=device, dtype=torch.float32)

    opt = torch.optim.Adam(trainable, lr=lr)
    history = {"step": [], "loss": [], "L_k": []}

    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(x_in)
        x_hat = out["x_hat"]
        x_hat_denorm = (vmin + (vmax - vmin) * x_hat).mean(dim=1).squeeze(0)  # (n, n)

        loss_leak, diag_R, _ = compute_leakage_loss(x_hat_denorm)

        # Low-band regularization: penalize regression from baseline
        curr_low = diag_R.index_select(0, idx_low)
        low_deficit = torch.relu(base_diag_low - curr_low - 0.02).mean()
        loss_low_reg = low_reg_weight * low_deficit

        # Low-band MSE guard
        x_denorm_cast = x_hat_denorm.to(D_target.dtype)
        loss_low_mse = low_mse_weight * F.mse_loss(
            x_denorm_cast.index_select(1, idx_low),
            D_target.index_select(1, idx_low)
        )

        freq = torch.arange(n, device=device, dtype=torch.float32) / max(1.0, n - 1)
        w = freq.pow(2.0) + 0.2
        w[:low_n] = 0.0
        w = w / (w.sum() + 1e-8)
        leak_vec = 1.0 - diag_R
        loss_weighted = (w * leak_vec).sum()

        loss = loss_weighted + loss_low_reg + loss_low_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        L_k_curr = float(leak_vec.detach().median().item())
        history["step"].append(step)
        history["loss"].append(float(loss.item()))
        history["L_k"].append(L_k_curr)

        if verbose and step % 100 == 0:
            print(f"  step {step:4d}/{steps}  loss={loss.item():.5f}  L_k={L_k_curr:.4f}")

    # Final evaluation
    model.eval()
    res_after = evaluate_codec(
        model, size=size, device=device,
        model_name=model_name, num_runs=1,
    )
    leak_after = res_after["leakage"].copy()
    if verbose:
        pct = 100 * (res_before["L_k"] - res_after["L_k"]) / (res_before["L_k"] + 1e-12)
        print(f"After  fine-tuning: median L_k = {res_after['L_k']:.4f}  ({pct:+.1f}%)")

    return {
        "leak_before": leak_before,
        "leak_after": leak_after,
        "L_k_before": res_before["L_k"],
        "L_k_after": res_after["L_k"],
        "R_before": res_before["R"],
        "R_after": res_after["R"],
        "recon_before": res_before["recon"],
        "recon_after": res_after["recon"],
        "history": history,
    }


def plot_results(result: dict, out_path: Path, size: int):
    """Reproduce Fig. 2c / Fig. S3 style leakage-before-vs-after plot."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    k = np.arange(size)
    ax.plot(k, result["leak_before"], label="Before", color="#7E489E", linewidth=1.2)
    ax.plot(k, result["leak_after"],  label="After",  color="#D82F30", linewidth=1.2)

    n = size
    bands = [(0, n // 4, "low"), (n // 4, 3 * n // 4, "mid"), (3 * n // 4, n, "high")]
    for lo, hi, label in bands:
        delta = result["leak_before"][lo:hi].mean() - result["leak_after"][lo:hi].mean()
        pct = 100 * delta / (result["leak_before"][lo:hi].mean() + 1e-12)
        ax.axvspan(lo, hi, alpha=0.04, color="gray")
        ax.text(0.5 * (lo + hi), 0.95, f"−{pct:.0f}%", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=7, color="gray")

    ax.set_xlabel("Frequency index k", fontsize=8)
    ax.set_ylabel(r"Leakage $L_k$", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, ls=":", lw=0.4, alpha=0.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="cheng2020-anchor")
    parser.add_argument("--size", type=int, default=128,
                        help="DCT basis size (128 for Fig. 2c, 1024 for Fig. S3)")
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/finetune")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(Path(__file__).resolve().parent.parent / "third_party")

    print(f"Loading {args.model} q={args.quality} ...")
    model = load_model(args.model, args.quality, device, base_dir=base)

    result = finetune(
        model, size=args.size, steps=args.steps, lr=args.lr,
        device=device, model_name=args.model,
    )

    tag = f"{args.model}_{args.size}_q{args.quality}"
    plot_results(result, out_dir / f"finetune_{tag}.pdf", args.size)

    # Save reconstruction images
    from PIL import Image as PILImage
    import numpy as _np
    for key in ("recon_before", "recon_after"):
        if result[key] is not None:
            img = _np.clip((result[key] - result[key].min()) /
                           (result[key].max() - result[key].min() + 1e-9) * 255, 0, 255).astype(_np.uint8)
            PILImage.fromarray(img.mean(axis=2).astype(_np.uint8) if img.ndim == 3 else img).save(
                out_dir / f"{key}_{tag}.png"
            )

    print(f"\nSummary:")
    print(f"  Before: L_k = {result['L_k_before']:.4f}")
    print(f"  After:  L_k = {result['L_k_after']:.4f}")
    pct = 100 * (result["L_k_before"] - result["L_k_after"]) / (result["L_k_before"] + 1e-12)
    print(f"  Reduction: {pct:.1f}%")


if __name__ == "__main__":
    main()
