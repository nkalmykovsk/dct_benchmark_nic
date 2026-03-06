"""DCT benchmark evaluation pipeline.

Main entry points:
    evaluate_codec(model, size, device, num_runs) -> dict
    run_benchmark(models, sizes, qualities, device, out_dir) -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .metrics import build_dct_basis_rgb, compute_response_matrix, all_metrics

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def _pad_for_model(x: torch.Tensor, model_name: str) -> tuple[torch.Tensor, int, int]:
    """Pad tensor to model-required stride."""
    _, _, H, W = x.shape
    tag = model_name.lower()
    compressai_models = {
        "cheng2020-anchor", "cheng2020-attn", "bmshj2018-hyperprior",
        "bmshj2018-factorized", "mbt2018-mean", "mbt2018",
    }
    if tag == "ftic":
        mult = 256
    elif tag == "tcm":
        mult = 128
    elif tag in compressai_models:
        mult = 64
    else:
        return x, 0, 0

    th = -(-H // mult) * mult
    tw = -(-W // mult) * mult
    ph, pw = th - H, tw - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)
    return x, ph, pw


def _model_forward(model, x: torch.Tensor, model_name: str = "") -> tuple[torch.Tensor, float | None]:
    """Run one codec forward pass; return (x_hat, bpp)."""
    x_pad, ph, pw = _pad_for_model(x, model_name)
    _, _, H_orig, W_orig = x.shape

    with torch.no_grad():
        out = model(x_pad)

    if not isinstance(out, dict) or "x_hat" not in out:
        raise TypeError("model(x) must return dict with key 'x_hat'")

    x_hat = out["x_hat"].detach().cpu()
    if ph or pw:
        x_hat = x_hat[:, :, :H_orig, :W_orig]

    bpp = None
    if "bpp" in out and out["bpp"] is not None:
        bpp = float(out["bpp"])
    elif "likelihoods" in out:
        num_pixels = H_orig * W_orig
        bpp = sum(
            (-torch.log2(lk.clamp(min=1e-9))).sum().item()
            for lk in out["likelihoods"].values()
        ) / num_pixels

    return x_hat, bpp


def evaluate_codec(
    model,
    size: int = 256,
    device: str | torch.device = "cuda",
    model_name: str = "",
    num_runs: int = 1,
) -> dict:
    """Evaluate a codec on the DCT benchmark.

    Compresses the n×n DCT basis image, computes the frequency-response matrix
    R, and returns all five metrics (leakage, ODR, centroid shift, spread,
    entropy) plus bpp.

    Args:
        model:      Any callable that accepts a [1, 3, H, W] float32 tensor in
                    [0, 1] and returns {"x_hat": tensor, ...}.
                    CompressAI, TCM, FTIC, and classical wrappers all satisfy
                    this interface.
        size:       DCT basis dimension n.
        device:     PyTorch device string or object.
        model_name: Used for architecture-specific padding (optional).
        num_runs:   Number of forward passes to average (for stochastic codecs).

    Returns:
        dict with keys:
            R           – (n, n) mean frequency-response matrix
            leakage     – (n,) per-frequency leakage L_k
            odr         – (n,) ODR_k
            centroid_shift – (n,) Δ_k
            spread      – (n,) σ_k
            entropy     – (n,) H_k
            L_k, L_low, L_high, ODR_k, |Δc_k|, s_k, H_k  – scalar summaries
            bpp         – mean bits-per-pixel (or None)
            orig        – (n, n, 3) normalized DCT basis image
            recon       – (n, n, 3) last reconstruction (de-normalized)
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

    basis_image, vmin, vmax = build_dct_basis_rgb(size)
    x = torch.from_numpy(basis_image).float().permute(2, 0, 1).unsqueeze(0).to(device)

    R_accum: list[np.ndarray] = []
    bpp_list: list[float] = []
    recon_last = None

    for _ in range(max(1, num_runs)):
        x_hat, run_bpp = _model_forward(model, x, model_name)
        if run_bpp is not None:
            bpp_list.append(run_bpp)

        # De-normalize to original DCT range for metric computation
        recon_rgb = x_hat.squeeze().permute(1, 2, 0).numpy()
        recon_rgb = vmin + (vmax - vmin) * recon_rgb
        recon_last = recon_rgb

        R = compute_response_matrix(recon_rgb.mean(axis=2))
        R_accum.append(R)

    R_mean = np.mean(np.stack(R_accum), axis=0)
    bpp = float(np.mean(bpp_list)) if bpp_list else None

    result = all_metrics(R_mean)
    result["bpp"] = bpp
    result["orig"] = basis_image
    result["recon"] = recon_last
    return result


# ---------------------------------------------------------------------------
# Batch benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    models: dict,
    sizes: list[int],
    device: str | torch.device = "cuda",
    num_runs: int = 1,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the DCT benchmark across multiple codecs and sizes.

    Args:
        models:   {name: callable} mapping. Use loaders.load_model() to build.
        sizes:    List of DCT basis sizes, e.g. [64, 128, 256, 512].
        device:   PyTorch device.
        num_runs: Runs to average per setting.
        out_dir:  If set, save results CSV to out_dir/benchmark_results.csv.
        verbose:  Print progress.

    Returns:
        pd.DataFrame with columns: model, size, L_k, L_low, L_high,
        ODR_k, |Δc_k|, s_k, H_k, bpp.
    """
    rows = []
    total = len(models) * len(sizes)
    done = 0

    for name, model in models.items():
        for sz in sizes:
            done += 1
            if verbose:
                print(f"[{done:3d}/{total}] {name:30s} size={sz}", end=" ... ", flush=True)
            try:
                res = evaluate_codec(model, size=sz, device=device,
                                     model_name=name, num_runs=num_runs)
                row = {"model": name, "size": sz}
                for k in ("L_k", "L_low", "L_high", "ODR_k", "|Δc_k|", "s_k", "H_k", "bpp"):
                    row[k] = res.get(k)
                rows.append(row)
                if verbose:
                    print(f"L_k={res['L_k']:.4f}  bpp={res['bpp']:.3f}" if res["bpp"] else f"L_k={res['L_k']:.4f}")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")

    df = pd.DataFrame(rows)
    if out_dir is not None:
        out_path = Path(out_dir) / "benchmark_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"\nSaved → {out_path}")
    return df
