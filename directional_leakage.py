#!/usr/bin/env python3
"""Directional frequency leakage analysis for image codecs.

Implements the professor's rotated DCT basis approach:

  Standard DCT:   D[n, k] = alpha(k) * cos(pi/N * (n + 0.5) * k)
  Rotated DCT:    D_theta[n, k] = alpha(k') * cos(pi/N * (n' + 0.5) * k')

  where  [n', k'] = R(theta) * [n, k]
         n' = n*cos(theta) + k*sin(theta)
         k' = -n*sin(theta) + k*cos(theta)

For each angle theta:
  1. Build the NxN rotated DCT basis image  D_theta      (one matrix per angle)
  2. Send through codec                     D_hat_theta   (one codec pass per angle)
  3. Per-column correlation analysis         L_k = 1 - rho_k^2

Key property: at theta=0 this metric EXACTLY equals R[k,k] from
compute_dct_smearing_metrics (proved via Parseval's theorem).

Usage:
    python directional_leakage.py                                   # all models
    python directional_leakage.py --models jpeg cheng2020-anchor    # specific
    python directional_leakage.py --size 128 --quality 3            # quick test
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.fft import dct

from utils.loaders import load_model, get_available_models

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 1. Build rotated DCT basis  (professor's formula)
# ---------------------------------------------------------------------------

def build_rotated_dct_basis(size: int, angle_deg: float) -> np.ndarray:
    """Build NxN rotated DCT-II basis matrix.

    D_theta[n, k] = alpha(k') * cos(pi / N * (n' + 0.5) * k')

    n' = n*cos(theta) + k*sin(theta)
    k' = -n*sin(theta) + k*cos(theta)

    theta=0   => standard DCT matrix
    theta=90  => transposed DCT matrix  (D^T)
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    nn, kk = np.meshgrid(
        np.arange(size, dtype=np.float64),
        np.arange(size, dtype=np.float64),
        indexing="ij",
    )

    n_rot = nn * c + kk * s          # rotated spatial coordinate
    k_rot = -nn * s + kk * c         # rotated frequency coordinate

    alpha = np.where(
        np.abs(k_rot) < 1e-12,
        math.sqrt(1.0 / size),
        math.sqrt(2.0 / size),
    )
    return alpha * np.cos(math.pi * (n_rot + 0.5) * k_rot / size)


# ---------------------------------------------------------------------------
# 2. Codec roundtrip  (normalise -> pad -> forward -> unpad -> denorm)
# ---------------------------------------------------------------------------

def codec_roundtrip(model, img_2d: np.ndarray, device, model_name: str = ""):
    """Send a 2-D grayscale image through *model*.

    Returns (grayscale_2d, rgb_3d):
      grayscale_2d -- channel-mean of the output (for analysis)
      rgb_3d       -- full HxWx3 denormalised output (for visualisation)
    """
    rgb = np.stack([img_2d] * 3, axis=-1).astype(np.float32)
    lo, hi = float(rgb.min()), float(rgb.max())
    norm = (rgb - lo) / (hi - lo + 1e-9)

    x = torch.from_numpy(norm).float().permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, H, W = x.shape

    # architecture-specific padding
    tag = model_name.lower()
    _ca = {
        "cheng2020-anchor", "cheng2020-attn", "bmshj2018-hyperprior",
        "bmshj2018-factorized", "mbt2018-mean", "mbt2018",
    }
    mult = (256 if tag == "ftic" else
            128 if tag == "tcm" else
            64  if tag in _ca else
            None)

    ph = pw = 0
    if mult:
        th = max(256, -(-H // mult) * mult) if tag in _ca else -(-H // mult) * mult
        tw = max(256, -(-W // mult) * mult) if tag in _ca else -(-W // mult) * mult
        ph, pw = th - H, tw - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)

    with torch.no_grad():
        out = model(x)

    hat = out["x_hat"].detach().cpu().squeeze().permute(1, 2, 0).numpy()
    hat = lo + (hi - lo) * hat
    if ph or pw:
        hat = hat[:H, :W, :]
    return hat.mean(axis=2), hat


# ---------------------------------------------------------------------------
# 3. Analysis: per-column correlation + standard DCT heatmap
# ---------------------------------------------------------------------------

def analyze_rotated_basis(D_theta: np.ndarray, D_hat: np.ndarray) -> dict:
    """Per-column correlation leakage + standard DCT response matrix.

    Primary metric:
        rho_k  = <D[:,k], D_hat[:,k]> / (||D[:,k]|| * ||D_hat[:,k]||)
        L_k    = 1 - rho_k^2

    At theta=0 this equals R[k,k] from compute_dct_smearing_metrics.
    """
    N = D_theta.shape[0]
    indices = np.arange(N, dtype=np.float64)
    eps = 1e-12

    # --- per-column correlation ---
    leakage = np.zeros(N)
    rho_arr = np.zeros(N)

    for k in range(N):
        d_in  = D_theta[:, k].astype(np.float64)
        d_out = D_hat[:, k].astype(np.float64)
        ni, no = np.linalg.norm(d_in), np.linalg.norm(d_out)

        if ni < eps or no < eps:
            leakage[k] = 1.0
            continue

        rho = float(np.clip(np.dot(d_in, d_out) / (ni * no), -1.0, 1.0))
        rho_arr[k] = rho
        leakage[k] = 1.0 - rho ** 2

    # --- standard DCT R-heatmap (for visualisation) ---
    R = np.zeros((N, N), dtype=np.float64)
    for k in range(N):
        c = dct(D_hat[:, k].astype(np.float64), norm="ortho")
        power = c ** 2
        total = power.sum()
        R[:, k] = power / (total + eps)

    diag = np.diag(R)
    odr = np.tanh(0.5 * (R.sum(axis=0) - diag) / (diag + eps))

    centroids = R.T @ indices
    centroid_shift = centroids - indices
    variance = np.array([
        np.sum(((indices - centroids[k]) ** 2) * R[:, k]) for k in range(N)
    ])
    spread = np.sqrt(variance)
    max_spread = np.maximum(np.abs(centroids), np.abs((N - 1) - centroids))
    spread = spread / (max_spread + eps)

    entropy = -np.sum(R * np.log(R + eps), axis=0) / np.log(N)

    return {
        "leakage": leakage,
        "rho": rho_arr,
        "R": R,
        "odr": odr,
        "spread": spread,
        "entropy": entropy,
        "L_k":    float(np.median(leakage)),
        "L_low":  float(np.mean(leakage[: N // 4])),
        "L_high": float(np.mean(leakage[3 * N // 4 :])),
    }


# ---------------------------------------------------------------------------
# 4. Image / plot saving helpers
# ---------------------------------------------------------------------------

def _save_angle_artifacts(D_theta, D_hat, D_hat_rgb, metrics, angle_deg,
                          out_dir):
    """Save per-angle images using the ORIGINAL basis range for normalisation.

    This matches the standard benchmark: both original and decompressed are
    normalised by (mx, Mx) of the input DCT basis, so the frequency structure
    is always visible.
    """
    d = out_dir / f"angle_{int(angle_deg)}"
    d.mkdir(parents=True, exist_ok=True)

    # Normalise everything relative to the original basis range
    # (same approach as save_experiment_images in utils/functions.py)
    D_rgb = np.stack([D_theta] * 3, axis=-1).astype(np.float32)
    mx, Mx = float(D_rgb.min()), float(D_rgb.max())
    denom = Mx - mx + 1e-9

    orig_norm = np.clip((D_rgb - mx) / denom, 0, 1).astype(np.float32)
    hat_norm  = np.clip((D_hat_rgb - mx) / denom, 0, 1).astype(np.float32)

    plt.imsave(str(d / "original_rotated_dct.png"), orig_norm)
    plt.imsave(str(d / "decompressed_dct_rgb.png"), hat_norm)

    # difference (absolute, own range for visibility)
    diff = np.abs(D_hat - D_theta)
    dlo, dhi = diff.min(), diff.max()
    diff_norm = np.clip((diff - dlo) / (dhi - dlo + 1e-9), 0, 1).astype(np.float32)
    plt.imsave(str(d / "difference.png"), diff_norm, cmap="gray")

    N = D_theta.shape[0]
    idx = np.arange(N)

    # R heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(metrics["R"], origin="lower", cmap="viridis",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Normalised power")
    ax.set_xlabel("input column k")
    ax.set_ylabel("observed DCT frequency i")
    ax.set_title(f"DCT Response Matrix  R   (θ = {angle_deg}°)")
    fig.tight_layout()
    fig.savefig(d / "R_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Metrics grid (4 subplots — like compute_dct_smearing_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(idx, metrics["leakage"], label="Leakage (1 − ρ²)")
    axes[0, 0].set_xlabel("column k")
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Per-column leakage  (θ = {angle_deg}°)")

    axes[0, 1].plot(idx, metrics["rho"], label="ρ")
    axes[0, 1].set_xlabel("column k")
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].legend()
    axes[0, 1].set_title(f"Correlation ρ  (θ = {angle_deg}°)")

    axes[1, 0].plot(idx, metrics["odr"], label="ODR (tanh)")
    axes[1, 0].set_xlabel("column k")
    axes[1, 0].legend()
    axes[1, 0].set_title(f"Off-diagonal ratio  (θ = {angle_deg}°)")

    axes[1, 1].plot(idx, metrics["spread"],  label="Spread (norm)")
    axes[1, 1].plot(idx, metrics["entropy"], label="Entropy (norm)")
    axes[1, 1].set_xlabel("column k")
    axes[1, 1].legend()
    axes[1, 1].set_title(f"Spread & Entropy  (θ = {angle_deg}°)")

    fig.tight_layout()
    fig.savefig(d / "metrics_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_leakage_by_angle(results: dict, out_dir: Path):
    """Per-model plot: leakage vs column index, one curve per angle."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for ang in sorted(results.keys()):
        ax.plot(results[ang]["leakage"],
                label=f"θ={ang}°", linewidth=0.8, alpha=0.85)
    ax.set_xlabel("column k")
    ax.set_ylabel("leakage (1 − ρ²)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Leakage vs column by angle")
    fig.tight_layout()
    fig.savefig(out_dir / "leakage_vs_column.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Main measurement loop
# ---------------------------------------------------------------------------

def measure_directional_leakage(
    model,
    size: int = 256,
    angles: list[float] | None = None,
    device: str = "cuda",
    model_name: str = "",
    save_images: bool = False,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Build rotated DCT basis -> codec -> analyse.  One pass per angle.

    Returns  {angle: {leakage, rho, R, L_k, L_low, L_high, ...}}
    """
    if angles is None:
        angles = [0, 15, 30, 45, 60, 75, 90]
    if isinstance(device, str):
        device = torch.device(device)

    results = {}
    for ang in angles:
        if verbose:
            print(f"  angle {ang:3.0f}°: ", end="", flush=True)

        D_theta = build_rotated_dct_basis(size, ang)
        D_hat, D_hat_rgb = codec_roundtrip(model, D_theta, device, model_name)
        metrics = analyze_rotated_basis(D_theta, D_hat)

        results[ang] = metrics
        if verbose:
            print(f"L_k={metrics['L_k']:.4f}  "
                  f"L_low={metrics['L_low']:.4f}  "
                  f"L_high={metrics['L_high']:.4f}")

        if save_images and out_dir:
            _save_angle_artifacts(D_theta, D_hat, D_hat_rgb, metrics,
                                  ang, out_dir)

    if save_images and out_dir:
        _save_leakage_by_angle(results, out_dir)

    return results


# ---------------------------------------------------------------------------
# 6. CSV & global plots
# ---------------------------------------------------------------------------

def results_to_dataframe(
    results: dict,
    model_name: str,
    size: int,
    quality: int | None = None,
    p: int | None = None,
) -> pd.DataFrame:
    rows = []
    for ang, m in results.items():
        row = {
            "model": model_name,
            "size": f"{size}x{size}",
            "angle_deg": ang,
            "L_k": m["L_k"],
            "L_low": m["L_low"],
            "L_high": m["L_high"],
        }
        if quality is not None:
            row["q"] = quality
        if p is not None:
            row["p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def plot_leakage_polar(all_results: dict[str, dict], out_path: Path):
    """Polar plot: median leakage vs angle for every model."""
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    for name, res in all_results.items():
        angs = sorted(res.keys())
        theta = [math.radians(a) for a in angs]
        lvals = [res[a]["L_k"] for a in angs]
        theta_full = theta + [math.radians(180 - a) for a in reversed(angs)]
        lvals_full = lvals + list(reversed(lvals))
        ax.plot(theta_full, lvals_full, "o-", label=name, markersize=4)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_title("Directional Leakage  L_k = 1 − ρ²", va="bottom", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_leakage_vs_angle(all_results: dict[str, dict], out_path: Path):
    """Bar-style plot: L_k, L_low, L_high vs angle for each model."""
    angles = sorted(next(iter(all_results.values())).keys())
    models = list(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, key, title in zip(
        axes,
        ["L_k", "L_low", "L_high"],
        ["Median leakage (L_k)", "Low-freq leakage (L_low)",
         "High-freq leakage (L_high)"],
    ):
        for name in models:
            vals = [all_results[name][a][key] for a in angles]
            ax.plot(angles, vals, "o-", label=name, markersize=5)
        ax.set_xlabel("angle θ (deg)")
        ax.set_ylabel("leakage")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "tcm": {"uses_p": True, "p": 128},
}


def run_model(name: str, quality: int, device, size: int,
              angles: list[float], base_out: Path):
    cfg = MODEL_CONFIG.get(name, {})
    if cfg.get("uses_p"):
        p_val = cfg["p"]
        model = load_model(name, 1, device, p=p_val,
                           base_dir=str(Path(__file__).resolve().parent / "third_party"))
        q_param, p_param, q_tag = None, p_val, f"p_{p_val}"
    else:
        model = load_model(name, quality, device,
                           base_dir=str(Path(__file__).resolve().parent / "third_party"))
        q_param, p_param, q_tag = quality, None, f"q_{quality}"

    model_dir = base_out / name / str(size) / q_tag

    res = measure_directional_leakage(
        model, size=size, angles=angles,
        device=str(device), model_name=name,
        save_images=True, out_dir=model_dir,
    )
    return res, results_to_dataframe(res, name, size, quality=q_param, p=p_param)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to test (default: all)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--angles", type=str, default="0,15,30,45,60,75,90")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="results/directional")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    angles = [float(a) for a in args.angles.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or get_available_models(include_codecs=True)

    # ---- 0) Identity baseline ------------------------------------------------
    print("=" * 70)
    print("BASELINE: identity codec  (must show leakage = 0 at every angle)")
    print("=" * 70)

    class _Identity:
        def __call__(self, x):
            return {"x_hat": x}

    id_dir = out_dir / "identity" / str(args.size)
    id_res = measure_directional_leakage(
        _Identity(), size=args.size, angles=angles,
        device=str(device), model_name="identity",
        save_images=True, out_dir=id_dir,
    )
    all_results: dict[str, dict] = {"identity": id_res}
    all_dfs = [results_to_dataframe(id_res, "identity", args.size)]

    # ---- 1) Test each codec --------------------------------------------------
    for i, name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {name}")
        print("-" * 50)
        try:
            res, df = run_model(name, args.quality, device,
                                args.size, angles, base_out=out_dir)
            all_results[name] = res
            all_dfs.append(df)
        except Exception as exc:
            print(f"  FAILED: {exc}")

    # ---- 2) CSV --------------------------------------------------------------
    df_all = pd.concat(all_dfs, ignore_index=True)
    csv_path = out_dir / "directional_leakage.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nCSV  -> {csv_path}")

    # ---- 3) Global plots -----------------------------------------------------
    plot_leakage_polar(all_results, out_dir / "polar_leakage.png")
    plot_leakage_vs_angle(all_results, out_dir / "leakage_vs_angle.png")
    print(f"Plots -> {out_dir}/")

    # ---- 4) Summary table ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Model':25s}", end="")
    for a in angles:
        print(f"  {a:5.0f}°", end="")
    print()
    print("-" * 70)
    for name in ["identity"] + models:
        if name not in all_results:
            continue
        print(f"{name:25s}", end="")
        for a in angles:
            print(f"  {all_results[name][a]['L_k']:.4f}", end="")
        print()


if __name__ == "__main__":
    main()
