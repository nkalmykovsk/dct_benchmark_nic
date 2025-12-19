"""Utilities to evaluate NIC frequency response using DCT basis inputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct


def compute_dct_smearing_metrics(
    d_hat,
    axis=0,
    normalize_freq=True,
    show_plots=True,
    title_prefix="",
    save_path=None,
):
    """Compute frequency smearing metrics from reconstructed DCT basis image."""
    if getattr(d_hat, "ndim", None) != 2 or d_hat.shape[0] != d_hat.shape[1]:
        raise ValueError("d_hat must be a square 2D array")

    n = d_hat.shape[0]
    r = np.zeros((n, n), dtype=float)  # r[i, k] = power at frequency i for input basis k

    for k in range(n):
        if axis == 0:
            d_k_hat = d_hat[:, k].astype(float)
        else:
            d_k_hat = d_hat[k, :].astype(float)
        c = dct(d_k_hat, norm="ortho")
        power = np.abs(c) ** 2
        total = power.sum()
        p = np.zeros_like(power) if total == 0 else power / total
        r[:, k] = p

    indices = np.arange(n)
    centroids = r.T @ indices  # shape (n,)
    centroid_shift = centroids - indices

    if normalize_freq:
        centroids_norm = centroids / (n - 1)
        max_shift = np.maximum(indices, (n - 1) - indices)
        centroid_shift_norm = centroid_shift / (max_shift + 1e-12)
    else:
        centroids_norm = centroids
        centroid_shift_norm = centroid_shift

    leakage = 1.0 - np.diag(r)

    eps = 1e-12
    diag_vals = np.diag(r)
    odr = (np.sum(r, axis=0) - diag_vals) / (diag_vals + eps)
    odr = np.tanh(0.5 * odr)

    variance = np.array(
        [np.sum(((indices - centroids[k]) ** 2) * r[:, k]) for k in range(n)]
    )
    spread = np.sqrt(variance)
    max_spread_per_k = np.maximum(
        np.abs(0 - centroids),
        np.abs((n - 1) - centroids),
    )
    spread = spread / (max_spread_per_k + 1e-12)

    entropy = -np.sum(r * np.log(r + eps), axis=0)
    entropy = entropy / np.log(n)

    def cumulative_energy(k, w):
        lo = max(0, k - w)
        hi = min(n - 1, k + w)
        return r[lo : hi + 1, k].sum()

    windows = [0, 1, 2, 4, 8]
    cum_energy = {
        w: np.array([cumulative_energy(k, w) for k in range(n)])
        for w in windows
    }

    metrics = {
        "R": r,
        "leakage": leakage,
        "odr": odr,
        "centroids": centroids_norm,
        "centroid_shift": centroid_shift_norm,
        "spread": spread,
        "entropy": entropy,
        "cum_energy": cum_energy,
        "indices": indices,
    }

    if show_plots:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        ax = axs[0, 0]
        im = ax.imshow(r, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, n - 0.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Normalized power")
        ax.set_xlabel("input basis k")
        ax.set_ylabel("observed frequency i")
        ax.set_title(f"{title_prefix} Frequency-response matrix R")

        ax = axs[0, 1]
        ax.plot(indices, leakage, label="Leakage (1 - p_k)")
        ax.plot(
            indices,
            np.abs(centroid_shift_norm),
            label="|Centroid shift| (normalized)",
        )
        ax.plot(indices, spread, label="Spread (normalized)")
        ax.plot(indices, entropy, label="Entropy (normalized)")
        ax.set_xlabel("basis k")
        ax.legend()
        ax.set_title(f"{title_prefix} Leakage / shift / spread / Entropy")

        ax = axs[1, 0]
        ax.plot(indices, odr, label="ODR (tanh scaled)")
        ax.set_xlabel("basis k")
        ax.legend()
        ax.set_title(f"{title_prefix} Off-diag ratio")

        ax = axs[1, 1]
        for w in windows:
            ax.plot(indices, cum_energy[w], label=f"within +/-{w}")
        ax.set_xlabel("basis k")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.set_title(f"{title_prefix} Cumulative energy in neighbor windows")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        plt.show()

    return metrics


def set_random_seed(seed):
    """Set seeds for reproducibility (torch + numpy)."""
    if seed is not None:
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
        except Exception:
            pass


def create_dct_basis_tensor(size, device):
    """Create DCT basis image (HxWx3) and normalized input tensor [1,3,H,W]."""
    # Construct DCT matrix (apply 1D DCT to identity)
    x = np.eye(size)
    x_dct = dct(x, axis=1, norm="ortho")

    # Replicate into 3 channels
    x_dct_rgb = np.stack([x_dct, x_dct, x_dct], axis=-1)

    # Normalize to [0,1]
    mx, Mx = x_dct_rgb.min(), x_dct_rgb.max()
    x_dct_norm = (x_dct_rgb - mx) / (Mx - mx + 1e-9)

    # Convert to tensor [1,3,H,W]
    x_tensor = (
        torch.from_numpy(x_dct_norm)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    return x_dct_rgb, x_tensor, mx, Mx


def _forward_x_hat(model, x_tensor):
    out = model(x_tensor)
    if not isinstance(out, dict) or "x_hat" not in out:
        raise TypeError("model(x) must return a dict containing key 'x_hat'")
    return out["x_hat"].detach().cpu()


def _denormalize_to_dct_rgb(x_hat_norm, mx, Mx):
    x_hat_rgb = x_hat_norm.squeeze().permute(1, 2, 0).numpy()
    return mx + (Mx - mx) * x_hat_rgb


def _idct_rgb(x_dct_rgb):
    x_idct = np.zeros_like(x_dct_rgb)
    for c in range(3):
        x_idct[..., c] = idct(x_dct_rgb[..., c], axis=1, norm="ortho")
    return x_idct


def _plot_frequency_response(
    x_dct_rgb,
    x_hat_norm,
    x_idct,
    avg_metrics,
    show_plots=True,
    show_metric_plots=True,
):
    """Plot reconstruction images and metric curves."""

    if show_plots:
        _, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(x_dct_rgb.mean(axis=2), cmap="gray")
        axs[0].set_title("Original DCT matrix (RGB)")
        axs[0].axis("off")

        axs[1].imshow(
            x_hat_norm.squeeze().permute(1, 2, 0).numpy().mean(axis=2),
            cmap="gray",
        )
        axs[1].set_title("Decompressed DCT (RGB)")
        axs[1].axis("off")

        axs[2].imshow(x_idct.mean(axis=2), cmap="gray")
        axs[2].set_title("iDCT of decompressed (RGB)")
        axs[2].axis("off")
        plt.show()

    if show_metric_plots:
        indices = avg_metrics["indices"]
        _, axs2 = plt.subplots(2, 3, figsize=(16, 8))

        axs2[0, 0].plot(indices, avg_metrics["leakage"])
        axs2[0, 0].set_title("Leakage L_k")
        axs2[0, 0].set_xlabel("k")

        axs2[0, 1].plot(indices, avg_metrics["odr"])
        axs2[0, 1].set_title("Off–diagonal ratio ODR_k")
        axs2[0, 1].set_xlabel("k")

        axs2[0, 2].plot(indices, np.abs(avg_metrics["centroid_shift"]))
        axs2[0, 2].set_title("Centroid shift Δc_k")
        axs2[0, 2].set_xlabel("k")

        axs2[1, 0].plot(indices, avg_metrics["spread"])
        axs2[1, 0].set_title("Spread s_k")
        axs2[1, 0].set_xlabel("k")

        axs2[1, 1].plot(indices, avg_metrics["entropy"])
        axs2[1, 1].set_title("Entropy H_k")
        axs2[1, 1].set_xlabel("k")

        # CE_k(w) for several windows
        cum_energy = avg_metrics.get("cum_energy", {})
        for w_key in sorted(cum_energy.keys()):
            axs2[1, 2].plot(indices, cum_energy[w_key], label=f"w={w_key}")
        axs2[1, 2].set_title("Cumulative energy CE_k(w)")
        axs2[1, 2].set_xlabel("k")
        axs2[1, 2].set_ylim(-0.05, 1.05)
        axs2[1, 2].legend()

        plt.tight_layout()
        plt.show()


def _stack_mean(arr_list, axis=0):
    return np.mean(np.stack(arr_list, axis=0), axis=axis)


def evaluate_frequency_response(
    model,
    size=64,
    device="cuda",
    show_plots=True,
    num_runs=10,
    show_metric_plots=True,
    seed=None,
):
    """
    Evaluate frequency response of a NIC model using DCT basis matrix input.

    Args:
        model: Trained NIC model (expects input in [0,1], shape [1,3,H,W])
        size: Image size (N x N)
        device: "cuda" or "cpu"
        show_plots: Whether to visualize inputs/outputs
        num_runs: Number of runs to average metrics over
        show_metric_plots: Whether to visualize metric plots
        seed: Random seed for reproducibility
    Returns:
        x_dct_rgb: Original DCT basis (H,W,3)
        x_hat: Decompressed DCT basis from last run (H,W,3)
        metrics: Dictionary of averaged frequency smearing metrics over num_runs
    """

    set_random_seed(seed)
    x_dct_rgb, x_tensor, mx, Mx = create_dct_basis_tensor(size, device)

    # Run multiple forwards and aggregate metrics
    leakage_list = []
    odr_list = []
    centroid_shift_list = []
    centroids_list = []
    spread_list = []
    entropy_list = []
    cum_energy_accumulator = {}
    R_list = []
    x_hat = None
    x_idct = None
    x_hat_norm = None

    # Compress & decompress with NIC model
    runs = int(max(1, num_runs))
    for _ in range(runs):
        with torch.no_grad():
            x_hat_norm = _forward_x_hat(model, x_tensor)
        x_hat_run = _denormalize_to_dct_rgb(x_hat_norm, mx, Mx)
        x_hat = x_hat_run  # save last run's outputs for visualization
        x_idct = _idct_rgb(x_hat_run)  # iDCT per channel

        # Metrics on grayscale
        m = compute_dct_smearing_metrics(
            x_hat_run.mean(axis=2),
            axis=0,
            normalize_freq=True,
            show_plots=False,
            title_prefix="NIC Response"
        )
        # Shapes: vectors (n,), R (n, n).
        for key, acc in (
            ("leakage", leakage_list),
            ("odr", odr_list),
            ("centroid_shift", centroid_shift_list),
            ("centroids", centroids_list),
            ("spread", spread_list),
            ("entropy", entropy_list),
            ("R", R_list),
        ):
            acc.append(m[key])

        # Cum energy windows
        for w_key, ce_arr in m["cum_energy"].items():
            cum_energy_accumulator.setdefault(w_key, []).append(ce_arr)

    # Average across runs
    leakage_avg = _stack_mean(leakage_list)
    odr_avg = _stack_mean(odr_list)
    centroid_shift_avg = _stack_mean(centroid_shift_list)
    centroids_avg = _stack_mean(centroids_list)
    spread_avg = _stack_mean(spread_list)
    entropy_avg = _stack_mean(entropy_list)
    R_mean = _stack_mean(R_list)

    cum_energy_avg = {
        w_key: _stack_mean(arrs)
        for w_key, arrs in cum_energy_accumulator.items()
    }

    # Build metrics dictionary
    indices_arr = np.arange(size)
    avg_metrics = {
        "num_runs": runs,
        "indices": indices_arr,
        "leakage": 1.0 - np.diag(R_mean),
        "odr": odr_avg,
        "centroid_shift": centroid_shift_avg,
        "spread": spread_avg,
        "entropy": entropy_avg,
        "cum_energy": cum_energy_avg,
        "R": R_mean,
    }

    summary_window = 2
    try:
        ce_w = cum_energy_avg.get(summary_window, None)
        # Compute entropy in bits from R_mean to keep CSV units consistent
        eps = 1e-12
        entropy_bits_from_R = -np.sum(
            R_mean * np.log(R_mean + eps),
            axis=0,
        ) / np.log(2)
        summary = {
            "L_k": float(np.median(1.0 - np.diag(R_mean))),
            "ODR_k": float(np.median(odr_avg)),
            "|Delta_c_k|": float(np.median(np.abs(centroid_shift_avg))),
            "s_k": float(np.median(spread_avg)),
            "H_k_bits": float(np.median(entropy_bits_from_R)),
            f"CE_k(w={summary_window})": (
                float(np.median(ce_w)) if ce_w is not None else None
            ),
        }
    except Exception:
        summary = None
    avg_metrics["summary"] = summary

    _plot_frequency_response(
        x_dct_rgb=x_dct_rgb,
        x_hat_norm=x_hat_norm,
        x_idct=x_idct,
        avg_metrics=avg_metrics,
        show_plots=show_plots,
        show_metric_plots=show_metric_plots,
    )

    if avg_metrics.get("summary") is not None:
        s = avg_metrics["summary"]
        print(
            "Summary (median over k; H in bits; CE window=2; "
            f"runs={avg_metrics['num_runs']}): "
            f"L_k={s['L_k']:.4f}, ODR_k={s['ODR_k']:.4f}, "
            f"|Δc_k|={s['|Delta_c_k|']:.4f}, s_k={s['s_k']:.4f}, "
            f"H_k={s['H_k_bits']:.4f}, CE_k(w=2)={s['CE_k(w=2)']:.4f}"
        )

    return x_dct_rgb, x_hat, avg_metrics


# CSV and Results Utilities

def build_row_key(row: dict, df_columns: list = None) -> str:
    """Build unique key for a results row (Model|Size|q:X or p:X)."""
    model = row.get('Model', '')
    size_s = row.get('Size', '')
    if df_columns and 'q' in df_columns and pd.notna(row.get('q', np.nan)):
        qp = f"q:{int(row['q'])}"
    elif df_columns and 'p' in df_columns and pd.notna(row.get('p', np.nan)):
        qp = f"p:{int(row['p'])}"
    elif 'q' in row and pd.notna(row.get('q', np.nan)):
        qp = f"q:{int(row['q'])}"
    elif 'p' in row and pd.notna(row.get('p', np.nan)):
        qp = f"p:{int(row['p'])}"
    else:
        qp = 'q:NA'
    return f"{model}|{size_s}|{qp}"


def merge_and_save_csv(
    new_rows: list,
    csv_path: Path,
    preferred_cols: list = None,
    sort_by: str = None,
) -> pd.DataFrame:
    """
    Merge new results with existing CSV, deduplicating by row key.

    Args:
        new_rows: List of dicts with new results
        csv_path: Path to CSV file
        preferred_cols: Column order preference
        sort_by: Column to sort by (auto-detects 'q' or 'p' if None)

    Returns:
        Merged DataFrame
    """
    if preferred_cols is None:
        preferred_cols = [
            'Model', 'Size', 'p', 'q',
            'L_k', 'L_low', 'L_high', 'ODR_k', '|Delta_c_k|',
            's_k', 'H_k_bits', 'CE_k(w=2)',
        ]

    df_new = pd.DataFrame(new_rows)

    # Add keys for deduplication
    df_new['__key__'] = [build_row_key(r, df_new.columns.tolist()) for _, r in df_new.iterrows()]

    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        # Align columns
        all_cols = list(set(df_old.columns.tolist()) | set(df_new.columns.tolist()))
        for c in all_cols:
            if c not in df_old.columns:
                df_old[c] = np.nan
            if c not in df_new.columns:
                df_new[c] = np.nan
        df_old['__key__'] = [build_row_key(r, df_old.columns.tolist()) for _, r in df_old.iterrows()]
        df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset='__key__', keep='last')
    else:
        df_merged = df_new

    df_merged = df_merged.drop(columns='__key__')

    # Sort
    if sort_by is None:
        if 'q' in df_merged.columns and df_merged['q'].notna().any():
            sort_by = 'q'
        elif 'p' in df_merged.columns and df_merged['p'].notna().any():
            sort_by = 'p'
    if sort_by and sort_by in df_merged.columns:
        df_merged = df_merged.sort_values(sort_by)

    # Select and order columns
    out_cols = [c for c in preferred_cols if c in df_merged.columns]
    df_out = df_merged[out_cols].round(4)
    df_out.to_csv(csv_path, index=False)

    return df_out


def compute_band_leakage(leakage: np.ndarray) -> dict:
    """Compute band-wise leakage medians (low/high thirds)."""
    n = len(leakage)
    one_third = n // 3
    two_third = 2 * n // 3

    return {
        'L_low': float(np.median(leakage[:one_third])) if one_third > 0 else np.nan,
        'L_high': float(np.median(leakage[two_third:])) if two_third < n else np.nan,
    }


def build_summary_row(
    model_name: str,
    size: int,
    metrics: dict,
    quality: int = None,
    p: int = None,
    ce_window: int = 2,
) -> dict:
    """Build a summary row dict from metrics."""
    R = metrics['R']
    leakage = metrics['leakage']
    odr = metrics['odr']
    centroid_shift = metrics['centroid_shift']
    spread = metrics['spread']
    entropy_bits = metrics['entropy'] / np.log(2)
    cum_energy = metrics['cum_energy']

    row = {
        'Model': model_name,
        'Size': f'{size}x{size}',
        'L_k': float(np.median(1.0 - np.diag(R))),
        'ODR_k': float(np.median(odr)),
        '|Delta_c_k|': float(np.median(np.abs(centroid_shift))),
        's_k': float(np.median(spread)),
        'H_k_bits': float(np.median(entropy_bits)),
    }

    # Add CE for window
    ce_w = cum_energy.get(ce_window)
    row[f'CE_k(w={ce_window})'] = float(np.median(ce_w)) if ce_w is not None else np.nan

    # Add band-wise leakage
    row.update(compute_band_leakage(leakage))

    # Add quality/p parameter
    if quality is not None:
        row['q'] = int(quality)
    if p is not None:
        row['p'] = int(p)

    return row


# Artifact Saving Utilities


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] for display."""
    mn, mx = arr.min(), arr.max()
    return np.clip((arr - mn) / (mx - mn + 1e-9), 0.0, 1.0).astype(np.float32)


def save_experiment_images(
    output_dir: Path,
    x_dct_rgb: np.ndarray,
    x_hat: np.ndarray,
):
    """Save original DCT, decompressed, and iDCT images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize using original DCT range
    mx, Mx = x_dct_rgb.min(), x_dct_rgb.max()
    denom = Mx - mx + 1e-9

    x_dct_norm = np.clip((x_dct_rgb - mx) / denom, 0.0, 1.0).astype(np.float32)
    x_hat_norm = np.clip((x_hat - mx) / denom, 0.0, 1.0).astype(np.float32)

    # iDCT of decompressed
    x_idct = _idct_rgb(x_hat)
    x_idct_disp = normalize_for_display(x_idct)

    plt.imsave(output_dir / 'original_dct_rgb.png', x_dct_norm)
    plt.imsave(output_dir / 'decompressed_dct_rgb.png', x_hat_norm)
    plt.imsave(output_dir / 'idct_of_decompressed_rgb.png', x_idct_disp)


def save_metrics_plots(output_dir: Path, metrics: dict, dpi: int = 150):
    """Save metrics grid plot and R heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = metrics['indices']
    leakage = metrics['leakage']
    odr = metrics['odr']
    centroid_shift = metrics['centroid_shift']
    spread = metrics['spread']
    entropy_bits = metrics['entropy'] / np.log(2)
    cum_energy = metrics['cum_energy']
    R = metrics['R']

    # 2x3 metrics grid
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    axs[0, 0].plot(indices, leakage)
    axs[0, 0].set_title('Leakage L_k')
    axs[0, 0].set_xlabel('k')

    axs[0, 1].plot(indices, odr)
    axs[0, 1].set_title('Off-diagonal ratio ODR_k')
    axs[0, 1].set_xlabel('k')

    axs[0, 2].plot(indices, np.abs(centroid_shift))
    axs[0, 2].set_title('Centroid shift Δc_k')
    axs[0, 2].set_xlabel('k')

    axs[1, 0].plot(indices, spread)
    axs[1, 0].set_title('Spread s_k')
    axs[1, 0].set_xlabel('k')

    axs[1, 1].plot(indices, entropy_bits)
    axs[1, 1].set_title('Entropy H_k (bits)')
    axs[1, 1].set_xlabel('k')

    for w_key in sorted(cum_energy.keys()):
        axs[1, 2].plot(indices, cum_energy[w_key], label=f"w={w_key}")
    axs[1, 2].set_title('Cumulative energy CE_k(w)')
    axs[1, 2].set_xlabel('k')
    axs[1, 2].set_ylim(-0.05, 1.05)
    axs[1, 2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / 'metrics_grid.png', dpi=dpi)
    plt.close(fig)

    # R heatmap
    fig_hm = plt.figure(figsize=(6, 5))
    plt.imshow(R, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Normalized power')
    plt.xlabel('input basis k')
    plt.ylabel('observed frequency i')
    plt.title('Frequency-response matrix R')
    fig_hm.tight_layout()
    fig_hm.savefig(output_dir / 'R_heatmap.png', dpi=dpi)
    plt.close(fig_hm)


def save_all_artifacts(
    output_dir: Path,
    x_dct_rgb: np.ndarray,
    x_hat: np.ndarray,
    metrics: dict,
    dpi: int = 150,
):
    """Save all experiment artifacts (images + plots)."""
    save_experiment_images(output_dir, x_dct_rgb, x_hat)
    save_metrics_plots(output_dir, metrics, dpi=dpi)