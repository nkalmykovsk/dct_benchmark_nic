"""Core DCT frequency-response metrics.

All functions operate on numpy arrays and are model-agnostic.

Paper reference:
    Kalmykov et al., "DCT Basis Benchmarks for Neural Image Compression"
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, dctn


# DCT basis generation

def build_dct_basis(n: int) -> np.ndarray:
    """Return the n×n orthonormal DCT-II basis matrix D.

    Column k of D is the spatial-domain image of the k-th DCT basis vector.
    D = DCT(I_n), shape (n, n), values in [-sqrt(2/n), sqrt(2/n)].
    """
    return dct(np.eye(n, dtype=np.float64), axis=1, norm="ortho")


def build_dct_basis_rgb(n: int) -> tuple[np.ndarray, float, float]:
    """Return DCT basis replicated across 3 channels, normalized to [0, 1].

    Returns:
        image: (n, n, 3) float32 array in [0, 1]
        vmin:  original minimum (for de-normalization)
        vmax:  original maximum (for de-normalization)
    """
    D = build_dct_basis(n)
    rgb = np.stack([D, D, D], axis=-1).astype(np.float32)
    vmin, vmax = float(rgb.min()), float(rgb.max())
    image = (rgb - vmin) / (vmax - vmin + 1e-9)
    return image, vmin, vmax


# Frequency-response matrix

def compute_response_matrix(d_hat: np.ndarray) -> np.ndarray:
    """Compute the n×n frequency-response matrix R from a reconstructed basis.

    R[i, k] = |DCT(d_hat[:, k])_i|² / sum_j |DCT(d_hat[:, k])_j|²

    R is the identity for a perfect codec; deviations quantify distortion.

    Args:
        d_hat: (n, n) reconstructed DCT basis matrix (grayscale mean of RGB output).

    Returns:
        R: (n, n) normalized power matrix, each column sums to 1.
    """
    n = d_hat.shape[0]
    R = np.zeros((n, n), dtype=np.float64)
    eps = 1e-12
    for k in range(n):
        c = dct(d_hat[:, k].astype(np.float64), norm="ortho")
        power = c ** 2
        total = power.sum()
        R[:, k] = power / (total + eps)
    return R


# Scalar metrics from R

def leakage(R: np.ndarray) -> np.ndarray:
    """Per-frequency leakage L_k = 1 - R[k, k] ∈ [0, 1].

    Low values → strong retention; high values → energy attenuation.
    """
    return 1.0 - np.diag(R)


def odr(R: np.ndarray) -> np.ndarray:
    """Off-diagonal ratio ODR_k = tanh(Σ_{i≠k} R[i,k] / (2 R[k,k])) ∈ [0, 1].

    Measures relative energy redistribution to other frequency bins.
    """
    eps = 1e-12
    diag = np.diag(R)
    off = R.sum(axis=0) - diag
    return np.tanh(0.5 * off / (diag + eps))


def centroid_shift(R: np.ndarray) -> np.ndarray:
    """Normalized signed centroid shift Δ_k = (μ_k - k) / max(k, n-1-k).

    Measures systematic drift of energy from the target frequency.
    """
    n = R.shape[0]
    idx = np.arange(n, dtype=np.float64)
    mu = R.T @ idx  # centroid per column
    raw_shift = mu - idx
    denom = np.maximum(idx, (n - 1) - idx)
    return raw_shift / (denom + 1e-12)


def spread(R: np.ndarray) -> np.ndarray:
    """Normalized spread σ_k = std(R[:, k]) / max(μ_k, n-1-μ_k).

    Measures broadening of the response around the centroid.
    """
    n = R.shape[0]
    idx = np.arange(n, dtype=np.float64)
    mu = R.T @ idx
    variance = np.array([np.sum((idx - mu[k]) ** 2 * R[:, k]) for k in range(n)])
    std = np.sqrt(variance)
    max_spread = np.maximum(np.abs(mu), np.abs((n - 1) - mu))
    return std / (max_spread + 1e-12)


def entropy(R: np.ndarray) -> np.ndarray:
    """Normalized entropy H_k = -Σ_i R[i,k] log(R[i,k]) / log(n) ∈ [0, 1].

    High values indicate indiscriminate energy redistribution.
    """
    n = R.shape[0]
    eps = 1e-12
    return -np.sum(R * np.log(R + eps), axis=0) / np.log(n)


def all_metrics(R: np.ndarray) -> dict:
    """Compute all five metrics from response matrix R.

    Returns dict with per-frequency arrays (leakage, odr, centroid_shift,
    spread, entropy) and scalar summaries (L_k, L_low, L_high, ODR_k,
    |Δc_k|, s_k, H_k).
    """
    n = R.shape[0]
    L = leakage(R)
    O = odr(R)
    C = centroid_shift(R)
    S = spread(R)
    H = entropy(R)
    return {
        "R": R,
        "leakage": L,
        "odr": O,
        "centroid_shift": C,
        "spread": S,
        "entropy": H,
        "L_k": float(np.median(L)),
        "L_low": float(np.mean(L[: n // 4])),
        "L_high": float(np.mean(L[3 * n // 4 :])),
        "ODR_k": float(np.median(O)),
        "|Δc_k|": float(np.median(np.abs(C))),
        "s_k": float(np.median(S)),
        "H_k": float(np.median(H)),
    }


# Spectral leakage coupling

def spectral_leakage_coupling(
    orig_gray: np.ndarray,
    recon_gray: np.ndarray,
    leakage_profile: np.ndarray,
    num_bins: int = 512,
) -> dict:
    """Compute spectral leakage coupling L̃ on a natural image.

    Args:
        orig_gray:       (H, W) original grayscale image in [0, 1].
        recon_gray:      (H, W) reconstructed grayscale image.
        leakage_profile: (n,) per-frequency leakage L_k from DCT benchmark.
        num_bins:        Num of radial frequency bins.

    Returns:
        dict with keys: L_tilde, rho_bar, ratio (L̃/ρ̄), rho (per-bin array).
    """
    H, W = orig_gray.shape
    eps = 1e-12

    Y_orig = dctn(orig_gray.astype(np.float64), norm="ortho")
    E = orig_gray.astype(np.float64) - recon_gray.astype(np.float64)
    Y_err = dctn(E, norm="ortho")

    fi = np.arange(H, dtype=np.float64).reshape(-1, 1)
    fj = np.arange(W, dtype=np.float64).reshape(1, -1)
    radius = np.sqrt(fi ** 2 + fj ** 2)

    r_max = radius.max() + eps
    bin_edges = np.linspace(0.0, r_max, num_bins + 1)
    bin_idx = np.clip(np.digitize(radius, bin_edges) - 1, 0, num_bins - 1)

    S = np.zeros(num_bins)
    D = np.zeros(num_bins)
    for b in range(num_bins):
        mask = bin_idx == b
        if mask.any():
            S[b] = (Y_orig ** 2)[mask].mean()
            D[b] = (Y_err ** 2)[mask].mean()

    rho = D / (S + D + eps)  # radial distortion fraction ρ(f)

    # Map 1D leakage profile to radial grid via linear interpolation
    n = len(leakage_profile)
    L_profile_clipped = np.clip(leakage_profile, 0.0, 1.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    L_r = np.interp(
        bin_centers,
        np.linspace(0.0, r_max, n),
        L_profile_clipped,
        left=L_profile_clipped[0],
        right=L_profile_clipped[-1],
    )

    L_tilde = float(np.mean(rho * L_r))
    rho_bar = float(np.mean(rho))
    ratio = L_tilde / rho_bar if rho_bar > 1e-12 else 0.0

    return {
        "L_tilde": L_tilde,
        "rho_bar": rho_bar,
        "ratio": ratio,
        "rho": rho,
        "L_r": L_r,
    }
