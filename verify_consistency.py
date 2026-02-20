"""Consistency check: L̃(DCT_basis, DCT_basis_compressed) vs mean(L_k).

Professor's question: if X = DCT basis matrix (the same input used to derive L_k),
does the weighted leakage L̃ reduce to something directly related to L_k?

We test this for several codecs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.fft import dct, dctn
from utils.functions import compute_radial_spectrum, compute_radial_distortion

NUM_BINS = 512
SIZE = 512
eps = 1e-12

def make_dct_basis(n):
    """Create n×n DCT basis matrix (same as in the benchmark)."""
    I = np.eye(n)
    D = dct(I, axis=0, norm='ortho')
    return D

def compute_Le(orig_gray, recon_gray, L_k):
    """L̃ = (1/N) Σ ρ(f)·L_k(f), with L_k interpolated onto radial grid."""
    freqs, S_f, _ = compute_radial_spectrum(orig_gray, num_bins=NUM_BINS)
    _, D_f, _ = compute_radial_distortion(orig_gray, recon_gray, num_bins=NUM_BINS)
    rho = D_f / (S_f + D_f + eps)
    k_axis = np.arange(len(L_k), dtype=np.float64)
    L_k_radial = np.interp(freqs, k_axis, L_k)
    return float(np.mean(rho * L_k_radial)), rho

def main():
    from pathlib import Path
    import torch
    from compressai.zoo import (bmshj2018_factorized, bmshj2018_hyperprior,
                                 mbt2018_mean, mbt2018, cheng2020_anchor,
                                 cheng2020_attn)

    RESULTS = Path("results/comprehensive_eval_full")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    codecs = {
        "cheng2020-anchor":     (cheng2020_anchor, 6),
        "bmshj2018-factorized": (bmshj2018_factorized, 6),
        "mbt2018":              (mbt2018, 6),
    }

    D = make_dct_basis(SIZE)
    D_rgb = np.stack([D, D, D], axis=-1)
    D_norm = (D - D.min()) / (D.max() - D.min() + eps)
    D_rgb_norm = np.stack([D_norm, D_norm, D_norm], axis=-1)

    print("=" * 72)
    print("CONSISTENCY CHECK: L̃(DCT_basis, compressed_DCT_basis) vs mean(L_k)")
    print("=" * 72)
    print(f"DCT basis size: {SIZE}×{SIZE}, radial bins: {NUM_BINS}")
    print()

    for name, (factory, q) in codecs.items():
        print(f"--- {name} (q={q}) ---")

        # Load cached L_k
        tag = f"Lk_{name}_q{q}_s{SIZE}.npy"
        L_k = np.load(RESULTS / tag)
        mean_Lk = np.mean(L_k)
        median_Lk = np.median(L_k)

        # Compress DCT basis matrix
        model = factory(quality=q, pretrained=True).eval().to(device)
        x = torch.from_numpy(D_rgb_norm.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            x_hat = out["x_hat"].clamp(0, 1)

        orig_gray = D_norm.astype(np.float64)
        recon_gray = x_hat[0, 0].cpu().numpy().astype(np.float64)

        # Compute L̃
        Le, rho = compute_Le(orig_gray, recon_gray, L_k)

        # Also compute PSNR for reference
        mse = np.mean((orig_gray - recon_gray) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + eps))

        print(f"  mean(L_k)   = {mean_Lk:.6f}")
        print(f"  median(L_k) = {median_Lk:.6f}")
        print(f"  L̃(DCT,DCT̂) = {Le:.6f}")
        print(f"  ratio L̃/mean(L_k) = {Le / (mean_Lk + eps):.4f}")
        print(f"  PSNR(DCT,DCT̂) = {psnr:.2f} dB")
        print(f"  mean(ρ)     = {np.mean(rho):.6f}")
        print(f"  max(ρ)      = {np.max(rho):.6f}")
        print()

        # Detailed: ρ(f) vs L_k(f) correlation
        from scipy.stats import spearmanr, pearsonr
        n = min(len(rho), len(L_k))
        rs, _ = spearmanr(rho[:n], L_k[:n])
        rp, _ = pearsonr(rho[:n], L_k[:n])
        print(f"  Spearman(ρ(f), L_k(f)) = {rs:.4f}")
        print(f"  Pearson(ρ(f), L_k(f))  = {rp:.4f}")
        print()

        del model
        torch.cuda.empty_cache()

    # === THEORETICAL ANALYSIS ===
    print("=" * 72)
    print("THEORETICAL ANALYSIS")
    print("=" * 72)
    print()
    print("For X = DCT basis matrix:")
    print("  S(f) = radial power spectrum of the DCT basis image")
    print("  D(f) = radial power spectrum of (DCT - compressed_DCT)")
    print("  ρ(f) = D(f) / (S(f) + D(f))")
    print()
    print("Key insight: the DCT basis matrix has energy at ALL radial")
    print("frequencies, so S(f) > 0 everywhere → ρ(f) is well-defined")
    print("and bounded away from 1.")
    print()
    print("If the codec distorts frequency f proportionally to L_k(f),")
    print("then ρ(f) ∝ L_k(f), and:")
    print("  L̃ = (1/N) Σ ρ(f)·L_k(f) ∝ (1/N) Σ L_k(f)² ≈ mean(L_k²)")
    print()
    print("So L̃(DCT, DCT̂) should correlate strongly with mean(L_k),")
    print("confirming self-consistency of the metric.")


if __name__ == "__main__":
    main()
