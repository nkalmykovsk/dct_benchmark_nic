"""dct_nic — DCT Basis Benchmarks for Neural Image Compression.

Quick start:

    from dct_nic import evaluate_codec, load_model

    model = load_model("cheng2020-anchor", quality=4, device="cuda")
    result = evaluate_codec(
        model, size=256, device="cuda", model_name="cheng2020-anchor"
    )

    print(f"Median leakage L_k: {result['L_k']:.4f}")
    print(f"Median ODR_k:       {result['ODR_k']:.4f}")
    print(f"Bitrate:            {result['bpp']:.3f} bpp")

Evaluating a custom model:

    # Any callable(x) -> {"x_hat": tensor, ...} works:
    result = evaluate_codec(my_model, size=256, device="cuda")
"""

from .evaluate import evaluate_codec, run_benchmark
from .loaders import load_model, get_available_models
from .metrics import (
    build_dct_basis,
    build_dct_basis_rgb,
    compute_response_matrix,
    all_metrics,
    leakage,
    odr,
    centroid_shift,
    spread,
    entropy,
    spectral_leakage_coupling,
)

__all__ = [
    # High-level API
    "evaluate_codec",
    "run_benchmark",
    "load_model",
    "get_available_models",
    # Metric primitives
    "build_dct_basis",
    "build_dct_basis_rgb",
    "compute_response_matrix",
    "all_metrics",
    "leakage",
    "odr",
    "centroid_shift",
    "spread",
    "entropy",
    "spectral_leakage_coupling",
]

__version__ = "1.0.0"
