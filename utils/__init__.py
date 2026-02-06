"""Utilities for DCT-based NIC frequency response benchmarking."""

from .functions import (
    compute_dct_smearing_metrics,
    evaluate_frequency_response,
    adaptive_finetune_on_leakage,
    build_summary_row,
    merge_and_save_csv,
    compare_metrics_by_angle,
    save_all_artifacts,
    save_experiment_images,
    save_metrics_plots,
    compute_band_leakage,
    build_row_key,
    normalize_for_display,
    set_random_seed,
    create_dct_basis_tensor,
    build_dct_matrix,
    build_extended_dct_field,
    rotate_extended_full,
    crop_containing_square,
    crop_size_for_angle,
    rotate_back_and_crop_n,
    compute_leakage_loss_from_reconstruction,
    make_dct_basis_rgb_normalized,
)

from .loaders import (
    load_model,
    load_compressai_model,
    load_tcm_model,
    get_available_models,
    ImageCodecModel,
)

from .tcm_setup import (
    ensure_tcm_assets,
    ensure_tcm_repo,
    ensure_tcm_weights,
    default_third_party_dir,
    default_tcm_dir,
)

__all__ = [
    # functions
    "compute_dct_smearing_metrics",
    "evaluate_frequency_response",
    "adaptive_finetune_on_leakage",
    "build_summary_row",
    "merge_and_save_csv",
    "compare_metrics_by_angle",
    "save_all_artifacts",
    "save_experiment_images",
    "save_metrics_plots",
    "compute_band_leakage",
    "build_row_key",
    "normalize_for_display",
    "set_random_seed",
    "create_dct_basis_tensor",
    "build_dct_matrix",
    "build_extended_dct_field",
    "rotate_extended_full",
    "crop_containing_square",
    "crop_size_for_angle",
    "rotate_back_and_crop_n",
    "compute_leakage_loss_from_reconstruction",
    "make_dct_basis_rgb_normalized",
    # loaders
    "load_model",
    "load_compressai_model",
    "load_tcm_model",
    "get_available_models",
    "ImageCodecModel",
    # tcm_setup
    "ensure_tcm_assets",
    "ensure_tcm_repo",
    "ensure_tcm_weights",
    "default_third_party_dir",
    "default_tcm_dir",
]