#!/usr/bin/env python3
"""Comprehensive evaluation of Spectral Leakage Coupling L̃(X, X̂).

All models × 24 Kodak images × q=6 × 512.

Distortion types per model:
  1. Clean codec roundtrip
  2. Codec + Gaussian noise on reconstruction (σ = 0.01, 0.03, 0.05, 0.10, 0.20)
  3. Pure Gaussian noise — no codec (σ = 0.01, 0.03, 0.05, 0.10, 0.20)
  4. Quantization — reduce bit depth (6, 4, 3, 2 bits)
  5. JPEG re-compression at low quality (q = 10, 30, 50, 70)

Metric: L(X, X̂) = Σ ρ(f)·L(f) / Σ L(f),  ρ(f) = D_f / (S_f + D_f) ∈ [0,1]

Usage:
    python run_comprehensive_eval.py --single   # 1 image, 1 model (quick test)
    python run_comprehensive_eval.py --full     # 24 images, all models
"""

import argparse
import gc
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.loaders import load_model
from utils.functions import (
    compute_radial_spectrum,
    compute_radial_distortion,
    evaluate_frequency_response,
)

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda")
QUALITY = 6
SIZE = 512
NUM_BINS = 512
KODAK = Path("kodak")
eps = 1e-12

MODELS = [
    "cheng2020-anchor",
    "cheng2020-attn",
    "mbt2018-mean",
    "mbt2018",
    "bmshj2018-hyperprior",
    "bmshj2018-factorized",
    "ftic",
    "tcm",
    "jpeg",
    "webp",
    "jpegxl",
]
TRADITIONAL = {"jpeg", "webp", "jpegxl", "jpeg2000"}
TCM_P = 128
PAD_MULT = {"ftic": 256, "tcm": 128}

GAUSS_SIGMAS = [0.01, 0.03, 0.05, 0.10, 0.20]
QUANT_BITS = [6, 4, 3, 2]
JPEG_QUALITIES = [10, 30, 50, 70]

N_DISTORTIONS_PER_IMAGE = 1 + len(GAUSS_SIGMAS) + len(GAUSS_SIGMAS) + len(QUANT_BITS) + len(JPEG_QUALITIES)

# ═══════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════
def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ═══════════════════════════════════════════════════════════════════════
# Core metric
# ═══════════════════════════════════════════════════════════════════════
def compute_Le(orig_gray: np.ndarray, recon_gray: np.ndarray, L_k: np.ndarray) -> float:
    """Spectral Leakage Coupling: L̃ = (1/N) Σ ρ(f)·L_k(f), ρ = D/(S+D)."""
    _, S_f, _ = compute_radial_spectrum(orig_gray, num_bins=NUM_BINS)
    _, D_f, _ = compute_radial_distortion(orig_gray, recon_gray, num_bins=NUM_BINS)
    n = min(len(S_f), len(L_k), len(D_f))
    rho = D_f[:n] / (S_f[:n] + D_f[:n] + eps)
    return float(np.mean(rho * L_k[:n]))


def psnr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(-10.0 * np.log10(F.mse_loss(x, y).item() + eps))


def to_gray_np(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().mean(dim=0).numpy()


def save_img(tensor: torch.Tensor, path: Path):
    arr = (tensor.squeeze().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


# ═══════════════════════════════════════════════════════════════════════
# Padding helpers
# ═══════════════════════════════════════════════════════════════════════
def pad_params(H, W, model_name=""):
    mult = PAD_MULT.get(model_name, 64)
    th = -(-H // mult) * mult
    tw = -(-W // mult) * mult
    if model_name not in PAD_MULT:
        th = max(256, th)
        tw = max(256, tw)
    return th - H, tw - W


# ═══════════════════════════════════════════════════════════════════════
# Distortion generators
# ═══════════════════════════════════════════════════════════════════════
def codec_roundtrip(model, x_dev, ph, pw, H, W, is_trad):
    if is_trad:
        out = model(x_dev)
        return out["x_hat"].to(x_dev.device)[:, :, :H, :W].clamp(0, 1).cpu()
    model.eval()
    with torch.no_grad():
        out = model(F.pad(x_dev, (0, pw, 0, ph), value=0.0))
        return out["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()


def add_gaussian_noise(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    return (tensor + sigma * torch.randn_like(tensor)).clamp(0, 1)


def quantize_to_bits(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2 ** bits - 1
    return (tensor * levels).round() / levels


def jpeg_recompress(tensor: torch.Tensor, quality: int) -> torch.Tensor:
    arr = (tensor.squeeze().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════
# Get / cache leakage profile
# ═══════════════════════════════════════════════════════════════════════
def get_leakage_profile(model, model_name, cache_dir, logger):
    if model_name == "tcm":
        tag = f"Lk_{model_name}_p{TCM_P}_s{SIZE}.npy"
    else:
        tag = f"Lk_{model_name}_q{QUALITY}_s{SIZE}.npy"
    lk_path = cache_dir / tag
    prev = Path(f"results/full_leakage_eval/{tag}")
    if lk_path.exists():
        L_k = np.load(lk_path)
        logger.info(f"  L_k loaded from cache ({len(L_k)} bins)")
        return L_k
    if prev.exists():
        L_k = np.load(prev)
        np.save(lk_path, L_k)
        logger.info(f"  L_k loaded from prev cache ({len(L_k)} bins)")
        return L_k
    logger.info("  Computing L_k (frequency response)...")
    t0 = time.time()
    _, _, metrics = evaluate_frequency_response(
        model, size=SIZE, device=DEVICE,
        show_plots=False, num_runs=3, show_metric_plots=False,
        seed=42, verbose=False)
    L_k = metrics["leakage"]
    np.save(lk_path, L_k)
    logger.info(f"  L_k computed in {time.time()-t0:.1f}s, {len(L_k)} bins")
    return L_k


# ═══════════════════════════════════════════════════════════════════════
# Evaluate single image with one model
# ═══════════════════════════════════════════════════════════════════════
def evaluate_image(img_path, model, model_name, L_k, is_trad, logger, save_dir=None):
    stem = img_path.stem
    rows = []

    img = Image.open(img_path).convert("RGB")
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0
                         ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    _, _, H, W = x.shape
    ph, pw = pad_params(H, W, model_name)
    x_cpu = x.cpu()
    orig_gray = to_gray_np(x_cpu)

    if save_dir:
        d = save_dir / model_name / stem
        d.mkdir(parents=True, exist_ok=True)
        save_img(x_cpu, d / "original.png")

    def rec(dist_type, param, x_hat_cpu):
        p = psnr_db(x_hat_cpu, x_cpu)
        le = compute_Le(orig_gray, to_gray_np(x_hat_cpu), L_k)
        rows.append(dict(model=model_name, image=stem, distortion=dist_type,
                         param=str(param), psnr=p, Le=le))
        if save_dir:
            tag = f"{dist_type}_{param}".replace(".", "p")
            save_img(x_hat_cpu, save_dir / model_name / stem / f"{tag}.png")

    # 1) Clean codec roundtrip
    x_hat_clean = codec_roundtrip(model, x, ph, pw, H, W, is_trad)
    param_tag = f"p{TCM_P}" if model_name == "tcm" else f"q{QUALITY}"
    rec("codec_clean", param_tag, x_hat_clean)

    # 2) Codec + Gaussian noise on reconstruction
    for sigma in GAUSS_SIGMAS:
        rec("codec+gauss", sigma, add_gaussian_noise(x_hat_clean, sigma))

    # 3) Pure Gaussian noise (no codec)
    for sigma in GAUSS_SIGMAS:
        rec("gauss_only", sigma, add_gaussian_noise(x_cpu, sigma))

    # 4) Quantization (reduce bit depth)
    for bits in QUANT_BITS:
        rec("quantization", f"{bits}bit", quantize_to_bits(x_cpu, bits))

    # 5) JPEG re-compression
    for jq in JPEG_QUALITIES:
        rec("jpeg_recomp", f"q{jq}", jpeg_recompress(x_cpu, jq))

    return rows


# ═══════════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "mathtext.fontset": "cm",
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8,
})

MODEL_COLORS = {
    "cheng2020-anchor": "#1b9e77", "cheng2020-attn": "#d95f02",
    "mbt2018-mean": "#7570b3", "mbt2018": "#e7298a",
    "bmshj2018-hyperprior": "#66a61e", "bmshj2018-factorized": "#e6ab02",
    "ftic": "#e41a1c", "tcm": "#984ea3",
    "jpeg": "#a6761d", "webp": "#666666", "jpegxl": "#1f78b4",
}
MODEL_MARKERS = {
    "cheng2020-anchor": "o", "cheng2020-attn": "s",
    "mbt2018-mean": "D", "mbt2018": "^",
    "bmshj2018-hyperprior": "v", "bmshj2018-factorized": "<",
    "ftic": "h", "tcm": "p",
    "jpeg": "P", "webp": "X", "jpegxl": "*",
}

DIST_COLORS = {
    "codec_clean": "#2ca02c", "codec+gauss": "#1f77b4",
    "gauss_only": "#ff7f0e", "quantization": "#9467bd",
    "jpeg_recomp": "#d62728",
}
DIST_MARKERS = {
    "codec_clean": "s", "codec+gauss": "o", "gauss_only": "^",
    "quantization": "D", "jpeg_recomp": "P",
}
DIST_LABELS = {
    "codec_clean": "Codec (clean)", "codec+gauss": "Codec + Gaussian",
    "gauss_only": "Gaussian only", "quantization": "Quantization",
    "jpeg_recomp": "JPEG recomp.",
}


def make_figures(df: pd.DataFrame, out_dir: Path, logger):
    """Generate publication-quality figures from full results."""
    n_img = len(df["image"].unique())
    models_in_data = df["model"].unique().tolist()

    df_clean = df[df["distortion"] == "codec_clean"]

    # ════════════════════════════════════════════════════════════════════
    # Fig 1 (MAIN PAPER FIGURE): PSNR vs L — clean codec, all models
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models_in_data:
        sub = df_clean[df_clean["model"] == m]
        ax.scatter(sub["psnr"], sub["Le"],
                   c=MODEL_COLORS.get(m, "gray"),
                   marker=MODEL_MARKERS.get(m, "o"),
                   s=40, alpha=0.75, edgecolors="black", lw=0.3,
                   label=m, zorder=5)
    rho_s, _ = spearmanr(df_clean["psnr"], df_clean["Le"])
    rho_p, _ = pearsonr(df_clean["psnr"], df_clean["Le"])
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel(r"$\tilde{\mathcal{L}}(X, \hat{X})$")
    ax.set_title(f"Codec Clean — {n_img} Kodak images, $q={QUALITY}$\n"
                 f"Spearman $\\rho_s$={rho_s:.3f}, Pearson $r$={rho_p:.3f}")
    ax.set_ylim(-0.02, max(0.6, df_clean["Le"].max() * 1.1))
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(True, ls=":", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_clean_psnr_vs_Le.pdf", dpi=300)
    fig.savefig(out_dir / "fig1_clean_psnr_vs_Le.png", dpi=300)
    plt.close(fig)
    logger.info("  fig1_clean_psnr_vs_Le — main scatter (clean codec)")

    # ════════════════════════════════════════════════════════════════════
    # Fig 2: Model-level means (clean codec) — bar chart
    # ════════════════════════════════════════════════════════════════════
    mm = df_clean.groupby("model").agg(
        psnr_m=("psnr", "mean"), psnr_s=("psnr", "std"),
        Le_m=("Le", "mean"), Le_s=("Le", "std")).reset_index()
    mm = mm.sort_values("psnr_m", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x_pos = np.arange(len(mm))
    colors = [MODEL_COLORS.get(m, "gray") for m in mm["model"]]

    ax1.bar(x_pos, mm["psnr_m"], yerr=mm["psnr_s"], color=colors,
            edgecolor="black", lw=0.4, alpha=0.8, capsize=3)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(mm["model"], rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("Mean PSNR per model (clean codec)")
    ax1.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)

    ax2.bar(x_pos, mm["Le_m"], yerr=mm["Le_s"], color=colors,
            edgecolor="black", lw=0.4, alpha=0.8, capsize=3)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(mm["model"], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel(r"$\tilde{\mathcal{L}}(X, \hat{X})$")
    ax2.set_title(r"Mean $\tilde{\mathcal{L}}$ per model (clean codec)")
    ax2.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)

    fig.suptitle(f"Kodak {n_img} images — $q={QUALITY}$, clean compression", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_model_bars.pdf", dpi=300)
    fig.savefig(out_dir / "fig2_model_bars.png", dpi=300)
    plt.close(fig)
    logger.info("  fig2_model_bars — per-model mean ± std")

    # ════════════════════════════════════════════════════════════════════
    # Fig 3: Box plot per model (clean codec) — ordered by PSNR
    # ════════════════════════════════════════════════════════════════════
    order = mm["model"].tolist()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data_boxes = [df_clean[df_clean["model"] == m]["Le"].values for m in order]
    bp = ax.boxplot(data_boxes, positions=range(len(order)), widths=0.6,
                    patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for i, (patch, m) in enumerate(zip(bp["boxes"], order)):
        patch.set_facecolor(MODEL_COLORS.get(m, "lightgray"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$\tilde{\mathcal{L}}(X, \hat{X})$")
    ax.set_title(f"Spectral leakage coupling — {n_img} Kodak, $q={QUALITY}$ (ordered by PSNR ↓)")
    ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_boxplot_models.pdf", dpi=300)
    fig.savefig(out_dir / "fig3_boxplot_models.png", dpi=300)
    plt.close(fig)
    logger.info("  fig3_boxplot_models — clean codec box plot per model")

    # ════════════════════════════════════════════════════════════════════
    # Fig 4: Heatmap Image × Model (clean codec)
    # ════════════════════════════════════════════════════════════════════
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
    piv_psnr = df_clean.pivot_table(index="image", columns="model", values="psnr")
    piv_le = df_clean.pivot_table(index="image", columns="model", values="Le")
    co = piv_psnr.mean().sort_values(ascending=False).index.tolist()
    piv_psnr, piv_le = piv_psnr[co], piv_le[co]
    im1 = a1.imshow(piv_psnr.values, aspect="auto", cmap="RdYlGn")
    a1.set_xticks(range(len(co)))
    a1.set_xticklabels(co, rotation=45, ha="right", fontsize=7)
    a1.set_yticks(range(len(piv_psnr.index)))
    a1.set_yticklabels(piv_psnr.index, fontsize=6)
    a1.set_title("PSNR (dB)")
    plt.colorbar(im1, ax=a1, shrink=0.8)
    im2 = a2.imshow(piv_le.values, aspect="auto", cmap="RdYlGn_r")
    a2.set_xticks(range(len(co)))
    a2.set_xticklabels(co, rotation=45, ha="right", fontsize=7)
    a2.set_yticks(range(len(piv_le.index)))
    a2.set_yticklabels(piv_le.index, fontsize=6)
    a2.set_title(r"$\tilde{\mathcal{L}}(X, \hat{X})$")
    plt.colorbar(im2, ax=a2, shrink=0.8)
    fig.suptitle(f"Kodak × Models — $q={QUALITY}$ (clean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_heatmap.pdf", dpi=300)
    fig.savefig(out_dir / "fig4_heatmap.png", dpi=300)
    plt.close(fig)
    logger.info("  fig4_heatmap — image × model PSNR and L")

    # ════════════════════════════════════════════════════════════════════
    # Fig 5: Distortion consistency — for one reference model (cheng2020-anchor)
    #   Shows L vs PSNR for all distortion types, with clean baseline
    # ════════════════════════════════════════════════════════════════════
    ref_model = "cheng2020-anchor"
    if ref_model in models_in_data:
        df_ref = df[df["model"] == ref_model]
        ref_clean = df_ref[df_ref["distortion"] == "codec_clean"]
        bl_Le = ref_clean["Le"].mean()
        bl_psnr = ref_clean["psnr"].mean()

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.axhline(bl_Le, color="#2ca02c", ls="--", lw=1.2, alpha=0.7,
                    label=f"Clean baseline L={bl_Le:.3f}")

        for dist in ["codec_clean", "codec+gauss", "gauss_only",
                      "quantization", "jpeg_recomp"]:
            sub = df_ref[df_ref["distortion"] == dist]
            if sub.empty:
                continue
            grp = sub.groupby("param").agg(
                pm=("psnr", "mean"), ps=("psnr", "std"),
                lm=("Le", "mean"), ls_=("Le", "std")).reset_index()
            ax.errorbar(grp["pm"], grp["lm"],
                        xerr=grp["ps"].fillna(0), yerr=grp["ls_"].fillna(0),
                        fmt=DIST_MARKERS.get(dist, "o"), ms=7,
                        color=DIST_COLORS.get(dist, "gray"),
                        capsize=2, capthick=0.8, elinewidth=0.7,
                        label=DIST_LABELS.get(dist, dist), zorder=5)

        rho_s_r, _ = spearmanr(df_ref["psnr"], df_ref["Le"])
        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel(r"$\tilde{\mathcal{L}}(X, \hat{X})$")
        ax.set_title(f"Metric consistency across distortion types — {ref_model}\n"
                     f"{n_img} Kodak, $q={QUALITY}$, Spearman $\\rho_s$={rho_s_r:.3f}")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        ax.grid(True, ls=":", lw=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "fig5_distortion_consistency.pdf", dpi=300)
        fig.savefig(out_dir / "fig5_distortion_consistency.png", dpi=300)
        plt.close(fig)
        logger.info("  fig5_distortion_consistency — all distortions for ref model")

    # ════════════════════════════════════════════════════════════════════
    # Fig 6: Monotonicity — L curves per distortion type (ref model)
    # ════════════════════════════════════════════════════════════════════
    if ref_model in models_in_data:
        df_ref = df[df["model"] == ref_model]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes = axes.flat
        for ax, dist in zip(axes, ["codec+gauss", "gauss_only",
                                    "quantization", "jpeg_recomp"]):
            sub = df_ref[df_ref["distortion"] == dist]
            if sub.empty:
                ax.set_visible(False)
                continue
            for img_name in sub["image"].unique():
                ss = sub[sub["image"] == img_name].sort_values("psnr", ascending=False)
                ax.plot(ss["psnr"], ss["Le"], "-", alpha=0.25, lw=0.7, color="gray")
            grp = sub.groupby("param").agg(
                pm=("psnr", "mean"), lm=("Le", "mean")).reset_index()
            grp = grp.sort_values("pm", ascending=False)
            ax.plot(grp["pm"], grp["lm"], "o-",
                    color=DIST_COLORS.get(dist), ms=7, lw=2, label="Mean")
            for _, r in grp.iterrows():
                ax.annotate(r["param"], (r["pm"], r["lm"]),
                            fontsize=7, xytext=(5, 5), textcoords="offset points")
            ax.set_xlabel("PSNR (dB)")
            ax.set_ylabel(r"$\tilde{\mathcal{L}}$")
            ax.set_title(DIST_LABELS.get(dist, dist),
                         color=DIST_COLORS.get(dist, "black"))
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, ls=":", lw=0.3, alpha=0.5)
            ax.legend(fontsize=8)

        fig.suptitle(f"Monotonicity: lower PSNR → higher Leakage ({ref_model})",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "fig6_monotonicity.pdf", dpi=300)
        fig.savefig(out_dir / "fig6_monotonicity.png", dpi=300)
        plt.close(fig)
        logger.info("  fig6_monotonicity — per-distortion monotonicity curves")

    # ════════════════════════════════════════════════════════════════════
    # Fig 7: Model-level scatter — mean PSNR vs mean L (clean codec)
    # ════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, r in mm.iterrows():
        m = r["model"]
        ax.scatter(r["psnr_m"], r["Le_m"],
                   c=MODEL_COLORS.get(m), marker=MODEL_MARKERS.get(m, "o"),
                   s=100, edgecolors="black", lw=0.6, zorder=5)
        ax.annotate(m, (r["psnr_m"], r["Le_m"]), fontsize=7,
                    xytext=(6, 6), textcoords="offset points")
    rs_m, _ = spearmanr(mm["psnr_m"], mm["Le_m"])
    rp_m, _ = pearsonr(mm["psnr_m"], mm["Le_m"])
    ax.set_xlabel("Mean PSNR (dB)")
    ax.set_ylabel(r"Mean $\tilde{\mathcal{L}}(X, \hat{X})$")
    ax.set_title(f"Model-level correlation\n"
                 f"Spearman $\\rho_s$={rs_m:.3f}, Pearson $r$={rp_m:.3f}")
    ax.grid(True, ls=":", lw=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_model_scatter.pdf", dpi=300)
    fig.savefig(out_dir / "fig7_model_scatter.png", dpi=300)
    plt.close(fig)
    logger.info("  fig7_model_scatter — model-level means correlation")


def generate_latex_table(df: pd.DataFrame, out_dir: Path, logger):
    """Generate LaTeX table for paper."""
    df_clean = df[df["distortion"] == "codec_clean"]
    mm = df_clean.groupby("model").agg(
        psnr_m=("psnr", "mean"), psnr_s=("psnr", "std"),
        Le_m=("Le", "mean"), Le_s=("Le", "std"),
        n=("Le", "count")).reset_index()
    mm = mm.sort_values("psnr_m", ascending=False)

    rho_s, _ = spearmanr(df_clean["psnr"], df_clean["Le"])

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Spectral Leakage Coupling $\tilde{\mathcal{L}}(X, \hat{X})$ for clean codec compression "
        f"on Kodak dataset ($q={QUALITY}$, $512\\times512$). "
        f"Spearman $\\rho_s = {rho_s:.3f}$.}}",
        r"  \label{tab:leakage_results}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Model & PSNR (dB) & $\tilde{\mathcal{L}}(X, \hat{X})$ & $\Delta$PSNR & N \\",
        r"    \midrule",
    ]

    best_psnr = mm["psnr_m"].max()
    for _, r in mm.iterrows():
        name = r["model"].replace("_", r"\_").replace("-", r"-")
        psnr_str = f"{r['psnr_m']:.2f} $\\pm$ {r['psnr_s']:.2f}"
        le_str = f"{r['Le_m']:.4f} $\\pm$ {r['Le_s']:.4f}"
        delta = r["psnr_m"] - best_psnr
        delta_str = f"{delta:+.2f}" if abs(delta) > 0.01 else "---"
        lines.append(f"    {name} & {psnr_str} & {le_str} & {delta_str} & {int(r['n'])} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(lines)
    (out_dir / "table_results.tex").write_text(tex)
    logger.info(f"  LaTeX table saved to {out_dir / 'table_results.tex'}")

    # Also generate distortion consistency table
    ref_model = "cheng2020-anchor"
    df_ref = df[df["model"] == ref_model]
    summary = df_ref.groupby(["distortion", "param"]).agg(
        psnr_m=("psnr", "mean"), Le_m=("Le", "mean")).reset_index()
    summary = summary.sort_values("psnr_m", ascending=False)

    bl = summary[summary["distortion"] == "codec_clean"]["Le_m"].values[0]

    lines2 = [
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{Metric consistency across distortion types ({ref_model}, $q={QUALITY}$).}}",
        r"  \label{tab:distortion_consistency}",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Distortion & Param & PSNR (dB) & $\tilde{\mathcal{L}}$ & $\tilde{\mathcal{L}}/\tilde{\mathcal{L}}_0$ \\",
        r"    \midrule",
    ]
    for _, r in summary.iterrows():
        dist_name = DIST_LABELS.get(r["distortion"], r["distortion"])
        ratio = r["Le_m"] / bl if bl > 0 else 0
        tag = " $\\leftarrow$ baseline" if r["distortion"] == "codec_clean" else ""
        lines2.append(
            f"    {dist_name} & {r['param']} & {r['psnr_m']:.2f} "
            f"& {r['Le_m']:.4f} & {ratio:.1f}$\\times${tag} \\\\"
        )
    lines2 += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tex2 = "\n".join(lines2)
    (out_dir / "table_distortion.tex").write_text(tex2)
    logger.info(f"  LaTeX distortion table saved")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--single", action="store_true",
                     help="Quick test: 1 image, 1 model")
    grp.add_argument("--full", action="store_true",
                     help="Full: 24 images, all models")
    args = parser.parse_args()

    mode = "single" if args.single else "full"
    out_dir = Path(f"results/comprehensive_eval_{mode}")
    logger = setup_logging(out_dir)

    images = sorted(KODAK.glob("kodim*.png"))
    assert len(images) == 24, f"Expected 24 images, found {len(images)}"

    models_to_run = MODELS if mode == "full" else [MODELS[0]]
    if mode == "single":
        images = images[:1]

    total_model_images = len(models_to_run) * len(images)
    total_cases = total_model_images * N_DISTORTIONS_PER_IMAGE

    logger.info(f"{'='*70}")
    logger.info(f"Comprehensive Leakage Evaluation — {mode.upper()}")
    logger.info(f"Models: {len(models_to_run)} — {', '.join(models_to_run)}")
    logger.info(f"Images: {len(images)}, q={QUALITY}, size={SIZE}")
    logger.info(f"Distortions per image: {N_DISTORTIONS_PER_IMAGE}")
    logger.info(f"Total cases: {total_cases}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"{'='*70}\n")

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.time()
    all_rows = []
    model_img_done = 0

    for mi, mname in enumerate(models_to_run):
        logger.info(f"{'━'*70}")
        logger.info(f"MODEL [{mi+1}/{len(models_to_run)}]: {mname}")
        logger.info(f"{'━'*70}")
        sys.stdout.flush()

        try:
            model = load_model(mname, QUALITY, DEVICE,
                               p=TCM_P if mname == "tcm" else 128)
        except Exception as e:
            logger.error(f"  SKIP model {mname}: {e}")
            model_img_done += len(images)
            continue

        is_trad = mname in TRADITIONAL
        if not is_trad:
            model.eval()

        L_k = get_leakage_profile(model, mname, out_dir, logger)
        t_model = time.time()

        for ii, img_path in enumerate(images):
            t_img = time.time()
            rows = evaluate_image(img_path, model, mname, L_k, is_trad, logger,
                                  save_dir=img_dir)
            all_rows.extend(rows)
            model_img_done += 1
            dt_img = time.time() - t_img

            # Progress and ETA
            elapsed = time.time() - t_global
            rate = model_img_done / elapsed if elapsed > 0 else 0
            remaining = (total_model_images - model_img_done) / rate if rate > 0 else 0
            psnr_clean = rows[0]["psnr"]  # first row is always codec_clean
            le_clean = rows[0]["Le"]

            if (ii + 1) % 4 == 0 or ii == len(images) - 1:
                logger.info(
                    f"  [{ii+1:2d}/{len(images)}] {img_path.stem}  "
                    f"PSNR={psnr_clean:.1f} L={le_clean:.4f}  "
                    f"({dt_img:.1f}s)  "
                    f"[{model_img_done}/{total_model_images} done, "
                    f"ETA {remaining/60:.1f}min]"
                )
                sys.stdout.flush()

            # Intermediate save every 24 images (each model completion)
            if (ii + 1) == len(images):
                pd.DataFrame(all_rows).to_csv(out_dir / "results.csv", index=False)

        dt_model = time.time() - t_model
        logger.info(f"  Model {mname} done in {dt_model:.0f}s\n")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    t_total = time.time() - t_global

    # Save final results
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # ── Summary tables ──
    df_clean = df[df["distortion"] == "codec_clean"]

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS — Clean Codec Compression")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<25s} {'PSNR (dB)':>12s}  {'L(X,X̂)':>12s}  {'n':>4s}")
    logger.info(f"{'-'*70}")

    mm = df_clean.groupby("model").agg(
        pm=("psnr", "mean"), ps=("psnr", "std"),
        lm=("Le", "mean"), ls_=("Le", "std"),
        n=("Le", "count")).reset_index().sort_values("pm", ascending=False)

    for _, r in mm.iterrows():
        logger.info(f"  {r['model']:<23s} {r['pm']:5.2f}±{r['ps']:.2f}  "
                    f"{r['lm']:.4f}±{r['ls_']:.4f}  {int(r['n']):>4d}")

    rho_s, _ = spearmanr(df_clean["psnr"], df_clean["Le"])
    rho_p, _ = pearsonr(df_clean["psnr"], df_clean["Le"])
    logger.info(f"\nCorrelation (clean codec, N={len(df_clean)}):")
    logger.info(f"  Spearman ρ = {rho_s:.4f}")
    logger.info(f"  Pearson  r = {rho_p:.4f}")

    # Distortion consistency for reference model
    ref = "cheng2020-anchor"
    if ref in df["model"].unique():
        df_ref = df[df["model"] == ref]
        bl = df_ref[df_ref["distortion"] == "codec_clean"]["Le"].mean()

        logger.info(f"\n{'='*70}")
        logger.info(f"DISTORTION CONSISTENCY — {ref}")
        logger.info(f"Baseline (clean): L = {bl:.4f}")
        logger.info(f"{'='*70}")
        logger.info(f"{'Distortion':<22} {'Param':>8}  {'PSNR':>8}  {'L':>8}  {'L/L₀':>6}")
        logger.info(f"{'-'*60}")

        summary = df_ref.groupby(["distortion", "param"]).agg(
            pm=("psnr", "mean"), lm=("Le", "mean")).reset_index()
        summary = summary.sort_values("pm", ascending=False)
        for _, r in summary.iterrows():
            ratio = r["lm"] / bl if bl > 0 else 0
            tag = " <-- baseline" if r["distortion"] == "codec_clean" else ""
            logger.info(f"  {r['distortion']:<20s} {r['param']:>8s}  "
                        f"{r['pm']:>7.2f}  {r['lm']:>8.4f}  {ratio:>5.1f}×{tag}")

        rho_all, _ = spearmanr(df_ref["psnr"], df_ref["Le"])
        logger.info(f"\n  Overall Spearman ρ (all distortions, {ref}): {rho_all:.4f}")

    # Generate figures and LaTeX
    logger.info(f"\n{'='*70}")
    logger.info("Generating figures...")
    make_figures(df, out_dir, logger)

    logger.info("\nGenerating LaTeX tables...")
    generate_latex_table(df, out_dir, logger)

    logger.info(f"\n{'='*70}")
    logger.info(f"ALL DONE — {t_total:.0f}s ({t_total/60:.1f} min)")
    logger.info(f"Results CSV: {out_dir}/results.csv")
    logger.info(f"Figures:     {out_dir}/fig*.pdf")
    logger.info(f"LaTeX:       {out_dir}/table_*.tex")
    logger.info(f"Images:      {img_dir}/")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
