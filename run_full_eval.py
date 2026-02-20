#!/usr/bin/env python3
"""Full evaluation: Spectral Leakage Coupling.

Part 1: Clean compression — all 24 Kodak × 9 models × q=6 × 512
Part 2: T-MLA adversarial — cheng2020-anchor × all 24 Kodak × q=6 × 512

Metric: L̃(X, X̂) = (1/N) Σ ρ(f)·L_k(f),  ρ(f) = D_f/(S_f+D_f) ∈ [0,1]
"""
import gc
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.loaders import load_model
from utils.functions import (
    compute_radial_spectrum, compute_radial_distortion,
    evaluate_frequency_response,
)
from tmla.attacks import maxdistortion_logexp_multiscale

DEVICE = torch.device("cuda")
QUALITY = 6
SIZE = 512
NUM_BINS = 512
KODAK = Path("kodak")
OUT = Path("results/full_leakage_eval")
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT / "images"
IMG_DIR.mkdir(exist_ok=True)

MODELS = [
    "cheng2020-anchor",
    "cheng2020-attn",
    "mbt2018-mean",
    "mbt2018",
    "bmshj2018-hyperprior",
    "bmshj2018-factorized",
    "jpeg",
    "webp",
    "jpegxl",
]
TMLA_MODEL = "cheng2020-anchor"

IMAGES = sorted(KODAK.glob("kodim*.png"))
assert len(IMAGES) == 24, f"Expected 24 Kodak images, found {len(IMAGES)}"

eps = 1e-12


def compute_Le(orig_gray, recon_gray, L_k):
    freqs, S_f, _ = compute_radial_spectrum(orig_gray, num_bins=NUM_BINS)
    _, D_f, _ = compute_radial_distortion(orig_gray, recon_gray, num_bins=NUM_BINS)
    rho = D_f / (S_f + D_f + eps)
    k_axis = np.arange(len(L_k), dtype=np.float64)
    L_k_radial = np.interp(freqs, k_axis, L_k)
    return float(np.mean(rho * L_k_radial))


def save_img(tensor, path):
    arr = (tensor.squeeze().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def get_leakage_profile(model, model_name):
    lk_path = OUT / f"Lk_{model_name}_q{QUALITY}_s{SIZE}.npy"
    if lk_path.exists():
        return np.load(lk_path)
    _, _, metrics = evaluate_frequency_response(
        model, size=SIZE, device=DEVICE,
        show_plots=False, num_runs=3, show_metric_plots=False, seed=42, verbose=False,
    )
    L_k = metrics["leakage"]
    np.save(lk_path, L_k)
    return L_k


class PaddedModel(torch.nn.Module):
    def __init__(self, m, ph, pw, H, W):
        super().__init__()
        self.model = m
        self.ph, self.pw, self.H, self.W = ph, pw, H, W
        self.model.train()
    def forward(self, x):
        x_pad = F.pad(x, (0, self.pw, 0, self.ph), value=0.0)
        out = self.model(x_pad)
        out["x_hat"] = out["x_hat"][:, :, :self.H, :self.W]
        return out


def pad_params(H, W, mult=64):
    th = max(256, -(-H // mult) * mult)
    tw = max(256, -(-W // mult) * mult)
    return th - H, tw - W


# ════════════════════════════════════════════════════════════════════════════
# PART 1: Clean compression — all models × all images
# ════════════════════════════════════════════════════════════════════════════
print(f"{'='*80}")
print(f"  PART 1: Clean compression")
print(f"  {len(MODELS)} models × {len(IMAGES)} images, q={QUALITY}, size={SIZE}")
print(f"{'='*80}")
sys.stdout.flush()

rows_clean = []
t0 = time.time()

for mi, mname in enumerate(MODELS):
    print(f"\n[{mi+1}/{len(MODELS)}] {mname}")
    sys.stdout.flush()

    try:
        model = load_model(mname, QUALITY, DEVICE)
    except Exception as e:
        print(f"  SKIP: {e}")
        continue

    L_k = get_leakage_profile(model, mname)
    is_trad = mname in ("jpeg", "webp", "jpegxl", "jpeg2000")
    if not is_trad:
        model.eval()

    for ii, img_path in enumerate(IMAGES):
        img = Image.open(img_path).convert("RGB")
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        _, _, H, W = x.shape
        ph, pw = pad_params(H, W)

        if is_trad:
            out = model(x)
            x_hat = out["x_hat"].to(DEVICE)[:, :, :H, :W].clamp(0, 1).cpu()
        else:
            with torch.no_grad():
                out = model(F.pad(x, (0, pw, 0, ph), value=0.0))
                x_hat = out["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()

        x_cpu = x.cpu()
        psnr = -10 * np.log10(F.mse_loss(x_hat, x_cpu).item() + eps)
        Le = compute_Le(x_cpu.squeeze().mean(0).numpy(), x_hat.squeeze().mean(0).numpy(), L_k)

        mdir = IMG_DIR / mname
        mdir.mkdir(exist_ok=True)
        save_img(x_cpu, mdir / f"{img_path.stem}_orig.png")
        save_img(x_hat, mdir / f"{img_path.stem}_recon.png")

        rows_clean.append(dict(model=mname, image=img_path.stem, mode="clean",
                               psnr=psnr, Le=Le))

        if (ii + 1) % 8 == 0 or ii == 23:
            print(f"  [{ii+1:2d}/24] {img_path.stem} PSNR={psnr:.1f} L={Le:.4f}")
            sys.stdout.flush()

    del model; torch.cuda.empty_cache(); gc.collect()

t1 = time.time()
print(f"\nPart 1 done in {t1 - t0:.0f}s")

# ════════════════════════════════════════════════════════════════════════════
# PART 2: T-MLA adversarial — cheng2020-anchor × all images
# ════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print(f"  PART 2: T-MLA adversarial attack")
print(f"  {TMLA_MODEL} × {len(IMAGES)} images, q={QUALITY}")
print(f"{'='*80}")
sys.stdout.flush()

rows_adv = []
model = load_model(TMLA_MODEL, QUALITY, DEVICE)
L_k = get_leakage_profile(model, TMLA_MODEL)

adv_dir = IMG_DIR / f"{TMLA_MODEL}_tmla"
adv_dir.mkdir(exist_ok=True)

for ii, img_path in enumerate(IMAGES):
    print(f"\n  [{ii+1}/24] {img_path.stem} — T-MLA attack...")
    sys.stdout.flush()

    img = Image.open(img_path).convert("RGB")
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    _, _, H, W = x.shape
    ph, pw = pad_params(H, W)

    # Clean roundtrip for this model (already computed, but need tensors for comparison)
    model.eval()
    with torch.no_grad():
        out_c = model(F.pad(x, (0, pw, 0, ph), value=0.0))
        x_hat_clean = out_c["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()

    # T-MLA attack
    model.train()
    wrapped = PaddedModel(model, ph, pw, H, W)

    perturbed, _, _, _ = maxdistortion_logexp_multiscale(
        x=x, errbound=0.1, losstype="psnr", num_iterations=1000,
        model=wrapped, device=DEVICE, scales=2, learningrate=0.05,  # 1000 iters
        keep_perturbation_targeted=False, keep_low_outcomequality=False,
    )
    x_adv = perturbed.clamp(0, 1).detach()

    model.eval()
    with torch.no_grad():
        out_a = model(F.pad(x_adv, (0, pw, 0, ph), value=0.0))
        x_hat_adv = out_a["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()

    x_cpu = x.cpu()
    psnr_adv = -10 * np.log10(F.mse_loss(x_hat_adv, x_cpu).item() + eps)
    Le_adv = compute_Le(x_cpu.squeeze().mean(0).numpy(), x_hat_adv.squeeze().mean(0).numpy(), L_k)

    save_img(x_adv.cpu(), adv_dir / f"{img_path.stem}_adv_input.png")
    save_img(x_hat_adv, adv_dir / f"{img_path.stem}_adv_recon.png")

    rows_adv.append(dict(model=TMLA_MODEL, image=img_path.stem, mode="tmla",
                         psnr=psnr_adv, Le=Le_adv))

    # Also get clean values for this model
    psnr_c = -10 * np.log10(F.mse_loss(x_hat_clean, x_cpu).item() + eps)
    Le_c = compute_Le(x_cpu.squeeze().mean(0).numpy(), x_hat_clean.squeeze().mean(0).numpy(), L_k)
    print(f"    Clean: PSNR={psnr_c:.1f} L={Le_c:.4f}  |  T-MLA: PSNR={psnr_adv:.1f} L={Le_adv:.4f}")
    sys.stdout.flush()

del model; torch.cuda.empty_cache(); gc.collect()
t2 = time.time()
print(f"\nPart 2 done in {t2 - t1:.0f}s")

# ════════════════════════════════════════════════════════════════════════════
# Save results & generate figures
# ════════════════════════════════════════════════════════════════════════════
df_clean = pd.DataFrame(rows_clean)
df_adv = pd.DataFrame(rows_adv)
df_all = pd.concat([df_clean, df_adv], ignore_index=True)
df_all.to_csv(OUT / "results.csv", index=False)
df_clean.to_csv(OUT / "results_clean.csv", index=False)
df_adv.to_csv(OUT / "results_tmla.csv", index=False)
print(f"\nSaved CSVs to {OUT}")

# Summary table
print(f"\n{'='*70}")
print("CLEAN — per-model averages:")
print(df_clean.groupby("model")[["psnr", "Le"]].agg(["mean", "std"]).round(4).to_string())

print(f"\nT-MLA ({TMLA_MODEL}) averages:")
anchor_clean = df_clean[df_clean["model"] == TMLA_MODEL]
print(f"  Clean:  PSNR={anchor_clean['psnr'].mean():.2f}±{anchor_clean['psnr'].std():.2f}  L={anchor_clean['Le'].mean():.4f}±{anchor_clean['Le'].std():.4f}")
print(f"  T-MLA:  PSNR={df_adv['psnr'].mean():.2f}±{df_adv['psnr'].std():.2f}  L={df_adv['Le'].mean():.4f}±{df_adv['Le'].std():.4f}")

# Correlations
rho_s, _ = spearmanr(df_clean["psnr"], df_clean["Le"])
rho_p, _ = pearsonr(df_clean["psnr"], df_clean["Le"])
print(f"\nCorrelation (clean, all models, N={len(df_clean)}):")
print(f"  Spearman ρ = {rho_s:.4f}, Pearson r = {rho_p:.4f}")

# ── FIGURES ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "mathtext.fontset": "cm",
    "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 8,
})

MC = {
    "cheng2020-anchor": "#3182bd", "cheng2020-attn": "#2171b5",
    "mbt2018-mean": "#31a354", "mbt2018": "#74c476",
    "bmshj2018-hyperprior": "#756bb1", "bmshj2018-factorized": "#9e9ac8",
    "jpeg": "#e6550d", "webp": "#fd8d3c", "jpegxl": "#d62728",
}
MM = {
    "cheng2020-anchor": "o", "cheng2020-attn": "s",
    "mbt2018-mean": "D", "mbt2018": "^",
    "bmshj2018-hyperprior": "v", "bmshj2018-factorized": "<",
    "jpeg": "P", "webp": "X", "jpegxl": "*",
}

# Fig 1: PSNR vs L — all clean + T-MLA adversarial
fig, ax = plt.subplots(figsize=(7, 5))
for m in MODELS:
    sub = df_clean[df_clean["model"] == m]
    ax.scatter(sub["psnr"], sub["Le"], c=MC.get(m), marker=MM.get(m, "o"),
               s=35, alpha=0.7, edgecolors="black", lw=0.3, label=m, zorder=5)
ax.scatter(df_adv["psnr"], df_adv["Le"], c="red", marker="x", s=50, lw=1.5,
           label=f"T-MLA ({TMLA_MODEL})", zorder=6)
ax.set_xlabel("PSNR (dB)")
ax.set_ylabel(r"$\mathcal{L}(X, \hat{X}) \in [0,\,1]$")
ax.set_title(f"Normalized Weighted Leakage vs PSNR — Kodak 24, $q={QUALITY}$\n"
             f"Spearman $\\rho_s$={rho_s:.3f}, Pearson $r$={rho_p:.3f}")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
ax.grid(True, ls=":", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig1_psnr_vs_leakage.pdf", dpi=300)
fig.savefig(OUT / "fig1_psnr_vs_leakage.png", dpi=300)
plt.close(fig)

# Fig 2: Clean vs T-MLA paired comparison (cheng2020-anchor only)
fig, ax = plt.subplots(figsize=(6, 4))
ac = df_clean[df_clean["model"] == TMLA_MODEL].sort_values("image")
aa = df_adv.sort_values("image")
for _, (rc, ra) in enumerate(zip(ac.itertuples(), aa.itertuples())):
    ax.plot([rc.psnr, ra.psnr], [rc.Le, ra.Le], color="gray", lw=0.6, alpha=0.5, zorder=1)
ax.scatter(ac["psnr"], ac["Le"], c="#3182bd", s=50, zorder=5, edgecolors="black", lw=0.4,
           label=f"Clean (L={ac['Le'].mean():.3f})")
ax.scatter(aa["psnr"], aa["Le"], c="#cb181d", s=50, zorder=5, marker="X", edgecolors="black", lw=0.4,
           label=f"T-MLA (L={aa['Le'].mean():.3f})")
ax.set_xlabel("PSNR (dB)")
ax.set_ylabel(r"$\mathcal{L}(X, \hat{X})$")
ax.set_title(f"{TMLA_MODEL} $q={QUALITY}$ — Clean vs T-MLA, all 24 Kodak")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9)
ax.grid(True, ls=":", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_clean_vs_tmla.pdf", dpi=300)
fig.savefig(OUT / "fig2_clean_vs_tmla.png", dpi=300)
plt.close(fig)

# Fig 3: Box plot per model + T-MLA
fig, ax = plt.subplots(figsize=(8, 4))
order = df_clean.groupby("model")["psnr"].mean().sort_values(ascending=False).index.tolist()
order_plus = order + ["T-MLA"]
data_le = [df_clean[df_clean["model"] == m]["Le"].values for m in order]
data_le.append(df_adv["Le"].values)
bp = ax.boxplot(data_le, positions=range(len(order_plus)), widths=0.6,
                patch_artist=True, medianprops=dict(color="black", lw=1.5))
for i, (patch, m) in enumerate(zip(bp["boxes"], order_plus)):
    if m == "T-MLA":
        patch.set_facecolor("#cb181d")
    else:
        patch.set_facecolor(MC.get(m, "lightgray"))
    patch.set_alpha(0.6); patch.set_edgecolor("black")
ax.set_xticks(range(len(order_plus)))
ax.set_xticklabels(order_plus, rotation=35, ha="right", fontsize=8)
ax.set_ylabel(r"$\mathcal{L}(X, \hat{X})$")
ax.set_title(f"Leakage distribution — Kodak 24, $q={QUALITY}$ (T-MLA = {TMLA_MODEL} adversarial)")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_boxplot.pdf", dpi=300)
fig.savefig(OUT / "fig3_boxplot.png", dpi=300)
plt.close(fig)

# Fig 4: Heatmap image×model
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
piv_psnr = df_clean.pivot_table(index="image", columns="model", values="psnr")
piv_le = df_clean.pivot_table(index="image", columns="model", values="Le")
co = piv_psnr.mean().sort_values(ascending=False).index.tolist()
piv_psnr, piv_le = piv_psnr[co], piv_le[co]
im1 = a1.imshow(piv_psnr.values, aspect="auto", cmap="RdYlGn")
a1.set_xticks(range(len(co))); a1.set_xticklabels(co, rotation=45, ha="right", fontsize=7)
a1.set_yticks(range(24)); a1.set_yticklabels(piv_psnr.index, fontsize=6)
a1.set_title("PSNR (dB)"); plt.colorbar(im1, ax=a1, shrink=0.8)
im2 = a2.imshow(piv_le.values, aspect="auto", cmap="RdYlGn_r")
a2.set_xticks(range(len(co))); a2.set_xticklabels(co, rotation=45, ha="right", fontsize=7)
a2.set_yticks(range(24)); a2.set_yticklabels(piv_le.index, fontsize=6)
a2.set_title(r"$\mathcal{L}(X, \hat{X})$"); plt.colorbar(im2, ax=a2, shrink=0.8)
fig.suptitle(f"Kodak × Models — $q={QUALITY}$", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_heatmap.pdf", dpi=300)
fig.savefig(OUT / "fig4_heatmap.png", dpi=300)
plt.close(fig)

# Fig 5: Model-level means
fig, ax = plt.subplots(figsize=(5, 4))
mm = df_clean.groupby("model")[["psnr", "Le"]].mean()
for m in mm.index:
    ax.scatter(mm.loc[m, "psnr"], mm.loc[m, "Le"], c=MC.get(m), marker=MM.get(m, "o"),
               s=100, edgecolors="black", lw=0.6, zorder=5)
    ax.annotate(m, (mm.loc[m, "psnr"], mm.loc[m, "Le"]), fontsize=6.5, xytext=(5, 5),
                textcoords="offset points")
ax.scatter(df_adv["psnr"].mean(), df_adv["Le"].mean(), c="#cb181d", marker="X",
           s=120, edgecolors="black", lw=0.8, zorder=6)
ax.annotate("T-MLA", (df_adv["psnr"].mean(), df_adv["Le"].mean()),
            fontsize=7, xytext=(5, 5), textcoords="offset points", color="red", fontweight="bold")
rs_m, _ = spearmanr(mm["psnr"], mm["Le"])
ax.set_xlabel("Mean PSNR (dB)")
ax.set_ylabel(r"Mean $\mathcal{L}(X, \hat{X})$")
ax.set_title(f"Model-level correlation (Spearman $\\rho$={rs_m:.3f})")
ax.set_ylim(-0.02, 1.02)
ax.grid(True, ls=":", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig5_model_means.pdf", dpi=300)
fig.savefig(OUT / "fig5_model_means.png", dpi=300)
plt.close(fig)

print(f"\n{'='*60}")
print(f"ALL DONE in {time.time() - t0:.0f}s")
print(f"Results: {OUT}")
print(f"{'='*60}")
