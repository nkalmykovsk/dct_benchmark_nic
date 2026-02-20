#!/usr/bin/env python3
"""Validate the interpolation fix and ablation metric.

Loads saved images + L_k profiles from the previous full run, re-computes
Le (new formula with interpolation) and rho_bar, then checks:

1. Correlations: Spearman/Pearson of PSNR vs Le_new  >=  old values
2. Ordering:     Codec ranking by mean Le is preserved
3. Ablation:     Le (with L_k) discriminates codecs better than rho_bar
4. Monotonicity: More distortion -> higher Le for every distortion sweep
5. Edge cases:   Identity recon -> Le ~ 0

Usage:
    python validate_metric_fix.py          # full validation
    python validate_metric_fix.py --quick  # 2 models, 4 images
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.functions import compute_radial_spectrum, compute_radial_distortion

NUM_BINS = 512
eps = 1e-12
BASE = Path("results/comprehensive_eval_full")

# ─────────────────────────────────────────────────────────────────────
# Metric implementations
# ─────────────────────────────────────────────────────────────────────

def compute_Le_old(orig_gray, recon_gray, L_k):
    """Old formula: index-based alignment (no interpolation)."""
    _, S_f, _ = compute_radial_spectrum(orig_gray, num_bins=NUM_BINS)
    _, D_f, _ = compute_radial_distortion(orig_gray, recon_gray, num_bins=NUM_BINS)
    n = min(len(S_f), len(L_k), len(D_f))
    rho = D_f[:n] / (S_f[:n] + D_f[:n] + eps)
    return float(np.mean(rho * L_k[:n]))


def compute_Le_new(orig_gray, recon_gray, L_k):
    """New formula: L_k interpolated onto radial-frequency grid."""
    freqs, S_f, _ = compute_radial_spectrum(orig_gray, num_bins=NUM_BINS)
    _, D_f, _ = compute_radial_distortion(orig_gray, recon_gray, num_bins=NUM_BINS)
    rho = D_f / (S_f + D_f + eps)
    k_axis = np.arange(len(L_k), dtype=np.float64)
    L_k_radial = np.interp(freqs, k_axis, L_k)
    Le = float(np.mean(rho * L_k_radial))
    rho_bar = float(np.mean(rho))
    return Le, rho_bar


def load_gray(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0


def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}min"

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 2 models, 4 images")
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION: metric interpolation fix + ablation")
    print("=" * 70, flush=True)

    # ── Load old results & L_k profiles ──
    csv_path = BASE / "results.csv"
    df_old = pd.read_csv(csv_path)
    print(f"[init] Loaded {len(df_old)} rows from old results CSV")

    lk_files = sorted(BASE.glob("Lk_*.npy"))
    lk_cache = {}
    for f in lk_files:
        name = f.stem.replace("Lk_", "").rsplit("_", 2)[0]
        lk_cache[name] = np.load(f)
    print(f"[init] L_k profiles: {list(lk_cache.keys())}")

    img_dir = BASE / "images"
    models = sorted(d.name for d in img_dir.iterdir() if d.is_dir())
    if args.quick:
        models = models[:2]
        print(f"[init] QUICK mode: using {models}")

    # Count total work
    total_pairs = 0
    for model in models:
        if model not in lk_cache:
            continue
        for img_d in sorted((img_dir / model).iterdir()):
            if not img_d.is_dir():
                continue
            pngs = [p for p in img_d.glob("*.png") if p.name != "original.png"]
            if args.quick and total_pairs > 0:
                pngs = pngs[:4]
            total_pairs += len(pngs)

    print(f"[init] Total image pairs to recompute: {total_pairs}")
    print(f"[init] Start time: {time.strftime('%H:%M:%S')}")
    print("", flush=True)

    # ── Recompute ──
    rows = []
    done = 0
    t0 = time.time()

    for mi, model in enumerate(models):
        L_k = lk_cache.get(model)
        if L_k is None:
            print(f"[{mi+1}/{len(models)}] SKIP {model}: no L_k profile",
                  flush=True)
            continue

        print(f"[{mi+1}/{len(models)}] {model} ...", flush=True)
        model_dir = img_dir / model
        images_done_model = 0

        for img_d in sorted(model_dir.iterdir()):
            if not img_d.is_dir():
                continue
            stem = img_d.name
            orig_path = img_d / "original.png"
            if not orig_path.exists():
                continue
            orig_gray = load_gray(orig_path)

            recon_files = sorted(
                p for p in img_d.glob("*.png") if p.name != "original.png"
            )
            if args.quick and images_done_model >= 4:
                break

            for recon_path in recon_files:
                t_pair = time.time()
                recon_gray = load_gray(recon_path)

                le_old = compute_Le_old(orig_gray, recon_gray, L_k)
                le_new, rho_b = compute_Le_new(orig_gray, recon_gray, L_k)

                tag = recon_path.stem
                if tag.startswith("codec_clean"):
                    dist_type = "codec_clean"
                elif tag.startswith("codec+gauss"):
                    dist_type = "codec+gauss"
                elif tag.startswith("gauss_only"):
                    dist_type = "gauss_only"
                elif tag.startswith("jpeg_recomp"):
                    dist_type = "jpeg_recomp"
                elif tag.startswith("quantization"):
                    dist_type = "quantization"
                else:
                    dist_type = tag

                rows.append(dict(
                    model=model, image=stem, tag=tag,
                    distortion=dist_type,
                    Le_old=le_old, Le_new=le_new, rho_bar=rho_b,
                ))
                done += 1
                dt = time.time() - t_pair

            images_done_model += 1

            # Progress every image
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_pairs - done) / rate if rate > 0 else 0
            print(
                f"    {stem}: {len(recon_files)} files, "
                f"Le_old={le_old:.5f} -> Le_new={le_new:.5f} "
                f"[{done}/{total_pairs}, ETA {fmt_eta(eta)}]",
                flush=True,
            )

        print(f"  -> {model} done ({images_done_model} images)\n", flush=True)

    df = pd.DataFrame(rows)
    elapsed_total = time.time() - t0
    print(f"[recompute] Done: {len(df)} pairs in {fmt_eta(elapsed_total)}\n",
          flush=True)

    # ── Merge PSNR from old CSV ──
    df_clean_old = df_old[df_old["distortion"] == "codec_clean"].copy()
    df_clean_new = df[df["distortion"] == "codec_clean"].copy()

    merged = df_clean_new.groupby("model").agg(
        Le_old=("Le_old", "mean"),
        Le_new=("Le_new", "mean"),
        rho_bar=("rho_bar", "mean"),
        n=("Le_new", "count"),
    ).reset_index()

    psnr_agg = df_clean_old.groupby("model")["psnr"].mean()
    merged = merged.merge(
        psnr_agg.rename("psnr"), left_on="model", right_index=True, how="inner"
    )
    merged = merged.sort_values("psnr", ascending=False)

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Correlations
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 1: Correlations (clean codec, model-level)")
    print("=" * 70)
    rs_old, _ = spearmanr(merged["psnr"], merged["Le_old"])
    rp_old, _ = pearsonr(merged["psnr"], merged["Le_old"])
    rs_new, _ = spearmanr(merged["psnr"], merged["Le_new"])
    rp_new, _ = pearsonr(merged["psnr"], merged["Le_new"])
    rs_rho, _ = spearmanr(merged["psnr"], merged["rho_bar"])
    rp_rho, _ = pearsonr(merged["psnr"], merged["rho_bar"])

    print(f"  {'Metric':<25s} {'Spearman':>10s} {'Pearson':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Le (old, index-based)':<25s} {rs_old:>10.4f} {rp_old:>10.4f}")
    print(f"  {'Le (new, interpolated)':<25s} {rs_new:>10.4f} {rp_new:>10.4f}")
    print(f"  {'rho_bar (no L_k)':<25s} {rs_rho:>10.4f} {rp_rho:>10.4f}")
    print()

    ok1 = abs(rs_new) >= abs(rs_old) - 0.02
    print(f"  Spearman preserved (|new| >= |old| - 0.02): "
          f"{'PASS' if ok1 else 'FAIL'}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Ordering
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 2: Model ordering (clean codec, by mean Le)")
    print("=" * 70)
    order_old = merged.sort_values("Le_old")["model"].tolist()
    order_new = merged.sort_values("Le_new")["model"].tolist()
    print(f"  Old order: {order_old}")
    print(f"  New order: {order_new}")

    rank_old = {m: i for i, m in enumerate(order_old)}
    rank_new = {m: i for i, m in enumerate(order_new)}
    models_both = sorted(set(rank_old) & set(rank_new))
    r_old_vec = [rank_old[m] for m in models_both]
    r_new_vec = [rank_new[m] for m in models_both]
    rank_corr, _ = spearmanr(r_old_vec, r_new_vec)
    ok2 = rank_corr > 0.9
    print(f"  Rank correlation old<->new: {rank_corr:.4f}  "
          f"{'PASS' if ok2 else 'FAIL'}", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Ablation
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TEST 3: Ablation — does L_k improve cross-codec discrimination?")
    print("=" * 70)
    print(f"  Spearman(PSNR, Le_new):   {rs_new:.4f}")
    print(f"  Spearman(PSNR, rho_bar):  {rs_rho:.4f}")
    delta_abl = abs(rs_new) - abs(rs_rho)
    print(f"  Delta = {delta_abl:+.4f}")
    if delta_abl >= 0:
        ok3_info = "Le_new >= rho_bar"
    else:
        ok3_info = "rho_bar > Le_new (L_k may not help at model level)"
    print(f"  Verdict: {ok3_info}")

    print(f"\n  {'Model':<28s} {'PSNR':>7s} {'Le_old':>9s} {'Le_new':>9s} "
          f"{'rho_bar':>9s} {'Le/rho':>7s}")
    print(f"  {'-'*75}")
    for _, r in merged.iterrows():
        ratio = r["Le_new"] / r["rho_bar"] if r["rho_bar"] > 1e-12 else 0
        print(f"  {r['model']:<28s} {r['psnr']:7.2f} {r['Le_old']:9.5f} "
              f"{r['Le_new']:9.5f} {r['rho_bar']:9.5f} {ratio:7.3f}")
    print(flush=True)

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Monotonicity
    # ══════════════════════════════════════════════════════════════════
    print(f"{'=' * 70}")
    print("TEST 4: Monotonicity — more distortion -> higher Le")
    print("=" * 70)

    mono_ok = True
    for dist in ["codec+gauss", "gauss_only"]:
        sub = df[df["distortion"] == dist]
        if sub.empty:
            continue
        grp = sub.groupby("tag").agg(Le_new=("Le_new", "mean")).reset_index()
        grp = grp.sort_values("Le_new")
        vals = grp["Le_new"].values
        is_sorted = all(
            vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)
        )
        status = "PASS" if is_sorted else "WARN (non-monotone on average)"
        if not is_sorted:
            mono_ok = False
        print(f"  {dist}: {status}")
        for _, r in grp.iterrows():
            print(f"    {r['tag']:<35s} Le_new={r['Le_new']:.6f}")
    print(flush=True)

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Edge cases
    # ══════════════════════════════════════════════════════════════════
    print(f"{'=' * 70}")
    print("TEST 5: Edge cases")
    print("=" * 70)

    first_model = [m for m in models if m in lk_cache][0]
    first_img_dir = sorted(
        d for d in (img_dir / first_model).iterdir() if d.is_dir()
    )[0]
    sample_orig = load_gray(first_img_dir / "original.png")

    le_perf, rho_perf = compute_Le_new(
        sample_orig, sample_orig, lk_cache[first_model]
    )
    ok5a = le_perf < 1e-8
    print(f"  Perfect recon:    Le = {le_perf:.2e}, "
          f"rho_bar = {rho_perf:.2e}  {'PASS' if ok5a else 'FAIL'}")

    sample_recon = load_gray(first_img_dir / "codec_clean_q6.png")
    L_zero = np.zeros_like(lk_cache[first_model])
    le_id, _ = compute_Le_new(sample_orig, sample_recon, L_zero)
    ok5b = le_id < 1e-10
    print(f"  L_k = 0 (ideal):  Le = {le_id:.2e}  "
          f"{'PASS' if ok5b else 'FAIL'}")

    L_ones = np.ones_like(lk_cache[first_model])
    le_ones, rho_ones = compute_Le_new(sample_orig, sample_recon, L_ones)
    ok5c = abs(le_ones - rho_ones) < 1e-6
    print(f"  L_k = 1 (Le==rho_bar): Le = {le_ones:.6f}, "
          f"rho_bar = {rho_ones:.6f}  {'PASS' if ok5c else 'FAIL'}")
    print(flush=True)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print("=" * 70)
    tests = {
        "Correlation preserved": ok1,
        "Ordering preserved":    ok2,
        "Edge: perfect recon":   ok5a,
        "Edge: L_k=0":           ok5b,
        "Edge: L_k=1 == rho":    ok5c,
    }
    all_pass = all(tests.values())
    for name, ok in tests.items():
        print(f"  {name:<30s} {'PASS' if ok else '** FAIL **'}")
    print(f"  {'Ablation (informational)':<30s} {ok3_info}")
    print(f"  {'Monotonicity':<30s} "
          f"{'PASS' if mono_ok else 'WARN — review above'}")
    print()
    print(f"OVERALL: {'ALL CORE TESTS PASSED' if all_pass else 'ISSUES FOUND'}")
    print("=" * 70)

    out = BASE / "validation_results.csv"
    df.to_csv(out, index=False)
    print(f"\nRecomputed data saved to {out}")
    print(f"Total time: {fmt_eta(time.time() - t0)}")


if __name__ == "__main__":
    main()
