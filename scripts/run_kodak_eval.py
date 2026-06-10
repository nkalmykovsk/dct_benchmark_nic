#!/usr/bin/env python3
"""Evaluate spectral leakage coupling L̃(X, X̂) on natural images (Table I).

Compresses each image with each codec (NICs at fixed quality q=6, classical
codecs rate-matched to --target-bpp), computes PSNR, MS-SSIM, LPIPS, and the
spectral leakage coupling L̃.

Paper settings (Table I):
    115 images = 24 Kodak + 41 CLIC Pro + 50 DIV2K; DCT size 512×512,
    N_b=512 bins; pool the per-dataset CSVs with scripts/aggregate_r15_table.py.
    Additional distortions (Kodak only): Gaussian noise (σ∈[0.01,0.20]),
    bit-depth quantization (2–6 bits), JPEG re-compression (q∈[10–70])

Usage:
    python scripts/run_kodak_eval.py --data-dir data/kodak
    python scripts/run_kodak_eval.py --data-dir data/clic  --dataset-name clic  --no-distortions
    python scripts/run_kodak_eval.py --data-dir data/div2k --dataset-name div2k --no-distortions
    python scripts/run_kodak_eval.py --data-dir data/kodak --single  # quick test
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dct_nic import evaluate_codec, load_model, get_available_models
from dct_nic.metrics import spectral_leakage_coupling

QUALITY = 6
DCT_SIZE = 512
NUM_BINS = 512
BASE_DIR = str(Path(__file__).resolve().parent.parent / "third_party")
TCM_P = 128

GAUSS_SIGMAS = [0.01, 0.03, 0.05, 0.10, 0.20]
QUANT_BITS = [6, 4, 3, 2]
JPEG_QUALITIES = [10, 30, 50, 70]

def load_image_tensor(path: Path, device: torch.device,
                      max_side: int = 768, mult: int = 64) -> torch.Tensor:
    """Load image as [1, 3, H, W] in [0, 1], center-cropped to a multiple of `mult`.

    Caps the crop at `max_side` so arbitrary-size natural images (CLIC/DIV2K) stay
    bounded and divisible by 64 for NIC codecs. Kodak (768x512) is unchanged.
    """
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    ch = (min(h, max_side) // mult) * mult
    cw = (min(w, max_side) // mult) * mult
    t, l = (h - ch) // 2, (w - cw) // 2
    img = img[t:t + ch, l:l + cw]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def codec_roundtrip(model, x: torch.Tensor, model_name: str):
    """Compress–decompress; return ([1,3,H,W] in [0,1] on CPU, bpp or None)."""
    is_trad = model_name in {"jpeg", "webp", "jpegxl"}
    if is_trad:
        out = model(x)
    else:
        with torch.no_grad():
            out = model(x)
    x_hat = out["x_hat"].clamp(0, 1).cpu()
    bpp = None
    if out.get("bpp") is not None:                      # classical wrappers
        bpp = float(out["bpp"])
    elif "likelihoods" in out:                          # NIC: bits from likelihoods
        _, _, H, W = x.shape
        bpp = sum((-torch.log2(lk.clamp(min=1e-9))).sum().item()
                  for lk in out["likelihoods"].values()) / (H * W)
    return x_hat, bpp


def psnr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    return float(-10.0 * np.log10(mse + 1e-12))


CLASSICAL_CANDIDATES = {            # quality / distance grids spanning the low-bpp regime
    "jpeg":   [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60],
    "webp":   [5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90],
    "jpegxl": [6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
}


def pick_classical_quality(model_name, images, device, target_bpp, base_dir, max_side, n_cal=6):
    """Select the classical setting whose mean bpp is nearest `target_bpp` (matched-rate)."""
    cal = images[:min(n_cal, len(images))]
    best = None
    for c in CLASSICAL_CANDIDATES[model_name]:
        m = load_model(model_name, QUALITY, device, base_dir=base_dir,
                       classical_overrides={model_name: c})
        bpps = []
        for p in cal:
            x = load_image_tensor(p, device, max_side=max_side)
            _, b = codec_roundtrip(m, x, model_name)
            if b is not None:
                bpps.append(b)
        mean_b = float(np.mean(bpps)) if bpps else float("inf")
        if best is None or abs(mean_b - target_bpp) < abs(best[1] - target_bpp):
            best = (c, mean_b)
    return best  # (setting, mean_bpp)


def add_gaussian_noise(t: torch.Tensor, sigma: float) -> torch.Tensor:
    return (t + sigma * torch.randn_like(t)).clamp(0, 1)


def quantize_bits(t: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2 ** bits - 1
    return (t * levels).round() / levels


def jpeg_recompress(t: torch.Tensor, quality: int) -> torch.Tensor:
    arr = (t.squeeze().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


def eval_image(
    img_path: Path,
    model,
    model_name: str,
    L_k: np.ndarray,
    device: torch.device,
    lpips_fn=None,
    with_distortions: bool = True,
    max_side: int = 768,
) -> list[dict]:
    """Evaluate one image with one model. Returns list of row dicts."""
    x = load_image_tensor(img_path, device, max_side=max_side)
    x_cpu = x.cpu()
    orig_gray = x_cpu.squeeze().mean(dim=0).numpy()
    rows = []

    def _record(dist_type: str, param: str, x_hat_cpu: torch.Tensor, bpp=None):
        recon_gray = x_hat_cpu.squeeze().mean(dim=0).numpy()
        p = psnr_db(x_cpu, x_hat_cpu)
        sc = spectral_leakage_coupling(orig_gray, recon_gray, L_k, num_bins=NUM_BINS)
        row = {
            "model": model_name, "image": img_path.stem,
            "distortion": dist_type, "param": param, "bpp": bpp,
            "psnr": p, "L_tilde": sc["L_tilde"],
            "rho_bar": sc["rho_bar"], "ratio": sc["ratio"],
        }
        try:
            from pytorch_msssim import ms_ssim
            row["ms_ssim"] = float(ms_ssim(x_cpu, x_hat_cpu, data_range=1.0))
        except ImportError:
            pass
        if lpips_fn is not None:
            with torch.no_grad():
                row["lpips"] = float(lpips_fn(
                    x_cpu.to(device) * 2 - 1,
                    x_hat_cpu.to(device) * 2 - 1,
                ).item())
        rows.append(row)

    x_hat, bpp = codec_roundtrip(model, x, model_name)
    _record("codec_clean", f"q{QUALITY}", x_hat, bpp=bpp)

    if with_distortions:
        for sigma in GAUSS_SIGMAS:
            _record("codec+gauss", f"{sigma}", add_gaussian_noise(x_hat, sigma))
        for sigma in GAUSS_SIGMAS:
            _record("gauss_only", f"{sigma}", add_gaussian_noise(x_cpu, sigma))
        for bits in QUANT_BITS:
            _record("quantization", f"{bits}bit", quantize_bits(x_cpu, bits))
        for jq in JPEG_QUALITIES:
            _record("jpeg_recomp", f"q{jq}", jpeg_recompress(x_cpu, jq))

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", help="Path to dataset images (Kodak/CLIC/DIV2K)")
    parser.add_argument("--kodak-dir", help="(alias of --data-dir)")
    parser.add_argument("--dataset-name", default=None,
                        help="output-file prefix (default: directory name)")
    parser.add_argument("--max-side", type=int, default=768,
                        help="center-crop cap; keeps Kodak unchanged, bounds CLIC/DIV2K")
    parser.add_argument("--target-bpp", type=float, default=0.9,
                        help="classical codecs are matched to this bpp (Table I uses 0.8-1.0)")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--single", action="store_true", help="Quick test: 1 image, 1 model")
    parser.add_argument("--max-images", type=int, default=None,
                        help="evenly subsample to this many images (CLIC/DIV2K generality check)")
    parser.add_argument("--no-distortions", action="store_true",
                        help="Skip additional distortion types")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/kodak_eval")
    args = parser.parse_args()

    data_dir = args.data_dir or args.kodak_dir
    if not data_dir:
        parser.error("provide --data-dir (or --kodak-dir)")
    data_dir = Path(data_dir)
    dataset = args.dataset_name or data_dir.name

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(data_dir.glob("kodim*.png"))
    if not images:
        images = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                        for p in data_dir.glob(ext))
    if not images:
        print(f"No images found in {data_dir}")
        return
    if args.max_images and len(images) > args.max_images:
        idx = np.linspace(0, len(images) - 1, args.max_images).round().astype(int)
        images = [images[i] for i in idx]
        print(f"Subsampled to {len(images)} images (--max-images)")
    print(f"Found {len(images)} images in {data_dir} (dataset='{dataset}')")

    models_list = args.models or get_available_models(include_codecs=True)
    if args.single:
        models_list = models_list[:1]
        images = images[:1]

    # Load LPIPS
    try:
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net="alex").to(device).eval()
    except ImportError:
        lpips_fn = None
        print("Warning: lpips not installed, skipping LPIPS computation")

    all_rows = []

    for mi, model_name in enumerate(models_list, 1):
        print(f"\n[{mi}/{len(models_list)}] {model_name}")

        # Load model. Classical codecs are matched to --target-bpp (like Table I);
        # NICs use their q-level.
        try:
            if model_name in CLASSICAL_CANDIDATES:
                setting, mean_b = pick_classical_quality(
                    model_name, images, device, args.target_bpp, BASE_DIR, args.max_side)
                print(f"  matched to ~{args.target_bpp} bpp: setting={setting} (mean bpp={mean_b:.3f})")
                model = load_model(model_name, QUALITY, device, base_dir=BASE_DIR,
                                   classical_overrides={model_name: setting})
            elif model_name == "tcm":
                model = load_model(model_name, QUALITY, device, p=TCM_P, base_dir=BASE_DIR)
            else:
                model = load_model(model_name, QUALITY, device, base_dir=BASE_DIR)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Get DCT leakage profile
        print("  Computing DCT leakage profile...", flush=True)
        res = evaluate_codec(model, size=DCT_SIZE, device=device, model_name=model_name)
        L_k = res["leakage"]
        print(f"  Median L_k = {res['L_k']:.4f}")

        # Evaluate on Kodak images
        for ii, img_path in enumerate(images, 1):
            print(f"  [{ii}/{len(images)}] {img_path.name}", end=" ... ", flush=True)
            try:
                rows = eval_image(
                    img_path, model, model_name, L_k, device,
                    lpips_fn=lpips_fn,
                    with_distortions=not args.no_distortions,
                    max_side=args.max_side,
                )
                all_rows.extend(rows)
                clean = [r for r in rows if r["distortion"] == "codec_clean"]
                if clean:
                    print(f"PSNR={clean[0]['psnr']:.2f}  L̃={clean[0]['L_tilde']:.4f}")
                else:
                    print("done")
            except Exception as e:
                print(f"ERROR: {e}")

    df = pd.DataFrame(all_rows)
    df["dataset"] = dataset
    csv_path = out_dir / f"{dataset}_per_image.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n→ Saved {csv_path} ({len(df)} rows)")

    # Aggregated summary (Table I style)
    clean = df[df["distortion"] == "codec_clean"]
    if not clean.empty:
        agg = clean.groupby("model").agg(
            n_images=("psnr", "count"),
            mean_bpp=("bpp", "mean"),
            std_bpp=("bpp", "std"),
            mean_psnr=("psnr", "mean"),
            std_psnr=("psnr", "std"),
            mean_L_tilde=("L_tilde", "mean"),
            std_L_tilde=("L_tilde", "std"),
            mean_ratio=("ratio", "mean"),
        )
        if "ms_ssim" in clean.columns:
            agg["mean_ms_ssim"] = clean.groupby("model")["ms_ssim"].mean()
        if "lpips" in clean.columns:
            agg["mean_lpips"] = clean.groupby("model")["lpips"].mean()
        agg["dataset"] = dataset
        agg_path = out_dir / f"{dataset}_aggregated.csv"
        agg.to_csv(agg_path)
        print(f"→ Saved {agg_path}")
        print("\n" + agg.to_string())


if __name__ == "__main__":
    main()
