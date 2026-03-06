#!/usr/bin/env python3
"""Evaluate spectral leakage coupling L̃(X, X̂) on the Kodak dataset (Table I).

Compresses each Kodak image with each codec at the highest quality, computes
PSNR, MS-SSIM, LPIPS, and the spectral leakage coupling L̃ (Eq. 6 in paper).

Paper settings (Table I):
    24 Kodak images, q=6 (highest quality), DCT size 512×512, N_b=512 bins
    Additional distortions: Gaussian noise (σ∈[0.01,0.20]), bit-depth
    quantization (2–6 bits), JPEG re-compression (q∈[10–70])

Usage:
    python scripts/run_kodak_eval.py --kodak-dir data/kodak
    python scripts/run_kodak_eval.py --kodak-dir data/kodak --models jpeg cheng2020-anchor
    python scripts/run_kodak_eval.py --kodak-dir data/kodak --single  # quick test
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

COMPRESSAI_PAD_MULT = 64
PAD_MULT = {"ftic": 256, "tcm": 128}


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """Load image as [1, 3, H, W] float32 tensor in [0, 1]."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def pad_for_codec(x: torch.Tensor, model_name: str) -> tuple[torch.Tensor, int, int]:
    """Pad to codec stride. Returns (padded, pad_h, pad_w)."""
    _, _, H, W = x.shape
    mult = PAD_MULT.get(model_name, COMPRESSAI_PAD_MULT)
    th = -(-H // mult) * mult
    tw = -(-W // mult) * mult
    if model_name not in PAD_MULT:
        th, tw = max(256, th), max(256, tw)
    ph, pw = th - H, tw - W
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)
    return x, ph, pw


def codec_roundtrip(model, x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Compress–decompress; return [1, 3, H, W] in [0, 1] on CPU."""
    _, _, H, W = x.shape
    is_trad = model_name in {"jpeg", "webp", "jpegxl"}
    if is_trad:
        out = model(x)
        return out["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()
    x_pad, ph, pw = pad_for_codec(x, model_name)
    with torch.no_grad():
        out = model(x_pad)
    return out["x_hat"][:, :, :H, :W].clamp(0, 1).cpu()


def psnr_db(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y).item()
    return float(-10.0 * np.log10(mse + 1e-12))


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
) -> list[dict]:
    """Evaluate one image with one model. Returns list of row dicts."""
    x = load_image_tensor(img_path, device)
    x_cpu = x.cpu()
    orig_gray = x_cpu.squeeze().mean(dim=0).numpy()
    rows = []

    def _record(dist_type: str, param: str, x_hat_cpu: torch.Tensor):
        recon_gray = x_hat_cpu.squeeze().mean(dim=0).numpy()
        p = psnr_db(x_cpu, x_hat_cpu)
        sc = spectral_leakage_coupling(orig_gray, recon_gray, L_k, num_bins=NUM_BINS)
        row = {
            "model": model_name, "image": img_path.stem,
            "distortion": dist_type, "param": param,
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

    x_hat = codec_roundtrip(model, x, model_name)
    _record("codec_clean", f"q{QUALITY}", x_hat)

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
    parser.add_argument("--kodak-dir", required=True, help="Path to Kodak images")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--single", action="store_true", help="Quick test: 1 image, 1 model")
    parser.add_argument("--no-distortions", action="store_true",
                        help="Skip additional distortion types")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/kodak_eval")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    kodak_dir = Path(args.kodak_dir)
    images = sorted(kodak_dir.glob("kodim*.png"))
    if not images:
        images = sorted(kodak_dir.glob("*.png"))
    if not images:
        print(f"No images found in {kodak_dir}")
        return
    print(f"Found {len(images)} images in {kodak_dir}")

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

        # Load model
        try:
            if model_name == "tcm":
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
    csv_path = out_dir / "kodak_per_image.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n→ Saved {csv_path} ({len(df)} rows)")

    # Aggregated summary (Table I style)
    clean = df[df["distortion"] == "codec_clean"]
    if not clean.empty:
        agg = clean.groupby("model").agg(
            bpp=("psnr", "count"),  # placeholder — bpp from DCT eval not per-image
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
        agg_path = out_dir / "kodak_aggregated.csv"
        agg.to_csv(agg_path)
        print(f"→ Saved {agg_path}")
        print("\n" + agg.to_string())


if __name__ == "__main__":
    main()
