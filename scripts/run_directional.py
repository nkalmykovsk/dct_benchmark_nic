#!/usr/bin/env python3
"""Directional leakage analysis using rotated DCT bases (Supplementary Fig. S1).

For each rotation angle θ, constructs a rotated DCT basis D_θ, compresses it,
and computes per-column correlation leakage L_k(θ) = 1 - ρ_k².

Paper settings (Fig. S1):
    size=512, matched bitrate ≈ 2.5 bpp, angles: 0°, 15°, 30°, 45°, 60°, 75°, 90°
    Classical codecs: jpeg q=43, webp q=70, jpegxl distance=2.5

Usage:
    python scripts/run_directional.py                    # default: size=512, matched bpp
    python scripts/run_directional.py --size 256 --quality 6
    python scripts/run_directional.py --models jpeg cheng2020-anchor
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dct_nic.loaders import load_model, get_available_models, ImageCodecModel
from dct_nic.evaluate import _model_forward

# Matched-bpp classical codec settings (≈ 2.5 bpp, 512×512)
MATCHED_BPP_CLASSICAL = {"jpeg": 43, "webp": 70, "jpegxl": 2.5}


def build_rotated_dct_basis(size: int, angle_deg: float) -> np.ndarray:
    """n×n rotated DCT-II basis at angle θ (degrees).

    D_θ[n, k] = α(k') · cos(π/N · (n' + 0.5) · k')
    where n' = n·cos θ + k·sin θ,  k' = −n·sin θ + k·cos θ.
    """
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    nn, kk = np.meshgrid(np.arange(size, dtype=np.float64),
                         np.arange(size, dtype=np.float64), indexing="ij")
    n_rot = nn * c + kk * s
    k_rot = -nn * s + kk * c
    alpha = np.where(np.abs(k_rot) < 1e-12, math.sqrt(1.0 / size), math.sqrt(2.0 / size))
    return alpha * np.cos(math.pi * (n_rot + 0.5) * k_rot / size)


def analyze_angle(D: np.ndarray, D_hat: np.ndarray) -> dict:
    """Per-column correlation leakage for one angle."""
    N = D.shape[0]
    leakage = np.zeros(N)
    eps = 1e-12
    for k in range(N):
        d_in, d_out = D[:, k].astype(np.float64), D_hat[:, k].astype(np.float64)
        ni, no = np.linalg.norm(d_in), np.linalg.norm(d_out)
        if ni < eps or no < eps:
            leakage[k] = 1.0
            continue
        rho = float(np.clip(np.dot(d_in, d_out) / (ni * no), -1.0, 1.0))
        leakage[k] = 1.0 - rho ** 2
    N4 = N // 4
    return {
        "leakage": leakage,
        "L_k": float(np.median(leakage)),
        "L_low": float(np.mean(leakage[:N4])),
        "L_high": float(np.mean(leakage[3 * N4:])),
    }


def run_model_angles(model, size: int, angles: list[float], device,
                     model_name: str = "") -> dict[float, dict]:
    """Run directional analysis for one model across all angles."""
    results = {}
    for ang in angles:
        D = build_rotated_dct_basis(size, ang)
        # Normalize to [0, 1] for codec
        lo, hi = float(D.min()), float(D.max())
        norm = (D - lo) / (hi - lo + 1e-9)
        rgb = np.stack([norm, norm, norm], axis=-1).astype(np.float32)
        x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)

        x_hat, _ = _model_forward(model, x, model_name)
        recon_rgb = x_hat.squeeze().permute(1, 2, 0).numpy()
        recon_gray = (lo + (hi - lo) * recon_rgb).mean(axis=2)
        results[ang] = analyze_angle(D, recon_gray)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--angles", default="0,15,30,45,60,75,90")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--matched-bpp", action="store_true", default=True,
                        help="Use matched-bpp settings for classical codecs (default)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="results/directional")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    angles = [float(a) for a in args.angles.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_list = args.models or get_available_models(include_codecs=True)
    base = str(Path(__file__).resolve().parent.parent / "third_party")

    all_rows = []
    for name in models_list:
        print(f"\n{name}")
        try:
            if args.matched_bpp and name in MATCHED_BPP_CLASSICAL:
                v = MATCHED_BPP_CLASSICAL[name]
                if name == "jpeg":
                    model = ImageCodecModel("jpeg", jpeg_quality=int(v))
                elif name == "webp":
                    model = ImageCodecModel("webp", webp_quality=int(v))
                elif name == "jpegxl":
                    model = ImageCodecModel("jpegxl", jpegxl_distance=float(v))
            elif name == "tcm":
                model = load_model(name, 1, device, p=128, base_dir=base)
            else:
                model = load_model(name, args.quality, device, base_dir=base)

            results = run_model_angles(model, args.size, angles, device, name)
            for ang, m in results.items():
                row = {
                    "model": name, "size": f"{args.size}x{args.size}",
                    "angle_deg": ang, "L_k": m["L_k"],
                    "L_low": m["L_low"], "L_high": m["L_high"],
                    "q": args.quality if name not in MATCHED_BPP_CLASSICAL else None,
                    "p": 128 if name == "tcm" else None,
                    "matched_bpp": 1 if name in MATCHED_BPP_CLASSICAL else 0,
                }
                all_rows.append(row)
                print(f"  {ang:5.1f}° → L_k={m['L_k']:.4f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "directional_leakage.csv"
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df]).drop_duplicates(
            subset=["model", "size", "angle_deg"], keep="last"
        )
    df.to_csv(csv_path, index=False)
    print(f"\n→ Saved {csv_path}")


if __name__ == "__main__":
    main()
