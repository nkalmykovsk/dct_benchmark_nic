#!/usr/bin/env python3
"""Run the full DCT frequency-response benchmark across all codecs.

Generates results/leakage_vs_bpp/scan_leakage_vs_bpp_{size}.csv for each size.

Usage:
    python scripts/run_benchmark.py                          # all models, size=256
    python scripts/run_benchmark.py --size 128 256 512       # multiple sizes
    python scripts/run_benchmark.py --models jpeg cheng2020-anchor  # subset
    python scripts/run_benchmark.py --quality-sweep          # all q levels

Codec configurations used for the paper:
    CompressAI / FTIC  : q ∈ {1, 2, 3, 4, 5, 6}
    TCM                : λ ∈ {0.0025, 0.05}, p ∈ {64, 128}
    JPEG / WebP        : quality ∈ {20, 40, 55, 70, 85, 95}
    JPEG XL            : distance ∈ {4.0, 2.0, 1.0, 0.6, 0.3, 0.1}
    DCT sizes          : n ∈ {64, 128, 256, 512, 1024}; FTIC, TCM n ≥ 256
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dct_nic import evaluate_codec, load_model, get_available_models
from dct_nic.loaders import ImageCodecModel

CLASSICAL_SCAN = {
    "jpeg":   list(range(20, 100, 5)) + [95],
    "webp":   list(range(20, 100, 5)) + [95],
    "jpegxl": [4.0, 2.0, 1.0, 0.6, 0.3, 0.1],
}
COMPRESSAI_QUALITIES = [1, 2, 3, 4, 5, 6]
TCM_P_VALUES = [64, 128]

# Models that need n ≥ 256
MIN_SIZE = {"ftic": 256, "tcm": 256}


def probe_one(
    name: str, quality: int, device, size: int,
    classical_q: int | float | None = None,
    tcm_p: int | None = None,
    ) -> dict | None:
    """Run a single (model, size, quality) evaluation."""
    base = str(Path(__file__).resolve().parent.parent / "third_party")
    try:
        if classical_q is not None:
            if name == "jpeg":
                model = ImageCodecModel("jpeg", jpeg_quality=int(classical_q))
            elif name == "webp":
                model = ImageCodecModel("webp", webp_quality=int(classical_q))
            elif name == "jpegxl":
                model = ImageCodecModel(
                    "jpegxl", jpegxl_distance=float(classical_q)
                )
            label = str(classical_q)
        elif tcm_p is not None:
            model = load_model("tcm", quality, device, p=tcm_p, base_dir=base)
            label = f"p={tcm_p}"
        else:
            model = load_model(name, quality, device, base_dir=base)
            label = f"q={quality}"

        res = evaluate_codec(model, size=size, device=device, model_name=name)
        return {
            "model": name, "size": size, "setting": label,
            "bpp": res["bpp"], "L_k": res["L_k"],
            "L_low": res["L_low"], "L_high": res["L_high"],
            "ODR_k": res["ODR_k"], "|Δc_k|": res["|Δc_k|"],
            "s_k": res["s_k"], "H_k": res["H_k"],
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--size", type=int, nargs="+", default=[256],
                        help="DCT basis size(s) (default: 256)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Codec names (default: all available)")
    parser.add_argument("--quality", type=int, default=6,
                        help="Quality level for single-q run (default: 6)")
    parser.add_argument("--quality-sweep", action="store_true",
                        help="Sweep all q (generates rate-leakage curves)")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Runs to average (default: 1)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")
    parser.add_argument("--out", default="results/leakage_vs_bpp",
                        help="Output dir (default: results/leakage_vs_bpp)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = args.models or get_available_models(include_codecs=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for size in args.size:
        print(f"\n{'='*60}\nSize: {size}×{size}\n{'='*60}")
        rows = []
        size_models = [m for m in models if size >= MIN_SIZE.get(m, 0)]
        skipped = set(models) - set(size_models)
        if skipped:
            print(f"Skipping {skipped} (require larger size)")

        for name in size_models:
            print(f"\n  [{name}]")
            if args.quality_sweep and name in CLASSICAL_SCAN:
                for q_val in CLASSICAL_SCAN[name]:
                    print(f"    q={q_val}", end=" ... ", flush=True)
                    row = probe_one(name, 1, device, size, classical_q=q_val)
                    if row:
                        rows.append(row)
                        print(f"bpp={row['bpp']:.3f}  L_k={row['L_k']:.4f}")

            elif name == "tcm":
                for p in TCM_P_VALUES:
                    print(f"    p={p}", end=" ... ", flush=True)
                    row = probe_one(name, 1, device, size, tcm_p=p)
                    if row:
                        rows.append(row)
                        print(f"bpp={row['bpp']:.3f}  L_k={row['L_k']:.4f}")

            elif args.quality_sweep:
                for q in COMPRESSAI_QUALITIES:
                    print(f"    q={q}", end=" ... ", flush=True)
                    row = probe_one(name, q, device, size)
                    if row:
                        rows.append(row)
                        print(f"bpp={row['bpp']:.3f}  L_k={row['L_k']:.4f}")

            else:
                q = args.quality
                print(f"    q={q}", end=" ... ", flush=True)
                row = probe_one(name, q, device, size)
                if row:
                    rows.append(row)
                    bpp_s = f"bpp={row['bpp']:.3f}  " if row["bpp"] else ""
                    print(f"{bpp_s}L_k={row['L_k']:.4f}")

        df = pd.DataFrame(rows)
        csv_path = out_dir / f"scan_leakage_vs_bpp_{size}.csv"
        if csv_path.exists():
            df_old = pd.read_csv(csv_path)
            df = pd.concat([df_old, df]).drop_duplicates(
                subset=["model", "setting"], keep="last"
            )
        df.to_csv(csv_path, index=False)
        print(f"\n→ Saved {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()