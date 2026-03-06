#!/usr/bin/env python3
"""Generate Fig. 4b: Cheng2020-Anchor reconstruction at 1024×1024 (q=6).

Shows visible artifacts that emerge at large DCT sizes, illustrating the
non-monotonic leakage behavior of Cheng2020-Anchor at n > 256.

Usage:
    python scripts/plot_fig4b_artifact.py
    python scripts/plot_fig4b_artifact.py --model cheng2020-anchor --size 1024 --quality 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dct_nic import evaluate_codec, load_model, build_dct_basis_rgb


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="cheng2020-anchor")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} q={args.quality} ...")
    model = load_model(args.model, args.quality, device)

    print(f"Evaluating at {args.size}×{args.size} ...")
    result = evaluate_codec(
        model, size=args.size, device=device, model_name=args.model,
    )

    _, vmin, vmax = build_dct_basis_rgb(args.size)
    recon = result["recon"]
    recon_disp = np.clip((recon - vmin) / (vmax - vmin + 1e-9), 0, 1)

    from PIL import Image
    img_u8 = (recon_disp * 255).clip(0, 255).astype(np.uint8)
    tag = f"dct_{args.model}_q{args.quality}_sz{args.size}_output"
    out_png = args.out / f"{tag}.png"
    Image.fromarray(img_u8).save(out_png)
    print(f"→ Saved {out_png}")
    print(f"  Median L_k = {result['L_k']:.4f}")


if __name__ == "__main__":
    main()
