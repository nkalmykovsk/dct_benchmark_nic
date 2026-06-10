#!/usr/bin/env python3
"""Compose the natural-image fine-tuning figure from saved reconstructions (R1.4).

Post-processes the outputs of run_finetune_natural.py: reads the original crops and
the baseline / leakage-fine-tuned reconstructions, computes PSNR, high-frequency PSNR
and (optionally) LPIPS, and assembles a CLEAN panel grid — no bitrate or metric text
baked in, so captions / metric overlays can be added afterwards. Columns are
Original | baseline | leakage-FT; rows are the selected images.

Usage:
    # paper figure: only the two artifact-free images
    python3 scripts/make_ft_natural_figure.py --images 0772 26f350af0f6ee2fb314606ebc2b56e56
    # all three (README / supplementary) + LPIPS
    python3 scripts/make_ft_natural_figure.py --lpips
    python3 scripts/make_ft_natural_figure.py --smoke
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dctn
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent
if not (_REPO / "paper").is_dir():
    _REPO = _REPO.parent

PATCH_MULT = 64
GAP = 6  # white separator (px) between panels


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else _REPO / p


def load_rgb(path, mult=PATCH_MULT):
    """Load RGB uint8 image center-cropped to a multiple of `mult` (matches the run)."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = img.shape
    ch, cw = (h // mult) * mult, (w // mult) * mult
    t, l = (h - ch) // 2, (w - cw) // 2
    return img[t:t + ch, l:l + cw]


def psnr(a, b):
    a, b = a.astype(np.float64) / 255, b.astype(np.float64) / 255
    return 10.0 * np.log10(1.0 / (np.mean((a - b) ** 2) + 1e-12))


def hf_psnr(a, b, frac=1.0 / 3.0):
    """PSNR over the top-`frac` radial DCT band (grayscale, Parseval, peak 1.0)."""
    g = lambda x: (x.astype(np.float64) / 255).mean(axis=2)
    A, B = dctn(g(a), norm="ortho"), dctn(g(b), norm="ortho")
    h, w = A.shape
    r = np.sqrt(np.arange(h)[:, None] ** 2 + np.arange(w)[None, :] ** 2)
    mask = r >= (1.0 - frac) * r.max()
    return 10.0 * np.log10(1.0 / (np.mean((A[mask] - B[mask]) ** 2) + 1e-12))


def lpips_fn():
    """Return f(a_uint8, b_uint8)->float, or None if lpips/torch unavailable."""
    try:
        import torch
        import lpips
    except ImportError:
        return None
    net = lpips.LPIPS(net="alex", verbose=False)

    def f(a, b):
        def t(x):
            x = torch.from_numpy(x.astype(np.float32) / 255 * 2 - 1)
            return x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(net(t(a), t(b)).item())
    return f


def msssim_fn():
    """Return f(a_uint8, b_uint8)->float, or None if pytorch_msssim/torch unavailable."""
    try:
        import torch
        from pytorch_msssim import ms_ssim
    except ImportError:
        return None

    def f(a, b):
        def t(x):
            x = torch.from_numpy(x.astype(np.float32) / 255)
            return x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(ms_ssim(t(a), t(b), data_range=1.0).item())
    return f


def hstrip(imgs, gap=GAP):
    """Horizontally concatenate same-height uint8 images with white gaps."""
    h = imgs[0].shape[0]
    sep = np.full((h, gap, 3), 255, np.uint8)
    out = []
    for i, im in enumerate(imgs):
        if i:
            out.append(sep)
        out.append(im)
    return np.concatenate(out, axis=1)


def vstrip(rows, gap=GAP):
    """Vertically stack rows, resizing narrower rows up to the max width (aspect kept)."""
    W = max(r.shape[1] for r in rows)
    sep = np.full((gap, W, 3), 255, np.uint8)
    out = []
    for i, r in enumerate(rows):
        if r.shape[1] != W:
            h = round(r.shape[0] * W / r.shape[1])
            r = np.asarray(Image.fromarray(r).resize((W, h), Image.LANCZOS))
        if i:
            out.append(sep)
        out.append(r)
    return np.concatenate(out, axis=0)


def auto_zoom_region(orig, base, win_frac=0.18):
    """Locate the artifact: window of size win_frac*W with max |base-orig| energy."""
    diff = np.abs(orig.astype(np.float64) - base.astype(np.float64)).sum(axis=2)
    H, W = diff.shape
    wsz = max(16, int(W * win_frac))
    ii = np.zeros((H + 1, W + 1))
    ii[1:, 1:] = np.cumsum(np.cumsum(diff, 0), 1)

    def wsum(y, x):
        y2, x2 = y + wsz, x + wsz
        return ii[y2, x2] - ii[y, x2] - ii[y2, x] + ii[y, x]

    stride = max(4, wsz // 6)
    best, by, bx = -1.0, 0, 0
    for y in range(0, H - wsz + 1, stride):
        for x in range(0, W - wsz + 1, stride):
            s = wsum(y, x)
            if s > best:
                best, by, bx = s, y, x
    return bx, by, wsz, wsz


def add_zoom(img, region, target_frac=0.42, pos="tr", color=(255, 212, 0), margin_frac=0.02):
    """Draw `region` box on img and paste a magnified crop inset in corner `pos`."""
    im = Image.fromarray(img).convert("RGB")
    W, H = im.size
    x, y, w, h = region
    crop = im.crop((x, y, x + w, y + h))
    tw = int(W * target_frac)
    th = int(tw * h / w)
    inset = crop.resize((tw, th), Image.NEAREST)  # NEAREST keeps the glitch crisp

    d = ImageDraw.Draw(im)
    lw = max(3, int(W * 0.006))
    d.rectangle([x, y, x + w, y + h], outline=color, width=lw)

    m = int(W * margin_frac)
    px = m if pos in ("tl", "bl") else W - tw - m
    py = m if pos in ("tl", "tr") else H - th - m
    im.paste(inset, (px, py))
    ImageDraw.Draw(im).rectangle([px, py, px + tw - 1, py + th - 1], outline=color, width=lw)
    return np.asarray(im)


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def annotate(img, lines, scale=0.08, corner="tl"):
    """Draw white text lines with a dark outline in `corner` ("tl" or "bl").

    Font scales with image *width*: README normalizes all cells to one width, so
    a width-proportional size renders identically across portrait/landscape crops.
    """
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    fs = max(24, int(im.width * scale))
    font = _font(fs)
    x = int(fs * 0.4)
    line_h = int(fs * 1.2)
    y = int(fs * 0.3) if corner == "tl" else im.height - line_h * len(lines) - int(fs * 0.3)
    for ln in lines:
        d.text((x, y), ln, font=font, fill="white",
               stroke_width=max(3, fs // 8), stroke_fill="black")
        y += line_h
    return np.asarray(im)


def load_metrics_csv(path):
    """name -> {col: float} from a metrics CSV (for re-annotation without recompute)."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["name"]] = {k: (float(v) if v not in ("", None) else None)
                              for k, v in r.items() if k != "name"}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", default="results/finetune_natural_heldout",
                    help="dir with recon_base/ and recon_ft/ from run_finetune_natural.py")
    ap.add_argument("--orig-dir", default="paper/div2k_clic_examples_crop")
    ap.add_argument("--images", nargs="+",
                    default=["0769", "0772", "26f350af0f6ee2fb314606ebc2b56e56"],
                    help="image stems, in row order")
    ap.add_argument("--out", default="results/finetune_natural_heldout/figure_clean")
    ap.add_argument("--name", default="ft_natural_clean")
    ap.add_argument("--lpips", action="store_true",
                    help="also compute perceptual metrics: MS-SSIM + LPIPS (needs torch)")
    ap.add_argument("--metrics-csv", default=None,
                    help="load metrics from a prior CSV instead of recomputing (for overlay)")
    ap.add_argument("--annotate", action="store_true",
                    help="bake PSNR/LPIPS (white, top-left) onto the baseline and FT cells")
    ap.add_argument("--label-scale", type=float, default=0.08,
                    help="overlay font size as a fraction of image width")
    ap.add_argument("--bottom-left-images", nargs="*", default=["0769"],
                    help="stems whose overlay goes bottom-left (clear of the artifact)")
    ap.add_argument("--zoom", action="store_true",
                    help="add a magnified inset of the artifact region (auto-located) to each panel")
    ap.add_argument("--zoom-frac", type=float, default=0.18, help="artifact window size / image width")
    ap.add_argument("--zoom-target", type=float, default=0.42, help="inset width / image width")
    ap.add_argument("--zoom-pos", default="tr", choices=["tl", "tr", "bl", "br"],
                    help="inset corner")
    ap.add_argument("--no-zoom-images", nargs="*", default=["0769"],
                    help="stems to skip the zoom inset for (artifact already visible)")
    ap.add_argument("--zoom-corner-overrides", nargs="*", default=["0772:bl"],
                    help="per-stem inset corner, e.g. 0772:bl 0769:tr")
    ap.add_argument("--base-dir", default="paper/div2k_clic_examples_reconstructed",
                    help="where annotated baseline cells are written (README display dir)")
    ap.add_argument("--ft-dir", default="paper/div2k_clic_examples_finetuned",
                    help="where annotated FT cells are written (README display dir)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        a = np.zeros((64, 64, 3), np.uint8)
        b = a.copy(); b[0, 0] = 255
        assert psnr(a, a) > 100 and psnr(a, b) < 100
        grid = vstrip([hstrip([a, a, a]), hstrip([b, b, b])])
        assert grid.shape[0] > 64 and grid.shape[2] == 3
        print("[smoke] psnr/hf/strip OK")
        return

    res = _resolve(args.result_dir)
    orig_dir = _resolve(args.orig_dir)
    out = _resolve(args.out); out.mkdir(parents=True, exist_ok=True)
    csv_metrics = load_metrics_csv(_resolve(args.metrics_csv)) if args.metrics_csv else None
    lp = lpips_fn() if (args.lpips and not csv_metrics) else None
    ms = msssim_fn() if (args.lpips and not csv_metrics) else None
    if args.lpips and not csv_metrics and lp is None:
        print("WARN: lpips/torch unavailable -> skipping LPIPS")
    if args.lpips and not csv_metrics and ms is None:
        print("WARN: pytorch_msssim/torch unavailable -> skipping MS-SSIM")

    def find(d, stem):
        for ext in (".png", ".jpg", ".jpeg"):
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p
        raise FileNotFoundError(f"{stem} not found in {d}")

    base_out = _resolve(args.base_dir) if args.annotate else None
    ft_out = _resolve(args.ft_dir) if args.annotate else None

    def _lines(m, suffix):
        out = [f"PSNR {m['psnr_' + suffix]:.1f} dB"]
        if m.get("lpips_" + suffix) is not None:
            out.append(f"LPIPS {m['lpips_' + suffix]:.3f}")
        return out

    rows, summary = [], []
    for stem in args.images:
        orig = load_rgb(find(orig_dir, stem))
        base = load_rgb(find(res / "recon_base", stem))
        ft = load_rgb(find(res / "recon_ft", stem))
        assert orig.shape == base.shape == ft.shape, f"shape mismatch on {stem}"

        if csv_metrics:
            row = {"name": stem, **csv_metrics[stem]}
        else:
            row = {"name": stem,
                   "psnr_base": psnr(orig, base), "psnr_ft": psnr(orig, ft),
                   "hf_base": hf_psnr(orig, base), "hf_ft": hf_psnr(orig, ft)}
            if ms is not None:
                row["msssim_base"] = ms(orig, base); row["msssim_ft"] = ms(orig, ft)
            if lp is not None:
                row["lpips_base"] = lp(orig, base); row["lpips_ft"] = lp(orig, ft)
        summary.append(row)

        text_corner = "bl" if stem in args.bottom_left_images else "tl"
        # inset goes to the right of the metrics (opposite horizontal corner) unless overridden
        overrides = dict(o.split(":") for o in args.zoom_corner_overrides if ":" in o)
        zoom_corner = overrides.get(stem,
                                    {"tl": "tr", "bl": "br"}[text_corner] if args.annotate else args.zoom_pos)

        if args.zoom and stem not in args.no_zoom_images:
            region = auto_zoom_region(orig, base, args.zoom_frac)
            orig = add_zoom(orig, region, args.zoom_target, zoom_corner)
            base = add_zoom(base, region, args.zoom_target, zoom_corner)
            ft = add_zoom(ft, region, args.zoom_target, zoom_corner)
            print(f"  zoom {stem}: region(x,y,w,h)={region}  inset={zoom_corner}")

        if args.annotate:
            base = annotate(base, _lines(row, "base"), args.label_scale, text_corner)
            ft = annotate(ft, _lines(row, "ft"), args.label_scale, text_corner)
            base_out.mkdir(parents=True, exist_ok=True)
            ft_out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(base).save(base_out / f"{stem}.png")
            Image.fromarray(ft).save(ft_out / f"{stem}.png")
            print(f"  annotated {stem}: base->{base_out.name}/  ft->{ft_out.name}/")
        rows.append(hstrip([orig, base, ft]))
        extra = ""
        if ms is not None:
            extra += f"  MS-SSIM {row['msssim_base']:.4f}->{row['msssim_ft']:.4f}"
        if lp is not None:
            extra += f"  LPIPS {row['lpips_base']:.4f}->{row['lpips_ft']:.4f}"
        print(f"  {stem:34s} PSNR {row['psnr_base']:.2f}->{row['psnr_ft']:.2f}  "
              f"HF {row['hf_base']:.2f}->{row['hf_ft']:.2f}{extra}")

    grid = vstrip(rows)
    Image.fromarray(grid).save(out / f"{args.name}.png")
    print(f"Saved: {out / f'{args.name}.png'}  ({grid.shape[1]}x{grid.shape[0]})")

    cols = ["name", "psnr_base", "psnr_ft", "hf_base", "hf_ft"]
    if ms is not None:
        cols += ["msssim_base", "msssim_ft"]
    if lp is not None:
        cols += ["lpips_base", "lpips_ft"]
    with open(out / f"{args.name}_metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in summary:
            wr.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    print(f"-> saved {args.name}.png + metrics.csv in {out}")


if __name__ == "__main__":
    main()
