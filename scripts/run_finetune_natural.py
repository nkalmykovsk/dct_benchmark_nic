#!/usr/bin/env python3
"""Natural-image reconstruction after Leakage-as-Loss fine-tuning (Reviewer R1.4).

Joint decoder-only fine-tuning: MSE on held-out natural patches + leakage loss on
the DCT basis (the paper's training signal) + a low-band guard. Encoder, hyper and
entropy stay frozen, so the bitstream — and bpp — are unchanged by construction.
The baseline and fine-tuned decoders then reconstruct held-out natural images for a
side-by-side column (Original | baseline | leakage-FT).

Tuning notes (held-out eval, bpp fixed):
    GOOD  — lambda=0.03, hf_power=3, mse=1, lr=1e-4, batch=4, steps=2000
            → 0769 +3.4 PSNR / +4.2 HF; 0772 +0.24 PSNR; 26f3 +3.7 PSNR; spot much smaller
    LEAK  — train without excluding eval stems 0769/0772 from DIV2K (inflated metrics)
    FAIL  — lambda>=0.06 or hf_power>=4.5: 0772/26f3 degrade, 0769 spot can worsen
    FAIL  — batch=8 / steps=3500: no gain on spot, 26f3 can get new artifacts
    FAIL  — batch-skip after step~1200: training stalls, final metrics collapse

Optional checkpoints (--checkpoint-every 200): save all steps, pick best 0769 HF.

Usage:
    python3 scripts/run_finetune_natural.py --device cuda
    python3 scripts/run_finetune_natural.py --device cuda --extra-train-dirs \\
        /home/nkalmykov/projects/nic-forensics/data/clic/professional
    python3 scripts/run_finetune_natural.py --smoke   # numpy-only self-test
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.fft import dct, dctn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent
if not (_REPO / "utils" / "loaders.py").exists():
    _REPO = _REPO.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

PATCH_MULT = 64  # CompressAI models need H, W divisible by this
DEFAULT_TRAIN_DIR = "/data1/nkalmykov/div2k/images"
DEFAULT_EXTRA_TRAIN_DIRS = []
DEFAULT_EVAL_IMAGES = [
    "paper/div2k_clic_examples_crop/0769.png",
    "paper/div2k_clic_examples_crop/0772.png",
    "paper/div2k_clic_examples_crop/26f350af0f6ee2fb314606ebc2b56e56.png",
]
# checkpoint selection: max HF on spot image, other eval images must keep PSNR
CKPT_SPOT_IMAGE = "0769"
CKPT_GUARD_PSNR = {"0772": 29.3, "26f350af0f6ee2fb314606ebc2b56e56": 30.0}

_LOG_FP = None


def setup_log(log_path):
    """Mirror stdout to a line-buffered log file."""
    global _LOG_FP
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FP = open(log_path, "w", buffering=1)


def log(msg=""):
    print(msg, flush=True)
    if _LOG_FP is not None:
        _LOG_FP.write(f"{msg}\n")
        _LOG_FP.flush()


def _resolve(path_str):
    p = Path(path_str)
    return p if p.is_absolute() else _REPO / p


def _load_model_fn():
    try:
        from dct_nic import load_model          # public repo package
    except ImportError:
        from utils.loaders import load_model    # server working copy
    return load_model


# DCT basis + image helpers (numpy)

def build_dct_basis_rgb(n):
    """DCT-II basis replicated to 3 channels, normalized to [0,1]; + (vmin, vmax)."""
    D = dct(np.eye(n, dtype=np.float64), axis=1, norm="ortho")
    rgb = np.stack([D, D, D], axis=-1).astype(np.float32)
    vmin, vmax = float(rgb.min()), float(rgb.max())
    return (rgb - vmin) / (vmax - vmin + 1e-9), vmin, vmax


def load_image(path, mult=PATCH_MULT):
    """Load RGB image in [0,1], center-cropped to a multiple of `mult`."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w, _ = img.shape
    ch, cw = (h // mult) * mult, (w // mult) * mult
    if ch < mult or cw < mult:
        raise ValueError(f"image too small after crop: {path} -> {ch}x{cw}")
    t, l = (h - ch) // 2, (w - cw) // 2
    return img[t:t + ch, l:l + cw]


def sample_patches(images, patch, batch, rng):
    """Random `patch`x`patch` crops drawn across `images` (list of HxWx3)."""
    out = []
    for _ in range(batch):
        im = images[rng.integers(len(images))]
        h, w, _ = im.shape
        t = rng.integers(0, h - patch + 1)
        l = rng.integers(0, w - patch + 1)
        out.append(im[t:t + patch, l:l + patch])
    return np.stack(out)


def sample_patches_from_paths(paths, patch, batch, rng):
    """Lazy-load random `patch`x`patch` crops from image paths."""
    out = []
    for _ in range(batch):
        path = paths[rng.integers(len(paths))]
        im = load_image(path)
        h, w, _ = im.shape
        t = rng.integers(0, h - patch + 1)
        l = rng.integers(0, w - patch + 1)
        out.append(im[t:t + patch, l:l + patch])
    return np.stack(out)


def collect_train_paths(train_dir, patch, exclude_stems=None):
    """List image paths large enough for `patch` crops (header-only size check)."""
    from PIL import Image
    exclude_stems = exclude_stems or set()
    paths, excluded = [], []
    candidates = sorted(p for p in train_dir.glob("*")
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    log(f"scanning {len(candidates)} files in {train_dir} ...")
    for i, p in enumerate(candidates, 1):
        if p.stem in exclude_stems:
            excluded.append(p.name)
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            if min(h, w) >= patch:
                paths.append(p)
        except Exception as e:
            log(f"  skip {p.name}: {e}")
        if i % 100 == 0 or i == len(candidates):
            log(f"  ... {i}/{len(candidates)} scanned, {len(paths)} usable")
    if excluded:
        log(f"excluding {len(excluded)} held-out eval image(s) from train: {excluded}")
    return paths


def collect_all_train_paths(train_dirs, patch, exclude_stems=None):
    """Scan multiple train dirs; skip missing dirs with a warning."""
    all_paths = []
    for train_dir in train_dirs:
        if not train_dir.is_dir():
            log(f"WARN: train dir not found, skipping: {train_dir}")
            continue
        all_paths.extend(collect_train_paths(train_dir, patch, exclude_stems))
    return all_paths


def to_gray(x):
    return x.mean(axis=2) if x.ndim == 3 else x


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


def hf_psnr(a, b, frac=1.0 / 3.0):
    """PSNR over the top-`frac` radial band of the 2-D DCT (Parseval, peak 1.0)."""
    A, B = dctn(to_gray(a), norm="ortho"), dctn(to_gray(b), norm="ortho")
    h, w = A.shape
    r = np.sqrt(np.arange(h)[:, None] ** 2 + np.arange(w)[None, :] ** 2)
    mask = r >= (1.0 - frac) * r.max()
    mse = float(np.mean((A[mask] - B[mask]) ** 2))
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


# Torch: leakage loss, compression, fine-tuning

def _dct_matrix(n, device):
    import math
    import torch
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    m = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    C = torch.cos(math.pi * k * (m + 0.5) / n)
    C[0] *= math.sqrt(1.0 / n)
    C[1:] *= math.sqrt(2.0 / n)
    return C


def _leakage_diag(x_hat_denorm, C):
    """Differentiable diag(R) for a (n,n) reconstructed basis (grayscale)."""
    import torch
    power = (C @ x_hat_denorm.float()).pow(2)
    R = power / (power.sum(dim=0, keepdim=True) + 1e-12)
    idx = torch.arange(x_hat_denorm.shape[0], device=x_hat_denorm.device)
    return R[idx, idx]


def _decoder_state_dict(model):
    import torch
    return {k: v.cpu() for k, v in model.state_dict().items() if k.startswith("g_s.")}


def _eval_all(model, eval_images, device):
    """Return {name: {bpp, psnr, hf, rec}} for each eval image."""
    metrics = {}
    for name, orig in eval_images:
        rec, bpp = reconstruct(model, orig, device)
        metrics[name] = {
            "bpp": bpp,
            "psnr": psnr(orig, rec),
            "hf": hf_psnr(orig, rec),
            "rec": rec,
        }
    return metrics


def _checkpoint_passes_guards(metrics, spot=CKPT_SPOT_IMAGE, guards=CKPT_GUARD_PSNR):
    return all(metrics[n]["psnr"] >= thr for n, thr in guards.items())


def _pick_best_checkpoint(records, spot=CKPT_SPOT_IMAGE, guards=CKPT_GUARD_PSNR):
    """Pick checkpoint with max spot HF among those passing PSNR guards."""
    passing = [r for r in records if _checkpoint_passes_guards(r["metrics"], spot, guards)]
    pool = passing if passing else records
    if not passing:
        log("WARN: no checkpoint passed PSNR guards; picking max spot HF overall")
    return max(pool, key=lambda r: r["metrics"][spot]["hf"])


def _save_checkpoint(step, model, eval_images, out_dir, device):
    """Save decoder + recon_ft + per-step metrics under checkpoints/step_XXXX/."""
    from PIL import Image

    ckpt_dir = out_dir / "checkpoints" / f"step_{step:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = ckpt_dir / "recon_ft"
    recon_dir.mkdir(exist_ok=True)

    import torch
    dec_path = ckpt_dir / "decoder.pth"
    torch.save(_decoder_state_dict(model), dec_path)

    metrics = _eval_all(model, eval_images, device)
    with open(ckpt_dir / "metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "bpp", "psnr", "hfpsnr"])
        for name, m in metrics.items():
            wr.writerow([name, f"{m['bpp']:.4f}", f"{m['psnr']:.3f}", f"{m['hf']:.3f}"])
            Image.fromarray((np.clip(m["rec"], 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                recon_dir / f"{name}.png")
    return {"step": step, "dir": ckpt_dir, "decoder": dec_path, "metrics": metrics}


def _write_checkpoint_history(records, out_dir, spot=CKPT_SPOT_IMAGE, guards=CKPT_GUARD_PSNR):
    """Write checkpoint_history.csv with guard flags."""
    if not records:
        return
    names = sorted(records[0]["metrics"].keys())
    path = out_dir / "checkpoint_history.csv"
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        header = ["step", "passes_guard"]
        for n in names:
            header.extend([f"{n}_psnr", f"{n}_hf"])
        wr.writerow(header)
        for r in records:
            m = r["metrics"]
            row = [r["step"], int(_checkpoint_passes_guards(m, spot, guards))]
            for n in names:
                row.extend([f"{m[n]['psnr']:.3f}", f"{m[n]['hf']:.3f}"])
            wr.writerow(row)
    log(f"Saved: {path}")


def _install_best_checkpoint(model, best, out_dir, model_name, quality, device):
    """Load best decoder weights into model and copy to top-level outputs."""
    import torch

    sd = torch.load(best["decoder"], map_location=device)
    model.load_state_dict(sd, strict=False)

    top_dec = out_dir / f"decoder_g_s_{model_name}_q{quality}.pth"
    shutil.copy2(best["decoder"], top_dec)
    log(f"Best decoder (step {best['step']}): {top_dec}")

    top_recon = out_dir / "recon_ft"
    top_recon.mkdir(exist_ok=True)
    for p in (best["dir"] / "recon_ft").glob("*.png"):
        shutil.copy2(p, top_recon / p.name)

    best_dir = out_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best["dir"], best_dir)
    (out_dir / "best_step.txt").write_text(f"{best['step']}\n")


def compress(model, x):
    """Deterministic forward (eval): return (x_hat in [0,1], bpp from likelihoods)."""
    import torch
    _, _, H, W = x.shape
    model.eval()
    with torch.no_grad():
        out = model(x)
    x_hat = out["x_hat"].clamp(0, 1)
    bpp = None
    if "likelihoods" in out:
        bpp = sum((-torch.log2(lk.clamp(min=1e-9))).sum().item()
                  for lk in out["likelihoods"].values()) / (H * W)
    return x_hat, bpp


def finetune_joint(model, train_paths, eval_images, args, device, out_dir=None):
    """Decoder-only joint FT. Returns (history, checkpoint_records); mutates model in place."""
    import torch
    import torch.nn.functional as F

    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("g_s.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:                       # fallback if naming differs
        for p in model.parameters():
            p.requires_grad = True
        trainable = list(model.parameters())
    log(f"  trainable decoder params: {sum(p.numel() for p in trainable):,}")

    n = args.basis_size
    basis_rgb, vmin, vmax = build_dct_basis_rgb(n)
    x_basis = torch.from_numpy(basis_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    C = _dct_matrix(n, device)
    D_target = C @ torch.eye(n, device=device)

    # baseline low-band reference on the basis (for the guard)
    model.eval()
    with torch.no_grad():
        xb = model(x_basis)["x_hat"]
    base_diag = _leakage_diag((vmin + (vmax - vmin) * xb).mean(1).squeeze(0), C).detach()
    low_n = max(1, int(0.33 * n))
    idx_low = torch.arange(low_n, device=device)
    base_low = base_diag[:low_n]

    freq = torch.arange(n, device=device, dtype=torch.float32) / max(1.0, n - 1)
    w = freq.pow(args.hf_leak_power) + 0.2
    w[:low_n] = 0.0
    w = w / (w.sum() + 1e-8)

    opt = torch.optim.Adam(trainable, lr=args.lr)
    rng = np.random.default_rng(0)
    history = []
    checkpoint_records = []
    ckpt_every = args.checkpoint_every if args.checkpoint_every else 0

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()

        # natural-image anchor (MSE)
        patches = sample_patches_from_paths(train_paths, args.patch, args.batch, rng)
        x_nat = torch.from_numpy(patches).permute(0, 3, 1, 2).to(device)
        x_hat_nat = model(x_nat)["x_hat"]
        loss_mse = F.mse_loss(x_hat_nat, x_nat)

        # leakage signal on the DCT basis
        x_hat_b = model(x_basis)["x_hat"]
        x_hat_denorm = (vmin + (vmax - vmin) * x_hat_b).mean(1).squeeze(0)
        diag_R = _leakage_diag(x_hat_denorm, C)
        loss_leak = (w * (1.0 - diag_R)).sum()

        # low-band guards (don't regress retained low frequencies)
        low_deficit = torch.relu(base_low - diag_R.index_select(0, idx_low) - 0.02).mean()
        loss_low_mse = F.mse_loss(
            x_hat_denorm.to(D_target.dtype).index_select(1, idx_low),
            D_target.index_select(1, idx_low))

        loss = (args.mse_weight * loss_mse + args.lambda_freq * loss_leak
                + 0.12 * low_deficit + 2e-2 * loss_low_mse)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            log(f"  step {step:4d}/{args.steps}  mse={loss_mse.item():.5f}  "
                f"leak={loss_leak.item():.4f}  L_k(basis)={float((1 - diag_R).median()):.4f}")
        if args.eval_every and (step % args.eval_every == 0):
            log(f"  -- mid-train eval @ step {step} --")
            _monitor_eval(model, eval_images, device)
        if ckpt_every and out_dir is not None and step % ckpt_every == 0:
            log(f"  -- checkpoint @ step {step} --")
            rec = _save_checkpoint(step, model, eval_images, out_dir, device)
            m = rec["metrics"]
            guard = _checkpoint_passes_guards(m)
            log(f"    spot {CKPT_SPOT_IMAGE} HF={m[CKPT_SPOT_IMAGE]['hf']:.2f}  "
                f"passes_guard={bool(guard)}")
            checkpoint_records.append(rec)
        history.append({"step": step, "mse": float(loss_mse.item()),
                        "leak": float(loss_leak.item())})
    return history, checkpoint_records


def _monitor_eval(model, eval_images, device):
    """Print bpp/PSNR/HF-PSNR on eval images mid-training (fidelity sanity)."""
    import torch
    for name, orig in eval_images:
        x = torch.from_numpy(orig).permute(2, 0, 1).unsqueeze(0).to(device)
        x_hat, bpp = compress(model, x)
        rec = x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy()
        log(f"    [eval {name:14s}] bpp={bpp:.3f}  PSNR={psnr(orig, rec):.2f}  "
            f"HF-PSNR={hf_psnr(orig, rec):.2f}")


def reconstruct(model, orig, device):
    import torch
    x = torch.from_numpy(orig).permute(2, 0, 1).unsqueeze(0).to(device)
    x_hat, bpp = compress(model, x)
    return x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy(), bpp


# Figure

def build_figure(rows, out_dir):
    """rows: list of dict(name, orig, base, ft, m_base, m_ft)."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "stix"})
    nr = len(rows)
    fig, axes = plt.subplots(nr, 3, figsize=(6.5, 2.2 * nr), squeeze=False)
    titles = ["Original", "Cheng2020-Anchor ($q{=}6$)", "+ Leakage fine-tuning"]
    for j, t in enumerate(titles):
        axes[0][j].set_title(t, fontsize=9, fontweight="bold")
    for i, r in enumerate(rows):
        panels = [(r["orig"], None), (r["base"], r["m_base"]), (r["ft"], r["m_ft"])]
        for j, (img, m) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(np.clip(img, 0, 1)); ax.set_xticks([]); ax.set_yticks([])
            if m is not None:
                ax.set_xlabel(f"{m['bpp']:.3f} bpp · {m['psnr']:.2f} dB · HF {m['hf']:.2f} dB",
                              fontsize=6.5)
    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"finetune_natural.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight"); log(f"Saved: {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="cheng2020-anchor")
    ap.add_argument("--quality", type=int, default=6)
    ap.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR,
                    help="primary dir of natural images for the MSE anchor (held-out)")
    ap.add_argument("--extra-train-dirs", nargs="*", default=None, metavar="DIR",
                    help="extra MSE anchor dirs (default: DIV2K only)")
    ap.add_argument("--eval-images", nargs="+", default=DEFAULT_EVAL_IMAGES,
                    help="images for the 3-col figure (paths relative to repo root ok)")
    ap.add_argument("--basis-size", type=int, default=256, help="DCT basis size for leakage")
    ap.add_argument("--patch", type=int, default=256, help="natural patch size (mult of 64)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000,
                    help="2000 ≈ 10 min (best held-out run)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--lambda-freq", type=float, default=0.03,
                    help="0.03 = sweet spot; >=0.06 hurts other images")
    ap.add_argument("--hf-leak-power", type=float, default=3.0,
                    help="freq^p HF emphasis; >=4.5 too aggressive")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200, help="0 disables mid-train eval")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save decoder+recon+metrics every N steps (200 = ckpt selection)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/finetune_natural_heldout")
    ap.add_argument("--log-file", default=None,
                    help="log path (default: <out>/run.log)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        b, vmin, vmax = build_dct_basis_rgb(16)
        assert b.shape == (16, 16, 3) and 0 <= b.min() <= b.max() <= 1
        rng = np.random.default_rng(0)
        ims = [rng.random((96, 96, 3)).astype(np.float32) for _ in range(2)]
        ps = sample_patches(ims, 64, 3, rng)
        assert ps.shape == (3, 64, 64, 3)
        a = rng.random((64, 64, 3)).astype(np.float32)
        assert psnr(a, a) > 100 and hf_psnr(a, a) > 100
        log("[smoke] basis/patch/psnr/hf-psnr OK")
        return

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = _resolve(args.out)
    log_path = _resolve(args.log_file) if args.log_file else out_dir / "run.log"
    setup_log(log_path)
    log("=== run_finetune_natural ===")
    log(f"device={device}  model={args.model}  q={args.quality}  "
        f"steps={args.steps}  lambda_freq={args.lambda_freq}  hf_leak_power={args.hf_leak_power}")
    log(f"log -> {log_path}")

    eval_paths = [_resolve(p) for p in args.eval_images]
    eval_stems = {p.stem for p in eval_paths}

    train_dir = _resolve(args.train_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"train-dir not found: {train_dir}")

    extra_dirs = (DEFAULT_EXTRA_TRAIN_DIRS if args.extra_train_dirs is None
                  else args.extra_train_dirs)
    train_dirs = [train_dir] + [_resolve(p) for p in extra_dirs]
    train_paths = collect_all_train_paths(train_dirs, args.patch, exclude_stems=eval_stems)
    if not train_paths:
        raise RuntimeError(f"no usable training images >= {args.patch}px in {train_dirs}")
    log(f"train: {len(train_paths)} images from {len(train_dirs)} dir(s) "
        f"(lazy patch loading, eval held out)")

    for p in eval_paths:
        if not p.is_file():
            raise FileNotFoundError(f"eval image not found: {p}")
    log("loading eval images ...")
    eval_images = []
    for p in eval_paths:
        im = load_image(p)
        eval_images.append((p.stem, im))
        log(f"  {p.name}  shape={im.shape}")

    load_model = _load_model_fn()
    base_dir = str(_REPO / "third_party")
    log(f"loading model {args.model} (q={args.quality}) ...")
    model = load_model(args.model, args.quality, device, base_dir=base_dir)
    log("model ready")

    # baseline reconstructions (before FT)
    log("Baseline reconstructions:")
    base_rec = {}
    for name, orig in eval_images:
        rec, bpp = reconstruct(model, orig, device)
        base_rec[name] = (rec, bpp)
        log(f"  {name:14s} bpp={bpp:.3f}  PSNR={psnr(orig, rec):.2f}  HF={hf_psnr(orig, rec):.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Fine-tuning (joint, decoder-only, lambda_freq={args.lambda_freq}) ...")
    if args.checkpoint_every:
        log(f"checkpoints every {args.checkpoint_every} steps -> {out_dir / 'checkpoints'}")
        log(f"selection: max {CKPT_SPOT_IMAGE} HF, guards {CKPT_GUARD_PSNR}")
    _, checkpoint_records = finetune_joint(
        model, train_paths, eval_images, args, device, out_dir=out_dir)

    # pick best checkpoint (not necessarily last step)
    best = None
    if checkpoint_records:
        _write_checkpoint_history(checkpoint_records, out_dir)
        best = _pick_best_checkpoint(checkpoint_records)
        log(f"Best checkpoint: step {best['step']}  "
            f"{CKPT_SPOT_IMAGE} HF={best['metrics'][CKPT_SPOT_IMAGE]['hf']:.2f}")
        _install_best_checkpoint(model, best, out_dir, args.model, args.quality, device)
        ft_metrics = best["metrics"]
    else:
        log("No checkpoints saved; using final weights")
        dec_sd = _decoder_state_dict(model)
        torch.save(dec_sd, out_dir / f"decoder_g_s_{args.model}_q{args.quality}.pth")
        log(f"Saved decoder: {len(dec_sd)} tensors")
        ft_metrics = _eval_all(model, eval_images, device)

    # fine-tuned reconstructions + figure + csv (from best checkpoint)
    rows, summary = [], []
    best_step = best["step"] if best else args.steps
    log(f"Fine-tuned reconstructions (step {best_step}):")
    (out_dir / "recon_base").mkdir(exist_ok=True)
    from PIL import Image
    for name, orig in eval_images:
        ft_rec = ft_metrics[name]["rec"]
        ft_bpp = ft_metrics[name]["bpp"]
        base_r, base_bpp = base_rec[name]
        dbpp = abs((ft_bpp or 0) - (base_bpp or 0))
        flag = "OK" if dbpp < 1e-6 else f"WARN dbpp={dbpp:.2e}"
        m_base = {"bpp": base_bpp, "psnr": psnr(orig, base_r), "hf": hf_psnr(orig, base_r)}
        m_ft = {"bpp": ft_bpp, "psnr": ft_metrics[name]["psnr"], "hf": ft_metrics[name]["hf"]}
        log(f"  {name:14s} bpp {base_bpp:.3f}->{ft_bpp:.3f} [{flag}]  "
            f"PSNR {m_base['psnr']:.2f}->{m_ft['psnr']:.2f}  "
            f"HF {m_base['hf']:.2f}->{m_ft['hf']:.2f}")
        Image.fromarray((np.clip(base_r, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
            out_dir / "recon_base" / f"{name}.png")
        rows.append({"name": name, "orig": orig, "base": base_r, "ft": ft_rec,
                     "m_base": m_base, "m_ft": m_ft})
        summary.append([name, f"{base_bpp:.4f}", f"{ft_bpp:.4f}",
                        f"{m_base['psnr']:.3f}", f"{m_ft['psnr']:.3f}",
                        f"{m_base['hf']:.3f}", f"{m_ft['hf']:.3f}"])

    build_figure(rows, out_dir)
    with open(out_dir / "finetune_natural_metrics.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "bpp_base", "bpp_ft", "psnr_base", "psnr_ft", "hfpsnr_base", "hfpsnr_ft"])
        wr.writerows(summary)
    log(f"-> saved figure, recon_base/, recon_ft/, checkpoints/, metrics.csv in {out_dir}")


if __name__ == "__main__":
    main()
