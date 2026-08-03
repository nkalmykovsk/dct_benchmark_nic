#!/usr/bin/env python3
"""S17b: main fine-tuning example.

The final layout shows the held-out image and a vertical sequence of the
reference crop, pretrained reconstruction, and Table 3 selected decoder.
Cached reconstructions make layout changes independent of model execution.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")
IMG = ROOT / "paper/div2k_clic_examples_crop/26f350af0f6ee2fb314606ebc2b56e56.png"
LOCAL_NPZ = ROOT / "results/analysis_s7/hero_recons.npz"
BACKUP_NPZ = WORKSPACE / "results_devbox_backup/analysis_s7/hero_recons.npz"
NPZ = LOCAL_NPZ if LOCAL_NPZ.exists() else BACKUP_NPZ
OUT = ROOT / "results/analysis_s7"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec

LP = None

img = np.array(Image.open(IMG).convert("RGB"), dtype=np.float32) / 255.0
h, w, _ = img.shape
ch, cw = (h // 64) * 64, (w // 64) * 64
img = img[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
x = None


@torch.no_grad()
def metrics(model):
    global LP
    if LP is None:
        import lpips as lpips_lib
        LP = lpips_lib.LPIPS(net="alex", verbose=False).to(DEV).eval()
    xh = model(x)["x_hat"].clamp(0, 1)
    mse = F.mse_loss(xh, x).item()
    return (xh.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            -10 * math.log10(max(mse, 1e-12)),
            float(LP(xh, x, normalize=True)))


if NPZ.exists():
    d = np.load(NPZ)
    rec0, rec2 = d["rec0"], d["rec2"]
    stats = d["stats"]
    if len(stats) == 6:  # Backward compatibility with the older four-panel cache.
        p0, l0, _, _, p2, l2 = stats
    else:
        p0, l0, p2, l2 = stats
    print("loaded cached reconstructions", flush=True)
else:
    from dct_nic import load_model
    from dct_nic.metrics import build_dct_basis_rgb

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEV)
    m0 = load_model("cheng2020-anchor", 6, DEV, base_dir=BASE)
    m0.eval()
    rec0, p0, l0 = metrics(m0)

    DIV2K_ALL = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                       for p in (ROOT / "data/div2k").glob(ext))
    _tab1 = set(np.linspace(0, len(DIV2K_ALL) - 1, 50).round().astype(int).tolist())
    MON = [p for i, p in enumerate(DIV2K_ALL) if i not in _tab1][3:100:12][:8]
    TRAIN = [p for p in DIV2K_ALL if p not in MON]

    def build_C(n):
        k = torch.arange(n, device=DEV, dtype=torch.float32).unsqueeze(1)
        mm = torch.arange(n, device=DEV, dtype=torch.float32).unsqueeze(0)
        C = torch.cos(math.pi * k * (mm + 0.5) / n)
        C[0] *= math.sqrt(1.0 / n)
        C[1:] *= math.sqrt(2.0 / n)
        return C

    class Pool:
        def __init__(s, paths, patch, mx, rng):
            s.p, s.rng, s.im = patch, rng, []
            for q in paths[:mx]:
                a = np.asarray(Image.open(q).convert("RGB"))
                if min(a.shape[:2]) >= patch:
                    s.im.append(a)
        def batch(s, bs):
            out = []
            for _ in range(bs):
                a = s.im[s.rng.integers(len(s.im))]
                hh, ww, _ = a.shape
                t = s.rng.integers(0, hh - s.p + 1)
                l = s.rng.integers(0, ww - s.p + 1)
                out.append(a[t:t + s.p, l:l + s.p])
            return torch.from_numpy(np.stack(out).astype(np.float32) / 255.0
                                    ).permute(0, 3, 1, 2).to(DEV)

    torch.manual_seed(0)
    np.random.seed(0)
    m2 = load_model("cheng2020-anchor", 6, DEV, base_dir=BASE)
    for pname, pp in m2.named_parameters():
        pp.requires_grad = pname.startswith("g_s.")
    tr = [pp for pp in m2.parameters() if pp.requires_grad]
    n = 256
    basis_rgb, vmin, vmax = build_dct_basis_rgb(n)
    xb = torch.from_numpy(basis_rgb).permute(2, 0, 1).unsqueeze(0).to(DEV)
    C = build_C(n)
    Dt = C @ torch.eye(n, device=DEV)
    m2.eval()
    def ldiag(xd):
        pw = (C @ xd.float()).pow(2)
        R = pw / (pw.sum(dim=0, keepdim=True) + 1e-12)
        ii = torch.arange(n, device=DEV)
        return R[ii, ii]
    with torch.no_grad():
        b0 = m2(xb)["x_hat"]
    bd = ldiag((vmin + (vmax - vmin) * b0).mean(1).squeeze(0)).detach()
    low_n = max(1, int(0.33 * n))
    il = torch.arange(low_n, device=DEV)
    bl = bd[:low_n]
    fr = torch.arange(n, device=DEV, dtype=torch.float32) / (n - 1.0)
    wv = fr.pow(3.0) + 0.2
    wv[:low_n] = 0.0
    wv = wv / (wv.sum() + 1e-8)
    pool = Pool(TRAIN, 256, 400, np.random.default_rng(0))
    opt = torch.optim.Adam(tr, lr=1e-4)
    for step in range(1, 501):
        m2.train()
        opt.zero_grad()
        xn = pool.batch(4)
        loss = F.mse_loss(m2(xn)["x_hat"], xn)
        xd = (vmin + (vmax - vmin) * m2(xb)["x_hat"]).mean(1).squeeze(0)
        dg = ldiag(xd)
        loss = loss + 0.03 * (wv * (1.0 - dg)).sum()
        loss = loss + 0.12 * torch.relu(bl - dg.index_select(0, il) - 0.02).mean()
        loss = loss + 2e-2 * F.mse_loss(xd.to(Dt.dtype).index_select(1, il),
                                        Dt.index_select(1, il))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr, 1.0)
        opt.step()
    m2.eval()
    rec2, p2, l2 = metrics(m2)
    np.savez_compressed(NPZ, rec0=rec0, rec2=rec2,
                        stats=np.array([p0, l0, p2, l2]))

print(f"pretrained: {p0:.2f}/{l0:.3f}  selected: {p2:.2f}/{l2:.3f}",
      flush=True)

# zoom location: strongest baseline error region (identical to s17)
err = np.abs(rec0 - img).mean(axis=2)
from scipy.ndimage import uniform_filter
es = uniform_filter(err, 40)
iy, ix = np.unravel_index(np.argmax(es), es.shape)
y0, y1 = max(0, iy - 60), min(ch, iy + 60)
x0, x1 = max(0, ix - 60), min(cw, ix + 60)

fig = plt.figure(figsize=(6.9, 5.55), constrained_layout=True)
gs = GridSpec(3, 2, figure=fig, width_ratios=[3.0 * cw / ch, 1])
ax_full = fig.add_subplot(gs[:, 0])
ax_full.imshow(np.clip(img, 0, 1))
ax_full.set_title("held-out CLIC image", fontsize=9)
ax_full.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                fill=False, ec="yellow", lw=1.4))
zooms = [(img, "reference crop", None, None, gs[0, 1]),
         (rec0, "pretrained decoder", p0, l0, gs[1, 1]),
         (rec2, "fine-tuned decoder", p2, l2, gs[2, 1])]
for im, title, pp, ll, cell in zooms:
    ax = fig.add_subplot(cell)
    ax.imshow(np.clip(im[y0:y1, x0:x1], 0, 1))
    ax.set_title(title, fontsize=8)
    if pp is not None:
        metric_text = ax.text(
            0.5, 0.97, f"PSNR={pp:.1f} dB, LPIPS={ll:.3f}",
            transform=ax.transAxes, ha="center", va="top",
            color="white", fontsize=6.8,
        )
        metric_text.set_path_effects([pe.withStroke(linewidth=1.4, foreground="black")])
    ax.set_xticks([]); ax.set_yticks([])
ax_full.set_xticks([]); ax_full.set_yticks([])
fig.savefig(OUT / "fig_ft_hero_v3.pdf")
fig.savefig(OUT / "fig_ft_hero_v3.png", dpi=200)
print("S17B_DONE", flush=True)
