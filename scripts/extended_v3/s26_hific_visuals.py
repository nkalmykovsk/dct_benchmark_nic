#!/usr/bin/env python3
"""S26: generate HiFiC visual candidates.
(1) DCT-basis reconstruction (for a possible Fig. 3 12th panel).
(2) natural before/after on texture-rich crops: original | faithful NIC
    (mbt2018 q6) | HiFiC-hi, to see where hallucination shows.
Saves side-by-side PNGs for inspection.
"""
import sys, os, glob
import numpy as np
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model
from dct_nic.metrics import build_dct_basis

DEV = torch.device("cuda")
OUT = "/root/dct_benchmark_nic/results/hific"
os.makedirs(OUT, exist_ok=True)

hific = load_model("hific-hi", 6, DEV)
mbt = load_model("mbt2018", 6, DEV, base_dir="/root/dct_benchmark_nic/third_party")
mbt.eval()


@torch.no_grad()
def run(model, x):
    out = model(x.to(DEV))
    xh = out["x_hat"] if isinstance(out, dict) else out
    return xh["x_hat"].clamp(0, 1).cpu() if isinstance(xh, dict) else xh.clamp(0, 1).cpu()


# (1) DCT basis reconstruction, 256, grayscale (matches Fig 3 rendering)
D = build_dct_basis(256)
img = 0.5 + 0.5 * (D / np.abs(D).max())  # normalized fringe pattern to [0,1]-ish
img = (img - img.min()) / (img.max() - img.min())
x = torch.from_numpy(np.repeat(img[:, :, None], 3, 2)).float().permute(2, 0, 1).unsqueeze(0)
rec = run(hific, x).squeeze(0).permute(1, 2, 0).numpy().mean(2)
side = np.concatenate([img, np.ones((256, 4)), rec], axis=1)
Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(f"{OUT}/basis_orig_vs_hific.png")

# (2) natural texture crops
CANDS = [
    ("/root/dct_benchmark_nic/data/clic/ales-krivec-15949.png", None),
    ("/root/dct_benchmark_nic/data/clic/alejandro-escamilla-6.png", None),
    ("/root/dct_benchmark_nic/data/kodak/kodim13.png", None),
    ("/root/dct_benchmark_nic/data/kodak/kodim19.png", None),
    ("/root/dct_benchmark_nic/data/kodak/kodim08.png", None),
]
for path, _ in CANDS:
    if not os.path.exists(path):
        print(f"skip {path}", flush=True); continue
    a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
    h, w, _ = a.shape
    ch, cw = (min(h, 512) // 64) * 64, (min(w, 512) // 64) * 64
    a = a[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    rh = run(hific, x).squeeze(0).permute(1, 2, 0).numpy()
    rm = run(mbt, x).squeeze(0).permute(1, 2, 0).numpy()
    # pick the 160px crop with the most original high-frequency energy
    from scipy.fft import dctn
    best, bs = None, -1
    step = 80
    for i in range(0, a.shape[0] - 160, step):
        for j in range(0, a.shape[1] - 160, step):
            g = a[i:i+160, j:j+160].mean(2)
            G = dctn(g, norm="ortho"); hf = float((G[80:, 80:] ** 2).sum())
            if hf > bs:
                bs, best = hf, (i, j)
    i, j = best
    trip = np.concatenate([a[i:i+160, j:j+160],
                           np.ones((160, 4, 3)),
                           rm[i:i+160, j:j+160],
                           np.ones((160, 4, 3)),
                           rh[i:i+160, j:j+160]], axis=1)
    name = os.path.basename(path).split('.')[0]
    Image.fromarray((np.clip(trip, 0, 1) * 255).astype(np.uint8)).save(
        f"{OUT}/nat_{name}_orig_mbt_hific.png")
    mse_h = float(np.mean((rh[i:i+160, j:j+160] - a[i:i+160, j:j+160]) ** 2))
    mse_m = float(np.mean((rm[i:i+160, j:j+160] - a[i:i+160, j:j+160]) ** 2))
    print(f"[{name}] crop@({i},{j}) HiFiC_cropPSNR={-10*np.log10(mse_h+1e-12):.1f} "
          f"MBT_cropPSNR={-10*np.log10(mse_m+1e-12):.1f}", flush=True)
print("S26_DONE", flush=True)
