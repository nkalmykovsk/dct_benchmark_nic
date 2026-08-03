#!/usr/bin/env python3
"""S24: evaluate HiFiC (Justin-Tan port, 3 operating points) on the DCT
benchmark, reusing dct_nic.evaluate_codec so the leakage math is identical
to every other codec. Reports basis leakage profile + bpp per checkpoint,
and standard natural-image metrics (PSNR/MS-SSIM/LPIPS/bpp) on Kodak-24.

Outputs -> /root/dct_benchmark_nic/results/hific/
"""
import sys, os, glob, logging, json, time
from pathlib import Path
import numpy as np
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as F
from PIL import Image

HIFIC = "/root/anchors/high-fidelity-generative-compression"
sys.path.insert(0, HIFIC)
os.chdir(HIFIC)
from src.helpers import utils
from default_config import ModelModes

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic.evaluate import evaluate_codec

DEV = torch.device("cuda")
OUT = Path("/root/dct_benchmark_nic/results/hific")
OUT.mkdir(parents=True, exist_ok=True)
CKPTS = {"low": "/root/anchors/hific_ckpts/hific_low.pt",
         "med": "/root/anchors/hific_ckpts/hific_med.pt",
         "hi": "/root/anchors/hific_ckpts/hific_hi.pt"}
_log = logging.getLogger("hific"); _log.addHandler(logging.NullHandler())


class HiFiC:
    def __init__(self, ckpt):
        self.args, self.model, _ = utils.load_model(
            ckpt, _log, DEV, model_mode=ModelModes.EVALUATION,
            current_args_d=None, prediction=True, strict=False, silent=True)
        self.model.eval()
        self.norm = bool(getattr(self.args, "normalize_input_image", True))

    @torch.no_grad()
    def __call__(self, x):
        xin = 2.0 * x - 1.0 if self.norm else x
        inter, _ = self.model.compression_forward(xin.to(DEV))
        rec = inter.reconstruction
        if self.norm:
            rec = (rec + 1.0) / 2.0
        return {"x_hat": rec.clamp(0, 1), "bpp": float(inter.q_bpp)}


try:
    from pytorch_msssim import ms_ssim
except Exception:
    ms_ssim = None
try:
    import lpips as lpips_lib
    LP = lpips_lib.LPIPS(net="alex", verbose=False).to(DEV).eval()
except Exception:
    LP = None

KOD = sorted(glob.glob("/root/dct_benchmark_nic/data/kodak/*.png"))
rows = {}
for name, ckpt in CKPTS.items():
    t0 = time.time()
    m = HiFiC(ckpt)
    # basis leakage profile at n=256 (identical pipeline to other codecs)
    res = evaluate_codec(m, size=256, device=DEV, model_name="hific")
    prof = {"L_k": float(res["L_k"]), "L_low": float(res["L_low"]),
            "L_high": float(res["L_high"]), "basis_bpp": float(res["bpp"])}
    np.savez_compressed(OUT / f"hific_{name}_basis256.npz",
                        leakage=res["leakage"], R=res["R"],
                        centroid_shift=res["centroid_shift"],
                        spread=res["spread"], entropy=res["entropy"],
                        bpp=res["bpp"])
    # natural-image standard metrics on Kodak-24
    bpps, psnrs, msss, lps = [], [], [], []
    for p in KOD:
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)
        out = m(x)
        xh = out["x_hat"]
        bpps.append(out["bpp"])
        mse = F.mse_loss(xh, x).item()
        psnrs.append(-10 * np.log10(max(mse, 1e-12)))
        if ms_ssim is not None:
            msss.append(float(ms_ssim(xh, x, data_range=1.0)))
        if LP is not None:
            lps.append(float(LP(xh, x, normalize=True)))
    prof.update(nat_bpp=float(np.mean(bpps)), nat_psnr=float(np.mean(psnrs)),
                nat_msssim=float(np.mean(msss)) if msss else None,
                nat_lpips=float(np.mean(lps)) if lps else None)
    rows[name] = prof
    print(f"[hific-{name}] basis: L_k={prof['L_k']:.4f} L_low={prof['L_low']:.4f} "
          f"L_high={prof['L_high']:.4f} bpp={prof['basis_bpp']:.3f} | "
          f"kodak: bpp={prof['nat_bpp']:.3f} PSNR={prof['nat_psnr']:.2f} "
          f"MS-SSIM={prof['nat_msssim']:.4f} LPIPS={prof['nat_lpips']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del m
    torch.cuda.empty_cache()

json.dump(rows, open(OUT / "hific_summary.json", "w"), indent=2)
print("S24_DONE", flush=True)
