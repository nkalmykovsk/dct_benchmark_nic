#!/usr/bin/env python3
"""S27: regenerate Fig. 3 (matched-bpp DCT-basis grid) with HiFiC added as a
12th panel (perceptual anchor). HiFiC has fixed rates; its highest point
(hific-hi, ~0.70 bpp) is the one nearest the ~1.0 bpp target, consistent
with the other panels' 0.68-1.24 bpp spread. Reuses the original renderer.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/root/dct_benchmark_nic")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import plot_fig3_matched_bpp as F3

# append HiFiC to the neural block
F3.CODECS = F3.CODECS + [("HiFiC\n(GAN)", "hific-hi")]

_orig_load = F3.load_for_setting
def load_for_setting(model, setting, device, base):
    if model == "hific-hi":
        from dct_nic import load_model
        return load_model("hific-hi", 6, device)
    return _orig_load(model, setting, device, base)
F3.load_for_setting = load_for_setting

_orig_select = F3.select_settings
def select_settings(scan_csv, target):
    sel = _orig_select(scan_csv, target)
    sel["hific-hi"] = ("fixed", 0.70, 0.618)  # fixed operating point
    return sel
F3.select_settings = select_settings

sys.argv = ["s27", "--target-bpp", "1.0", "--device", "cuda",
            "--out", str(REPO / "results/fig3_matched_hific"), "--dpi", "600"]
F3.main()
print("S27_DONE", flush=True)
