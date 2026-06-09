#!/usr/bin/env python3
"""2-D DCT-basis leakage (answers Reviewer comment R1.2).

The main paper builds the frequency-response matrix from the *1-D* DCT basis
(Eq. 1): each column is one 1-D cosine, so leakage L_k is indexed by a single
axis frequency. This script builds the *2-D* basis and computes the diagonal of
the frequency-response *tensor*:

    tiles  T_{k,l}[i,j] = a_k a_l cos(pi(i+1/2)k/s) cos(pi(j+1/2)l/s)
    map    L_{k,l} = 1 - |DCT2(recon T_{k,l})_{k,l}|^2 / sum |DCT2(...)|^2

Reports (i) that the 1-D L_k is the l=0 axis slice of L_{k,l} (ideal codec:
both zero), (ii) the across-codec rank-equivalence Spearman(median L_{k,l},
median L_k) — if >0.9 the compact 1-D presentation is faithful — and (iii) the
radial average L^r(f) = <L_{k,l}>_{sqrt(k^2+l^2)=f}, a principled radial leakage
for the coupling (Eq. 4) that needs no 1-D->radial interpolation. Saves an
L_{k,l} heatmap per codec for the supplementary.

Classical codecs (jpeg/webp) run locally via Pillow. On a GPU box (torch +
dct_nic) every codec — including the neural ones and jpegxl — is loaded through
dct_nic.load_model so settings match the paper exactly.

Usage:
    # local sanity (no torch needed):
    python scripts/run_2d_leakage.py --codecs jpeg webp --tile-n 16 --q 50
    # H100 — run everything at highest quality:
    python scripts/run_2d_leakage.py --tile-n 16 --q 6 --device cuda
    python scripts/run_2d_leakage.py --smoke
"""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.fft import dct, dctn
from PIL import Image

ALL_CODECS = ["jpeg", "webp", "jpegxl", "bmshj2018-factorized", "bmshj2018-hyperprior",
              "mbt2018-mean", "mbt2018", "cheng2020-anchor", "cheng2020-attn", "tcm", "ftic"]
CLASSICAL = {"jpeg", "webp", "jpegxl"}


# --- 2-D DCT basis -----------------------------------------------------------

def _alpha(k, s):
    return np.where(np.asarray(k) == 0, np.sqrt(1.0 / s), np.sqrt(2.0 / s))


def basis_tile(k, l, s):
    """(s,s) 2-D DCT-II basis function; DCT2(tile) is a unit impulse at (k,l)."""
    i = np.arange(s)[:, None]
    j = np.arange(s)[None, :]
    return (_alpha(k, s) * np.cos(np.pi * (i + 0.5) * k / s)) * \
           (_alpha(l, s) * np.cos(np.pi * (j + 0.5) * l / s))


def build_mosaic(s):
    """Tile all s^2 basis functions into one (s^2, s^2) image, normalized to [0,1]."""
    M = np.empty((s * s, s * s), dtype=np.float64)
    for a in range(s):
        for b in range(s):
            M[a * s:(a + 1) * s, b * s:(b + 1) * s] = basis_tile(a, b, s)
    vmin, vmax = float(M.min()), float(M.max())
    return ((M - vmin) / (vmax - vmin + 1e-9)).astype(np.float32), vmin, vmax


# --- leakage -----------------------------------------------------------------

def leakage_map(recon_gray, s):
    """L_{k,l} from a reconstructed mosaic (de-normalized, grayscale)."""
    L = np.zeros((s, s))
    for a in range(s):
        for b in range(s):
            P = dctn(recon_gray[a * s:(a + 1) * s, b * s:(b + 1) * s], norm="ortho") ** 2
            L[a, b] = 1.0 - P[a, b] / (P.sum() + 1e-12)
    return L


def leakage_1d(recon_basis_gray):
    """1-D L_k (column-wise), identical to the paper's Eq. (1)/(2) formula."""
    n = recon_basis_gray.shape[0]
    L = np.zeros(n)
    for k in range(n):
        P = dct(recon_basis_gray[:, k], norm="ortho") ** 2
        L[k] = 1.0 - P[k] / (P.sum() + 1e-12)
    return L


def radial_profile(L, num_bins):
    """L^r(f) = mean of L_{k,l} over the ring sqrt(k^2+l^2)=f."""
    s = L.shape[0]
    kk, ll = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    r = np.sqrt(kk ** 2 + ll ** 2)
    edges = np.linspace(0.0, r.max() + 1e-12, num_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, num_bins - 1).ravel()
    flat = L.ravel()
    return np.array([flat[idx == b].mean() if (idx == b).any() else 0.0
                     for b in range(num_bins)])


# --- codecs ------------------------------------------------------------------

def _pil_classical(img01, codec, q):
    u8 = np.clip(img01 * 255 + 0.5, 0, 255).astype(np.uint8)
    if u8.ndim == 2:
        u8 = np.stack([u8] * 3, -1)
    buf = BytesIO()
    if codec == "jpeg":
        Image.fromarray(u8, "RGB").save(buf, format="JPEG", quality=q, subsampling=0, optimize=True)
    else:
        Image.fromarray(u8, "RGB").save(buf, format="WEBP", quality=q, method=6)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"), np.float32) / 255.0


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def make_runner(codec, q, device):
    """Return a function ``img01 -> recon01``, loading the codec only ONCE.

    On a GPU box everything is routed through dct_nic.load_model (paper-exact
    settings). Without torch, only classical jpeg/webp run (local Pillow path).
    """
    if _torch_available():
        import sys, torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from dct_nic import load_model
        base = str(Path(__file__).resolve().parent.parent / "third_party")
        model = (load_model("tcm", q, device, p=128, base_dir=base) if codec == "tcm"
                 else load_model(codec, q, device, base_dir=base))

        def run(img01):
            if img01.ndim == 2:
                img01 = np.stack([img01] * 3, -1)
            x = torch.from_numpy(img01).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
            return out["x_hat"].clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return run

    if codec in ("jpeg", "webp"):
        return lambda img01: _pil_classical(img01, codec, q)
    raise RuntimeError(f"{codec} needs torch/dct_nic (run on the GPU box)")


# --- driver ------------------------------------------------------------------

def to_gray_denorm(rec, lo, hi):
    rec = (lo + (hi - lo) * rec)
    return rec.mean(-1) if rec.ndim == 3 else rec


def analyze_codec(codec, s, q, num_bins, device):
    run = make_runner(codec, q, device)  # load the codec once, reuse for both inputs

    # 2-D mosaic -> L_{k,l}
    mosaic, vmin, vmax = build_mosaic(s)
    L2 = leakage_map(to_gray_denorm(run(mosaic), vmin, vmax), s)

    # 1-D basis -> L_k. Neural codecs need >=256; classical can match s (enables
    # the exact l=0-slice check). NICs use 256 (paper's smallest reported size).
    n1d = 256 if (codec not in CLASSICAL and s < 256) else s
    D = dct(np.eye(n1d), axis=1, norm="ortho")
    d0, d1 = float(D.min()), float(D.max())
    L1 = leakage_1d(to_gray_denorm(run((D - d0) / (d1 - d0 + 1e-9)), d0, d1))

    # Exact l=0-slice check only when the 1-D size matches the tile grid
    # (classical). NICs are forced to n1d=256 (FTIC/TCM minimum) -> sizes differ,
    # so the exact slice is undefined for them; they are covered by the rank gate.
    slice_gap = float(np.max(np.abs(L2[:, 0] - L1))) if n1d == s else None
    return {
        "codec": codec, "L2_map": L2, "L_r": radial_profile(L2, num_bins),
        "med_2d": float(np.median(L2)), "med_1d": float(np.median(L1)),
        "slice_l0_gap": slice_gap,
        "low_2d": float(np.median(L2[:s // 2, :s // 2])),
        "high_2d": float(np.median(L2[s // 2:, s // 2:])),
    }


def save_heatmap(L2, codec, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    im = ax.imshow(L2, origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.set_xlabel("horizontal freq $l$"); ax.set_ylabel("vertical freq $k$")
    ax.set_title(f"{codec}: $L_{{k,l}}$", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out / f"L2map_{codec}.pdf", bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codecs", nargs="+", default=ALL_CODECS)
    p.add_argument("--tile-n", type=int, default=16, help="freq-grid size s; mosaic is s^2 x s^2 (>=16 for NICs)")
    p.add_argument("--q", type=int, default=6, help="quality for NIC (q-level) / classical (0-100)")
    p.add_argument("--num-bins", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/leakage_2d")
    p.add_argument("--smoke", action="store_true", help="ideal-codec invariant + 1 jpeg run, tile-n 8")
    args = p.parse_args()

    if args.smoke:
        s = 8
        mosaic, vmin, vmax = build_mosaic(s)
        ideal = leakage_map(vmin + (vmax - vmin) * mosaic, s)
        assert ideal.max() < 1e-8, f"ideal 2-D leakage must be ~0, got {ideal.max():.2e}"
        r = analyze_codec("jpeg", s, 50, 64, "cpu")
        print(f"[smoke] ideal L_map max={ideal.max():.2e} (PASS)  "
              f"jpeg med_2d={r['med_2d']:.4f} low={r['low_2d']:.4f} high={r['high_2d']:.4f}")
        return

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, profiles = [], {}
    for codec in args.codecs:
        try:
            r = analyze_codec(codec, args.tile_n, args.q, args.num_bins, args.device)
        except Exception as e:
            print(f"{codec:22s} FAILED: {e}")
            continue
        rows.append(r); profiles[codec] = r["L_r"]
        np.save(out / f"L2map_{codec}.npy", r["L2_map"])
        save_heatmap(r["L2_map"], codec, out)
        gap = r["slice_l0_gap"]
        gap_s = f"{gap:.4f}" if gap is not None else "  n/a "  # NIC: covered by rank gate
        print(f"{codec:22s} med_2d={r['med_2d']:.4f}  med_1d={r['med_1d']:.4f}  "
              f"low={r['low_2d']:.4f} high={r['high_2d']:.4f}  |L_(k,0)-L_k|max={gap_s}")

    np.savez(out / "radial_2d_profiles.npz", **profiles)
    if len(rows) >= 3:
        from scipy.stats import spearmanr
        rho = spearmanr([r["med_2d"] for r in rows], [r["med_1d"] for r in rows]).statistic
        verdict = "PASS (1-D faithful)" if rho > 0.9 else "REVIEW (1-D vs 2-D disagree)"
        print(f"\nRank-equivalence gate: Spearman(median 2-D, median 1-D) = {rho:+.3f}  -> {verdict}")
    else:
        print(f"\n(need >=3 codecs for the rank gate; got {len(rows)})")
    print(f"-> saved L_(k,l) maps (.npy/.pdf) + radial_2d_profiles.npz in {out}")


if __name__ == "__main__":
    main()
