#!/usr/bin/env python3
"""S10: corrections found in the audit prep.

A) Classical codecs: rate-match with the ORIGINAL fine candidate grids
   (run_kodak_eval.CLASSICAL_CANDIDATES) per dataset, re-run the natural
   cache (metrics + radial + blocks) for jpeg/webp/jpegxl, and measure
   their 256/512 basis profiles at the chosen settings.
B) TCM at matched basis-bpp: use p64 (0.71 bpp) instead of p128 (2.6 bpp)
   for all synthetic matched-~1bpp analyses; add its single-freq sweeps.
C) Regenerate: excess-leakage figure with an honest envelope policy
   (only codecs whose basis bpp is within [0.5, 1.5]; others shown greyed,
   excluded from the floor), single-freq sweep figure with tcm-p64,
   Table I (classical rows fixed; TCM rule placement fixed).
Old classical caches are copied to natural_cache/old_coarse_classical/.

Outputs overwrite: results/table1_tex/table1.tex, results/analysis_s4/fig_*,
results/analysis_s7/fig_singlefreq_sweep.png, coupling CSVs.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn.functional as F
from PIL import Image
from scipy.fft import dct, dctn
from scipy.stats import spearmanr

sys.path.insert(0, "/root/dct_benchmark_nic")
from dct_nic import load_model, evaluate_codec
from dct_nic.metrics import build_dct_basis, spectral_leakage_coupling

ROOT = Path("/root/dct_benchmark_nic")
NC = ROOT / "results/natural_cache"
PROF = ROOT / "results/profiles"
S4 = ROOT / "results/analysis_s4"
S7 = ROOT / "results/analysis_s7"
SF = ROOT / "results/singlefreq"
T1 = ROOT / "results/table1_tex"
DEV = torch.device("cuda")
BASE = str(ROOT / "third_party")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSICAL_CANDIDATES = {
    "jpeg":   [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60],
    "webp":   [5, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90],
    "jpegxl": [6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
}
CLS = ["jpeg", "webp", "jpegxl"]
NICS = ["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean",
        "mbt2018", "cheng2020-anchor", "cheng2020-attn", "tcm", "ftic"]
NUM_BINS, BLOCK, N_BANDS = 512, 32, 16

_LPIPS = None
def lpips_fn():
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    return _LPIPS


def load_image_tensor(path, max_side=768, mult=64):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = img.shape
    ch = (min(h, max_side) // mult) * mult
    cw = (min(w, max_side) // mult) * mult
    t, l = (h - ch) // 2, (w - cw) // 2
    return torch.from_numpy(img[t:t + ch, l:l + cw]).permute(2, 0, 1).unsqueeze(0)


def cls_model(name, setting):
    return load_model(name, 6, DEV, classical_overrides={name: setting})


def roundtrip_cls(model, x):
    out = model(x)
    return out["x_hat"].clamp(0, 1), float(out["bpp"])


def radial_profiles(og, rg):
    Y = dctn(og.astype(np.float64), norm="ortho")
    E = dctn(og.astype(np.float64) - rg.astype(np.float64), norm="ortho")
    H, W = Y.shape
    r = np.sqrt(np.arange(H)[:, None] ** 2 + np.arange(W)[None, :] ** 2)
    edges = np.linspace(0, r.max() + 1e-12, NUM_BINS + 1)
    b = np.clip(np.digitize(r, edges) - 1, 0, NUM_BINS - 1)
    S = np.zeros(NUM_BINS); D = np.zeros(NUM_BINS); c = np.zeros(NUM_BINS)
    np.add.at(S, b.ravel(), (Y ** 2).ravel())
    np.add.at(D, b.ravel(), (E ** 2).ravel())
    np.add.at(c, b.ravel(), 1.0)
    ok = c > 0
    S[ok] /= c[ok]; D[ok] /= c[ok]
    return S, D


def block_stats(og, rg):
    H, W = og.shape
    nh, nw = H // BLOCK, W // BLOCK
    mse = np.zeros((nh, nw)); hf = np.zeros((nh, nw))
    energy = np.zeros((nh, nw, N_BANDS))
    fi = np.arange(BLOCK)[:, None].astype(float)
    fj = np.arange(BLOCK)[None, :].astype(float)
    r = np.sqrt(fi ** 2 + fj ** 2)
    band = np.clip((r / (r.max() + 1e-12) * N_BANDS).astype(int), 0, N_BANDS - 1)
    hfm = r >= (2.0 / 3.0) * r.max()
    for i in range(nh):
        for j in range(nw):
            a = og[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]
            b_ = rg[i * BLOCK:(i + 1) * BLOCK, j * BLOCK:(j + 1) * BLOCK]
            d = a.astype(np.float64) - b_.astype(np.float64)
            mse[i, j] = np.mean(d ** 2)
            A = dctn(a.astype(np.float64), norm="ortho")
            Dd = dctn(d, norm="ortho")
            hf[i, j] = np.mean(Dd[hfm] ** 2)
            for bb in range(N_BANDS):
                energy[i, j, bb] = np.sum(A[band == bb] ** 2)
    return mse, hf, energy


def basis_profile_cls(name, setting, size):
    m = cls_model(name, setting)
    res = evaluate_codec(m, size=size, device=DEV, model_name=name)
    return res["leakage"], res["bpp"]


def main():
    # ------------------ A) classical fine rate-matching ------------------
    div2k_all = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                       for p in (ROOT / "data/div2k").glob(ext))
    idx = np.linspace(0, len(div2k_all) - 1, 50).round().astype(int)
    datasets = {
        "kodak": sorted((ROOT / "data/kodak").glob("*.png")),
        "clic": sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                       for p in (ROOT / "data/clic").glob(ext)),
        "div2k": [div2k_all[i] for i in idx],
    }
    tensors = {ds: [load_image_tensor(p) for p in paths]
               for ds, paths in datasets.items()}

    old_dir = NC / "old_coarse_classical"
    old_dir.mkdir(exist_ok=True)
    for name in CLS:
        for ds in datasets:
            for pref in ("radial", "blocks"):
                f = NC / f"{pref}_{name}_{ds}.npz"
                if f.exists() and not (old_dir / f.name).exists():
                    shutil.copy2(f, old_dir / f.name)

    picks = {}
    for name in CLS:
        for ds, paths in datasets.items():
            best = None
            for cand in CLASSICAL_CANDIDATES[name]:
                m = cls_model(name, cand)
                bpps = [roundtrip_cls(m, x)[1] for x in tensors[ds][:6]]
                mb = float(np.mean(bpps))
                if best is None or abs(mb - 0.9) < abs(best[1] - 0.9):
                    best = (cand, mb)
            picks[(name, ds)] = best[0]
            print(f"[pick] {name}/{ds}: setting={best[0]} (cal bpp={best[1]:.3f})",
                  flush=True)

    from pytorch_msssim import ms_ssim
    nat = pd.read_csv(NC / "natural_metrics.csv")
    nat = nat[~nat.model.isin(CLS)]              # drop old classical rows
    new_rows = []
    for name in CLS:
        for ds, paths in datasets.items():
            setting = picks[(name, ds)]
            m = cls_model(name, setting)
            prof_S, prof_D, blocks = [], [], []
            for p, x in zip(paths, tensors[ds]):
                xh, bpp = roundtrip_cls(m, x)
                og = x.squeeze(0).permute(1, 2, 0).numpy().mean(axis=2)
                rg = xh.squeeze(0).permute(1, 2, 0).numpy().mean(axis=2)
                S, D = radial_profiles(og, rg)
                bm, bh, be = block_stats(og, rg)
                mse = F.mse_loss(xh, x).item()
                new_rows.append({
                    "model": name, "dataset": ds, "image": p.stem,
                    "q": setting, "bpp": bpp,
                    "psnr": -10 * np.log10(max(mse, 1e-12)),
                    "ms_ssim": float(ms_ssim(xh.to(DEV), x.to(DEV), data_range=1.0)),
                    "lpips": float(lpips_fn()(xh.to(DEV), x.to(DEV), normalize=True)),
                })
                prof_S.append(S); prof_D.append(D)
                blocks.append({"image": p.stem, "mse": bm, "hf": bh, "energy": be})
            np.savez_compressed(NC / f"radial_{name}_{ds}.npz",
                                S=np.stack(prof_S), D=np.stack(prof_D),
                                images=np.array([b["image"] for b in blocks], dtype=str))
            np.savez_compressed(NC / f"blocks_{name}_{ds}.npz",
                                images=np.array([b["image"] for b in blocks], dtype=str),
                                **{f"b{i}_{k}": b[k] for i, b in enumerate(blocks)
                                   for k in ("mse", "hf", "energy")})
            print(f"[nat] {name}/{ds} done", flush=True)
    nat = pd.concat([nat, pd.DataFrame(new_rows)], ignore_index=True)
    nat.to_csv(NC / "natural_metrics.csv", index=False)

    # basis profiles at chosen settings (majority across datasets) 256 & 512
    fine_prof = {}
    fine_rows = []
    for name in CLS:
        vals = [picks[(name, ds)] for ds in datasets]
        setting = max(set(vals), key=vals.count)
        for size in (256, 512):
            L, bpp = basis_profile_cls(name, setting, size)
            fine_prof[f"{name}_n{size}"] = L
            fine_rows.append({"model": name, "setting": setting, "size": size,
                              "bpp": bpp, "L_med": float(np.median(L))})
            print(f"[basis] {name}@{setting} n={size}: L_med={np.median(L):.4f} "
                  f"bpp={bpp:.2f}", flush=True)
    np.savez_compressed(PROF / "classical_fine_profiles.npz", **fine_prof)
    pd.DataFrame(fine_rows).to_csv(PROF / "classical_fine_summary.csv", index=False)
    with open(PROF / "classical_fine_picks.json", "w") as f:
        json.dump({f"{k[0]}/{k[1]}": v for k, v in picks.items()}, f, indent=2)

    # ------------------ B) coupling recompute (S7 core) ------------------
    def leak_profile(model, ds):
        if model == "tcm":
            return np.load(PROF / "tcm-p128_q128_n512.npz")["leakage"]
        if model in NICS:
            return np.load(PROF / f"{model}_q6_n512.npz")["leakage"]
        return fine_prof[f"{model}_n512"]

    rows = []
    for model in NICS + CLS:
        for ds in datasets:
            f = NC / f"radial_{model}_{ds}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            S, D = z["S"], z["D"]
            rho = D / (S + D + 1e-12)
            Lk = np.clip(leak_profile(model, ds), 0, 1)
            centers = (np.arange(NUM_BINS) + 0.5) / NUM_BINS
            L_r = np.interp(centers, np.linspace(0, 1, len(Lk)), Lk)
            L_t = (rho * L_r[None, :]).mean(axis=1)
            rho_b = rho.mean(axis=1)
            for i, img in enumerate(z["images"]):
                rows.append({"model": model, "dataset": ds, "image": str(img),
                             "L_tilde": float(L_t[i]), "rho_bar": float(rho_b[i])})
    cp = pd.DataFrame(rows)
    cp.to_csv(S7 / "coupling_per_image.csv", index=False)

    met = nat.groupby("model")[["psnr", "ms_ssim", "lpips", "bpp"]].mean()
    agg = cp.groupby("model")[["L_tilde", "rho_bar"]].mean()
    tbl = met.join(agg)
    tbl["ratio"] = tbl["L_tilde"] / tbl["rho_bar"]
    tbl.to_csv(S7 / "coupling_table.csv")
    nic_t = tbl.loc[[m for m in NICS if m in tbl.index]]
    sp = {t: spearmanr(nic_t["L_tilde"], nic_t[t])[0]
          for t in ("psnr", "ms_ssim", "lpips")}
    print("[Spearman NIC-only]", {k: round(v, 3) for k, v in sp.items()}, flush=True)

    # per-dataset ranking consistency
    ranks = {}
    for ds in datasets:
        sub = cp[(cp.dataset == ds) & (cp.model.isin(NICS))].groupby(
            "model")["L_tilde"].mean()
        ranks[ds] = sub.rank().reindex(NICS).tolist()
    consist = all(np.allclose(ranks["kodak"], ranks[ds]) for ds in ("clic", "div2k"))
    print("[ranking identical across datasets]:", consist, flush=True)

    # ------------------ C) Table I regeneration (fixed) ------------------
    prof_sum = pd.read_csv(PROF / "profiles_summary.csv")

    def lk512(m):
        if m == "tcm":
            return float(prof_sum[(prof_sum.model == "tcm-p128") &
                                  (prof_sum["size"] == 512)]["L_k"].iloc[0])
        if m in NICS:
            return float(prof_sum[(prof_sum.model == m) & (prof_sum["size"] == 512) &
                                  (prof_sum.q == 6)]["L_k"].iloc[0])
        return float(np.median(fine_prof[f"{m}_n512"]))

    PRETTY = {"bmshj2018-factorized": "BMSHJ2018-Factorized",
              "bmshj2018-hyperprior": "BMSHJ2018-Hyperprior",
              "mbt2018-mean": "MBT2018-Mean", "mbt2018": "MBT2018",
              "cheng2020-anchor": "Cheng2020-Anchor",
              "cheng2020-attn": "Cheng2020-Attention",
              "tcm": "TCM", "ftic": "FTIC",
              "jpeg": "JPEG", "webp": "WebP", "jpegxl": "JPEG XL"}
    rows1 = []
    for m in NICS + CLS:
        a = nat[nat.model == m]
        c = cp[cp.model == m]
        rows1.append({"model": m, "L_k": lk512(m),
                      "bpp_m": a.bpp.mean(), "bpp_s": a.bpp.std(),
                      "psnr_m": a.psnr.mean(), "psnr_s": a.psnr.std(),
                      "ssim_m": a.ms_ssim.mean(), "ssim_s": a.ms_ssim.std(),
                      "lpips_m": a.lpips.mean(), "lpips_s": a.lpips.std(),
                      "lt_m": c.L_tilde.mean(), "lt_s": c.L_tilde.std(),
                      "ratio": c.L_tilde.mean() / c.rho_bar.mean()})
    df1 = pd.DataFrame(rows1)
    df1.to_csv(T1 / "table1_agg.csv", index=False)
    nicdf = df1[df1.model.isin(NICS)].sort_values("lt_m", ascending=False)
    clsdf = df1[df1.model.isin(CLS)].sort_values("lt_m", ascending=False)
    best = {"L_k": df1.L_k.min(), "psnr": df1.psnr_m.max(), "ssim": df1.ssim_m.max(),
            "lpips": df1.lpips_m.min(), "lt": df1.lt_m.min(), "ratio": df1.ratio.min()}

    def bold(v, b, s):
        return (r"$\mathbf{" + s + "}$") if np.isclose(v, b) else ("$" + s + "$")

    lines = [r"\begin{tabular}{@{}l|c|cccccc@{}}", r"\toprule",
             r"Codec & $L_k\downarrow$ & bpp & PSNR$\uparrow$ (dB) & MS-SSIM$\uparrow$ "
             r"& LPIPS$\downarrow$ & $\tilde{\mathcal{L}}\downarrow$ "
             r"& $\tilde{\mathcal{L}}/\bar{\rho}$ \\", r"\midrule"]
    for block, dfb in (("nic", nicdf), ("cls", clsdf)):
        for _, r in dfb.iterrows():
            cells = [PRETTY[r.model],
                     bold(r.L_k, best["L_k"], f"{r.L_k:.3f}"),
                     f"${r.bpp_m:.2f} \\pm {r.bpp_s:.2f}$",
                     bold(r.psnr_m, best["psnr"], f"{r.psnr_m:.2f} \\pm {r.psnr_s:.2f}"),
                     bold(r.ssim_m, best["ssim"], f"{r.ssim_m:.3f} \\pm {r.ssim_s:.3f}"),
                     bold(r.lpips_m, best["lpips"], f"{r.lpips_m:.3f} \\pm {r.lpips_s:.3f}"),
                     bold(r.lt_m, best["lt"], f"{r.lt_m:.4f} \\pm {r.lt_s:.4f}"),
                     bold(r.ratio, best["ratio"], f"{r.ratio:.4f}")]
            lines.append(" & ".join(cells) + r" \\")
        if block == "nic":
            lines.append(r"\cmidrule{1-8}")
    lines.append(r"\cmidrule(lr){1-8}")
    lines.append(
        r"\multicolumn{8}{@{}l}{\footnotesize\textit{NIC-only Spearman} (8 models):\quad"
        + f"$\\rho_s(\\tilde{{\\mathcal{{L}}}},\\mathrm{{PSNR}})={sp['psnr']:.2f}$,\\quad"
        + f"$\\rho_s(\\tilde{{\\mathcal{{L}}}},\\mathrm{{MS}}\\mbox{{-}}\\mathrm{{SSIM}})={sp['ms_ssim']:.2f}$,\\quad"
        + f"$\\rho_s(\\tilde{{\\mathcal{{L}}}},\\mathrm{{LPIPS}})={sp['lpips']:.2f}$.}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (T1 / "table1.tex").write_text("\n".join(lines))
    print(df1.round(4).to_string(), flush=True)

    # ------------------ D) tcm-p64 single-freq sweeps ------------------
    Dm = build_dct_basis(256)
    m = load_model("tcm", 1, DEV, p=64, base_dir=BASE)
    m.eval()
    prof64 = {}
    for which, s in (("x2", 0.225 * 256), ("x1", 0.45 * np.sqrt(256))):
        errs = []
        for k in range(256):
            if which == "x2":
                X = np.outer(Dm[:, k], Dm[:, k])
            else:
                X = np.outer(Dm[:, k], np.ones(256) / np.sqrt(256))
            img = 0.5 + s * X
            img3 = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
            x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).to(DEV)
            with torch.no_grad():
                xh = m(x)["x_hat"].clamp(0, 1)
            rec = (xh.squeeze(0).permute(1, 2, 0).cpu().numpy().mean(axis=2) - 0.5) / s
            errs.append(float(np.sum((X - rec) ** 2)))
        prof64[f"tcm64_{which}_frob2"] = np.array(errs)
        print(f"[tcm-p64 {which}] med={np.median(errs):.4f}", flush=True)
    np.savez_compressed(SF / "singlefreq_tcm64.npz", **prof64)
    del m
    torch.cuda.empty_cache()

    # ------------------ E) figures: floor + sweep ------------------
    # matched configs at 256 with honest bpp policy
    def prof256(model):
        if model == "tcm":
            return np.load(PROF / "tcm-p64_q64_n256.npz")["leakage"], 0.71
        if model in NICS:
            sub = prof_sum[(prof_sum.model == model) & (prof_sum["size"] == 256)].dropna(subset=["bpp"])
            r = sub.iloc[(sub["bpp"] - 1.0).abs().argsort().iloc[0]]
            return np.load(PROF / f"{model}_q{int(r.q)}_n256.npz")["leakage"], float(r.bpp)
        return fine_prof[f"{model}_n256"], [x for x in fine_rows
                                            if x["model"] == model and x["size"] == 256][0]["bpp"]

    profs, bpps = {}, {}
    for mname in NICS + CLS:
        profs[mname], bpps[mname] = prof256(mname)
    in_band = [mname for mname in profs if 0.5 <= bpps[mname] <= 1.5]
    floor = np.min(np.stack([profs[mname] for mname in in_band]), axis=0)
    print("[floor] in-band codecs:", {mname: round(bpps[mname], 2) for mname in in_band},
          flush=True)
    print("[floor] excluded:", {mname: round(bpps[mname], 2) for mname in profs
                                if mname not in in_band}, flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sm = lambda v: np.convolve(v, np.ones(9) / 9, "same")
    for mname in NICS + CLS:
        ls = "--" if mname in CLS else "-"
        alpha, lw = (0.35, 1.0) if mname not in in_band else (1.0, 1.2)
        ax.plot(sm(profs[mname] - floor), ls=ls, lw=lw, alpha=alpha,
                label=f"{mname} ({bpps[mname]:.2f}bpp"
                      f"{'*' if mname not in in_band else ''})")
    ax.set_xlabel("frequency index k")
    ax.set_ylabel(r"excess leakage $\Delta L_k$")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(S4 / "fig_excess_leakage_real.png", dpi=150)
    plt.close(fig)

    sfz = np.load(SF / "singlefreq_profiles.npz")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    show = ["cheng2020-anchor", "cheng2020-attn", "bmshj2018-factorized",
            "jpeg", "jpegxl"]
    sm5 = lambda v: np.convolve(v, np.ones(5) / 5, "same")
    for mname in show:
        axes[0].semilogy(sm5(sfz[f"{mname}_x2_frob2"]) + 1e-6, lw=1.1, label=mname)
        axes[1].semilogy(sm5(sfz[f"{mname}_x1_frob2"]) + 1e-6, lw=1.1, label=mname)
    axes[0].semilogy(sm5(prof64["tcm64_x2_frob2"]) + 1e-6, lw=1.1, label="tcm (p64)")
    axes[1].semilogy(sm5(prof64["tcm64_x1_frob2"]) + 1e-6, lw=1.1, label="tcm (p64)")
    axes[0].set_ylabel(r"$\|X_k^{(2)}-\hat X\|_F^2$")
    axes[1].set_ylabel(r"$\|X_k^{(1)}-\hat X\|_F^2$")
    axes[1].set_xlabel("frequency index $k$")
    for ax in axes:
        ax.legend(fontsize=6, ncol=3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(S7 / "fig_singlefreq_sweep.png", dpi=150)
    plt.close(fig)

    print("S10_DONE", flush=True)


if __name__ == "__main__":
    main()
