from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib as mpl


def pivot(df, model, value_col, x_col, sizes):
    g = df[df['Model'] == model]
    xs = sorted(g[x_col].dropna().unique())
    grid = np.full((len(sizes), len(xs)), np.nan)
    for i, size in enumerate(sizes):
        gs = g[g['Size'] == size]
        for j, x in enumerate(xs):
            v = gs.loc[gs[x_col] == x, value_col]
            if len(v) > 0:
                grid[i, j] = float(v.iloc[0])
    return xs, grid


def fig_heatmap_grid(df, models, sizes, x_col, value_col, title, cmap, fname):
    cols = len(models)
    fig, axes = plt.subplots(len(sizes), cols, figsize=(4*cols, 0.7+1.6*len(sizes)), sharex='col', sharey='row')
    if len(sizes) == 1:
        axes = np.expand_dims(axes, 0)
    if cols == 1:
        axes = np.expand_dims(axes, 1)
    vmin = df[value_col].min()
    vmax = df[value_col].max()
    for c, model in enumerate(models):
        xs, grid = pivot(df, model, value_col, x_col, sizes)
        im = axes[0, c].imshow(grid[0:1, :], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
        for r in range(len(sizes)):
            axes[r, c].imshow(grid[r:r+1, :], vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
            if c == 0:
                axes[r, c].set_ylabel(str(sizes[r]))
            else:
                axes[r, c].set_yticks([])
            if r == len(sizes) - 1:
                axes[r, c].set_xticks(range(len(xs)))
                axes[r, c].set_xticklabels(xs)
                axes[r, c].set_xlabel(x_col)
            else:
                axes[r, c].set_xticks([])
            axes[r, c].set_title(model if r == 0 else '')
    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
    out = Path(fname)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_ex1_dct_grid(root_dir: Path, size: int, models, q_list=(1, 6), pretty_names=None, out_base='ex1_dct_grid', metric_key='L_k'):
    """Create a compact grid: two rows (q=1 and q=6), columns: Original + models.
    Shows decompressed DCT images and a one-line metric summary under each.
    Images are read from results/<model>/<size>/q_<q>/decompressed_dct_rgb.png
    Metrics from results/<model>/<size>/metrics_summary.csv (matching q).
    """
    root = Path(root_dir)
    pretty = pretty_names or {}
    cols = 1 + len(models)
    rows = len(q_list)
    fig_h = 2.2*rows + 0.45*rows  # space for captions
    fig_w = 2.2*cols
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    # Helper to draw one cell
    def draw_image(ax, img_path, title=None):
        try:
            img = plt.imread(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
        ax.set_axis_off()
        if title is not None:
            ax.set_title(title, fontsize=10)

    # Try to fetch an original image from the first available model/q
    def get_original_for_q(q):
        for m in models:
            p = root / m / str(size) / f'q_{q}' / 'original_dct_rgb.png'
            if p.exists():
                return p
        # fallback
        return None

    # Read per-size metrics for a model into dict keyed by q/p
    def read_metrics_row(model, q):
        csv_path = root / model / str(size) / 'metrics_summary.csv'
        if not csv_path.exists():
            return None
        try:
            dfm = pd.read_csv(csv_path)
            if 'q' in dfm.columns:
                row = dfm[dfm['q'] == q]
            else:
                row = dfm[dfm['p'] == q]
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception:
            return None

    # Caption builder
    def caption_from_row(r):
        if r is None:
            return '—'
        def fmt_smart(x):
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return '—'
            try:
                v = float(x)
            except Exception:
                return '—'
            if v == 0.0:
                return '0'
            for dec in (3, 4, 5, 6, 7, 8, 9):
                s = f"{v:.{dec}f}"
                if float(s) != 0.0:
                    return s
            return f"{v:.9f}"
        label = {
            'L_k': 'L',
            'ODR_k': 'ODR',
            '|Delta_c_k|': '|Δc|',
            's_k': 's',
            'H_k_bits': 'H',
            'CE_k(w=2)': 'CE2',
        }.get(metric_key, metric_key)
        return f"{label}={fmt_smart(r.get(metric_key))}"

    # Draw grid
    for r, q in enumerate(q_list):
        # Original column
        orig_path = get_original_for_q(q)
        draw_image(axes[r, 0], orig_path, title=('Original' if r == 0 else None))
        axes[r, 0].text(0.5, -0.15, '—', transform=axes[r, 0].transAxes, ha='center', va='top', fontsize=8)
        # Models
        for c, m in enumerate(models, start=1):
            dct_path = root / m / str(size) / f'q_{q}' / 'decompressed_dct_rgb.png'
            title = pretty.get(m, m) if r == 0 else None
            draw_image(axes[r, c], dct_path, title=title)
            metrics_row = read_metrics_row(m, q)
            axes[r, c].text(0.5, -0.07, caption_from_row(metrics_row), transform=axes[r, c].transAxes,
                            ha='center', va='top', fontsize=8)
        # Left row label (Q1/Q6)
        axes[r, 0].text(-0.22, 0.5, f"q={q}", transform=axes[r, 0].transAxes, rotation=90,
                        va='center', ha='center', fontsize=10)

    # No global title; tighten layout and use tight bbox on save
    fig.tight_layout(rect=[0, 0, 1, 1])
    outdir = root / 'paper'
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{out_base}_size{size}.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(outdir / f'{out_base}_size{size}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_ex1_dct_grid_allmodels(root_dir: Path, size: int, models, pretty_names=None, out_base='ex1_dct_grid_allmodels', metric_key='L_k'):
    """Two rows (q=1 and q=6) × N model columns (10 models). Each cell shows
    the decompressed DCT image; below it, a compact two-line metric caption.
    Avoids overlap by using a dedicated caption row below each image.
    Special handling: TCM uses p, mapped as q=1 -> p=64, q=6 -> p=128.
    """
    root = Path(root_dir)
    pretty = pretty_names or {}
    q_list = [1, 6]
    ncols = len(models)
    nrows = len(q_list)*2  # image row + caption row per q
    # Height ratios: more space for images, thin for captions
    height_ratios = []
    for _ in q_list:
        height_ratios.extend([1.0, 0.12])
    fig_w = 2.0*ncols
    fig_h = 2.1*len(q_list) + 0.35*len(q_list)
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.08, wspace=0.05, height_ratios=height_ratios)

    def img_path_for(model, q):
        if model == 'tcm':
            pval = 64 if q == 1 else 128
            return root / model / str(size) / f'p_{pval}' / 'decompressed_dct_rgb.png'
        return root / model / str(size) / f'q_{q}' / 'decompressed_dct_rgb.png'

    def read_metrics_row(model, q):
        csv_path = root / model / str(size) / 'metrics_summary.csv'
        if not csv_path.exists():
            return None
        try:
            dfm = pd.read_csv(csv_path)
            if model == 'tcm' and 'p' in dfm.columns:
                pval = 64 if q == 1 else 128
                row = dfm[dfm['p'] == pval]
            else:
                row = dfm[dfm['q'] == q] if 'q' in dfm.columns else pd.DataFrame()
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception:
            return None

    def make_caption(r):
        if r is None:
            return ("—", '')
        f3 = lambda x: ('%.3f' % float(x)) if pd.notna(x) else '—'
        label = {
            'L_k': 'L',
            'ODR_k': 'ODR',
            '|Delta_c_k|': '|Δc|',
            's_k': 's',
            'H_k_bits': 'H',
            'CE_k(w=2)': 'CE2',
        }.get(metric_key, metric_key)
        return (f"{label}={f3(r.get(metric_key))}", '')

    for i, q in enumerate(q_list):
        img_row = 2*i
        cap_row = 2*i + 1
        for j, m in enumerate(models):
            # Image cell
            ax = fig.add_subplot(gs[img_row, j])
            p = img_path_for(m, q)
            try:
                ax.imshow(plt.imread(p))
            except Exception:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=9)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(pretty.get(m, m), fontsize=9, fontweight='bold')
            # Caption cell
            axc = fig.add_subplot(gs[cap_row, j])
            axc.set_axis_off()
            r = read_metrics_row(m, q)
            l1, l2 = make_caption(r)
            axc.text(0.5, 0.95, l1, ha='center', va='top', fontsize=7)
            if l2:
                axc.text(0.5, 0.2, l2, ha='center', va='top', fontsize=7)
        # Left label q on the first image axis to avoid extra margins
        # Locate first image axis of the row (column 0)
        first_ax = fig.add_subplot(gs[img_row, 0])
        first_ax.text(-0.22, 0.5, f"q={q}", transform=first_ax.transAxes, rotation=90,
                      va='center', ha='center', fontsize=10)
        first_ax.set_axis_off()

    # No global title; tighten and save with tight bbox
    fig.tight_layout(rect=[0, 0, 1, 1])
    outdir = root / 'paper'
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{out_base}_size{size}.png', dpi=250, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(outdir / f'{out_base}_size{size}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_heatmap_plus_bars(df, models, sizes, x_col, fname):
    # Left block: high CE heatmaps; Right block: bars (best high CE and high L per size per model)
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(len(sizes), len(models) + 1, width_ratios=[1]*len(models) + [1.1], wspace=0.15, hspace=0.2)

    # Heatmaps
    all_ce = df['high_CE_k(w=2)'].dropna()
    vmin, vmax = (all_ce.min(), all_ce.max()) if len(all_ce) else (0.0, 1.0)
    im = None
    for c, model in enumerate(models):
        axcol = [fig.add_subplot(gs[r, c]) for r in range(len(sizes))]
        xs, grid = pivot(df, model, 'high_CE_k(w=2)', x_col, sizes)
        for r, ax in enumerate(axcol):
            im = ax.imshow(grid[r:r+1, :], vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto')
            if c == 0:
                ax.set_ylabel(str(sizes[r]))
            else:
                ax.set_yticks([])
            if r == len(sizes) - 1:
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xs)
                ax.set_xlabel(x_col)
            else:
                ax.set_xticks([])
            ax.set_title(model if r == 0 else '')
    cax = fig.add_subplot(gs[:, -1])
    fig.colorbar(im, cax=cax, label='high_CE(w=2)')

    out = Path(fname)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle('High-frequency retention across models and sizes')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_mega_grids(df, models, sizes, x_col, out_path, include_gaps=True, nic_only=False):
    if nic_only:
        models = [m for m in models if m.startswith('cheng') or m == 'tcm']
    rows = len(sizes)
    cols = len(models)

    # Prepare CE grid values range
    ce_col = 'high_CE_k(w=2)'
    l_col = 'high_L_k'
    ce_min = float(df[ce_col].min()) if ce_col in df.columns else 0.0
    ce_max = float(df[ce_col].max()) if ce_col in df.columns else 1.0
    l_min = float(df[l_col].min()) if l_col in df.columns else 0.0
    l_max = float(df[l_col].max()) if l_col in df.columns else 1.0

    gap_ce_col = 'gap_CE_low_high'
    gap_l_col = 'gap_L_high_low'
    gmin = float(df[gap_ce_col].min()) if gap_ce_col in df.columns else -1.0
    gmax = float(df[gap_ce_col].max()) if gap_ce_col in df.columns else 1.0
    center = 0.0
    norm = TwoSlopeNorm(vmin=min(gmin, -1e-6), vcenter=center, vmax=max(gmax, 1e-6))

    # Layout: CE grid on top, leakage grid below, optional gap CE at bottom
    n_panels = 3 if include_gaps and gap_ce_col in df.columns else 2
    fig = plt.figure(figsize=(4*cols, 0.7 + 1.7*rows*n_panels))
    gs = fig.add_gridspec(n_panels*rows, cols, hspace=0.15, wspace=0.1)

    def draw_grid(value_col, row_block, vmin, vmax, cmap, norm_local=None):
        for c, model in enumerate(models):
            xs, grid = pivot(df, model, value_col, x_col, sizes)
            for r in range(rows):
                ax = fig.add_subplot(gs[row_block*rows + r, c])
                im = ax.imshow(grid[r:r+1, :], vmin=vmin, vmax=vmax, cmap=cmap,
                               aspect='auto', norm=norm_local)
                if c == 0:
                    ax.set_ylabel(str(sizes[r]))
                else:
                    ax.set_yticks([])
                if r == rows - 1:
                    ax.set_xticks(range(len(xs)))
                    ax.set_xticklabels(xs)
                    ax.set_xlabel(x_col)
                else:
                    ax.set_xticks([])
                ax.set_title(model if r == 0 else '')
        return im

    im1 = draw_grid(ce_col, 0, ce_min, ce_max, 'viridis', None)
    im2 = draw_grid(l_col, 1, l_min, l_max, 'magma_r', None)
    if n_panels == 3:
        im3 = draw_grid(gap_ce_col, 2, None, None, 'coolwarm', norm)

    # Colorbars
    cax1 = fig.add_axes([0.92, 0.67 if n_panels==3 else 0.58, 0.015, 0.2])
    fig.colorbar(im1, cax=cax1, label='high_CE(w=2)')
    cax2 = fig.add_axes([0.92, 0.38 if n_panels==3 else 0.3, 0.015, 0.2])
    fig.colorbar(im2, cax=cax2, label='high_L_k')
    if n_panels == 3:
        cax3 = fig.add_axes([0.92, 0.09, 0.015, 0.2])
        fig.colorbar(im3, cax=cax3, label='gap CE (low−high)')

    title = 'High-frequency CE, Leakage' + (' and CE gap' if n_panels==3 else '')
    if nic_only:
        title += ' — NIC only'
    fig.suptitle(title)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _build_hq_matrix(df, models, sizes, metric_col):
    # High-quality slice per model (max q or max p)
    mat = np.full((len(models), len(sizes)), np.nan, dtype=float)
    for i, model in enumerate(models):
        gm = df[df['Model'] == model]
        x_col = 'q' if 'q' in gm.columns and gm['q'].notna().any() else 'p'
        if x_col in gm.columns and gm[x_col].notna().any():
            try:
                target = int(np.nanmax(gm[x_col]))
            except Exception:
                target = gm[x_col].dropna().max()
            gq = gm[gm[x_col] == target]
        else:
            gq = gm
        for j, size in enumerate(sizes):
            gs = gq[gq['Size'] == size]
            if metric_col in gs.columns and not gs[metric_col].empty:
                mat[i, j] = float(gs[metric_col].median())
    return mat


def fig_double_heatmap_hq(df, models, sizes, pretty_names, out_png, out_pdf):
    # Matrices: rows=models, cols=sizes at high quality
    m_leak = _build_hq_matrix(df, models, sizes, 'L_k')
    m_ce = _build_hq_matrix(df, models, sizes, 'CE_k(w=2)')
    # Labels
    row_labels = [pretty_names.get(m, m) for m in models]
    col_labels = [str(s) for s in sizes]
    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.8 + 0.45*len(models)))
    im0 = axes[0].imshow(m_leak, cmap='magma_r', aspect='auto')
    im1 = axes[1].imshow(m_ce, cmap='viridis', aspect='auto')
    for ax, title in zip(axes, ['Median Leakage L_k (high quality) ↓', 'Median CE(w=2) (high quality) ↑']):
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_xlabel('size')
        ax.grid(False)
        ax.set_title(title)
    axes[0].set_ylabel('model')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)


def fig_double_bars_hq(df, models, sizes, pretty_names, out_png, out_pdf):
    # Aggregate across sizes at high quality
    vals_leak = []
    vals_ce = []
    labels = []
    for model in models:
        gm = df[df['Model'] == model]
        x_col = 'q' if 'q' in gm.columns and gm['q'].notna().any() else 'p'
        if x_col in gm.columns and gm[x_col].notna().any():
            hq = gm[gm[x_col] == gm[x_col].dropna().max()]
        else:
            hq = gm
        # Median over sizes
        vals_leak.append(float(hq['L_k'].median() if 'L_k' in hq.columns else np.nan))
        vals_ce.append(float(hq['CE_k(w=2)'].median() if 'CE_k(w=2)' in hq.columns else np.nan))
        labels.append(pretty_names.get(model, model))

    # Plot side-by-side horizontal bars for compactness
    y = np.arange(len(labels))
    height = 0.4
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.8 + 0.45*len(labels)))
    axes[0].barh(y, vals_leak, color='#a63603')
    axes[0].set_title('Median Leakage L_k at high quality ↓')
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    axes[1].barh(y, vals_ce, color='#1b9e77')
    axes[1].set_title('Median CE(w=2) at high quality ↑')
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)

def main():
    root = Path('/home/nkalmykov/compressai_project/results')
    df = pd.read_csv(root / 'all_metrics_summary.csv')
    available = set(df['Model'].dropna().unique())
    # Enforce requested order
    desired_order = [
        'jpeg', 'jpegxl', 'webp',
        'bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018',
        'cheng2020-anchor', 'cheng2020-attn', 'tcm'
    ]
    models = [m for m in desired_order if m in available]
    sizes = sorted(df['Size'].dropna().unique(), key=lambda s: int(str(s).split('x')[0]))
    x_col = 'q' if df['q'].notna().any() else 'p'

    # Optional heatmaps: only if band-wise columns exist
    if 'high_CE_k(w=2)' in df.columns and 'high_L_k' in df.columns:
        # Grid A: high-frequency CE heatmaps for all models
        fig_heatmap_grid(df, models, sizes, x_col, 'high_CE_k(w=2)', 'High-frequency CE (w=2) ↑', 'viridis', root / 'paper/overview_high_ce_grid.png')
        # Grid B: high-frequency leakage heatmaps for all models
        fig_heatmap_grid(df, models, sizes, x_col, 'high_L_k', 'High-frequency leakage ↓', 'magma_r', root / 'paper/overview_high_leakage_grid.png')
        # Composite C: only CE heatmaps with shared colorbar on the right
        fig_heatmap_plus_bars(df, models, sizes, x_col, root / 'paper/overview_high_ce_grid_wbar.png')

    # Mega composites (multi-panel): only if band-wise columns exist
    if 'high_CE_k(w=2)' in df.columns and 'high_L_k' in df.columns:
        fig_mega_grids(df, models, sizes, x_col, root / 'paper/mega_ce_leak.png', include_gaps=False)
        fig_mega_grids(df, models, sizes, x_col, root / 'paper/mega_ce_leak_gap.png', include_gaps=True)
        fig_mega_grids(df, models, sizes, x_col, root / 'paper/mega_ce_leak_nic.png', include_gaps=True, nic_only=True)

    # Compact median curves: 2 rows (L_k, CE), columns are models; x=q or p per model; lines are sizes
    ncols = len(models)
    fig, axes = plt.subplots(2, ncols, figsize=(3.3*ncols, 5.2), sharey='row')
    if ncols == 1:
        axes = np.expand_dims(axes, 1)
    # Global y-lims for consistency
    l_vals = df['L_k'].dropna()
    ce_vals = df['CE_k(w=2)'].dropna()
    l_min, l_max = (float(l_vals.min()), float(l_vals.max())) if len(l_vals) else (0.0, 1.0)
    ce_min, ce_max = (float(ce_vals.min()), float(ce_vals.max())) if len(ce_vals) else (0.0, 1.0)

    for c, model in enumerate(models):
        gm = df[df['Model'] == model]
        x_m = 'q' if 'q' in gm.columns and gm['q'].notna().any() else 'p'
        for size in sizes:
            gs = gm[gm['Size'] == size].sort_values(x_m)
            if gs.empty:
                continue
            x = gs[x_m]
            if 'L_k' in gs.columns:
                axes[0, c].plot(x, gs['L_k'], '-o', label=str(size))
            if 'CE_k(w=2)' in gs.columns:
                axes[1, c].plot(x, gs['CE_k(w=2)'], '-o', label=str(size))
        axes[0, c].set_title(f"{model}")
        axes[0, c].set_xlabel(x_m)
        axes[1, c].set_xlabel(x_m)
        axes[0, c].grid(True, alpha=0.3)
        axes[1, c].grid(True, alpha=0.3)
        if c == 0:
            axes[0, c].set_ylabel('L_k (median) ↓')
            axes[1, c].set_ylabel('CE_k(w=2) (median) ↑')
        axes[0, c].set_ylim(l_min, l_max)
        axes[1, c].set_ylim(ce_min, ce_max)
    # One legend outside
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Size', fontsize=9, loc='upper center', ncol=min(len(sizes), 6), bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Median leakage and median CE across qualities (q) or p (TCM)')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    (root / 'paper').mkdir(parents=True, exist_ok=True)
    fig.savefig(root / 'paper/compact_median_overview.png', dpi=200)
    plt.close(fig)

    # Multi-model 3D: Median leakage per model on one figure
    # y=size, x=q/p, z=L_k (median)
    # Typography: serif + mathtext (STIX)
    mpl.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
    })

    pretty = {
        'jpeg': 'JPEG',
        'jpegxl': 'JPEG XL',
        'webp': 'WebP',
        'bmshj2018-factorized': 'BMShj18-Fact',
        'bmshj2018-hyperprior': 'BMShj18-Hyper',
        'mbt2018-mean': 'MBT18-Mean',
        'mbt2018': 'MBT18',
        'cheng2020-anchor': 'Cheng2020-Anchor',
        'cheng2020-attn': 'Cheng2020-Attn',
        'tcm': 'TCM',
    }

    # Re-apply order for 3D panels with extended list
    desired_order = [
        'jpeg', 'jpegxl', 'webp',
        'bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018',
        'cheng2020-anchor', 'cheng2020-attn', 'tcm'
    ]
    models = [m for m in desired_order if m in models]
    n_models = len(models)
    # Force layout: 5 columns per row (2 rows for 10 models)
    ncols = 5 if n_models >= 5 else n_models
    nrows = int(np.ceil(n_models / ncols))
    fig3 = plt.figure(figsize=(4.6*ncols, 3.8*nrows))
    # Consistent colors/markers per size across panels
    cmap = plt.get_cmap('tab10')
    color_map = {s: cmap(i % 10) for i, s in enumerate(sizes)}
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*', 'h', '<']
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(sizes)}

    for idx, model in enumerate(models):
        ax = fig3.add_subplot(nrows, ncols, idx + 1, projection='3d')
        gm = df[df['Model'] == model]
        x_m = 'q' if 'q' in gm.columns and gm['q'].notna().any() else 'p'
        size_to_int = {s: int(str(s).split('x')[0]) for s in sizes}
        for size in sizes:
            gs = gm[gm['Size'] == size].sort_values(x_m)
            if gs.empty or 'L_k' not in gs.columns:
                continue
            x = gs[x_m].values
            y = [size_to_int[size]] * len(x)  # y=size
            z = gs['L_k'].values              # z=median leakage
            ax.plot(x, y, z, '-', color=color_map[size], marker=marker_map[size], label=str(size))
        # Dynamic labelpad and tick formatting for z
        zmin = float(np.nanmin(gm['L_k'])) if 'L_k' in gm.columns else 0.0
        zmax = float(np.nanmax(gm['L_k'])) if 'L_k' in gm.columns else 1.0
        z_span = abs(zmax - zmin)
        z_pad = 8 if z_span >= 0.1 else 16
        from matplotlib.ticker import FormatStrFormatter
        if z_span < 0.01:
            dec = 3
        elif z_span < 0.1:
            dec = 2
        else:
            dec = 1
        ax.zaxis.set_major_formatter(FormatStrFormatter(f'%0.{dec}f'))

        # Ticks: make sure we show q=1..6 or actual p values; y ticks at available sizes
        if x_m == 'q':
            ax.set_xticks([1, 2, 3, 4, 5, 6])
        else:
            xp = sorted(gm['p'].dropna().unique()) if 'p' in gm.columns else []
            ax.set_xticks(xp)
        sizes_in_model = sorted(gm['Size'].unique(), key=lambda s: int(str(s).split('x')[0]))
        # Force y ticks every 250 (250, 500, 750, 1000, ... ) within the present size range
        min_size = min(int(str(s).split('x')[0]) for s in sizes_in_model)
        max_size = max(int(str(s).split('x')[0]) for s in sizes_in_model)
        start = 250
        stop = ((max_size + 249) // 250) * 250
        y_ticks = list(range(start, stop + 1, 250))
        ax.set_yticks(y_ticks)

        ax.set_title(pretty.get(model, model), pad=1)
        ax.set_xlabel(r'$%s$' % ('q' if x_m == 'q' else 'p'), labelpad=2)
        ax.set_ylabel('size', labelpad=8)
        # Pull z-label closer as requested
        ax.set_zlabel(r'$L_k$ (median)', labelpad=max(2, z_pad - 6))
        ax.tick_params(axis='x', pad=2, labelsize=10)
        ax.tick_params(axis='y', pad=2, labelsize=9)
        ax.tick_params(axis='z', pad=2, labelsize=10)
        ax.view_init(elev=25, azim=-60)
    # Tighter layout: less spacing between subplots
    fig3.subplots_adjust(wspace=0.1, hspace=0.25)
    # Custom legend for all sizes (ensures 64 and 128 are present), placed at top-left under the title
    from matplotlib.lines import Line2D
    legend_handles = []
    for s in sizes:
        legend_handles.append(Line2D([0], [0], color=color_map[s], marker=marker_map[s], lw=2, ls='-', label=str(s)))
    # Title with extra vertical headroom
    fig3.suptitle('Median Leakage (x = quality, y = image size, z = L_k (leakage))', y=0.988)
    if legend_handles:
        fig3.legend(
            handles=legend_handles,
            fontsize=10,
            loc='upper center',
            ncol=len(sizes),
            bbox_to_anchor=(0.5, 0.962),
            frameon=False,
            borderaxespad=0.8,
            handlelength=2.2,
            columnspacing=1.5,
            labelspacing=0.9,
            handletextpad=0.7,
        )
    # Increase top margin gap so title+legend do not collide with plots
    fig3.subplots_adjust(top=0.86, bottom=0.06)
    # Save both PNG and high-quality PDF
    fig3.savefig(root / 'paper/overview_3d_median_leak_allmodels.png', dpi=250)
    fig3.savefig(root / 'paper/overview_3d_median_leak_allmodels.pdf', format='pdf')
    plt.close(fig3)

    # Additional 3D figures for other median metrics
    metrics_specs = [
        ('ODR_k', 'Median ODR (x = quality, y = image size, z = ODR_k)', r'$ODR_k$ (median)', 'overview_3d_median_odr_allmodels'),
        ('|Delta_c_k|', 'Median |Delta c_k| (x = quality, y = image size, z = |Δc_k|)', r'$|\Delta c_k|$ (median)', 'overview_3d_median_centroidshift_allmodels'),
        ('s_k', 'Median Spread (x = quality, y = image size, z = s_k)', r'$s_k$ (median)', 'overview_3d_median_spread_allmodels'),
        ('H_k_bits', 'Median Entropy (x = quality, y = image size, z = H_k (bits))', r'$H_k$ (bits, median)', 'overview_3d_median_entropy_allmodels'),
        ('CE_k(w=2)', 'Median CE (w=2) (x = quality, y = image size, z = CE_k(w=2))', r'$CE_k(w=2)$ (median)', 'overview_3d_median_ce_w2_allmodels'),
    ]

    for metric_col, suptitle, zlabel, out_base in metrics_specs:
        if metric_col not in df.columns:
            continue
        figm = plt.figure(figsize=(4.6*ncols, 3.8*nrows))
        cmap = plt.get_cmap('tab10')
        color_map = {s: cmap(i % 10) for i, s in enumerate(sizes)}
        markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*', 'h', '<']
        marker_map = {s: markers[i % len(markers)] for i, s in enumerate(sizes)}
        for idx, model in enumerate(models):
            ax = figm.add_subplot(nrows, ncols, idx + 1, projection='3d')
            gm = df[df['Model'] == model]
            x_m = 'q' if 'q' in gm.columns and gm['q'].notna().any() else 'p'
            size_to_int = {s: int(str(s).split('x')[0]) for s in sizes}
            for size in sizes:
                gs = gm[gm['Size'] == size].sort_values(x_m)
                if gs.empty or metric_col not in gs.columns:
                    continue
                x = gs[x_m].values
                y = [size_to_int[size]] * len(x)
                z = gs[metric_col].values
                ax.plot(x, y, z, '-', color=color_map[size], marker=marker_map[size], label=str(size))
            # Ticks configuration
            if x_m == 'q':
                ax.set_xticks([1, 2, 3, 4, 5, 6])
            else:
                xp = sorted(gm['p'].dropna().unique()) if 'p' in gm.columns else []
                ax.set_xticks(xp)
            sizes_in_model = sorted(gm['Size'].unique(), key=lambda s: int(str(s).split('x')[0]))
            if sizes_in_model:
                min_size = min(int(str(s).split('x')[0]) for s in sizes_in_model)
                max_size = max(int(str(s).split('x')[0]) for s in sizes_in_model)
                start = 250
                stop = ((max_size + 249) // 250) * 250
                y_ticks = list(range(start, stop + 1, 250))
                ax.set_yticks(y_ticks)
            ax.set_title(pretty.get(model, model), pad=1)
            ax.set_xlabel(r'$%s$' % ('q' if x_m == 'q' else 'p'), labelpad=2)
            ax.set_ylabel('size', labelpad=8)
            ax.set_zlabel(zlabel, labelpad=6)
            ax.tick_params(axis='x', pad=2, labelsize=10)
            ax.tick_params(axis='y', pad=2, labelsize=9)
            ax.tick_params(axis='z', pad=2, labelsize=10)
            ax.view_init(elev=25, azim=-60)
        figm.subplots_adjust(wspace=0.1, hspace=0.25)
        from matplotlib.lines import Line2D
        legend_handles = [Line2D([0], [0], color=color_map[s], marker=marker_map[s], lw=2, ls='-', label=str(s)) for s in sizes]
        figm.suptitle(suptitle, y=0.988)
        if legend_handles:
            figm.legend(
                handles=legend_handles,
                fontsize=10,
                loc='upper center',
                ncol=len(sizes),
                bbox_to_anchor=(0.5, 0.962),
                frameon=False,
                borderaxespad=0.8,
                handlelength=2.2,
                columnspacing=1.5,
                labelspacing=0.9,
                handletextpad=0.7,
            )
        figm.subplots_adjust(top=0.86, bottom=0.06)
        figm.savefig(root / f'paper/{out_base}.png', dpi=250)
        figm.savefig(root / f'paper/{out_base}.pdf', format='pdf')
        plt.close(figm)

    # Compact paper figures (space-saving): double heatmap and double bar chart at high quality
    pretty_map = {
        'jpeg': 'JPEG', 'jpegxl': 'JPEG XL', 'webp': 'WebP',
        'bmshj2018-factorized': 'BMShj18-Fact', 'bmshj2018-hyperprior': 'BMShj18-Hyper',
        'mbt2018-mean': 'MBT18-Mean', 'mbt2018': 'MBT18',
        'cheng2020-anchor': 'Cheng2020-Anchor', 'cheng2020-attn': 'Cheng2020-Attn', 'tcm': 'TCM'
    }
    fig_double_heatmap_hq(
        df, models, sizes, pretty_map,
        root / 'paper/mini_double_heatmap_hq.png',
        root / 'paper/mini_double_heatmap_hq.pdf'
    )
    fig_double_bars_hq(
        df, models, sizes, pretty_map,
        root / 'paper/mini_double_bars_hq.png',
        root / 'paper/mini_double_bars_hq.pdf'
    )

    # Ex1: compact DCT image grid for q=1 and q=6 (NIC-focused models if available)
    ex_models = [m for m in models if m in ['mbt2018-mean','mbt2018','cheng2020-anchor','cheng2020-attn']]
    if ex_models:
        # prefer a mid-large canonical size present in CSV
        size_candidates = [256, 128, 512]
        chosen = None
        for s in size_candidates:
            if f'{s}x{s}' in df['Size'].unique():
                chosen = s
                break
        if chosen is None:
            # fallback to first size parsed from Size strings
            try:
                chosen = int(str(sizes[0]).split('x')[0])
            except Exception:
                chosen = 256
        fig_ex1_dct_grid(root, chosen, ex_models, q_list=(1, 6), pretty_names=pretty_map, out_base='ex1_dct_grid')

    # Full 10-model Ex1 with separate caption rows (avoid overlap)
    all_models_order = [
        'jpeg', 'jpegxl', 'webp',
        'bmshj2018-factorized', 'bmshj2018-hyperprior', 'mbt2018-mean', 'mbt2018',
        'cheng2020-anchor', 'cheng2020-attn', 'tcm'
    ]
    ex_all = [m for m in all_models_order if m in models]
    if ex_all:
        # choose same size as above if possible
        chosen_all = chosen if 'chosen' in locals() else int(str(sizes[0]).split('x')[0])
        models_for_size = [m for m in ex_all if not (m == 'tcm' and chosen_all in (64, 128))]
        fig_ex1_dct_grid_allmodels(root, chosen_all, models_for_size, pretty_names=pretty_map, out_base='ex1_dct_grid_allmodels')

    # Also render compact versions for smaller sizes if available: 64 and 128
    for small in [64, 128]:
        if f'{small}x{small}' in df['Size'].unique():
            models_for_small = [m for m in ex_all if m != 'tcm']
            fig_ex1_dct_grid_allmodels(root, small, models_for_small, pretty_names=pretty_map, out_base=f'ex1_dct_grid_allmodels')


if __name__ == '__main__':
    main()


