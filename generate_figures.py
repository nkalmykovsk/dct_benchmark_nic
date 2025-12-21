"""Generate composite overview figures for the paper."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe


DPI = 300

DESIRED_ORDER = [
    'jpeg',
    'jpegxl',
    'webp',
    'bmshj2018-factorized',
    'bmshj2018-hyperprior',
    'mbt2018-mean',
    'mbt2018',
    'cheng2020-anchor',
    'cheng2020-attn',
    'tcm',
]

PRETTY_NAMES = {
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


def _default_results_root():
    """Return default results directory (repo_root/results)."""
    return Path(__file__).resolve().parents[1] / 'results'


def _fmt_smart(x):
    """Format numeric value with adaptive precision."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return '—'
    try:
        v = float(x)
    except (ValueError, TypeError):
        return '—'
    if v == 0.0:
        return '0'
    for dec in (3, 4, 5, 6, 7, 8, 9):
        s = f'{v:.{dec}f}'
        if float(s) != 0.0:
            return s
    return f'{v:.9f}'


def _img_path_for(root, model, size, q):
    """Get path to decompressed DCT image."""
    if model == 'tcm':
        pval = 64 if q == 1 else 128
        return root / model / str(size) / f'p_{pval}' / 'decompressed_dct_rgb.png'
    return root / model / str(size) / f'q_{q}' / 'decompressed_dct_rgb.png'


def _read_metrics_row(root, model, size, q):
    """Read metrics row from CSV for given model/size/q."""
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


def _make_caption(r):
    """Create caption tuple from metrics row."""
    if r is None:
        return ('—', '')
    l_med = _fmt_smart(r.get('L_k'))
    l_low = _fmt_smart(r.get('L_low')) if 'L_low' in r else '—'
    l_high = _fmt_smart(r.get('L_high')) if 'L_high' in r else '—'
    return (f'L={l_med}', f'(low={l_low}, high={l_high})')


def fig_ex1_dct_grid_allmodels(
    root_dir,
    size,
    models,
    pretty_names=None,
    out_name='ex1_dct_grid_allmodels',
):
    """
    Create grid with two rows (q=1 and q=6) x N model columns.

    Each cell shows the decompressed DCT image with a metric caption.
    TCM uses p: q=1 -> p=64, q=6 -> p=128.
    """
    from matplotlib.gridspec import GridSpec

    root = Path(root_dir)
    pretty = pretty_names or {}
    q_list = [1, 6]
    ncols = len(models)
    nrows = len(q_list) * 2

    height_ratios = []
    for _ in q_list:
        height_ratios.extend([1.0, 0.12])

    fig_w = 2.0 * ncols
    fig_h = 2.1 * len(q_list) + 0.35 * len(q_list)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        hspace=0.08,
        wspace=0.05,
        height_ratios=height_ratios,
    )

    for i, q in enumerate(q_list):
        img_row = 2 * i
        cap_row = 2 * i + 1

        for j, m in enumerate(models):
            ax = fig.add_subplot(gs[img_row, j])
            p = _img_path_for(root, m, size, q)
            try:
                ax.imshow(plt.imread(p))
            except Exception:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=9)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(pretty.get(m, m), fontsize=9, fontweight='bold')

            axc = fig.add_subplot(gs[cap_row, j])
            axc.set_axis_off()
            r = _read_metrics_row(root, m, size, q)
            l1, l2 = _make_caption(r)
            axc.text(
                0.5,
                0.95,
                l1,
                ha='center',
                va='top',
                fontsize=8,
                path_effects=[pe.withStroke(linewidth=2, foreground='white')],
            )
            if l2:
                axc.text(
                    0.5,
                    0.22,
                    l2,
                    ha='center',
                    va='top',
                    fontsize=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                )

        first_ax = fig.add_subplot(gs[img_row, 0])
        first_ax.text(
            -0.22,
            0.5,
            f'q={q}',
            transform=first_ax.transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
        )
        first_ax.set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 1])
    outdir = root / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outdir / f'{out_name}_size{size}.png',
        dpi=DPI,
        bbox_inches='tight',
        pad_inches=0.02,
    )
    plt.close(fig)


def fig_compact_median_overview(df, models, sizes, root):
    """Create 2-row plot: L_k and CE vs quality for each model."""
    ncols = len(models)
    fig, axes = plt.subplots(2, ncols, figsize=(3.3 * ncols, 5.2), sharey='row')
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    l_vals = df['L_k'].dropna()
    ce_vals = df['CE_k(w=2)'].dropna()
    l_min, l_max = (
        (float(l_vals.min()), float(l_vals.max()))
        if len(l_vals)
        else (0.0, 1.0)
    )
    ce_min, ce_max = (
        (float(ce_vals.min()), float(ce_vals.max()))
        if len(ce_vals)
        else (0.0, 1.0)
    )

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

        axes[0, c].set_title(model)
        axes[0, c].set_xlabel(x_m)
        axes[1, c].set_xlabel(x_m)
        axes[0, c].grid(True, alpha=0.3)
        axes[1, c].grid(True, alpha=0.3)

        if c == 0:
            axes[0, c].set_ylabel('L_k (median) ↓')
            axes[1, c].set_ylabel('CE_k(w=2) (median) ↑')

        axes[0, c].set_ylim(l_min, l_max)
        axes[1, c].set_ylim(ce_min, ce_max)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title='Size',
        fontsize=9,
        loc='upper center',
        ncol=min(len(sizes), 6),
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle('Median leakage and median CE across qualities (q) or p (TCM)')
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    outdir = root / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / 'compact_median_overview.png', dpi=DPI)
    plt.close(fig)


def fig_3d_metric(df, models, sizes, metric_col, suptitle, zlabel, out_name, root):
    """Generate a 3D surface plot grid for a given metric."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    n_models = len(models)
    ncols = 5 if n_models >= 5 else n_models
    nrows = int(np.ceil(n_models / ncols))

    fig = plt.figure(figsize=(4.6 * ncols, 3.8 * nrows))

    cmap = plt.get_cmap('tab10')
    color_map = {s: cmap(i % 10) for i, s in enumerate(sizes)}
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '*', 'h', '<']
    marker_map = {s: markers[i % len(markers)] for i, s in enumerate(sizes)}

    for idx, model in enumerate(models):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
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
            ax.plot(
                x,
                y,
                z,
                '-',
                color=color_map[size],
                marker=marker_map[size],
                label=str(size),
            )

        # Dynamic z-axis formatting
        if metric_col in gm.columns and not gm[metric_col].dropna().empty:
            zmin = float(np.nanmin(gm[metric_col]))
            zmax = float(np.nanmax(gm[metric_col]))
            z_span = abs(zmax - zmin)
        else:
            z_span = 1.0

        if z_span < 0.01:
            dec = 3
        elif z_span < 0.1:
            dec = 2
        else:
            dec = 1
        ax.zaxis.set_major_formatter(FormatStrFormatter(f'%0.{dec}f'))

        # X ticks
        if x_m == 'q':
            ax.set_xticks([1, 2, 3, 4, 5, 6])
        else:
            xp = sorted(gm['p'].dropna().unique()) if 'p' in gm.columns else []
            ax.set_xticks(xp)

        # Y ticks
        sizes_in_model = sorted(
            gm['Size'].unique(),
            key=lambda s: int(str(s).split('x')[0]),
        )
        if sizes_in_model:
            max_size = max(int(str(s).split('x')[0]) for s in sizes_in_model)
            stop = ((max_size + 249) // 250) * 250
            y_ticks = list(range(250, stop + 1, 250))
            ax.set_yticks(y_ticks)

        ax.set_title(PRETTY_NAMES.get(model, model), pad=1)
        ax.set_xlabel(rf'${x_m}$', labelpad=2)
        ax.set_ylabel('size', labelpad=8)
        ax.set_zlabel(zlabel, labelpad=6)
        ax.tick_params(axis='x', pad=2, labelsize=10)
        ax.tick_params(axis='y', pad=2, labelsize=9)
        ax.tick_params(axis='z', pad=2, labelsize=10)
        ax.view_init(elev=25, azim=-60)

    fig.subplots_adjust(wspace=0.1, hspace=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[s],
            marker=marker_map[s],
            lw=2,
            ls='-',
            label=str(s),
        )
        for s in sizes
    ]
    fig.suptitle(suptitle, y=0.988)

    if legend_handles:
        fig.legend(
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

    fig.subplots_adjust(top=0.86, bottom=0.06)

    outdir = root / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{out_name}.png', dpi=DPI)
    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate composite overview figures for the paper.',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=_default_results_root(),
        help='Path to results directory containing all_metrics_summary.csv.',
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    csv_path = root / 'all_metrics_summary.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Could not find {csv_path}. Pass the correct path via --root.',
        )

    df = pd.read_csv(csv_path)
    available = set(df['Model'].dropna().unique())
    models = [m for m in DESIRED_ORDER if m in available]
    sizes = sorted(
        df['Size'].dropna().unique(),
        key=lambda s: int(str(s).split('x')[0]),
    )

    # Set matplotlib style
    mpl.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
    })

    # 1. Compact median overview
    fig_compact_median_overview(df, models, sizes, root)

    # 2. DCT grid for size 256
    if '256x256' in df['Size'].unique():
        models_for_256 = [
            m for m in models
            if not (m == 'tcm' and 256 in (64, 128))
        ]
        fig_ex1_dct_grid_allmodels(root, 256, models_for_256, PRETTY_NAMES)

    # 3. 3D metric figures
    metrics_specs = [
        (
            'L_k',
            'Median Leakage (x=quality, y=size, z=L_k)',
            r'$L_k$ (median)',
            'overview_3d_median_leak_allmodels',
        ),
        (
            'ODR_k',
            'Median ODR (x=quality, y=size, z=ODR_k)',
            r'$ODR_k$ (median)',
            'overview_3d_median_odr_allmodels',
        ),
        (
            '|Delta_c_k|',
            r'Median |Δc_k| (x=quality, y=size, z=|Δc_k|)',
            r'$|\Delta c_k|$ (median)',
            'overview_3d_median_centroidshift_allmodels',
        ),
        (
            's_k',
            'Median Spread (x=quality, y=size, z=s_k)',
            r'$s_k$ (median)',
            'overview_3d_median_spread_allmodels',
        ),
        (
            'H_k_bits',
            'Median Entropy (x=quality, y=size, z=H_k)',
            r'$H_k$ (bits, median)',
            'overview_3d_median_entropy_allmodels',
        ),
        (
            'CE_k(w=2)',
            'Median CE (x=quality, y=size, z=CE_k(w=2))',
            r'$CE_k(w=2)$ (median)',
            'overview_3d_median_ce_w2_allmodels',
        ),
    ]

    for metric_col, suptitle, zlabel, out_name in metrics_specs:
        if metric_col in df.columns:
            fig_3d_metric(
                df,
                models,
                sizes,
                metric_col,
                suptitle,
                zlabel,
                out_name,
                root,
            )


if __name__ == '__main__':
    main()