"""Figure generation utilities for simulation results.

This module reads per-config simulation CSVs from a results directory, computes
theta = n / (2 * k * log(p - k)), and produces three plots:
  - recovery_vs_theta.png (bar)
  - exact_frac_scatter_theta.png (per-config scatter)
  - heatmap_theta_b.png (heatmap theta vs b)

Usage (from repo root):
    PYTHONPATH=. python3 -m src.figures --raw results/raw --out results/analysis --fig results/figures
"""
from pathlib import Path
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set(style='whitegrid', context='talk', rc={'figure.dpi': 150})
except Exception:
    sns = None
import shutil
import argparse


def load_sim_results(raw_dir: Path) -> pd.DataFrame:
    pattern = str(raw_dir / 'sim_n*_p*_k*_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No simulation CSVs found matching {pattern}')

    pat = re.compile(
        r'sim_n(?P<n>\d+)_p(?P<p>\d+)_k(?P<k>\d+)_rho(?P<rho>[^_]+)_b(?P<b>[^_]+)_'
        r'sigma(?P<sigma>[^_]+)_lf(?P<lf>[^_]+)_reps(?P<reps>\d+)_seed(?P<seed>[^_]+)'
    )
    rows = []
    for f in files:
        m = pat.search(Path(f).name)
        if not m:
            continue
        n = int(m.group('n'))
        p = int(m.group('p'))
        k = int(m.group('k'))
        rho = float(m.group('rho').replace('p', '.'))
        b = float(m.group('b').replace('p', '.'))
        sigma = float(m.group('sigma').replace('p', '.'))
        lf = float(m.group('lf').replace('p', '.'))
        try:
            df = pd.read_csv(f)
        except Exception:
            continue

        exact_frac = float(df['exact_support_recovery'].mean()) if 'exact_support_recovery' in df.columns else 0.0
        tpr = float(df['tpr'].mean()) if 'tpr' in df.columns else np.nan
        fdp = float(df['fdp'].mean()) if 'fdp' in df.columns else np.nan
        theta = n / (2 * k * np.log(p - k))
        rows.append({
            'file': Path(f).name,
            'n': n, 'p': p, 'k': k, 'rho': rho, 'b': b, 'sigma': sigma,
            'lam_factor': lf, 'exact_frac': exact_frac, 'tpr': tpr, 'fdp': fdp, 'theta': theta
        })

    return pd.DataFrame(rows)



def plot_focused_theta_search(df: pd.DataFrame, out_path: Path, b_values=None, lam_values=None):
    """Create a focused publication-quality plot: panels for each b, lines per lam_factor.

    Panels (cols) are different signal magnitudes `b`. Within each panel we plot
    mean exact_frac vs theta for each lam_factor, with 95% CI shaded.
    """
    df = df.copy()
    # choose unique sorted values if not provided
    if b_values is None:
        b_values = sorted(df['b'].unique())
    if lam_values is None:
        lam_values = sorted(df['lam_factor'].unique())

    ncols = len(b_values)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, b in zip(axes, b_values):
        sub_b = df[df['b'] == b]
        for i, lf in enumerate(lam_values):
            sub = sub_b[sub_b['lam_factor'] == lf]
            if sub.empty:
                continue
            agg = sub.groupby('theta').exact_frac.agg(['mean', 'std', 'count']).reset_index().sort_values('theta')
            theta = agg['theta'].values
            mean = agg['mean'].values
            se = agg['std'].fillna(0).values / np.sqrt(np.maximum(1, agg['count'].values))
            color = plt.cm.viridis(i / max(1, len(lam_values) - 1))
            ax.plot(theta, mean, marker='o', linestyle='-', linewidth=1.8, markersize=5, color=color, label=f'lam={lf}')
            ax.fill_between(theta, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.2)

        ax.set_title(f'b = {b}')
        ax.set_xlabel('theta')
        ax.grid(True, linestyle='--', alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel('mean exact support recovery')
        ax.set_ylim(-0.02, 1.02)
        ax.legend(title='lam_factor', fontsize='small')

    plt.suptitle('Focused theta search — recovery vs theta (panels by b, lines by lam_factor)', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    try:
        plt.savefig(str(out_path.with_suffix('.pdf')), bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def plot_heatmap_theta_b(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['theta_round'] = df['theta'].round(3)
    pivot = df.groupby(['theta_round', 'b']).exact_frac.mean().reset_index()
    piv = pivot.pivot(index='theta_round', columns='b', values='exact_frac')
    plt.figure(figsize=(7, 5))
    if sns is not None:
        ax = sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis', vmin=0, vmax=1, cbar_kws={'label': 'mean exact_frac'})
    else:
        # fallback: use imshow and annotate
        arr = piv.values
        im = plt.imshow(arr, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        cbar = plt.colorbar(im)
        cbar.set_label('mean exact_frac')
        # tick labels
        plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
        # annotate
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                plt.text(j, i, f'{val:.3f}', ha='center', va='center', color='white' if val > 0.5 else 'black')
    plt.xlabel('b (signal magnitude)')
    plt.ylabel('theta_round')
    plt.title('Mean exact_frac (averaged over rho & lam_factor)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    try:
        plt.savefig(str(out_path.with_suffix('.pdf')))
    except Exception:
        pass
    plt.close()


def main(raw_dir: str, out_dir: str, fig_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir)
    figs = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load_sim_results(raw)

    out1 = out / 'focused_theta_search.png'
    out2 = out / 'heatmap_theta_b.png'

    # focused theta plot: panels by b, lines by lam_factor
    # choose b and lam_factor ordering from available values
    b_values = sorted(df['b'].unique())
    lam_values = sorted(df['lam_factor'].unique())
    plot_focused_theta_search(df, out1, b_values=b_values, lam_values=lam_values)
    plot_heatmap_theta_b(df, out2)

    # Copy to figures folder
    shutil.copy(out1, figs / out1.name)
    shutil.copy(out2, figs / out2.name)

    print('Saved plots to', out)
    print('Copied plots to', figs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figures from simulation CSVs')
    parser.add_argument('--raw', default='results/raw', help='Directory containing per-config CSVs')
    parser.add_argument('--out', default='results/analysis', help='Directory to write analysis PNGs')
    parser.add_argument('--fig', default='results/figures', help='Directory to copy final figures')
    args = parser.parse_args()
    main(args.raw, args.out, args.fig)
