#!/usr/bin/env python3
"""Generate figures from simulation CSVs.

Reads per-config CSVs saved in `results/raw/`, computes theta = n/(2*k*log(p-k)),
and produces three plots (bar, scatter, heatmap) saved to an output directory.

Usage:
    python3 scripts/figures.py --raw results/raw --out results/analysis --fig results/figures
"""
from pathlib import Path
import argparse
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil


def load_sim_results(raw_dir: Path):
    pattern = str(raw_dir / 'sim_n*_p*_k*_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No simulation CSVs found matching {pattern}')

    pat = re.compile(r'sim_n(?P<n>\d+)_p(?P<p>\d+)_k(?P<k>\d+)_rho(?P<rho>[^_]+)_b(?P<b>[^_]+)_sigma(?P<sigma>[^_]+)_lf(?P<lf>[^_]+)_reps(?P<reps>\d+)_seed(?P<seed>[^_]+)')
    rows = []
    for f in files:
        m = pat.search(Path(f).name)
        if not m:
            # skip non-matching files
            continue
        n = int(m.group('n'))
        p = int(m.group('p'))
        k = int(m.group('k'))
        rho = float(m.group('rho').replace('p','.'))
        b = float(m.group('b').replace('p','.'))
        sigma = float(m.group('sigma').replace('p','.'))
        lf = float(m.group('lf').replace('p','.'))
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


def plot_recovery_vs_theta(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['theta_round'] = df['theta'].round(3)
    group = df.groupby('theta_round').exact_frac.mean().reset_index().sort_values('theta_round')
    plt.figure(figsize=(6,4))
    sns.barplot(data=group, x='theta_round', y='exact_frac', palette='viridis')
    plt.xlabel('theta (rounded)')
    plt.ylabel('mean exact recovery')
    plt.title('Mean exact support recovery vs theta')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter_theta(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['theta_round'] = df['theta'].round(3)
    plt.figure(figsize=(8,4))
    sns.stripplot(data=df, x='theta_round', y='exact_frac', hue='lam_factor', dodge=True)
    plt.xlabel('theta (rounded)')
    plt.ylabel('exact_frac')
    plt.title('Per-config exact_frac vs theta (colored by lam_factor)')
    plt.ylim(0,1)
    plt.legend(title='lam_factor')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap_theta_b(df: pd.DataFrame, out_path: Path):
    df = df.copy()
    df['theta_round'] = df['theta'].round(3)
    pivot = df.groupby(['theta_round','b']).exact_frac.mean().reset_index()
    piv = pivot.pivot(index='theta_round', columns='b', values='exact_frac')
    plt.figure(figsize=(6,4))
    sns.heatmap(piv, annot=True, fmt='.3f', cmap='rocket_r', vmin=0, vmax=1)
    plt.xlabel('b (signal magnitude)')
    plt.ylabel('theta_round')
    plt.title('Mean exact_frac (averaged over rho & lam_factor)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(raw_dir: str, out_dir: str, fig_dir: str):
    raw = Path(raw_dir)
    out = Path(out_dir)
    figs = Path(fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load_sim_results(raw)

    out1 = out / 'recovery_vs_theta.png'
    out2 = out / 'exact_frac_scatter_theta.png'
    out3 = out / 'heatmap_theta_b.png'

    plot_recovery_vs_theta(df, out1)
    plot_scatter_theta(df, out2)
    plot_heatmap_theta_b(df, out3)

    # Copy to figures folder
    shutil.copy(out1, figs / out1.name)
    shutil.copy(out2, figs / out2.name)
    shutil.copy(out3, figs / out3.name)

    print('Saved plots to', out)
    print('Copied plots to', figs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate figures from simulation CSVs')
    parser.add_argument('--raw', default='results/raw', help='Directory containing per-config CSVs')
    parser.add_argument('--out', default='results/analysis', help='Directory to write analysis PNGs')
    parser.add_argument('--fig', default='results/figures', help='Directory to copy final figures')
    args = parser.parse_args()
    main(args.raw, args.out, args.fig)
