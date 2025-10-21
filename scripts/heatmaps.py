#!/usr/bin/env python3
"""Generate heatmaps of exact recovery rate vs (lam_factor, sigma) and vs (lam_factor, k).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from src.simulation import run_simulation


def sweep_lam_sigma(lam_values, sigma_values, n_reps=50, n=100, p=200, k=5, rho=0.0, b=1.0, base_seed=3000):
    records = []
    for i, (lf, sigma) in enumerate(product(lam_values, sigma_values), start=1):
        seed = base_seed + i
        df = run_simulation(n=n, p=p, k=k, rho=rho, b=b, sigma=sigma, lam_factor=lf, n_reps=n_reps, seed=seed, save=False)
        records.append({
            'lam_factor': float(lf), 'sigma': float(sigma), 'exact_frac': float(df['exact_support_recovery'].mean())
        })
    return pd.DataFrame(records)


def sweep_lam_k(lam_values, k_values, n_reps=50, n=100, p=200, rho=0.0, b=1.0, sigma=0.1, base_seed=4000):
    records = []
    for i, (lf, k) in enumerate(product(lam_values, k_values), start=1):
        seed = base_seed + i
        df = run_simulation(n=n, p=p, k=k, rho=rho, b=b, sigma=sigma, lam_factor=lf, n_reps=n_reps, seed=seed, save=False)
        records.append({
            'lam_factor': float(lf), 'k': int(k), 'exact_frac': float(df['exact_support_recovery'].mean())
        })
    return pd.DataFrame(records)


def plot_heatmap(df, x_col, y_col, z_col, out_path, xlabel=None, ylabel=None, title=None):
    pivot = df.pivot_table(index=y_col, columns=x_col, values=z_col)
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label=z_col)
    plt.xticks(range(len(pivot.columns)), [f"{c:.3f}" for c in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel(xlabel or x_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or z_col)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_values = np.round(np.arange(0.01, 0.201, 0.01), 3)
    sigma_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    df_ls = sweep_lam_sigma(lam_values, sigma_values, n_reps=50)
    csv1 = out_dir / "heatmap_lam_sigma.csv"
    df_ls.to_csv(csv1, index=False)
    plot_heatmap(df_ls, x_col='lam_factor', y_col='sigma', z_col='exact_frac', out_path=out_dir / "heatmap_lam_sigma.png", xlabel='lambda factor', ylabel='sigma', title='Exact recovery rate')

    k_values = [3,5,10,20]
    df_lk = sweep_lam_k(lam_values, k_values, n_reps=50)
    csv2 = out_dir / "heatmap_lam_k.csv"
    df_lk.to_csv(csv2, index=False)
    plot_heatmap(df_lk, x_col='lam_factor', y_col='k', z_col='exact_frac', out_path=out_dir / "heatmap_lam_k.png", xlabel='lambda factor', ylabel='k', title='Exact recovery rate')

    print(f"Saved analysis outputs to {out_dir}")


if __name__ == '__main__':
    main()
