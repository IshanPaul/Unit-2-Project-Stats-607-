#!/usr/bin/env python3
"""Focused search over lambda factors for sigma=0.1 and k=5.
Saves CSV and a line plot of exact recovery fraction vs lambda factor.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.simulation import run_simulation


def main():
    lam_values = np.round(np.arange(0.05, 0.201, 0.01), 3)
    n_reps = 200
    n, p, k = 100, 200, 5
    rho, b = 0.0, 1.0
    sigma = 0.1
    base_seed = 2025

    out_dir = Path(__file__).resolve().parents[1] / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, lf in enumerate(lam_values, start=1):
        seed = base_seed + i
        print(f"Running lam_factor={lf} (seed={seed}) with {n_reps} reps")
        df = run_simulation(n=n, p=p, k=k, rho=rho, b=b, sigma=sigma, lam_factor=lf, n_reps=n_reps, seed=seed, save=False)
        rec = {
            "lam_factor": float(lf),
            "exact_count": int(df['exact_support_recovery'].sum()),
            "exact_frac": float(df['exact_support_recovery'].mean()),
            "mean_tpr": float(df['tpr'].mean()),
            "mean_fdp": float(df['fdp'].mean()),
            "mean_mse": float(df['mse'].mean()),
        }
        records.append(rec)

    out_csv = out_dir / f"focused_lam_search_sigma{sigma}_k{k}.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

    # plot
    plt.figure(figsize=(8, 4))
    xs = [r['lam_factor'] for r in records]
    ys = [r['exact_frac'] for r in records]
    plt.plot(xs, ys, marker='o')
    plt.xlabel('lambda factor')
    plt.ylabel('exact recovery fraction')
    plt.title(f'Exact recovery vs lambda (sigma={sigma}, k={k})')
    plt.grid(True)
    out_png = out_dir / f"focused_lam_search_sigma{sigma}_k{k}.png"
    plt.savefig(out_png, bbox_inches='tight')
    print(f"Saved plot: {out_png}")


if __name__ == '__main__':
    main()
