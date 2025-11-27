#!/usr/bin/env python3
"""
Computational Complexity Analysis Script

Analyzes how runtime scales with key parameters (n, p) by timing
individual components of the simulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import simulation components
from src.dgps import generate_X, generate_beta
from src.methods import theoretical_lambda
from sklearn.linear_model import Lasso
from src.metrics import mse, tpr_fdp

# Create output directory
Path("results/figures").mkdir(parents=True, exist_ok=True)

def time_generate_X(n, p, rho=0.5, n_reps=10):
    """Time X generation for given (n, p)"""
    times = []
    for i in range(n_reps):
        start = time.time()
        X = generate_X(n, p, rho, seed=i)
        elapsed = time.time() - start
        times.append(elapsed)
    return np.mean(times), np.std(times)

def time_lasso_fit(n, p, k=40, n_reps=10):
    """Time Lasso fitting for given (n, p)"""
    rho, b, sigma = 0.5, 1.0, 1.0
    lam = theoretical_lambda(sigma, n, p)
    
    times = []
    for i in range(n_reps):
        # Generate data
        X = generate_X(n, p, rho, seed=i)
        beta_true, _ = generate_beta(p, k, b, seed=i)
        y = X @ beta_true + np.random.normal(0, sigma, size=n)
        
        # Time Lasso fit
        start = time.time()
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1000, tol=1e-4)
        lasso.fit(X, y)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

def time_metrics(p, k=40, n_reps=10):
    """Time metric computation for given p"""
    times = []
    for i in range(n_reps):
        beta_true, _ = generate_beta(p, k, 1.0, seed=i)
        beta_est, _ = generate_beta(p, k, 1.0, seed=i+1000)
        
        start = time.time()
        _ = mse(beta_true, beta_est)
        _ = tpr_fdp(beta_true, beta_est)
        elapsed = time.time() - start
        times.append(elapsed)
    
    return np.mean(times), np.std(times)

def main():
    print("Running Computational Complexity Analysis")
    print("=" * 60)
    
    # Parameters to test
    p = 1000  # Fixed feature dimension
    k = 40    # Fixed sparsity
    n_values = [300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650]
    n_reps = 5  # Replications per timing
    
    results = []
    
    # Test scaling with n
    print("\nTesting scaling with sample size n:")
    print(f"{'n':>6} {'X gen':>10} {'Lasso':>10} {'Metrics':>10} {'Total':>10}")
    print("-" * 60)
    
    for n in n_values:
        print(f"{n:>6}", end="", flush=True)
        
        # Time components
        t_X, _ = time_generate_X(n, p, n_reps=n_reps)
        print(f" {t_X:>9.3f}s", end="", flush=True)
        
        t_lasso, _ = time_lasso_fit(n, p, k, n_reps=n_reps)
        print(f" {t_lasso:>9.3f}s", end="", flush=True)
        
        t_metrics, _ = time_metrics(p, k, n_reps=n_reps)
        print(f" {t_metrics:>9.3f}s", end="", flush=True)
        
        total = t_X + t_lasso + t_metrics
        print(f" {total:>9.3f}s")
        
        results.append({
            'n': n,
            'p': p,
            'k': k,
            'time_X': t_X,
            'time_lasso': t_lasso,
            'time_metrics': t_metrics,
            'time_total': total
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/complexity_results.csv', index=False)
    print(f"\nResults saved to results/complexity_results.csv")
    
    # Create visualizations
    print("\nGenerating plots...")
    create_complexity_plots(df)
    print("Plots saved to results/figures/")
    
    # Empirical complexity analysis
    print("\n" + "=" * 60)
    print("EMPIRICAL COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    # Fit linear models on log-log scale
    log_n = np.log10(df['n'].values)
    
    for component in ['time_X', 'time_lasso', 'time_metrics', 'time_total']:
        log_time = np.log10(df[component].values)
        slope = np.polyfit(log_n, log_time, 1)[0]
        
        print(f"\n{component}:")
        print(f"  Empirical complexity: O(n^{slope:.2f})")
        
        if 'lasso' in component:
            print(f"  Expected: O(n × p²) ≈ O(n) for fixed p")
            print(f"  Match: {'✓ Good' if 0.8 < slope < 1.2 else '✗ Unexpected'}")
        elif 'X' in component:
            print(f"  Expected: O(n × p) ≈ O(n) for fixed p")
            print(f"  Match: {'✓ Good' if 0.8 < slope < 1.2 else '✗ Unexpected'}")
        elif 'metrics' in component:
            print(f"  Expected: O(p) ≈ O(1) for fixed p")
            print(f"  Match: {'✓ Good' if -0.2 < slope < 0.2 else '✗ Unexpected'}")

def create_complexity_plots(df):
    """Create complexity analysis plots"""
    
    # Plot 1: Individual components
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    components = [
        ('time_X', 'X Generation', 'blue'),
        ('time_lasso', 'Lasso Fitting', 'red'),
        ('time_metrics', 'Metrics', 'green'),
        ('time_total', 'Total per Rep', 'black')
    ]
    
    for (col, title, color), ax in zip(components, axes.flat):
        # Linear scale
        ax.plot(df['n'], df[col], 'o-', color=color, linewidth=2, markersize=6)
        ax.set_xlabel('Sample Size (n)', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['n'], df[col], 1)
        p = np.poly1d(z)
        ax.plot(df['n'], p(df['n']), '--', color=color, alpha=0.5, 
                label=f'Linear fit: {z[0]:.2e}n + {z[1]:.3f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/figures/complexity_components.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: complexity_components.png")
    plt.close()
    
    # Plot 2: Log-log plot for complexity analysis
    fig, ax = plt.subplots(figsize=(10, 7))
    
    log_n = np.log10(df['n'])
    
    for col, title, color in components:
        log_time = np.log10(df[col])
        ax.plot(log_n, log_time, 'o-', color=color, linewidth=2, 
                markersize=6, label=title)
        
        # Fit line and show slope
        slope, intercept = np.polyfit(log_n, log_time, 1)
        fit_line = slope * log_n + intercept
        ax.plot(log_n, fit_line, '--', color=color, alpha=0.5)
        
        # Add slope annotation
        mid_idx = len(log_n) // 2
        ax.text(log_n.iloc[mid_idx], fit_line.iloc[mid_idx] + 0.1, 
                f'O(n^{slope:.2f})', fontsize=9, color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('log₁₀(n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('log₁₀(Time)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Complexity Analysis (Log-Log Scale)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/complexity_loglog.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: complexity_loglog.png")
    plt.close()
    
    # Plot 3: Stacked area chart showing time breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(df['n'], 0, df['time_X'], 
                     color='blue', alpha=0.6, label='X Generation')
    ax.fill_between(df['n'], df['time_X'], 
                     df['time_X'] + df['time_lasso'],
                     color='red', alpha=0.6, label='Lasso Fitting')
    ax.fill_between(df['n'], df['time_X'] + df['time_lasso'],
                     df['time_total'],
                     color='green', alpha=0.6, label='Metrics')
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Time Breakdown by Component', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/complexity_breakdown.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: complexity_breakdown.png")
    plt.close()
    
    # Plot 4: Percentage breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pct_X = 100 * df['time_X'] / df['time_total']
    pct_lasso = 100 * df['time_lasso'] / df['time_total']
    pct_metrics = 100 * df['time_metrics'] / df['time_total']
    
    ax.plot(df['n'], pct_X, 'o-', color='blue', linewidth=2, 
            markersize=6, label='X Generation')
    ax.plot(df['n'], pct_lasso, 'o-', color='red', linewidth=2, 
            markersize=6, label='Lasso Fitting')
    ax.plot(df['n'], pct_metrics, 'o-', color='green', linewidth=2, 
            markersize=6, label='Metrics')
    
    ax.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Total Time (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Time Distribution vs Sample Size', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('results/figures/complexity_percentages.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: complexity_percentages.png")
    plt.close()

if __name__ == "__main__":
    main()