# src/simulation.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse, os, time

from src.sim_runner import run_simulation
from src.sim_helpers import run_simulation_grid

# -------------------------------
# Utility functions
# -------------------------------

def n_from_theta(theta, p, k, b=1.0, sigma=1.0, scale=30):
    """Compute sample size n given theoretical threshold theta."""
    return int(scale * round(((sigma ** 2) * np.log(p - k) * theta) / (b ** 2)))

# -------------------------------
# Main function
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="small")
    parser.add_argument("--n_reps", type=int, default=1000)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel workers (default: all cores)")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of parameter combos per worker")
    args = parser.parse_args()

    start_time = time.time()

    if args.mode != "large":
        print("No large mode specified. Use `--mode large` to run full experiments.")
        return

    # -------------------------------
    # Parameter grid
    # -------------------------------
    theta_list = np.arange(0.5, 2.05, 0.05).round(2).tolist()
    p, k = 1000, 40
    rho_list = [0.0, 0.3, 0.6]
    b_list = [0.5, 1.0]
    sigma_list = [1.0]
    lam_factors = [1.0]

    combos = [
        (n_from_theta(t, p, k, b, sigma), p, k, rho, b, sigma, lam_factor)
        for rho in rho_list
        for b in b_list
        for sigma in sigma_list
        for lam_factor in lam_factors
        for t in theta_list
    ]

    print(f"Launching {len(combos)} parameter combinations using {args.n_jobs or os.cpu_count()} workers...")

    # -------------------------------
    # Run simulations in parallel
    # -------------------------------
    all_results = []
    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        # submit jobs
        futures = [
            executor.submit(run_simulation_grid, combo, args.n_reps, rng.integers(0, 1_000_000))
            for combo in combos
        ]

        # collect results with progress bar
        for future in tqdm(futures, total=len(futures), desc="Parameter grid"):
            all_results.append(future.result())

    # -------------------------------
    # Save results
    # -------------------------------
    if all_results:
        results_df = pd.DataFrame(all_results)
        if args.save:
            os.makedirs("results/raw", exist_ok=True)
            csv_path = "results/raw/large_experiment_parallel.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
    else:
        print("No results generated — check logs above.")

    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time/60:.2f} minutes")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()

