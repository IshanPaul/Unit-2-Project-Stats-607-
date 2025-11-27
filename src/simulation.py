# src/simulation.py
import cProfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import time
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
    parser.add_argument("--batch_size", type=int, default=10, help="Save results every N batches")
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
    # Run simulations in parallel with batching
    # -------------------------------
    os.makedirs("results/raw", exist_ok=True)
    csv_path = "results/raw/large_experiment_parallel.csv"
    
    # Generate unique seeds for each combination
    base_rng = np.random.default_rng(42)
    seeds = base_rng.integers(0, 2**31, size=len(combos))

    batch_results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        # Submit all jobs with unique seeds
        # batch_size controls memory per worker: 100=~240MB, 50=~120MB, 200=~480MB
        future_to_combo = {
            executor.submit(run_simulation_grid, combo, args.n_reps, seed, 
                          batch_size=100, low_memory=False): (combo, seed)
            for combo, seed in zip(combos, seeds)
        }

        # Process results as they complete
        with tqdm(total=len(combos), desc="Parameter grid") as pbar:
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    batch_results.append(result)
                    completed_count += 1
                    pbar.update(1)

                    # Save incrementally every batch_size iterations
                    if completed_count % args.batch_size == 0:
                        results_df = pd.DataFrame(batch_results)
                        if completed_count == args.batch_size:
                            # First batch: create new file
                            results_df.to_csv(csv_path, index=False, mode='w')
                        else:
                            # Append to existing file
                            results_df.to_csv(csv_path, index=False, mode='a', header=False)
                        
                        print(f"\nSaved batch ({completed_count}/{len(combos)} completed)")
                        batch_results = []  # Clear memory
                        
                except Exception as e:
                    print(f"\nError in simulation: {e}")
                    pbar.update(1)

    # Save any remaining results
    if batch_results:
        results_df = pd.DataFrame(batch_results)
        if completed_count <= args.batch_size:
            results_df.to_csv(csv_path, index=False, mode='w')
        else:
            results_df.to_csv(csv_path, index=False, mode='a', header=False)
        print(f"\nSaved final batch ({completed_count}/{len(combos)} completed)")

    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time/60:.2f} minutes")
    print(f"Results saved to {csv_path}")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    profiler.dump_stats('profile.prof')
    print("Profile saved to profile.prof")