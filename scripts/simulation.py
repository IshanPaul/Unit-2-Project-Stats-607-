import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.dgps import generate_data
from scripts.methods import fit_lasso, theoretical_lambda
from scripts.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.sim_helpers import run_simulation
import argparse, os, time



def n_from_theta(theta, p, k, b=1.0, sigma=1.0, scale=30):
    """Compute sample size n given theoretical threshold theta."""
    return int(scale * round(((sigma ** 2) * np.log(p - k) * theta) / (b ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="small")
    parser.add_argument("--n_reps", type=int, default=1000)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel workers (default: all cores)")
    args = parser.parse_args()

    start_time = time.time()

    if args.mode == "large":
        theta_list = np.arange(0.5, 2.05, 0.05).round(2).tolist()  # around threshold θ=1
        p, k = 1000, 40                                # lower p for clearer recovery
        rho_list = [0.0, 0.3, 0.6]                     # moderate correlations
        b_list = [0.5, 1.0]                            # signal strengths high enough
        sigma_list = [1.0]                             # noise level
        lam_factors = [1.0]                            # theoretical λ scaling

        sample_sizes = [n_from_theta(t, p, k, b_list[0], sigma_list[0]) for t in theta_list]
        print(f"Theta list: {theta_list}")
        print(f"Computed n_list (p={p}, k={k}, b={b_list[0]}): {sample_sizes}")

        rng = np.random.default_rng(42)

        combos = [
            (n_from_theta(t, p, k, b, sigma), p, k, rho, b, sigma, lam_factor, args.n_reps)
            for rho in rho_list
            for b in b_list
            for sigma in sigma_list
            for lam_factor in lam_factors
            for t in theta_list
        ]

        print(f"Launching {len(combos)} parallel simulations using {args.n_jobs or os.cpu_count()} workers...")

        all_results = []
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(run_simulation, *combo, seed=rng.integers(0, 1e6)): combo
                for combo in combos
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parameter grid"):
                try:
                    all_results.append(future.result())
                except Exception as e:
                    print(f"Error: {e}")

        if all_results:
            results_df = pd.DataFrame(all_results)
            if args.save:
                os.makedirs("results/raw", exist_ok=True)
                csv_path = "results/benchmark_results.csv"
                results_df.to_csv(csv_path, index=False)
                print(f"Results saved to {csv_path}")
        else:
            print("No results generated — check logs above.")

    else:
        print("No large mode specified. Use `--mode large` to run full experiments.")

    total_time = time.time() - start_time
    print(f"\nTotal simulation time: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()