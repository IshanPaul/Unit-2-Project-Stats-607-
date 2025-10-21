import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dgps import generate_data
from src.methods import fit_lasso, theoretical_lambda
from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, os, time


def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None):
    """Run simulation study for Lasso regression and return averaged metrics."""
    rng = np.random.default_rng(seed)
    results = {
        'theta': (n * b ** 2) / ((sigma ** 2) * np.log(p - k)),
        'rho': rho,
        'beta_min': b,
        'average_mse': 0,
        'average_tpr': 0,
        'average_fdp': 0,
        'exact_support_recovery_rate': 0,
        'unsigned_support_recovery_rate': 0,
        'lambda_used': 0
    }

    mse_list, tpr_list, fdp_list, exact_list, unsigned_list = [], [], [], [], []

    for rep in range(n_reps):  # no tqdm to avoid clutter in parallel mode
        sim_seed = rng.integers(0, 1e6)
        X, y, beta_true, support_true = generate_data(n, p, k, rho, b, sigma, seed=sim_seed)
        lam_theoretical = theoretical_lambda(sigma, n, p) * lam_factor
        beta_est = fit_lasso(X, y, lam_theoretical)

        mse_list.append(mse(beta_true, beta_est))
        tpr, fdp = tpr_fdp(beta_true, beta_est)
        tpr_list.append(tpr)
        fdp_list.append(fdp)
        exact_list.append(exact_support_recovery(beta_true, beta_est))
        unsigned_list.append(support_recovery(beta_true, beta_est))

    results.update({
        'average_mse': np.mean(mse_list),
        'average_tpr': np.mean(tpr_list),
        'average_fdp': np.mean(fdp_list),
        'exact_support_recovery_rate': np.mean(exact_list),
        'unsigned_support_recovery_rate': np.mean(unsigned_list),
        'lambda_used': lam_theoretical
    })

    return results


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
        theta_list = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # around threshold θ=1
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
                csv_path = "results/raw/large_experiment_parallel.csv"
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
