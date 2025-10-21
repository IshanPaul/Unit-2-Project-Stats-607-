import numpy as np
import pandas as pd
from tqdm import tqdm
from src.dgps import generate_data
from src.methods import fit_lasso, theoretical_lambda
from src.metrics import mse, tpr_fdp, exact_support_recovery
from pathlib import Path
from datetime import datetime
from typing import Optional

def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None, save: Optional[bool] = True):
    """
    Run simulation study for Lasso regression.
    
    Parameters:
    n : int
        Number of samples.
    p : int
        Number of features.
    k : int
        Number of non-zero coefficients in true beta.
    rho : float
        Correlation parameter for Toeplitz covariance.
    b : float
        Magnitude of non-zero coefficients in true beta.
    sigma : float
        Standard deviation of Gaussian noise.
    lam_factor : float
        Factor to scale theoretical lambda for Lasso.
    n_reps : int
        Number of simulation repetitions.
    seed : int or None
        Random seed for reproducibility.
        
    Returns:
    pd.DataFrame
        DataFrame containing simulation results for each repetition.
    """
    results = []
    rng = np.random.default_rng(seed)
    
    for rep in tqdm(range(n_reps), desc="Simulation Repetitions"):
        sim_seed = rng.integers(0, 1e6)
        X, y, beta_true, support_true = generate_data(
            n, p, k, rho, b, sigma, seed=sim_seed
        )
        
        lam_theoretical = theoretical_lambda(sigma, n, p) * lam_factor
        beta_est = fit_lasso(X, y, lam_theoretical)
        
        mse_val = mse(beta_true, beta_est)
        tpr, fdp = tpr_fdp(beta_true, beta_est)
        exact_recovery = exact_support_recovery(beta_true, beta_est)
        
        results.append({
            'replication': rep + 1,
            'mse': mse_val,
            'tpr': tpr,
            'fdp': fdp,
            'exact_support_recovery': exact_recovery,
            'lambda_used': lam_theoretical
        })
    
    df = pd.DataFrame(results)

    # Save results to results/raw/ by default (repository root relative to this file)
    if save:
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = repo_root / "results" / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(x: object) -> str:
            s = str(x)
            return s.replace(".", "p").replace("/", "_")

        seed_str = _sanitize(seed) if seed is not None else "None"
        fname = (
            f"sim_n{n}_p{p}_k{k}_rho{_sanitize(rho)}_b{_sanitize(b)}_sigma{_sanitize(sigma)}"
            f"_lf{_sanitize(lam_factor)}_reps{n_reps}_seed{seed_str}_"
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        )
        out_path = out_dir / fname
        df.to_csv(out_path, index=False)
        print(f"Saved simulation results to: {out_path}")

    return df

def n_from_theta(theta, p, k):
    return int(round(2 * k * np.log(p - k) * theta))


if __name__ == "__main__":
    import argparse
    from itertools import product

    parser = argparse.ArgumentParser(description="Run Lasso simulation sweeps and save results.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "large"],
        default="quick",
        help=(
            "Preset parameter grids: 'quick' is small and fast; 'full' runs a larger sweep; "
            "'large' runs an extensive sweep (expensive)."
        ),
    )
    parser.add_argument("--save", action="store_true", help="Save results to results/raw/ (default: False for CLI runs)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Recovery fraction threshold to flag high-recovery configs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--n-reps", type=int, default=None, help="Override number of repetitions per config")
    args = parser.parse_args()

    # Define parameter grids
    if args.mode == "quick":
        n_list = [100]
        p_list = [200]
        k_list = [5, 10]
        rho_list = [0.0, 0.5]
        b_list = [1.0]
        sigma_list = [1.0]
        lam_factors = [0.5, 1.0]
        default_reps = 10
    elif args.mode == "full":
        n_list = [100, 200]
        p_list = [200, 500]
        k_list = [5, 10]
        rho_list = [0.0, 0.5, 0.9]
        b_list = [0.5, 1.0, 2.0]
        sigma_list = [0.1, 0.5, 1.0]
        lam_factors = [0.1, 0.5, 1.0]
        default_reps = 50
    else:  # large
        # placeholder values; will be overwritten below
        n_list = []
        p_list = []
        k_list = []
        rho_list = []
        b_list = []
        sigma_list = []
        lam_factors = []
        default_reps = args.n_reps if args.n_reps is not None else 1000

    n_reps = args.n_reps if args.n_reps is not None else default_reps

    save_flag = args.save

    # Iterate over grid and run simulations
    base_seed = args.seed

    def _run_grid(n_list, p_list, k_list, rho_list, b_list, sigma_list, lam_factors, default_reps, save_flag, base_seed, threshold):
        from itertools import product
        import pandas as pd
        from pathlib import Path

        grid = list(product(n_list, p_list, k_list, rho_list, b_list, sigma_list, lam_factors))
        grid_size = len(grid)
        print(f"Running grid with {grid_size} configs, {default_reps} reps each (save={save_flag})")
        cfg_idx = 0
        records = []
        for (n, p, k, rho, b, sigma, lam_factor) in grid:
            cfg_idx += 1
            seed = base_seed + cfg_idx
            print(f"[{cfg_idx}/{grid_size}] n={n}, p={p}, k={k}, rho={rho}, b={b}, sigma={sigma}, lam_factor={lam_factor}, seed={seed}")
            # run simulation (do not rely on internal saving for aggregation)
            df = run_simulation(
                n=n,
                p=p,
                k=k,
                rho=rho,
                b=b,
                sigma=sigma,
                lam_factor=lam_factor,
                n_reps=default_reps,
                seed=seed,
                save=save_flag,
            )

            exact_frac = float(df['exact_support_recovery'].mean())
            rec = {
                'n': int(n), 'p': int(p), 'k': int(k), 'rho': float(rho), 'b': float(b),
                'sigma': float(sigma), 'lam_factor': float(lam_factor), 'seed': int(seed),
                'exact_count': int(df['exact_support_recovery'].sum()), 'exact_frac': exact_frac,
                'mean_tpr': float(df['tpr'].mean()), 'mean_fdp': float(df['fdp'].mean()), 'mean_mse': float(df['mse'].mean()),
                'recovery_above_threshold': bool(exact_frac >= threshold)
            }
            records.append(rec)

        # save aggregated summary CSV
        out_dir = Path(__file__).resolve().parents[1] / 'results' / 'raw'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%dT%H%M%S')
        summary_fname = out_dir / f"sim_summary_mode_{args.mode}_{ts}.csv"
        pd.DataFrame(records).to_csv(summary_fname, index=False)
        print(f"Saved aggregated summary to: {summary_fname}")
        return records

    if args.mode in ("quick", "full"):
        _run_grid(n_list, p_list, k_list, rho_list, b_list, sigma_list, lam_factors, n_reps, save_flag, base_seed, args.threshold)

    else:  # large mode: run larger sweeps (be careful — this can be expensive)
        # Large sweep parameters (similar to earlier manual runs)
        # Use theta mapping to compute n from theta (n_from_theta defined above)
        theta_list = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        p = 1000
        k = 50
        rho_list = [0.0, 0.5, 0.9]
        b_list = [1.0, 10.0, 100.0]
        lam_factors = [1.0]  # keep lam_factor=1.0 as in earlier experiments; adjust as needed
        default_reps_large = args.n_reps if args.n_reps is not None else 1000

        n_list_large = [n_from_theta(t, p, k) for t in theta_list]
        print(f"Running large sweep: p={p}, k={k}, thetas={theta_list} -> n={n_list_large}, reps={default_reps_large}")

        # Run grid over (n_list_large x rho_list x b_list x lam_factors)
        _run_grid(n_list_large, [p], [k], rho_list, b_list, [1.0], lam_factors, default_reps_large, save_flag, base_seed, args.threshold)

    print("All simulations completed.")