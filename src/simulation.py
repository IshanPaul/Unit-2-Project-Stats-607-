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

if __name__ == "__main__":
    # Example usage
    df_results = run_simulation(
        n=100,
        p=200,
        k=10,
        rho=0.5,
        b=1.0,
        sigma=1.0,
        lam_factor=1.0,
        n_reps=10,
        seed=42,
        save=True
    )