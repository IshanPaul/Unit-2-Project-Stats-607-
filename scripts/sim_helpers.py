import numpy as np
from sklearn.linear_model import Lasso
from scripts.dgps import generate_data
from scripts.methods import theoretical_lambda, fit_lasso
from scripts.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery


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
