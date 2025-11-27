from src.dgps import generate_data
from src.methods import fit_lasso, theoretical_lambda
from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery
import numpy as np

def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None):
    rng = np.random.default_rng(seed)

    mse_list, tpr_list, fdp_list, exact_list, unsigned_list = [], [], [], [], []
    for _ in range(n_reps):
        sim_seed = rng.integers(0, 1e6)
        X, y, beta_true, support_true = generate_data(n, p, k, rho, b, sigma, seed=sim_seed)
        lam = theoretical_lambda(sigma, n, p)*lam_factor
        beta_est = fit_lasso(X, y, lam)

        mse_list.append(mse(beta_true, beta_est))
        tpr, fdp   = tpr_fdp(beta_true, beta_est)
        exact_list.append(exact_support_recovery(beta_true, beta_est))
        unsigned_list.append(support_recovery(beta_true, beta_est))

    return dict(
        theta = (n * b**2) / (sigma**2 * np.log(p-k)),
        rho=rho, beta_min=b,
        average_mse=np.mean(mse_list),
        average_tpr=np.mean(tpr_list),
        average_fdp=np.mean(fdp_list),
        exact_support_recovery_rate=np.mean(exact_list),
        unsigned_support_recovery_rate=np.mean(unsigned_list),
        lambda_used=lam
    )
