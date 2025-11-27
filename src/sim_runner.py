# from src.dgps import generate_data
# from src.methods import fit_lasso, theoretical_lambda
# from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery
# import numpy as np

# def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None):
#     rng = np.random.default_rng(seed)

#     mse_list, tpr_list, fdp_list, exact_list, unsigned_list = [], [], [], [], []
#     for _ in range(n_reps):
#         sim_seed = rng.integers(0, 1e6)
#         X, y, beta_true, support_true = generate_data(n, p, k, rho, b, sigma, seed=sim_seed)
#         lam = theoretical_lambda(sigma, n, p)*lam_factor
#         beta_est = fit_lasso(X, y, lam)

#         mse_list.append(mse(beta_true, beta_est))
#         tpr, fdp   = tpr_fdp(beta_true, beta_est)
#         exact_list.append(exact_support_recovery(beta_true, beta_est))
#         unsigned_list.append(support_recovery(beta_true, beta_est))

#     return dict(
#         theta = (n * b**2) / (sigma**2 * np.log(p-k)),
#         rho=rho, beta_min=b,
#         average_mse=np.mean(mse_list),
#         average_tpr=np.mean(tpr_list),
#         average_fdp=np.mean(fdp_list),
#         exact_support_recovery_rate=np.mean(exact_list),
#         unsigned_support_recovery_rate=np.mean(unsigned_list),
#         lambda_used=lam
#     )

import numpy as np
from src.dgps import generate_data
from src.methods import fit_lasso, theoretical_lambda
from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery, batch_metrics

def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None):
    """
    Vectorized simulation runner for n_reps repetitions.
    
    Parameters
    ----------
    n, p, k : int
        Sample size, number of features, number of non-zero coefficients.
    rho : float
        Feature correlation for Toeplitz covariance.
    b : float
        Signal magnitude for non-zero coefficients.
    sigma : float
        Noise standard deviation.
    lam_factor : float
        Multiplicative factor on theoretical lambda.
    n_reps : int
        Number of simulation repetitions.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict
        Summary metrics of simulation.
    """
    rng = np.random.default_rng(seed)
    
    # Preallocate arrays
    beta_true_array = np.zeros((n_reps, p))
    beta_est_array = np.zeros((n_reps, p))
    
    lam = theoretical_lambda(sigma, n, p) * lam_factor
    
    # Vectorized simulation loop
    for i in range(n_reps):
        sim_seed = rng.integers(0, 1_000_000)
        X, y, beta_true, _ = generate_data(n, p, k, rho, b, sigma, seed=sim_seed)
        beta_est = fit_lasso(X, y, lam)
        
        beta_true_array[i] = beta_true
        beta_est_array[i] = beta_est

    # Compute all metrics in batch
    metrics = batch_metrics(beta_true_array, beta_est_array)
    
    return dict(
        theta = (n * b**2) / (sigma**2 * np.log(p - k)),
        rho=rho,
        beta_min=b,
        average_mse=np.mean(metrics['mse']),
        average_tpr=np.mean(metrics['TPR']),
        average_fdp=np.mean(metrics['FDP']),
        exact_support_recovery_rate=np.mean(metrics['exact_recovery']),
        unsigned_support_recovery_rate=np.mean([support_recovery(beta_true_array[i], beta_est_array[i]) for i in range(n_reps)]),
        lambda_used=lam
    )

