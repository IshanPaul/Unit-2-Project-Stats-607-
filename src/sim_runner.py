import numpy as np
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed
from src.dgps import generate_X, generate_beta
from src.methods import theoretical_lambda
from src.metrics import batch_metrics, support_recovery

def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None, n_jobs=-1):
    """
    Fully vectorized simulation runner with parallel Lasso fitting.
    """
    rng = np.random.default_rng(seed)
    
    # Compute lambda
    lam = theoretical_lambda(sigma, n, p) * lam_factor
    
    # Generate all beta_true vectors and supports
    beta_true_array = np.zeros((n_reps, p))
    supports = []
    for i in range(n_reps):
        beta_true, support = generate_beta(p, k, b, seed=rng.integers(0, 1_000_000))
        beta_true_array[i] = beta_true
        supports.append(support)
    
    # Generate all design matrices X and noise vectors y
    X_array = np.array([generate_X(n, p, rho, seed=rng.integers(0, 1_000_000)) for _ in range(n_reps)])
    y_array = X_array @ beta_true_array.T + rng.normal(0, sigma, size=(n, n_reps))
    y_array = y_array.T  # shape: (n_reps, n)

    # Solve Lasso in parallel
    def fit_lasso_row(X, y):
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1000)
        lasso.fit(X, y)
        return lasso.coef_
    
    beta_est_array = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(fit_lasso_row)(X_array[i], y_array[i]) for i in range(n_reps)
        )
    )

    # Compute metrics fully vectorized
    metrics = batch_metrics(beta_true_array, beta_est_array)

    # Vectorized unsigned support recovery
    support_true_mask = np.abs(beta_true_array) > 1e-8
    support_est_mask = np.abs(beta_est_array) > 1e-8
    unsigned_recovery = np.all(support_true_mask == support_est_mask, axis=1)

    return dict(
        theta=(n * b**2) / (sigma**2 * np.log(p - k)),
        rho=rho,
        beta_min=b,
        average_mse=np.mean(metrics['mse']),
        average_tpr=np.mean(metrics['TPR']),
        average_fdp=np.mean(metrics['FDP']),
        exact_support_recovery_rate=np.mean(metrics['exact_recovery']),
        unsigned_support_recovery_rate=np.mean(unsigned_recovery),
        lambda_used=lam
    )
