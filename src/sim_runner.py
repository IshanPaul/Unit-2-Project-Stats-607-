# src/sim_runner.py
import numpy as np
from sklearn.linear_model import Lasso
from src.dgps import generate_X, generate_beta
from src.methods import theoretical_lambda
from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery

def run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None, 
                   batch_size=100, n_jobs=1):
    """
    Optimal time-memory tradeoff: Process reps in mini-batches.
    
    Strategy:
    - Batches small enough to fit in memory (default: 100 reps)
    - Vectorized operations within each batch for speed
    - Sequential batch processing to limit memory
    
    Memory usage: ~batch_size × n × p × 8 bytes for X
                  With batch_size=100, n=300, p=1000: ~240MB per batch
    
    Args:
        batch_size: Number of reps per batch (tune based on RAM)
                    100 = ~240MB, 50 = ~120MB, 200 = ~480MB
        n_jobs: Should be 1 (parallelization happens at outer level)
    """
    rng = np.random.default_rng(seed)
    lam = theoretical_lambda(sigma, n, p) * lam_factor
    
    # Accumulators
    mse_sum = 0.0
    tpr_sum = 0.0
    fdp_sum = 0.0
    exact_recovery_count = 0
    unsigned_recovery_count = 0
    
    n_batches = int(np.ceil(n_reps / batch_size))
    
    for batch_idx in range(n_batches):
        # Determine batch size (last batch may be smaller)
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_reps)
        current_batch_size = end_idx - start_idx
        
        # --- Generate batch data ---
        beta_true_batch = np.zeros((current_batch_size, p))
        
        for i in range(current_batch_size):
            beta_true, _ = generate_beta(p, k, b, seed=rng.integers(0, 2**31))
            beta_true_batch[i] = beta_true
        
        # Generate X and y, fit Lasso
        X_list = []
        y_list = []
        
        for i in range(current_batch_size):
            X_i = generate_X(n, p, rho, seed=rng.integers(0, 2**31))
            y_i = X_i @ beta_true_batch[i] + rng.normal(0, sigma, size=n)
            X_list.append(X_i)
            y_list.append(y_i)
        
        # --- Fit Lasso for batch ---
        beta_est_batch = np.zeros((current_batch_size, p))
        
        for i in range(current_batch_size):
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1000, 
                         tol=1e-4, warm_start=False)
            lasso.fit(X_list[i], y_list[i])
            beta_est_batch[i] = lasso.coef_
        
        # --- Compute metrics using existing functions ---
        # MSE (can be vectorized)
        mse_batch = np.mean((beta_true_batch - beta_est_batch) ** 2, axis=1)
        mse_sum += np.sum(mse_batch)
        
        # TPR, FDP, and recovery metrics
        for i in range(current_batch_size):
            # TPR and FDP
            tpr_i, fdp_i = tpr_fdp(beta_true_batch[i], beta_est_batch[i])
            tpr_sum += tpr_i
            fdp_sum += fdp_i
            
            # Exact recovery (with signs)
            exact_recovery_count += int(exact_support_recovery(beta_true_batch[i], 
                                                                beta_est_batch[i]))
            
            # Unsigned recovery (ignoring signs)
            unsigned_recovery_count += int(support_recovery(beta_true_batch[i], 
                                                            beta_est_batch[i]))
        
        # --- Free memory immediately ---
        del beta_true_batch, beta_est_batch, X_list, y_list, mse_batch
    
    # Return averaged metrics
    return dict(
        theta=(n * b**2) / (sigma**2 * np.log(p - k)),
        rho=rho,
        beta_min=b,
        n=n,
        p=p,
        k=k,
        sigma=sigma,
        lambda_factor=lam_factor,
        average_mse=mse_sum / n_reps,
        average_tpr=tpr_sum / n_reps,
        average_fdp=fdp_sum / n_reps,
        exact_support_recovery_rate=exact_recovery_count / n_reps,
        unsigned_support_recovery_rate=unsigned_recovery_count / n_reps,
        lambda_used=lam,
        n_reps=n_reps
    )


def run_simulation_ultralow_memory(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=None):
    """
    Ultra-low memory version: Process completely one-at-a-time.
    Use this if you have < 4GB RAM available.
    
    Memory usage: ~n × p × 8 bytes = with n=300, p=1000: ~2.4MB
    """
    rng = np.random.default_rng(seed)
    lam = theoretical_lambda(sigma, n, p) * lam_factor
    
    # Accumulators
    mse_sum = 0.0
    tpr_sum = 0.0
    fdp_sum = 0.0
    exact_recovery_count = 0
    unsigned_recovery_count = 0
    
    # Process one rep at a time
    for rep in range(n_reps):
        # Generate
        beta_true, _ = generate_beta(p, k, b, seed=rng.integers(0, 2**31))
        X = generate_X(n, p, rho, seed=rng.integers(0, 2**31))
        y = X @ beta_true + rng.normal(0, sigma, size=n)
        
        # Fit
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1000, tol=1e-4)
        lasso.fit(X, y)
        beta_est = lasso.coef_
        
        # Metrics using existing functions
        mse_sum += mse(beta_true, beta_est)
        
        tpr_i, fdp_i = tpr_fdp(beta_true, beta_est)
        tpr_sum += tpr_i
        fdp_sum += fdp_i
        
        exact_recovery_count += int(exact_support_recovery(beta_true, beta_est))
        unsigned_recovery_count += int(support_recovery(beta_true, beta_est))
        
        # Clean up
        del X, y, beta_true, beta_est
    
    return dict(
        theta=(n * b**2) / (sigma**2 * np.log(p - k)),
        rho=rho,
        beta_min=b,
        n=n,
        p=p,
        k=k,
        sigma=sigma,
        lambda_factor=lam_factor,
        average_mse=mse_sum / n_reps,
        average_tpr=tpr_sum / n_reps,
        average_fdp=fdp_sum / n_reps,
        exact_support_recovery_rate=exact_recovery_count / n_reps,
        unsigned_support_recovery_rate=unsigned_recovery_count / n_reps,
        lambda_used=lam,
        n_reps=n_reps
    )