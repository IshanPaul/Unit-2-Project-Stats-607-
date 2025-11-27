# src/metrics.py
import numpy as np

def support(beta, tol=1e-8):
    """
    Return indices of non-zero coefficients in beta, using a tolerance.
    
    Parameters
    ----------
    beta : np.ndarray, shape (n_features,)
    tol : float, optional
        Threshold below which coefficients are considered zero.
    
    Returns
    -------
    np.ndarray
        Indices of non-zero coefficients.
    """
    beta = np.asarray(beta)
    return np.flatnonzero(np.abs(beta) > tol)


def support_recovery(beta_true, beta_est, tol=1e-8):
    """
    Check if the estimated support matches the true support (ignoring signs).
    This is "unsigned support recovery".
    
    Parameters
    ----------
    beta_true : np.ndarray
    beta_est : np.ndarray
    tol : float
    
    Returns
    -------
    bool
    """
    s_true = np.sort(support(beta_true, tol))
    s_est = np.sort(support(beta_est, tol))
    return np.array_equal(s_true, s_est)


def exact_support_recovery(beta_true, beta_est, tol=1e-8):
    """
    Check if the estimated support matches the true support including signs.
    This is "exact support recovery" or "signed support recovery".
    
    Parameters
    ----------
    beta_true : np.ndarray
    beta_est : np.ndarray
    tol : float
    
    Returns
    -------
    bool
    """
    s_true = support(beta_true, tol)
    s_est = support(beta_est, tol)
    
    if not support_recovery(beta_true, beta_est, tol):
        return False
    
    # Check signs match
    return np.all(np.sign(beta_true[s_true]) == np.sign(beta_est[s_est]))


def mse(beta_true, beta_est):
    """
    Compute Mean Squared Error between true and estimated coefficients.
    
    Parameters
    ----------
    beta_true : np.ndarray
    beta_est : np.ndarray
    
    Returns
    -------
    float
    """
    beta_true = np.asarray(beta_true)
    beta_est = np.asarray(beta_est)
    return np.mean((beta_true - beta_est) ** 2)


def tpr_fdp(beta_true, beta_est, tol=1e-8):
    """
    Compute True Positive Rate (TPR) and False Discovery Proportion (FDP) 
    for support recovery.
    
    TPR = TP / (TP + FN)  - proportion of true non-zeros correctly identified
    FDP = FP / (TP + FP)  - proportion of identified non-zeros that are false
    
    Parameters
    ----------
    beta_true : np.ndarray
    beta_est : np.ndarray
    tol : float
    
    Returns
    -------
    tuple of floats: (TPR, FDP)
    """
    s_true = support(beta_true, tol)
    s_est = support(beta_est, tol)
    
    # Vectorized counts
    TP = np.intersect1d(s_true, s_est).size
    FP = np.setdiff1d(s_est, s_true).size
    FN = np.setdiff1d(s_true, s_est).size
    
    TPR = TP / max(TP + FN, 1)  # avoid division by zero
    FDP = FP / max(TP + FP, 1)   # avoid division by zero
    
    return TPR, FDP


def batch_metrics(beta_true_array, beta_est_array, tol=1e-8):
    """
    Vectorized computation of metrics over multiple simulations.
    
    Note: This function is kept for compatibility but is NOT used in the
    optimized sim_runner.py (which processes in smaller batches for memory).
    
    Parameters
    ----------
    beta_true_array : np.ndarray, shape (n_simulations, n_features)
    beta_est_array : np.ndarray, shape (n_simulations, n_features)
    tol : float
    
    Returns
    -------
    dict of np.ndarrays
        'mse', 'TPR', 'FDP', 'exact_recovery' for all simulations.
    """
    beta_true_array = np.asarray(beta_true_array)
    beta_est_array = np.asarray(beta_est_array)
    n_sim = beta_true_array.shape[0]
    
    mse_vals = np.mean((beta_true_array - beta_est_array) ** 2, axis=1)
    TPR_vals = np.zeros(n_sim)
    FDP_vals = np.zeros(n_sim)
    exact_vals = np.zeros(n_sim, dtype=bool)
    
    for i in range(n_sim):
        TPR_vals[i], FDP_vals[i] = tpr_fdp(beta_true_array[i], beta_est_array[i], tol)
        exact_vals[i] = exact_support_recovery(beta_true_array[i], beta_est_array[i], tol)
    
    return {'mse': mse_vals, 'TPR': TPR_vals, 'FDP': FDP_vals, 'exact_recovery': exact_vals}