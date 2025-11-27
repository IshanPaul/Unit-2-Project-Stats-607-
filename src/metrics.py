# import numpy as np

# def support(beta, tol=1e-8):
#     """
#     parameters:
#     beta : np.ndarray
#         Coefficient vector.
#     tol : float
#         Tolerance level to consider a coefficient as non-zero.
#     Returns: np.ndarray
#         Indices of non-zero coefficients in beta.
#     """
#     return np.where(np.abs(beta) > tol)[0]

# def support_recovery(beta_true, beta_est, tol=1e-8):
#     """
#     parameters:
#     beta_true : np.ndarray
#         True coefficient vector.
#     beta_est : np.ndarray
#         Estimated coefficient vector.
#     tol : float
#         Tolerance level to consider a coefficient as non-zero.
    
#     Returns: bool
#         True if supports match exactly, False otherwise.
#     """
#     support_true = support(beta_true, tol)
#     support_est = support(beta_est, tol)
#     return np.array_equal(np.sort(support_true), np.sort(support_est))

# def exact_support_recovery(beta_true, beta_est, tol=1e-8):
#     """
#     parameters:
#     beta_true : np.ndarray
#         True coefficient vector.
#     beta_est : np.ndarray
#         Estimated coefficient vector.
#     tol : float
#         Tolerance level to consider a coefficient as non-zero.
    
#     Returns: bool
#         True if supports match exactly including sign of coefficients, False otherwise.
#     """
#     support_true = support(beta_true, tol)
#     support_est = support(beta_est, tol)
#     return support_recovery(beta_true, beta_est, tol) and np.all(np.sign(beta_true[support_true]) == np.sign(beta_est[support_est]))

# def mse(beta_true, beta_est):
#     """
#     parameters:
#     beta_true : np.darray
#         True coefficient vector.
#     beta_est : np.darray
#         Estimated coefficient vector.
    
#     Returns: float
#         Mean Squared Error between beta_true and beta_est.
#     """
#     return np.mean((beta_true - beta_est) ** 2)

# def tpr_fdp(beta_true, beta_est, tol=1e-8):
#     """
#     parameters: 
#     beta_true : np.ndarray
#         True coefficient vector.
#     beta_est : np.ndarray
#         Estimated coefficient vector.
#     tol : float
#         Tolerance level to consider a coefficient as non-zero.

#     Returns: tuple (TPR, FDP)
#     TPR : float 
#         True Positive Rate
#     FDP : float
#         False Discovery Proportion
#     TPR = TP / (TP + FN)
#     FDP = FP / (TP + FP)
#     """
#     support_true = support(beta_true, tol)
#     support_est = support(beta_est, tol)
    
#     TP = len(np.intersect1d(support_true, support_est))
#     FP = len(np.setdiff1d(support_est, support_true))
#     FN = len(np.setdiff1d(support_true, support_est))
    
#     TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     FDP = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    
#     return TPR, FDP

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
    
    # Check signs
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
    Compute True Positive Rate (TPR) and False Discovery Proportion (FDP) for support recovery.
    
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
