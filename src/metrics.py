import numpy as np

def support(beta, tol=1e-8):
    """
    parameters:
    beta : np.ndarray
        Coefficient vector.
    tol : float
        Tolerance level to consider a coefficient as non-zero.
    Returns: np.ndarray
        Indices of non-zero coefficients in beta.
    """
    return np.where(np.abs(beta) > tol)[0]

def support_recovery(beta_true, beta_est, tol=1e-8):
    """
    parameters:
    beta_true : np.ndarray
        True coefficient vector.
    beta_est : np.ndarray
        Estimated coefficient vector.
    tol : float
        Tolerance level to consider a coefficient as non-zero.
    
    Returns: bool
        True if supports match exactly, False otherwise.
    """
    support_true = support(beta_true, tol)
    support_est = support(beta_est, tol)
    return np.array_equal(np.sort(support_true), np.sort(support_est))

def exact_support_recovery(beta_true, beta_est, tol=1e-8):
    """
    parameters:
    beta_true : np.ndarray
        True coefficient vector.
    beta_est : np.ndarray
        Estimated coefficient vector.
    tol : float
        Tolerance level to consider a coefficient as non-zero.
    
    Returns: bool
        True if supports match exactly including sign of coefficients, False otherwise.
    """
    support_true = support(beta_true, tol)
    support_est = support(beta_est, tol)
    return support_recovery(beta_true, beta_est, tol) and np.all(np.sign(beta_true[support_true]) == np.sign(beta_est[support_est]))

def mse(beta_true, beta_est):
    """
    Compute Mean Squared Error (MSE) between true coefficients
    beta_true and estimated coefficients beta_est.
    """
    return np.mean((beta_true - beta_est) ** 2)

def tpr_fdp(beta_true, beta_est, tol=1e-8):
    """
    Compute True Positive Rate (TPR) and False Discovery Proportion (FDP)
    for the estimated coefficients beta_est compared to true coefficients beta_true.
    TPR = TP / (TP + FN)
    FDP = FP / (TP + FP)
    """
    support_true = support(beta_true, tol)
    support_est = support(beta_est, tol)
    
    TP = len(np.intersect1d(support_true, support_est))
    FP = len(np.setdiff1d(support_est, support_true))
    FN = len(np.setdiff1d(support_true, support_est))
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FDP = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    return TPR, FDP