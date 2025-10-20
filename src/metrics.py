import numpy as np

def support(beta, tol=1e-8):
    """
    Compute the support of a coefficient vector beta.
    The support is the set of indices where beta is non-zero
    (within a tolerance level tol).
    """
    return np.where(np.abs(beta) > tol)[0]

def exact_support_recovery(beta_true, beta_est, tol=1e-8):
    """
    Check if the estimated coefficient vector beta_est
    exactly recovers the support of the true coefficient vector beta_true.
    Returns True if supports match, False otherwise.
    """
    support_true = support(beta_true, tol)
    support_est = support(beta_est, tol)
    return np.array_equal(np.sort(support_true), np.sort(support_est))

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