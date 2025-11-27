# src/methods.py
import numpy as np
from sklearn.linear_model import Lasso

def fit_lasso(X, y, lam, max_iter=1000):
    """
    Fit Lasso regression model to data (X, y) with regularization parameter lam.
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix.
    y : np.ndarray of shape (n_samples,)
        Response vector.
    lam : float
        Regularization parameter for Lasso.
    max_iter : int
        Maximum number of iterations.
    
    Returns
    -------
    np.ndarray of shape (n_features,)
        Estimated coefficients.
    
    Notes
    -----
    - Columns of X are standardized to unit variance for numerical stability.
    - y is centered to improve conditioning.
    """
    # Center y
    y_centered = y - np.mean(y)
    
    # Standardize X columns safely (avoid divide by zero)
    col_std = X.std(axis=0, keepdims=True)
    col_std_safe = np.where(col_std == 0, 1.0, col_std)
    X_scaled = (X - X.mean(axis=0, keepdims=True)) / col_std_safe

    # Fit Lasso
    lasso = Lasso(alpha=lam, max_iter=max_iter, fit_intercept=False)
    lasso.fit(X_scaled, y_centered)

    # Rescale coefficients back to original scale
    coef = lasso.coef_ / col_std_safe.ravel()
    return coef


def theoretical_lambda(sigma, n, p, c=2.0):
    """
    Compute theoretical Lasso lambda (Wainwright 2009).
    Vectorized over sigma, n, p.
    
    Parameters
    ----------
    sigma : float or array-like
        Noise standard deviation.
    n : int or array-like
        Number of samples.
    p : int or array-like
        Number of features.
    c : float, optional
        Constant factor.
    
    Returns
    -------
    float or np.ndarray
        Theoretical lambda values.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    # avoid log(0) or division by zero
    p_safe = np.where(p <= 1, 2.0, p)  # p must be >1
    n_safe = np.where(n <= 0, 1.0, n)  # n must be >0

    return c * sigma * np.sqrt(2 * np.log(p_safe) / n_safe)
