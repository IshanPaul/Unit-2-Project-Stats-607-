# src/dgps.py
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=32)
def make_sigma_cached(p, rho):
    """
    Create Toeplitz covariance matrix S with entries S_ij = rho^|i-j|.
    Cached for efficiency since same (p, rho) pairs are reused.
    
    For your simulation: p=1000, rho in {0.0, 0.3, 0.6}
    This means only 3 matrices are ever created and cached!
    """
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return Sigma


@lru_cache(maxsize=32)
def get_cholesky_cached(p, rho):
    """
    Compute and cache Cholesky decomposition of Toeplitz matrix.
    This is the expensive operation - cache it!
    
    For p=1000, this saves ~0.1-0.2 seconds per call.
    Over 246,000 calls, this is a HUGE speedup!
    
    With 3 rho values, this function is called only 3 times total,
    then returns cached results for all subsequent calls.
    """
    Sigma = make_sigma_cached(p, rho)
    L = np.linalg.cholesky(Sigma)
    return L


def make_sigma(p, rho):
    """
    Create Toeplitz covariance matrix S with entries S_ij = rho^|i-j|.
    Vectorized and cached.
    """
    return make_sigma_cached(p, rho)


def generate_X(n, p, rho=0.5, seed=None):
    """
    Generate design matrix X with n samples and p features,
    features have Toeplitz covariance structure.
    Columns are centered and scaled (numerically stable).
    
    Uses cached Cholesky decomposition for speed.
    
    Args:
        n: number of samples
        p: number of features
        rho: correlation parameter for Toeplitz structure
        seed: random seed for reproducibility
    
    Returns:
        X: (n, p) design matrix with standardized columns
    """
    rng = np.random.default_rng(seed)
    
    # Use cached Cholesky decomposition (MAJOR speedup!)
    L = get_cholesky_cached(p, rho)
    
    Z = rng.standard_normal(size=(n, p))  # vectorized standard normals
    X = Z @ L.T  # correlated variables
    
    # Column center and scale safely
    col_mean = X.mean(axis=0, keepdims=True)
    col_std = X.std(axis=0, keepdims=True)
    
    # Avoid division by zero
    col_std_safe = np.where(col_std == 0, 1.0, col_std)
    
    X = (X - col_mean) / col_std_safe
    return X


def generate_beta(p, k, b=1.0, seed=None):
    """
    Generate sparse coefficient vector beta of length p with k non-zero entries.
    Non-zeros are ±b randomly.
    
    Args:
        p: dimension of beta
        k: number of non-zero entries (sparsity)
        b: magnitude of non-zero coefficients
        seed: random seed for reproducibility
    
    Returns:
        beta: (p,) sparse coefficient vector
        support: indices of non-zero entries
    """
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    if k > 0:
        support = rng.choice(p, size=k, replace=False)
        beta[support] = rng.choice([-b, b], size=k)
    else:
        support = np.array([], dtype=int)
    return beta, support


def generate_data(n, p, k, rho=0.0, b=1.0, sigma=1.0, seed=None):
    """
    Generate synthetic regression data (X, y) with n samples,
    p features, k non-zero coefficients in beta,
    Toeplitz covariance structure with correlation rho,
    and Gaussian noise with standard deviation sigma.
    
    Args:
        n: number of samples
        p: number of features
        k: number of non-zero coefficients (sparsity)
        rho: correlation parameter for Toeplitz structure
        b: magnitude of non-zero coefficients
        sigma: noise standard deviation
        seed: random seed for reproducibility
    
    Returns:
        X: (n, p) design matrix
        y: (n,) response vector
        beta_true: (p,) true coefficient vector
        support: indices of non-zero entries in beta_true
    """
    rng = np.random.default_rng(seed)
    
    X = generate_X(n, p, rho, seed)
    beta_true, support = generate_beta(p, k, b, seed)
    
    # Gaussian noise
    noise = rng.normal(0, sigma, size=n)
    
    y = X @ beta_true + noise
    return X, y, beta_true, support