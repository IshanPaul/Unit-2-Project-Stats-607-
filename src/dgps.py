import numpy as np

def make_sigma(p, rho):
    """
    Create Toeplitz covariance matrix 
    S_ij = rho^|i-j| 
    for a given dimension p and correlation rho."""
    return np.fromfunction(lambda i, j: rho ** np.abs(i - j), (p, p))

def generate_X(n, p, rho=0.5, seed=None):
    """
    Generate design matrix X with n samples and p features,
    where features have Toeplitz covariance structure.
    """
    if seed is not None:
        np.random.seed(seed)
    
    Sigma = make_sigma(p, rho)
    L = np.linalg.cholesky(Sigma)
    Z = np.random.normal(size=(n, p))
    X = Z @ L.T
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    return X

def generate_beta(p, k, b=1.0, seed=None):
    """
    Generate sparse true coefficient vector
    beta of length p with k non-zero entries,
    each set to value less than or b.
    """
    
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    support = rng.choice(p, size=k, replace=False)
    beta[support] = rng.choice([-b, b], size=k)
    return beta, support

def generate_data(n, p, k, rho=0.0, b=1.0, sigma=1.0, seed=None):
    """
    Generate synthetic regression data (X, y) with n samples,
    p features, k non-zero coefficients in beta,
    Toeplitz covariance structure with correlation rho,
    and Gaussian noise with standard deviation sigma.
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = generate_X(n, p, rho, seed)
    beta_true, support = generate_beta(p, k, b, seed)
    noise = np.random.normal(0, sigma, size=n)
    y = X @ beta_true + noise
    return X, y, beta_true, support