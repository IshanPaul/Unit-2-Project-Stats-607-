# import numpy as np

# def make_sigma(p, rho):
#     """
#     Create Toeplitz covariance matrix 
#     S_ij = rho^|i-j| 
#     for a given dimension p and correlation rho."""
#     return np.fromfunction(lambda i, j: rho ** np.abs(i - j), (p, p))
# # lambda i,j: rho ** np.abs(i-j) is a function that returns rho^|i-j| for given i,j
# # np.fromfunction makes an array ((lambda(i,j)))_pxp


# def generate_X(n, p, rho=0.5, seed=None):
#     """
#     Generate design matrix X with n samples and p features,
#     where features have Toeplitz covariance structure.
#     """
#     if seed is not None:    # to ensure reproducibility
#         np.random.seed(seed)
    
#     Sigma = make_sigma(p, rho) # create covariance matrix
#     L = np.linalg.cholesky(Sigma) # Cholesky decomposition to get lower triangular matrix L
#     Z = np.random.normal(size=(n, p)) # standard normal random variables
#     X = Z @ L.T # transform Z to have covariance Sigma
#     # Center columns
#     X -= X.mean(axis=0, keepdims=True)

#     # Scale columns
#     X /= X.std(axis=0, keepdims=True)

#     return X

# def generate_beta(p, k, b=1.0, seed=None):
#     """
#     Generate sparse true coefficient vector
#     beta of length p with k non-zero entries(choice of supports is random),
#     each set to value b or -b randomly.
#     """
    
#     rng = np.random.default_rng(seed)
#     beta = np.zeros(p)
#     support = rng.choice(p, size=k, replace=False) # SRSWR from {1,...,p}
#     beta[support] = rng.choice([-b, b], size=k)  # assign b or -b randomly to support indices
#     return beta, support

# def generate_data(n, p, k, rho=0.0, b=1.0, sigma=1.0, seed=None):
#     """
#     Generate synthetic regression data (X, y) with n samples,
#     p features, k non-zero coefficients in beta,
#     Toeplitz covariance structure with correlation rho,
#     and Gaussian noise with standard deviation sigma.
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     X = generate_X(n, p, rho, seed) # generate design matrix
#     beta_true, support = generate_beta(p, k, b, seed) # generate true beta
#     noise = np.random.normal(0, sigma, size=n) # iid Gaussian noise
#     y = X @ beta_true + noise
#     return X, y, beta_true, support

import numpy as np

def make_sigma(p, rho):
    """
    Create Toeplitz covariance matrix S with entries S_ij = rho^|i-j|.
    Vectorized.
    """
    idx = np.arange(p)
    # |i-j| is broadcasted
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return Sigma


def generate_X(n, p, rho=0.5, seed=None):
    """
    Generate design matrix X with n samples and p features,
    features have Toeplitz covariance structure.
    Columns are centered and scaled (numerically stable).
    """
    rng = np.random.default_rng(seed)
    
    Sigma = make_sigma(p, rho)
    # Cholesky decomposition (lower-triangular)
    L = np.linalg.cholesky(Sigma)
    
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
    """
    rng = np.random.default_rng(seed)
    
    X = generate_X(n, p, rho, seed)
    beta_true, support = generate_beta(p, k, b, seed)
    
    # Gaussian noise
    noise = rng.normal(0, sigma, size=n)
    
    y = X @ beta_true + noise
    return X, y, beta_true, support
