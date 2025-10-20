import numpy as np
from sklearn.linear_model import Lasso

def fit_lasso(X, y, lam, max_iter=1000):
    """
    Fit Lasso regression model to data (X, y) with regularization parameter lam.
    Returns the estimated coefficients.
    """
    lasso = Lasso(alpha=lam, max_iter=max_iter, fit_intercept=False)
    lasso.fit(X, y)
    return lasso.coef_

def theoretical_lambda(sigma, n, p, c=1.0):
    """
    Compute theoretical lambda for Lasso based on noise level sigma,
    number of samples n, and number of features p.
    """
    return c * sigma * np.sqrt(2 * np.log(p) / n)
