import pandas as pd
from src.simulation import run_simulation
import numpy as np
from src import metrics
from src import methods
from src.dgps import generate_data

def test_run_simulation_smoke():
	# small smoke test: don't save to disk during test
	df = run_simulation(
		n=20,
		p=50,
		k=5,
		rho=0.2,
		b=1.0,
		sigma=1.0,
		lam_factor=1.0,
		n_reps=3,
		seed=123
	)

	assert isinstance(df, dict)


def test_support_and_exact_recovery_and_tpr_fdp():
	beta_true = np.array([-0.5, 0.0, 0.5, 0.0])
	# exact same estimate -> exact recovery
	beta_est_same = beta_true.copy()
	supp = metrics.support(beta_true)
	assert np.array_equal(supp, np.array([0, 2]))
	assert metrics.exact_support_recovery(beta_true, beta_est_same)

	# estimate with one small extra nonzero (above default tol=1e-8)
	beta_est_extra = beta_true.copy()
	beta_est_extra[1] = 1e-6
	assert not metrics.exact_support_recovery(beta_true, beta_est_extra)

	tpr, fdp = metrics.tpr_fdp(beta_true, beta_est_extra)
	# support_true = {0,2}, support_est = {0,1,2} => TP=2, FP=1, FN=0
	assert np.isclose(tpr, 1.0)
	assert np.isclose(fdp, 1.0 / 3.0)


def test_mse():
	a = np.array([1.0, 2.0, 3.0])
	b = np.array([1.0, 2.5, 2.5])
	# MSE = mean((0, -0.5, 0.5)^2) = (0 + 0.25 + 0.25)/3 = 0.166...
	val = metrics.mse(a, b)
	assert np.isclose(val, (0.25 + 0.25) / 3.0)


def test_theoretical_lambda_and_fit_lasso_recovery():
	# small regression problem where Lasso with small alpha should recover coefficients
	n, p, k = 500, 10, 3
	rho, b, sigma = 0.1, 1.0, 0.1
	seed = 0
	X, y, beta_true, support = generate_data(n, p, k, rho, b, sigma, seed=seed)

	lam_theory = methods.theoretical_lambda(sigma, n, p)
	assert lam_theory > 0

	# use a small multiplier to avoid over-penalization
	lam = lam_theory * 0.1
	beta_est = methods.fit_lasso(X, y, lam)

	assert isinstance(beta_est, np.ndarray)
	assert beta_est.shape[0] == p

	# check MSE of coefficients is reasonably small for this easy problem
	mse_val = metrics.mse(beta_true, beta_est)
	assert mse_val < 0.1


def test_tpr_fdp_edge_cases_empty_supports():
	beta_true = np.zeros(5)
	beta_est = np.zeros(5)
	tpr, fdp = metrics.tpr_fdp(beta_true, beta_est)
	assert tpr == 0.0 and fdp == 0.0


def test_mutual_incoherence_of_design():
	"""Compute mutual incoherence mu = max_{i!=j} |<X_i, X_j>| for generated X.
	We check that mu is in [0,1] and that increasing the Toeplitz rho increases mu (monotonicity check).
	"""
	def mutual_incoherence(X: np.ndarray) -> float:
		# columns should be normalized by generate_X, but normalize defensively
		Xc = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-12)
		G = np.abs(Xc.T @ Xc)
		np.fill_diagonal(G, 0.0)
		return float(np.max(G))

	n, p = 500, 20
	seed = 12345

	X_low = generate_data(n, p, k=3, rho=0.0, b=1.0, sigma=0.1, seed=seed)[0]
	X_high = generate_data(n, p, k=3, rho=0.9, b=1.0, sigma=0.1, seed=seed)[0]

	mu_low = mutual_incoherence(X_low)
	mu_high = mutual_incoherence(X_high)

	assert 0.0 <= mu_low <= 1.0
	assert 0.0 <= mu_high <= 1.0
	# high rho should produce more correlated columns (loose monotonicity check)
	assert mu_high >= mu_low


def test_generate_X_normalization():
	# columns produced by generate_X should be normalized to unit L2 norm
	n, p = 100, 30
	seed = 42
	X = generate_data(n, p, k=3, rho=0.3, b=1.0, sigma=0.5, seed=seed)[0]
	sd = np.std(X, axis=0)
	# allow small numerical tolerance
	assert np.allclose(sd, 1.0, rtol=1e-6, atol=1e-6)


def test_lasso_coefficient_scale_and_sign_recovery():
	# Large n, small sigma -> Lasso with small alpha should recover coefficients closely
	n, p, k = 1500, 40, 6
	rho, b, sigma = 0.2, 1.5, 0.05
	seed = 2025

	X, y, beta_true, support_true = generate_data(n, p, k, rho, b, sigma, seed=seed)
	lam_theory = methods.theoretical_lambda(sigma, n, p)
	lam = lam_theory * 0.05
	beta_est = methods.fit_lasso(X, y, lam)

	# Check norms: if columns normalized, beta_est should be on same scale as beta_true
	mse_val = metrics.mse(beta_true, beta_est)
	tpr, fdp = metrics.tpr_fdp(beta_true, beta_est)

	# low noise and many samples -> expect reasonably accurate recovery
	assert mse_val < 0.05
	assert tpr >= 0.6
	# check sign agreement on true support (fraction of correctly signed nonzeros)
	true_idx = support_true
	est_signs = np.sign(beta_est[true_idx])
	true_signs = np.sign(beta_true[true_idx])
	sign_agree = float(np.mean(est_signs == true_signs))
	assert sign_agree >= 0.6


if __name__ == "__main__":
    test_run_simulation_smoke()
    test_support_and_exact_recovery_and_tpr_fdp()
    test_mse()
    test_theoretical_lambda_and_fit_lasso_recovery()
    test_tpr_fdp_edge_cases_empty_supports()
    test_mutual_incoherence_of_design()
    test_generate_X_normalization()
    test_lasso_coefficient_scale_and_sign_recovery()
    print("All tests passed.")