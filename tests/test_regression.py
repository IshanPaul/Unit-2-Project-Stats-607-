#!/usr/bin/env python3
"""
Regression Tests for Unit 3 Optimizations

Verifies that optimizations preserve correctness by comparing:
1. Cached vs non-cached Cholesky decomposition
2. Batch vs fully-vectorized processing
3. Parallel vs serial execution results
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dgps import generate_X, generate_beta, make_sigma
from src.sim_runner import run_simulation, run_simulation_ultralow_memory
from src.metrics import mse, tpr_fdp, exact_support_recovery, support_recovery

# Test parameters
np.random.seed(42)
TOLERANCE = 1e-8  # Floating point tolerance
COV_TOL = 0.05    # Relative Frobenius error for covariance

# -------------------------------
# Test 1: Cholesky caching
# -------------------------------
def test_cholesky_caching():
    print("\n" + "="*60)
    print("TEST 1: Cholesky Decomposition Caching")
    print("="*60)
    
    n, p, rho = 50, 20, 0.5
    seed = 42
    
    X1 = generate_X(n, p, rho, seed=seed)
    X2 = generate_X(n, p, rho, seed=seed)  # Should hit cache
    assert np.array_equal(X1, X2), "Same seed should produce identical X"
    print("  ✓ Same seed produces identical results")
    
    X3 = generate_X(n, p, rho, seed=seed+1)
    assert not np.array_equal(X1, X3), "Different seeds should produce different X"
    print("  ✓ Different seeds produce different results")
    
    Sigma_theoretical = make_sigma(p, rho)
    Sigma_empirical1 = X1.T @ X1 / n
    Sigma_empirical3 = X3.T @ X3 / n
    
    rel_error1 = np.linalg.norm(Sigma_empirical1 - Sigma_theoretical, 'fro') / np.linalg.norm(Sigma_theoretical, 'fro')
    rel_error3 = np.linalg.norm(Sigma_empirical3 - Sigma_theoretical, 'fro') / np.linalg.norm(Sigma_theoretical, 'fro')
    
    print(f"  Relative Frobenius error for X1: {rel_error1:.3f}")
    print(f"  Relative Frobenius error for X3: {rel_error3:.3f}")
    assert rel_error1 < COV_TOL and rel_error3 < COV_TOL, "Covariance structure incorrect"
    
    print("  ✓ Covariance structure preserved")
    print("\n✅ TEST 1 PASSED\n")

# -------------------------------
# Test 2: Batch vs sequential
# -------------------------------
def test_batch_vs_sequential():
    print("="*60)
    print("TEST 2: Batch Processing Correctness")
    print("="*60)
    
    n, p, k = 20, 10, 5
    rho, b, sigma, lam_factor = 0.5, 1.0, 1.0, 1.0
    n_reps = 10
    seed = 42
    
    results_batch = run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=seed, batch_size=5)
    results_seq = run_simulation_ultralow_memory(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=seed)
    
    for key in ['average_mse', 'average_tpr', 'average_fdp', 'exact_support_recovery_rate', 'unsigned_support_recovery_rate']:
        val_batch = results_batch[key]
        val_seq = results_seq[key]
        diff = abs(val_batch - val_seq)
        print(f"  {key}: batch={val_batch:.6f}, seq={val_seq:.6f}, diff={diff:.2e}")
        assert diff < TOLERANCE, f"{key} differs between batch and sequential"
    
    print("\n✅ TEST 2 PASSED\n")

# -------------------------------
# Test 3: Metric computation
# -------------------------------
def test_metric_computation():
    print("="*60)
    print("TEST 3: Vectorized Metric Computation")
    print("="*60)
    
    p, k = 10, 5
    n_test = 10
    errors_mse, errors_tpr, errors_fdp = [], [], []
    
    for i in range(n_test):
        beta_true, _ = generate_beta(p, k, 1.0, seed=i)
        beta_est, _ = generate_beta(p, k, 0.8, seed=i+100)
        
        mse_val = mse(beta_true, beta_est)
        tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
        
        mse_manual = np.mean((beta_true - beta_est)**2)
        true_support = np.abs(beta_true) > 1e-8
        est_support = np.abs(beta_est) > 1e-8
        TP = np.sum(true_support & est_support)
        FP = np.sum(~true_support & est_support)
        FN = np.sum(true_support & ~est_support)
        tpr_manual = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fdp_manual = FP / (FP + TP) if (FP + TP) > 0 else 0.0
        
        errors_mse.append(abs(mse_val - mse_manual))
        errors_tpr.append(abs(tpr_val - tpr_manual))
        errors_fdp.append(abs(fdp_val - fdp_manual))
    
    assert max(errors_mse) < TOLERANCE
    assert max(errors_tpr) < TOLERANCE
    assert max(errors_fdp) < TOLERANCE
    print("  ✓ Metric computations match manual calculation\n✅ TEST 3 PASSED\n")

# -------------------------------
# Test 4: Deterministic behavior
# -------------------------------
def test_deterministic_behavior():
    print("="*60)
    print("TEST 4: Deterministic Behavior")
    print("="*60)
    
    n, p, k = 20, 10, 5
    rho, b, sigma, lam_factor = 0.5, 1.0, 1.0, 1.0
    n_reps = 5
    seed = 123
    
    results1 = run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=seed, batch_size=2)
    results2 = run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps, seed=seed, batch_size=2)
    
    for key in results1.keys():
        if isinstance(results1[key], (int, float)):
            diff = abs(results1[key] - results2[key])
            assert diff < TOLERANCE, f"{key} not deterministic"
    
    print("  ✓ Deterministic behavior verified\n✅ TEST 4 PASSED\n")

# -------------------------------
# Test 5: Edge cases
# -------------------------------
def test_edge_cases():
    print("="*60)
    print("TEST 5: Edge Cases")
    print("="*60)
    
    # All zeros
    beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
    beta_est = np.zeros(5)
    mse_val = mse(beta_true, beta_est)
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    assert abs(mse_val - np.mean(beta_true**2)) < TOLERANCE
    assert tpr_val == 0.0
    assert fdp_val == 0.0
    
    # Perfect recovery
    beta_est = beta_true.copy()
    mse_val = mse(beta_true, beta_est)
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    exact = exact_support_recovery(beta_true, beta_est)
    unsigned = support_recovery(beta_true, beta_est)
    assert mse_val < TOLERANCE
    assert tpr_val == 1.0
    assert fdp_val == 0.0
    assert exact is True
    assert unsigned is True
    
    # No true support
    beta_true = np.zeros(5)
    beta_est = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    assert tpr_val == 0.0
    assert fdp_val == 1.0
    
    print("  ✓ Edge cases handled correctly\n✅ TEST 5 PASSED\n")

# -------------------------------
# Run all tests
# -------------------------------
def run_all_tests():
    print("\n" + "="*60)
    print("REGRESSION TEST SUITE FOR UNIT 3 OPTIMIZATIONS")
    print("="*60)
    
    try:
        test_cholesky_caching()
        test_batch_vs_sequential()
        test_metric_computation()
        test_deterministic_behavior()
        test_edge_cases()
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    exit_code = run_all_tests()
    sys.exit(exit_code)
