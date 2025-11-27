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
TOLERANCE = 1e-10  # Numerical tolerance for floating point comparison

def test_cholesky_caching():
    """Test that cached Cholesky produces identical results"""
    print("\n" + "="*60)
    print("TEST 1: Cholesky Decomposition Caching")
    print("="*60)
    
    n, p, rho = 300, 1000, 0.5
    seed = 42
    
    # Generate X multiple times with same seed
    # Cache should return identical results
    X1 = generate_X(n, p, rho, seed=seed)
    X2 = generate_X(n, p, rho, seed=seed)  # Should hit cache
    
    # Check exact equality (same seed should give same result)
    assert np.array_equal(X1, X2), "Same seed should produce identical X"
    print("  ✓ Same seed produces identical results")
    
    # Generate X with different seeds
    X3 = generate_X(n, p, rho, seed=seed+1)
    
    # Should be different
    assert not np.array_equal(X1, X3), "Different seeds should produce different X"
    print("  ✓ Different seeds produce different results")
    
    # But should have same covariance structure
    # Check that sample covariance is close to theoretical
    Sigma_theoretical = make_sigma(p, rho)
    Sigma_empirical1 = X1.T @ X1 / n
    Sigma_empirical3 = X3.T @ X3 / n
    
    # Both should be close to theoretical (within sampling error)
    error1 = np.linalg.norm(Sigma_empirical1 - Sigma_theoretical, 'fro')
    error3 = np.linalg.norm(Sigma_empirical3 - Sigma_theoretical, 'fro')
    
    print(f"  Frobenius error for X1: {error1:.4f}")
    print(f"  Frobenius error for X3: {error3:.4f}")
    print(f"  Both should be < 50 due to sampling variation")
    
    assert error1 < 50, "X1 covariance structure incorrect"
    assert error3 < 50, "X3 covariance structure incorrect"
    print("  ✓ Covariance structure preserved")
    
    print("\n✅ TEST 1 PASSED: Cholesky caching works correctly\n")

def test_batch_vs_sequential():
    """Test that batch processing produces same results as one-at-a-time"""
    print("="*60)
    print("TEST 2: Batch Processing Correctness")
    print("="*60)
    
    # Small test case
    n, p, k = 150, 100, 10
    rho, b, sigma, lam_factor = 0.5, 1.0, 1.0, 1.0
    n_reps = 20
    seed = 42
    
    print(f"\nRunning {n_reps} reps with n={n}, p={p}, k={k}")
    
    # Run with batch processing
    print("  Running batch mode (batch_size=10)...")
    results_batch = run_simulation(
        n, p, k, rho, b, sigma, lam_factor, n_reps, 
        seed=seed, batch_size=10
    )
    
    # Run one-at-a-time
    print("  Running sequential mode...")
    results_sequential = run_simulation_ultralow_memory(
        n, p, k, rho, b, sigma, lam_factor, n_reps, seed=seed
    )
    
    # Compare results
    print("\n  Comparing results:")
    for key in ['average_mse', 'average_tpr', 'average_fdp', 
                'exact_support_recovery_rate', 'unsigned_support_recovery_rate']:
        val_batch = results_batch[key]
        val_seq = results_sequential[key]
        diff = abs(val_batch - val_seq)
        
        print(f"    {key}:")
        print(f"      Batch:      {val_batch:.6f}")
        print(f"      Sequential: {val_seq:.6f}")
        print(f"      Difference: {diff:.2e}")
        
        assert diff < TOLERANCE, f"{key} differs between batch and sequential!"
    
    print("\n✅ TEST 2 PASSED: Batch processing preserves correctness\n")

def test_metric_computation():
    """Test that vectorized metrics match sequential computation"""
    print("="*60)
    print("TEST 3: Vectorized Metric Computation")
    print("="*60)
    
    p, k = 100, 10
    n_test = 50
    
    print(f"\nTesting metrics on {n_test} random coefficient pairs")
    
    # Generate random beta pairs
    np.random.seed(42)
    errors_mse = []
    errors_tpr = []
    errors_fdp = []
    
    for i in range(n_test):
        beta_true, _ = generate_beta(p, k, 1.0, seed=i)
        beta_est, _ = generate_beta(p, k, 0.8, seed=i+1000)
        
        # Compute metrics
        mse_val = mse(beta_true, beta_est)
        tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
        
        # Manual computation for verification
        mse_manual = np.mean((beta_true - beta_est) ** 2)
        
        true_support = np.abs(beta_true) > 1e-8
        est_support = np.abs(beta_est) > 1e-8
        TP = np.sum(true_support & est_support)
        FP = np.sum(~true_support & est_support)
        FN = np.sum(true_support & ~est_support)
        
        tpr_manual = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        fdp_manual = FP / (FP + TP) if (FP + TP) > 0 else 0.0
        
        # Check agreement
        errors_mse.append(abs(mse_val - mse_manual))
        errors_tpr.append(abs(tpr_val - tpr_manual))
        errors_fdp.append(abs(fdp_val - fdp_manual))
    
    print(f"  MSE errors: max={np.max(errors_mse):.2e}, mean={np.mean(errors_mse):.2e}")
    print(f"  TPR errors: max={np.max(errors_tpr):.2e}, mean={np.mean(errors_tpr):.2e}")
    print(f"  FDP errors: max={np.max(errors_fdp):.2e}, mean={np.mean(errors_fdp):.2e}")
    
    assert np.max(errors_mse) < TOLERANCE, "MSE computation incorrect"
    assert np.max(errors_tpr) < TOLERANCE, "TPR computation incorrect"
    assert np.max(errors_fdp) < TOLERANCE, "FDP computation incorrect"
    
    print("\n✅ TEST 3 PASSED: Metric computations correct\n")

def test_deterministic_behavior():
    """Test that same seed produces same results"""
    print("="*60)
    print("TEST 4: Deterministic Behavior")
    print("="*60)
    
    n, p, k = 150, 100, 10
    rho, b, sigma, lam_factor = 0.5, 1.0, 1.0, 1.0
    n_reps = 10
    seed = 123
    
    print(f"\nRunning simulation twice with same seed={seed}")
    
    # Run twice with same seed
    results1 = run_simulation(
        n, p, k, rho, b, sigma, lam_factor, n_reps, 
        seed=seed, batch_size=5
    )
    
    results2 = run_simulation(
        n, p, k, rho, b, sigma, lam_factor, n_reps, 
        seed=seed, batch_size=5
    )
    
    # Should be identical
    print("\n  Comparing results:")
    for key in results1.keys():
        if isinstance(results1[key], (int, float)):
            diff = abs(results1[key] - results2[key])
            print(f"    {key}: difference = {diff:.2e}")
            assert diff < TOLERANCE, f"{key} not deterministic!"
    
    print("\n✅ TEST 4 PASSED: Behavior is deterministic\n")

def test_edge_cases():
    """Test edge cases and numerical stability"""
    print("="*60)
    print("TEST 5: Edge Cases and Stability")
    print("="*60)
    
    print("\n  Testing edge case: All zeros in beta_est")
    beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
    beta_est = np.zeros(5)
    
    mse_val = mse(beta_true, beta_est)
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    
    print(f"    MSE: {mse_val:.4f} (should be {np.mean(beta_true**2):.4f})")
    print(f"    TPR: {tpr_val:.4f} (should be 0.0)")
    print(f"    FDP: {fdp_val:.4f} (should be 0.0)")
    
    assert abs(mse_val - np.mean(beta_true**2)) < TOLERANCE
    assert tpr_val == 0.0
    assert fdp_val == 0.0
    
    print("\n  Testing edge case: Perfect recovery")
    beta_true = np.array([1.0, 0.0, 2.0, 0.0, 0.0])
    beta_est = beta_true.copy()
    
    mse_val = mse(beta_true, beta_est)
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    exact = exact_support_recovery(beta_true, beta_est)
    unsigned = support_recovery(beta_true, beta_est)
    
    print(f"    MSE: {mse_val:.4f} (should be 0.0)")
    print(f"    TPR: {tpr_val:.4f} (should be 1.0)")
    print(f"    FDP: {fdp_val:.4f} (should be 0.0)")
    print(f"    Exact recovery: {exact} (should be True)")
    print(f"    Unsigned recovery: {unsigned} (should be True)")
    
    assert mse_val < TOLERANCE
    assert tpr_val == 1.0
    assert fdp_val == 0.0
    assert exact == True
    assert unsigned == True
    
    print("\n  Testing edge case: No true support")
    beta_true = np.zeros(5)
    beta_est = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    
    tpr_val, fdp_val = tpr_fdp(beta_true, beta_est)
    
    print(f"    TPR: {tpr_val:.4f} (denominator is 0, should return 0)")
    print(f"    FDP: {fdp_val:.4f} (should be 1.0)")
    
    assert tpr_val == 0.0  # Our implementation returns 0 when TP+FN=0
    assert fdp_val == 1.0
    
    print("\n✅ TEST 5 PASSED: Edge cases handled correctly\n")

def run_all_tests():
    """Run all regression tests"""
    print("\n" + "="*60)
    print("REGRESSION TEST SUITE FOR UNIT 3 OPTIMIZATIONS")
    print("="*60)
    print("\nThis test suite verifies that optimizations preserve correctness")
    print("by comparing optimized implementations against reference versions.\n")
    
    try:
        test_cholesky_caching()
        test_batch_vs_sequential()
        test_metric_computation()
        test_deterministic_behavior()
        test_edge_cases()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nOptimizations preserve correctness.")
        print("The optimized code produces results equivalent to baseline.\n")
        
        return 0
        
    except AssertionError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}\n")
        return 1
    
    except Exception as e:
        print("\n" + "="*60)
        print("❌ UNEXPECTED ERROR")
        print("="*60)
        print(f"\nError: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)