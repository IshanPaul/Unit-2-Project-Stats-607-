# scripts/stability_check_metrics.py
import numpy as np

from src.metrics import (
    support,
    support_recovery,
    exact_support_recovery,
    mse,
    tpr_fdp,
    batch_metrics,
)


# --------------------------------------------------
# Support function tests
# --------------------------------------------------
def check_support_basic():
    print("\n--- Checking support() basic behavior ---")
    beta = np.array([0, 0.1, -0.2, 0.0, 5e-9])
    s = support(beta, tol=1e-8)
    print(f"  support(beta) = {s}")

    if np.array_equal(s, np.array([1, 2])):
        print("  ✅ Correct support indices")
    else:
        print("  ❌ Incorrect support extraction")


def check_support_tolerance():
    print("\n--- Checking support tolerance behavior ---")
    beta = np.array([1e-9, 2e-8, -3e-7])

    s1 = support(beta, tol=1e-6)
    s2 = support(beta, tol=1e-9)

    print(f"  tol=1e-6 → {s1}")
    print(f"  tol=1e-9 → {s2}")

    if len(s1) == 0 and len(s2) == 2:
        print("  ✅ Tolerance threshold applied correctly")
    else:
        print("  ❌ Unexpected behavior with tolerance")


# --------------------------------------------------
# Support recovery tests
# --------------------------------------------------
def check_unsigned_support_recovery():
    print("\n--- Checking unsigned support recovery ---")
    beta_true = np.array([0, 1, -2, 0, 0])
    beta_est = np.array([0, -3, 4, 0, 0])

    if support_recovery(beta_true, beta_est):
        print("  ✅ Correct: support matches ignoring sign")
    else:
        print("  ❌ Support recovery incorrect")


def check_signed_support_recovery():
    print("\n--- Checking exact sign support recovery ---")
    beta_true = np.array([0, 1, -2, 0])
    beta_good = np.array([0, 0.5, -4, 0])
    beta_bad = np.array([0, -0.5, -4, 0])

    if exact_support_recovery(beta_true, beta_good):
        print("  ✅ Correct: signs and support match")
    else:
        print("  ❌ Should have matched")

    if not exact_support_recovery(beta_true, beta_bad):
        print("  ✅ Correct: wrong sign detected")
    else:
        print("  ❌ Sign mismatch not detected")


def check_empty_support():
    print("\n--- Checking empty support behavior ---")
    beta_true = np.zeros(10)
    beta_est = np.zeros(10)

    if support_recovery(beta_true, beta_est):
        print("  ✅ Correct: empty support matches")
    else:
        print("  ❌ Empty supports incorrectly treated as mismatch")

    if exact_support_recovery(beta_true, beta_est):
        print("  ✅ Correct: exact recovery with empty supports")
    else:
        print("  ❌ Exact empty support mismatch")


# --------------------------------------------------
# MSE tests
# --------------------------------------------------
def check_mse():
    print("\n--- Checking MSE calculation ---")
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 5])

    m = mse(a, b)
    print(f"  MSE = {m}")

    if np.isclose(m, (0 + 0 + 4) / 3):
        print("  ✅ MSE computed correctly")
    else:
        print("  ❌ Incorrect MSE")


# --------------------------------------------------
# TPR / FDP tests
# --------------------------------------------------
def check_tpr_fdp_basic():
    print("\n--- Checking TPR / FDP basic behavior ---")

    beta_true = np.array([0, 1, 0, -2, 0])
    beta_est = np.array([0, 3, 0, 0, 0])  # detect only one of two nonzeros

    TPR, FDP = tpr_fdp(beta_true, beta_est)
    print(f"  TPR={TPR:.3f}, FDP={FDP:.3f}")

    if np.isclose(TPR, 1/2) and np.isclose(FDP, 0):
        print("  ✅ Correct TPR/FDP calculation")
    else:
        print("  ❌ TPR/FDP incorrect")


def check_tpr_fdp_full_recovery():
    print("\n--- Checking perfect support recovery ---")
    beta_true = np.array([0, 1, -2])
    beta_est = np.array([0, 10, -8])

    TPR, FDP = tpr_fdp(beta_true, beta_est)

    if TPR == 1.0 and FDP == 0.0:
        print("  ✅ Perfect recovery metrics correct")
    else:
        print("  ❌ Incorrect metrics for perfect recovery")


def check_tpr_fdp_no_true_nonzeros():
    print("\n--- Checking TPR/FDP with no true non-zeros (edge case) ---")
    beta_true = np.zeros(5)
    beta_est = np.array([0, 0, 1, 0, 0])  # one false positive

    TPR, FDP = tpr_fdp(beta_true, beta_est)
    print(f"  TPR={TPR}, FDP={FDP}")

    if TPR == 0 and FDP == 1:
        print("  ✅ Correct behavior when true support empty")
    else:
        print("  ❌ Incorrect metrics for empty true support")


# --------------------------------------------------
# Batch metrics tests
# --------------------------------------------------
def check_batch_metrics():
    print("\n--- Checking batch_metrics() ---")
    rng = np.random.default_rng(0)

    beta_true = rng.normal(size=(5, 10))
    beta_est = beta_true + 0.1 * rng.normal(size=(5, 10))

    results = batch_metrics(beta_true, beta_est)

    if (
        "mse" in results
        and "TPR" in results
        and "FDP" in results
        and "exact_recovery" in results
        and results["mse"].shape == (5,)
    ):
        print("  ✅ batch_metrics returns correct structure")
    else:
        print("  ❌ batch_metrics structure wrong or incomplete")


# --------------------------------------------------
# Run all tests
# --------------------------------------------------
def run_all_checks():
    print("\n========================================")
    print(" Numerical Stability Checks for metrics.py ")
    print("========================================\n")

    check_support_basic()
    check_support_tolerance()

    check_unsigned_support_recovery()
    check_signed_support_recovery()
    check_empty_support()

    check_mse()

    check_tpr_fdp_basic()
    check_tpr_fdp_full_recovery()
    check_tpr_fdp_no_true_nonzeros()

    check_batch_metrics()

    print("\nAll checks completed.\n")


if __name__ == "__main__":
    run_all_checks()
