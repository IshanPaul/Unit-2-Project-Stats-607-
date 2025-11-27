# ADEMP — Reproduction of Wainwright (2009)

## A. Aims

The goal of this simulation is to reproduce and empirically validate the **sharp threshold phenomenon** for exact support recovery in high-dimensional linear regression using the **Lasso** estimator, as presented in *Wainwright (2009)*:  
> “Sharp thresholds for high-dimensional and noisy sparsity recovery using L1-Constrained Quadratic Programming (Lasso).”

We aim to investigate how the probability of exact and unsigned support recovery depends on the parameter  
\[
\theta = \frac{n b^2}{\sigma^2 \log(p - k)},
\]
where:
- \( n \): sample size  
- \( b \): minimum signal strength  
- \( \sigma^2 \): noise variance  
- \( p \): ambient dimensionality  
- \( k \): sparsity (number of nonzero coefficients)

**Hypotheses**
1. There exists a critical threshold near **θ = 1** separating recovery and failure regimes.
2. Higher feature correlations (ρ) degrade support recovery.
3. Stronger signal strengths (b) improve recovery.

---

## D. Data-Generating Mechanisms

We consider the high-dimensional linear model:
\[
Y = X\beta + \varepsilon,
\]
where:
- \( X \in \mathbb{R}^{n \times p} \) has rows \( x_i \sim \mathcal{N}(0, \Sigma) \)
- \( \Sigma_{ij} = \rho^{|i-j|} \) introduces Toeplitz-style correlations between features
- \( \beta \in \mathbb{R}^p \) has \( k \) nonzero entries equal to \( b or -b\)
- \( \varepsilon \sim \mathcal{N}(0, \sigma^2 I_n) \)

### Parameter grid

| Parameter | Values | Description |
|------------|---------|-------------|
| θ | np.arange(0.5, 2.05, 0.05) | Controls sample size \( n \) |
| p | 1000 | Number of predictors |
| k | 40 | True sparsity level |
| ρ | [0.0, 0.3, 0.6] | Feature correlation levels |
| b | [0.5, 1.0] | Minimum signal strength |
| σ | [1.0] | Noise level |
| λ-factor | [1.0] | Theoretical λ scaling factor |
| n | Computed from θ using `n_from_theta()` | Derived sample size |
| Repetitions | 50 (default) | Monte Carlo replications |

\[
    n_from_theta = \frac{scale\theta\sigma^2\log(p-k)}{b^2}
]
where scale = 30, otherwise values of n would be too small.
Each parameter combination defines one simulation condition.

---

## E. Estimands (Targets)

We evaluate two main recovery probabilities:

1. **Exact Support Recovery Rate**
   \[
   P(\hat{S} = S)
   \]
   where \( S = \text{supp}(\beta) \) and \( \hat{S} = \text{supp}(\hat{\beta}_{lasso}) \).

2. **Unsigned Support Recovery Rate**
   \[
   P(|\hat{S}| = |S| \text{ and } \text{signs ignored})
   \]
   — a relaxed measure that ignores coefficient signs.

These estimands quantify the empirical probability of correctly recovering the sparsity structure of the true model.

---

## M. Methods

We use the **Lasso estimator**:
\[
\hat{\beta}_{lasso} = \arg\min_{\beta} \frac{1}{2n} \|Y - X\beta\|_2^2 + \lambda \|\beta\|_1,
\]
where
\[
\lambda = \lambda_{\text{factor}} \cdot \sigma \sqrt{2 \log(p) / n}.
\]

**Implementation details:**
- Implemented via `sklearn.linear_model.Lasso`
- Random seeds generated using `numpy.random.default_rng(42)`
- Each replicate uses a new random seed for independence
- Correlated design matrices generated using Toeplitz covariance structure
- Simulations parallelized using `ProcessPoolExecutor`

---

## P. Performance Measures

Each simulation computes the following metrics:

| Metric | Description |
|---------|-------------|
| **Exact Support Recovery Rate** | Proportion of replications where the exact support is recovered |
| **Unsigned Support Recovery Rate** | Proportion where the correct support size is recovered, ignoring signs |
| **(Optional)** Mean Squared Error | \( \|\hat{\beta} - \beta\|_2^2 \) |
| **(Optional)** Runtime | Per-replication computation time |

Results are aggregated into a DataFrame with columns:
```
["rho", "theta", "beta_min", "sigma", "exact_support_recovery_rate",
 "unsigned_support_recovery_rate", "n", "rep"]
```

---

## Simulation Workflow

1. Generate parameter combinations for all values of ρ, b, σ, and θ.
2. Compute \( n \) from θ via `n_from_theta(θ, p, k, b, σ)`.
3. For each condition:
   - Generate data (X, β, Y)
   - Fit Lasso with theoretical λ
   - Record recovery indicators
4. Aggregate and save results.

---

## Output and Visualization

- **Raw results:**  
  `results/raw/large_experiment_parallel.csv`

- **Figures:**
  - `results/figures/phase_transition_wainwright2009.png` — Exact support recovery vs. θ  
  - `results/figures/unsigned_phase_transition_wainwright2009.png` — Unsigned recovery vs. θ

Expected outcomes:
- Recovery probability transitions sharply near **θ ≈ 1**.
- Higher ρ → lower recovery probability.
- Higher b → steeper, more successful recovery curve.

---

## Reproducibility Notes

- Random generator: `numpy.random.default_rng(42)`
- Parallel processing: `ProcessPoolExecutor` (`--n_jobs` parameter)
- All results saved automatically with timestamps
- Run full experiment with:
  ```bash
  python main.py --mode large --n_jobs 10 --save
  ```
- Runtime: 29.74 minutes on an 10-core machine

---

## References

- Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and noisy sparsity recovery using L1-constrained quadratic programming (Lasso).* IEEE Transactions on Information Theory, 55(5), 2183–2202.
