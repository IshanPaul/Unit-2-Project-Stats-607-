# ADEMP — Reproducing Wainwright (2009): Sharp Thresholds for Lasso Support Recovery

## A. Aims
- **Primary aim:** Reproduce and empirically verify Wainwright (2009)'s sharp sample-complexity threshold for exact support recovery by the Lasso:
  \[
  n_{\text{crit}} \approx 2k\log(p-k) + k.
  \]
- **Secondary aims:**
  - Measure sensitivity of the threshold to predictor correlation (ρ), noise level (σ), and minimum signal strength (β_min).
  - Compare Lasso with group-aware alternatives (Group Lasso) in settings with grouped categorical features.
  - Produce diagnostic and publication-quality visualizations that illustrate the phase transition.

**Research questions**
1. For iid Gaussian and AR(1) correlated designs, how does empirical exact signed-support recovery probability change as \(n\) crosses \(n_{\text{crit}}\)?
2. How do ρ, σ, and β_min shift the transition?
3. When predictors are grouped and group sparsity holds, does Group Lasso outperform Lasso in support/group recovery and MSE?

---

## D. Data-generating mechanisms (DGPs)

We simulate data from the linear model
\[
y = X\beta^* + w,\qquad w\sim N(0,\sigma^2 I_n).
\]

### Design matrix \(X\)
- **IID Gaussian:** \(X_{ij}\overset{iid}{\sim}N(0,1)\), then column-wise normalized so \(\|X_j\|_2/\sqrt{n}=1\).
- **AR(1) / Toeplitz covariance:** rows \(x_i\sim N(0,\Sigma)\) with \(\Sigma_{jk}=\rho^{|j-k|}\). Values of ρ: `{0.0, 0.3, 0.6}`. Columns normalized after sampling.
- **Grouped categorical + continuous design:** construct categorical variables, one-hot encode, and concatenate with continuous covariates for Group Lasso experiments.

### True coefficients \(\beta^*\)
- Dimension \(p\in\{500, 1000, 2000\}\).
- Sparsity \(k\in\{5, 10, 20\}\).
- Support \(S\) chosen uniformly at random of size k.
- Nonzero entries: either fixed amplitude \(b\in\{0.5, 1.0, 2.0\}\) (with random signs) or drawn from a small uniform range (e.g., `Unif(b/2, 3b/2)`).
- For group experiments, groups correspond to categorical factors (all dummy columns for a factor form a group).

### Noise
- Gaussian noise with variance σ². Values: `σ ∈ {0.5, 1.0, 2.0}` depending on SNR scenarios.

### Sample size grid (n)
- For each (p,k) compute theoretical threshold: `n0 = 2*k*log(p-k) + k`.
- Sweep `n` across `round([0.5, 0.75, 1.0, 1.25, 1.5, 2.0] * n0)`; adjust grid for runtime constraints.

### Replications
- Replicates per condition: `R = 500` (use `R = 200` for quick exploratory runs).
- RNG: use `base_seed + condition_id + replicate_id` to ensure reproducibility.

---

## E. Estimands (targets)
- **Primary estimand:** Empirical probability of *exact signed-support recovery*
  \[
  \hat{P}_{\text{exact}} = \frac{1}{R}\sum_{r=1}^R \mathbb{I}\{\operatorname{sign}(\hat\beta^{(r)})=\operatorname{sign}(\beta^*)\}.
  \]
- **Secondary estimands:**
  - Support recovery ignoring sign (set equality).
  - True Positive Rate (TPR) and False Discovery Proportion (FDP).
  - Parameter estimation error: MSE \(= \|\hat\beta - \beta^*\|_2^2\).
  - Computational runtime per replicate.
  - For group experiments: group-wise selection accuracy (group TPR, group FDP) and group coefficient norms.

---

## M. Methods
- **Primary estimator:** Lasso (ℓ₁-penalized least squares)
  - Implementation: `sklearn.linear_model.Lasso` for fits; `LassoCV` for a prediction-oriented λ baseline.
  - Fit with `fit_intercept=False` and `normalize=False` because we pre-normalize columns in the DGP.
  - Theoretical λ: \(\lambda_n = c \cdot \sigma \sqrt{\tfrac{2\log(p-k)}{n}}\) with `c ∈ {1.0, 1.5}` (tested).
- **Group-aware estimator:** Group Lasso (for categorical group experiments)
  - Implementation: `group-lasso` package (`GroupLasso`).
  - Cross-validated group penalty `group_reg` chosen over a log-spaced grid.
- **Baselines / checks:**
  - OLS (when `n>p`) for sanity checks.
  - Thresholded Lasso (post-selection hard thresholding) as an alternative.

**Implementation practices**
- Modular code: `src/dgps.py`, `src/methods.py`, `src/metrics.py`, `src/simulation.py`, `src/analyze.py`, `src/figures.py`.
- Logging: save parameter dictionary and git commit hash for each run in `results/runinfo.json`.
- Seed control for reproducibility.

---

## P. Performance measures (how we evaluate)
- **Exact signed-support recovery rate** (primary).
- **Support recovery rate** (set equality).
- **TPR, FDP**: per-replicate and averaged with standard errors.
- **MSE**: mean and sd across replicates.
- **Group-wise metrics:** ℓ₂ norm of group coefficients; group TPR and group FDP.
- **Calibration diagnostics:** histogram of estimated coefficients for active/inactive indices; ROC-like curves if appropriate.
- **Plots:** recovery probability vs `n / (2k log(p-k))`, curve with ±1 SE bands; heatmaps or contour plots over `(n, k)` grid.

---

## Simulation matrix & storage
- Programmatically generate full matrix and save as `results/design_matrix.csv` with fields:
  `design_type, rho, p, k, n, sigma, b, lambda_choice, R, condition_id`.
- Save per-replicate raw outputs in `results/raw/{condition_id}.pkl` or `{condition_id}.csv`.
- Aggregate summaries stored in `results/summary.csv`.
- Figures saved to `results/figures/` in `.png` and `.pdf`.

---

## Tests
Implement at least three tests under `tests/`:
1. **DGP verification:** `test_dgp_properties` — draws X and checks column norms near 1 and sample cov matches Σ.
2. **Method correctness:** `test_lasso_recovery_easy_case` — for `n >> p` and low noise, Lasso (or OLS) recovers support.
3. **Reproducibility:** `test_reproducible_runs` — rerun a small condition with fixed seed and confirm identical saved results.

Run tests with `pytest`.

---

## Makefile targets
- `make all` : run full pipeline (simulate, analyze, figures).
- `make simulate` : run simulations and save raw outputs.
- `make analyze` : summarize raw outputs into `results/summary.csv`.
- `make figures` : generate figures from summary outputs.
- `make test` : run the test-suite.
- `make clean` : remove generated results.

---

## Reproducibility notes
- Fixed seeds and deterministic RNG (`numpy.random.default_rng`) used.
- Save `results/runinfo.json` containing:
  - git commit hash, date/time, python package versions (`pip freeze`), base seed, and parameter grid.
- Cache raw simulation outputs to avoid re-running expensive experiments.

---

## Expected validation & interpretation
- For IID Gaussian design and moderate β amplitude, empirical recovery curves (recovery probability vs normalized n) should show an S-shaped transition around `n/(2k log(p-k)) ≈ 1`.
- Increasing ρ or reducing β_min moves the transition to larger n (harder recovery).
- Group Lasso should outperform Lasso (in MSE and group selection) when the ground truth is group-sparse and group penalty is tuned well; mismatch or poor λ grid can make Group Lasso perform worse.

---

## References
- Wainwright, M. J. (2009). Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ₁-constrained quadratic programming (Lasso). *IEEE Transactions on Information Theory*, 55(5), 2183–2202.
- Additional readings: Hastie et al., *Elements of Statistical Learning*; Tibshirani (1996) on the Lasso; papers on Group Lasso.
