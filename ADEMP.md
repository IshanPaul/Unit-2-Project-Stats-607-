
# ADEMP Framework: Simulation Design Document

## Aims
The goal of this simulation study is to evaluate statistical methods in high-dimensional regression settings, examining how correlation structures, signal strength, and sample size affect estimator performance.
We aim to answer the following questions:
1. How does correlation among predictors influence the bias and variance of estimators?
2. How do different signal strengths impact model recovery accuracy?
3. Which estimators are most robust under varying noise levels?

## Data-Generating Mechanisms (DGMs)
The data are generated as follows:
- Response: \( y = X\beta + \epsilon \)
- Predictors: \( X \sim N(0, \Sigma) \), where \( \Sigma_{ij} = \rho^{|i-j|} \)
- Error: \( \epsilon \sim N(0, \sigma^2 I) \)

**Parameters varied across simulation conditions:**
| n | p | ρ | β strength | σ | reps |
|---|---|---|-------------|---|------|
| 343 | 1000 | 0.0 | 1 | 1.0 | 1000 |
| 514 | 1000 | 0.5 | 10 | 1.0 | 1000 |
| 686 | 1000 | 0.9 | 100 | 1.0 | 1000 |
| 1028 | 1000 | 0.5 | 10 | 1.0 | 1000 |
| 1371 | 1000 | 0.9 | 100 | 1.0 | 1000 |

The design grid combines multiple values of `n`, `ρ`, and signal-to-noise ratios to assess robustness across complexity levels.

Seeds are fixed for reproducibility (`numpy.random.seed(42)`), and all raw outputs are cached under `results/raw/`.

## Estimands / Targets
The primary estimands include:
- Regression coefficients (\( \beta \))
- Mean squared error (MSE) of estimated vs. true coefficients
- Coverage probability of confidence intervals
- Power and Type I error where applicable

## Methods
The following methods are evaluated:
1. **Ordinary Least Squares (OLS)** — baseline method.
2. **Ridge Regression** — penalized estimator with L2 regularization.
3. **Lasso Regression** — sparse estimator using L1 penalty.
4. **Elastic Net** — combines L1 and L2 penalties.

All methods are implemented in `src/methods.py` using `scikit-learn` and `numpy`.

## Performance Measures
The following metrics are computed in `src/metrics.py`:
- Bias: \( E[\hat{\theta}] - \theta \)
- Variance: \( Var(\hat{\theta}) \)
- Mean Squared Error: \( E[(\hat{\theta} - \theta)^2] \)
- Coverage probability and empirical power (if applicable)

Results are aggregated across repetitions to summarize performance stability.

## Reproducibility Notes
- Number of replications: 1000 (default)
- Seeds are fixed for reproducibility
- Raw simulation output saved under `results/raw/`
- Figures are saved under `results/figures/`
- Simulation parameters and results are logged for transparency

