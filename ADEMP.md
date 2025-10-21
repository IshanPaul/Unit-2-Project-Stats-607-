
---

## 🧮 2. `ADEMP.md`

```markdown
# ADEMP — Simulation Study on Sharp Thresholds for Lasso Support Recovery

## A — Aims
- **Primary aim:** Empirically reproduce the phase transition for exact signed-support recovery of the Lasso, as derived in Wainwright (2009).
- **Secondary aims:**
  - Examine how the critical sample size threshold \( n_{\text{crit}} = 2k \log(p-k) + k \) shifts under correlation (ρ), noise (σ), and signal strength (β_min).
  - Compare theoretical vs. cross-validated λ selection.
  - Illustrate phase transition via recovery probability vs. normalized sample size \( θ = n / (2k \log(p-k)) \).

---

## D — Data-Generating Mechanisms
Simulated data follow the linear model:
\[
y = X\beta^* + w,\quad w \sim N(0, \sigma^2 I_n).
\]

### Parameters
| Symbol | Meaning | Typical values |
|---------|----------|----------------|
| n | sample size | determined via θ grid |
| p | total predictors | 1000 |
| k | nonzero β entries | {20, 50, 100, 200} |
| ρ | predictor correlation | {0.0, 0.3, 0.6} |
| σ | noise standard deviation | {0.5, 1.0} |
| β_min | signal magnitude | {0.5, 1.0, 2.0} |

### Design matrix
- \( X \sim N(0, \Sigma) \) with \( \Sigma_{ij} = \rho^{|i-j|} \).
- Columns normalized: \( \|X_j\|_2 / \sqrt{n} = 1. \)

### True coefficients
- Random support of size k.
- Signs ±1 with equal probability.
- Magnitude fixed at β_min.

### Sample size grid
For each (p,k), compute \( n = 2k\log(p-k)\theta \) for:
\[
\theta \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}.
\]

---

## E — Estimands
| Estimand | Description |
|-----------|--------------|
| \( P_{\text{exact}} \) | Probability of exact signed-support recovery |
| TPR/FDP | True positive & false discovery proportions |
| MSE | Mean squared error \( \|\hat\beta - \beta^*\|_2^2 \) |
| θ | Normalized sample ratio \( n / (2k\log(p-k)) \) |

---

## M — Methods
### Estimators
- **Lasso:** `sklearn.linear_model.Lasso`  
  - \( \lambda = c \cdot \sigma \sqrt{2 \log(p-k)/n} \), with c ∈ {1.0, 1.5}.
  - Also compare `LassoCV` (5-fold CV).
- **Group Lasso:** optional extension using `group-lasso` package for grouped designs.

### Simulation parameters
- R = 200 replicates per condition.
- Random seed fixed by condition ID + replicate ID.
- All code modularized:
  - `dgps.py` → data generation  
  - `methods.py` → estimators  
  - `simulation.py` → main loop  
  - `figures.py` → visualization

---

## P — Performance & Analysis
### Primary metric
- **Exact signed-support recovery rate** across replicates for each condition.

### Secondary metrics
- TPR, FDP, MSE
- Recovery probability vs θ plots

### Output
- Raw results: `results/raw/`
- Aggregated summaries: `results/summary.csv`
- Figures: `results/figures/`

### Makefile Targets
| Command | Description |
|----------|--------------|
| `make all` | Run simulation, analysis, and plotting |
| `make simulate` | Generate simulation data and results |
| `make analyze` | Aggregate results |
| `make figures` | Plot output figures |
| `make clean` | Remove intermediate results |

---

## Expected Outcome
- Sharp S-shaped transition in recovery probability around θ ≈ 1.
- Shift to larger θ as ρ or σ increase.
- Cross-validated λ performs worse for exact recovery but similar for prediction.

---
```
Unit-2-Project-Stats-607-/
├── Makefile
├── README.md
├── ADEMP.md
├── requirements.txt
├── src/
│   ├── dgps.py
│   ├── methods.py
│   ├── simulation.py
│   ├── figures.py
│   ├── __init__.py
│
├── scripts/
│   └── focused_search_lam.py
│   ├── heatmaps.py
│
│
├── tests/
│   └── test_basic.py
│
├── results/
│   ├── raw/
│   ├── figures/
│   └── analysis/
```

**Reference:**  
Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ₁-constrained quadratic programming (Lasso)*. IEEE Trans. Info. Theory, 55(5), 2183–2202.
