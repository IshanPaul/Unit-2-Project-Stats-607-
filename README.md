# Simulation Study — Reproducing Wainwright (2009)

## Overview

This repository reproduces and extends the key results of **Wainwright (2009)**:  
> *“Sharp thresholds for high-dimensional and noisy sparsity recovery using L1-Constrained Quadratic Programming (Lasso)”*  
> *IEEE Transactions on Information Theory, 55(5), 2183–2202.*

The goal is to empirically verify the **sharp threshold** in support recovery for the Lasso in high-dimensional linear regression and to explore how feature correlation (ρ) and signal strength (b) affect recovery probability.

The simulation is designed and documented under the **ADEMP** framework (`ADEMP.md`).

---

## Repository Structure

```
simulation-study/
├── data/
│   └── simulated/           # optional cached simulation data
├── results/
│   ├── raw/                 # raw simulation outputs (*.csv)
│   └── figures/             # generated visualizations (*.png)
├── src/
│   ├── dgps.py              # data-generating processes (DGP)
│   ├── methods.py           # Lasso estimator implementation
│   ├── metrics.py           # performance metrics
│   ├── simulation.py        # main simulation orchestration
│   ├── analyze.py           # post-processing and summaries
│   └── figures.py           # visualization scripts
├── tests/
│   └── test_basic.py        # unit tests for DGP, metrics, and output validity
├── ADEMP.md                 # simulation design description
├── Makefile                 # automation for full pipeline
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Simulation Design Summary

The simulation follows the model:
\[
Y = X\beta + \varepsilon, \quad X_{ij} \sim \mathcal{N}(0, \Sigma), \quad \Sigma_{ij} = \rho^{|i-j|}.
\]

We vary:
- \( \theta = \frac{n b^2}{\sigma^2 \log(p - k)} \) from **0.5 to 2.0** (in steps of 0.05)
- \( \rho \in \{0.0, 0.3, 0.6\} \)
- \( b \in \{0.5, 1.0\} \)
- \( \sigma = 1.0 \)
- \( p = 1000, k = 40 \)

Each combination defines a simulation condition.  
The Lasso regularization parameter follows the theoretical scaling:
\[
\lambda = \lambda_{\text{factor}} \cdot \sigma \sqrt{2 \log(p) / n}.
\]

For more detail, see [`ADEMP.md`](./ADEMP.md).

---

## Key Outputs

### Raw Results
- `results/raw/large_experiment_parallel.csv`  
  Contains all parameter combinations and recovery metrics.

### Figures
- `results/figures/phase_transition_wainwright2009.png`  
  Exact support recovery probability vs. θ.
- `results/figures/unsigned_phase_transition_wainwright2009.png`  
  Unsigned recovery probability vs. θ.

Expected behavior:
- Sharp transition near **θ ≈ 1**.
- Larger ρ → lower recovery rates.
- Larger b → improved recovery.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
make venv
```

### 2. Install Dependencies
```bash
make install
```

---

## Usage

### 🔹 Run the Full Pipeline
Runs simulation, analysis, and figure generation end-to-end:
```bash
make all
```

### 🔹 Run Parallel Simulation (using all cores)
```bash
make large
```

### 🔹 Run Individual Steps
- Run simulations (serial):  
  ```bash
  make simulate
  ```
- Analyze results:  
  ```bash
  make analyze
  ```
- Generate figures:  
  ```bash
  make figures
  ```

### 🔹 Run Tests
Check correctness of data generation, metrics, and outputs:
```bash
make test
```

### 🔹 Clean Outputs
Remove virtual environment, cached results, and figures:
```bash
make clean
```

---

## Example Output

After running:
```bash
make large
```

You should see progress updates such as:
```
Theta list: [0.5, 0.55, 0.6, ..., 2.0]
Computed n_list (p=1000, k=40, b=0.5): [...]
Launching 246 parallel simulations using 8 workers...
Parameter grid: 100%|████████████████████████████████| 246/246 [XX:XX<00:00, XX.XXit/s]
Results saved to results/raw/large_experiment_parallel.csv
```

Resulting plots will appear under:
```
results/figures/
├── phase_transition_wainwright2009.png
└── unsigned_phase_transition_wainwright2009.png
```

---

## Estimated Runtime

| Mode | Hardware | Runtime |
|------|-----------|----------|
| `make simulate` (serial) | 1 CPU | ~1–1.5 hrs |
| `make large` (parallel, 8 cores) | 8 CPUs | ~15–25 min |

---

## Citation

If reproducing or extending this work, please cite:

> Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and noisy sparsity recovery using L1-constrained quadratic programming (Lasso).* IEEE Transactions on Information Theory, 55(5), 2183–2202.

---

## Author & Acknowledgments

Prepared for **Unit 2 Project — Simulation Study**  
Course: *Advanced Statistical Computing / Simulation Methods*   
Author: *Ishan Paul*  
