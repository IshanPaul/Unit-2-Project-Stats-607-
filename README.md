
# Unit 2 Project — Simulation Study

## Overview
This project implements a reproducible simulation pipeline to evaluate regression estimators under high-dimensional, correlated predictor settings.
The simulation follows the ADEMP framework (Aims, Data-generating mechanisms, Estimands, Methods, and Performance measures).

## Project Structure
```
simulation-study/
├── data/
│   └── simulated/
├── src/
│   ├── dgps.py
│   ├── methods.py
│   ├── metrics.py
│   └── simulation.py
├── results/
│   ├── figures/
│   └── raw/
├── tests/
├── ADEMP.md
├── README.md
├── Makefile
├── requirements.txt
└── .gitignore
```

## How to Run
```bash
# Set up environment and dependencies
make install

# Run complete pipeline
make all

# Or run stages individually
make simulate
make figures
```

## Estimated Runtime
~10–15 minutes for 1000 replications on a standard laptop.

## Key Findings
Ridge and Lasso regression outperform OLS in high-correlation settings by reducing MSE and maintaining better coverage across simulation conditions.

## Reproducibility
- Random seeds are fixed, default = 42
- Intermediate and final results are stored in `results/raw/` and `results/figures/`.
- Figures and metrics can be regenerated with `make all`.

## Testing
To validate components (data generation, method correctness, metric validity):
```bash
make test
```

## Authors
- Ishan

## References
- Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and noisy sparsity recovery using L1-constrained quadratic programming (Lasso).* *IEEE Transactions on Information Theory*, 55(5), 2183–2202.
