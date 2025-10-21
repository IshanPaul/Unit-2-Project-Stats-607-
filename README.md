# Unit 2 Project — Simulation Study (Stats 607)

## Overview
This repository reproduces and extends the simulation results from **Wainwright (2009)**,  
> *"Sharp Thresholds for High-Dimensional and Noisy Sparsity Recovery Using ℓ₁-Constrained Quadratic Programming (Lasso)"*  
(*IEEE Transactions on Information Theory, 55(5)*).

The goal is to empirically verify the **sharp phase transition** in support recovery of the **Lasso estimator** and explore how it depends on:
- Sample size \( n \)
- Problem dimension \( p \)
- Sparsity \( k \)
- Correlation \( \rho \)
- Noise level \( \sigma \)
- Signal strength \( \beta_{\min} \)

The entire project follows the **ADEMP framework** (Aims, Data-generating mechanisms, Estimands, Methods, Performance measures).

---

## Directory Structure
├── src/
│ ├── init.py
│ ├── dgps.py # Data generation (X, β, y)
│ ├── methods.py # Lasso and Group Lasso estimators
│ ├── simulation.py # Main simulation pipeline
│ ├── figures.py # Visualization and analysis
│
├── results/
│ ├── raw/ # Raw simulation outputs
│ └── figures/ # Generated plots
│
├── data/ # Optional cached simulated datasets
├── tests/ # Unit tests
│
├── ADEMP.md # Full ADEMP description
├── Makefile # Run full pipeline (simulate → analyze → figures)
├── requirements.txt
└── README.md