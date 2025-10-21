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


**Reference:**  
Wainwright, M. J. (2009). *Sharp thresholds for high-dimensional and noisy sparsity recovery using ℓ₁-constrained quadratic programming (Lasso)*. IEEE Trans. Info. Theory, 55(5), 2183–2202.
