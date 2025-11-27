# src/sim_helpers.py
from src.sim_runner import run_simulation

def run_simulation_grid(combo, n_reps=1000, seed=None):
    n, p, k, rho, b, sigma, lam_factor = combo
    return run_simulation(n, p, k, rho, b, sigma, lam_factor, n_reps=n_reps, seed=seed)
