# src/sim_helpers.py
from src.sim_runner import run_simulation, run_simulation_ultralow_memory

def run_simulation_grid(combo, n_reps=1000, seed=None, batch_size=100, low_memory=False):
    """
    Wrapper for running a single parameter combination.
    
    Args:
        combo: tuple of (n, p, k, rho, b, sigma, lam_factor)
        n_reps: Number of Monte Carlo replications
        seed: Random seed for reproducibility
        batch_size: Number of reps to process per batch (memory control)
                   - 50: ~120MB per worker (very safe)
                   - 100: ~240MB per worker (optimal, recommended)
                   - 200: ~480MB per worker (faster, needs 8GB+ RAM)
        low_memory: If True, use ultra-low-memory mode (1 rep at a time)
                   - False: batch mode ~240MB (recommended for 8GB+ RAM)
                   - True: ultra-low mode ~3MB (for <4GB RAM)
    
    Returns:
        dict of averaged results across n_reps replications
    """
    n, p, k, rho, b, sigma, lam_factor = combo
    
    if low_memory:
        # Ultra-low memory: ~2-3MB per worker
        # Use this for laptops with <4GB RAM
        return run_simulation_ultralow_memory(
            n, p, k, rho, b, sigma, lam_factor, 
            n_reps=n_reps, 
            seed=seed
        )
    else:
        # Optimal batch mode: ~240MB per worker with batch_size=100
        # Use this for laptops with 8GB+ RAM
        return run_simulation(
            n, p, k, rho, b, sigma, lam_factor, 
            n_reps=n_reps, 
            seed=seed,
            batch_size=batch_size
        )