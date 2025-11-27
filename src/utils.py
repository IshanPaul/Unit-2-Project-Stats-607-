# src/sim_helpers.py
import numpy as np
from src.sim_runner import run_simulation

def run_batch(batch):
    """Run multiple simulations sequentially inside a single process."""
    results = []
    for combo in batch:
        seed = np.random.randint(0, 1_000_000)
        results.append(run_simulation(*combo, seed=seed))
    return results
