import pandas as pd
import numpy as np

df = pd.read_csv("results/raw/large_experiment_parallel.csv")

# Loop over rho values
for rho in df['rho'].unique():
    subset = df[df['rho'] == rho].sort_values(by='theta')
    print(f'\nRho = {rho}\n')
    print(subset[['theta', 'exact_support_recovery_rate']])

# Loop over beta_min values
for b in df['beta_min'].unique():
    subset = df[df['beta_min'] == b].sort_values(by='theta')
    print(f'\nSignal Strength b = {b}\n')
    print(subset[['theta', 'exact_support_recovery_rate']])
