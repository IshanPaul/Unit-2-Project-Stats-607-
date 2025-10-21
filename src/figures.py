import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/raw/large_experiment_parallel.csv")


plt.figure(figsize=(6,4))
for rho in df['rho'].unique():
    subset = df[df['rho'] == rho]
    plt.plot(subset['theta'], subset['exact_support_recovery_rate'], 'o-', label=f'rho={rho}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Exact Support Recovery Probability')
plt.legend()
plt.title('Phase Transition (Wainwright 2009)')
plt.tight_layout()
plt.show()
plt.savefig("results/figures/phase_transition_wainwright2009.png", bbox_inches='tight')

plt.figure(figsize=(6,4))
for rho in df['rho'].unique():
    subset = df[df['rho'] == rho]
    plt.plot(subset['theta'], subset['unsigned_support_recovery_rate'], 'o-', label=f'rho={rho}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Unsigned Support Recovery Probability')
plt.legend()
plt.title('Phase Transition (Wainwright 2009)')
plt.tight_layout()
plt.show()
plt.savefig("results/figures/unsigned_phase_transition_wainwright2009.png", bbox_inches='tight')