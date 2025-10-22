import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/raw/large_experiment_parallel.csv")


plt.figure(figsize=(6,4))
for rho in df['rho'].unique():
    subset = df[df['rho'] == rho].sort_values(by='theta')
    plt.plot(subset['theta'], subset['exact_support_recovery_rate'], 'o-', label=f'rho={rho}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Exact Support Recovery Probability')
plt.legend()
plt.title('Exact Support Recovery Rate Phase Transition By Rho(Wainwright 2009)')
plt.tight_layout()
plt.savefig("results/figures/exact_phase_transition_byrho_wainwright2009.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,4))
for rho in df['rho'].unique():
    subset = df[df['rho'] == rho].sort_values(by='theta')
    plt.plot(subset['theta'], subset['unsigned_support_recovery_rate'], 'o-', label=f'rho={rho}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Support Recovery Probability')
plt.legend()
plt.title('Support Recovery Rate Phase Transition By Rho(Wainwright 2009)')
plt.tight_layout()
plt.savefig("results/figures/unsigned_phase_transition_byrho_wainwright2009.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,4))
for b in df['beta_min'].unique():
    subset = df[df['beta_min'] == b].sort_values(by='theta')
    plt.plot(subset['theta'], subset['exact_support_recovery_rate'], 'o-', label=f'beta_min={b}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Exact Support Recovery Probability')
plt.legend()
plt.title('Exact Support Recovery Rate Phase Transition By Beta_min(Wainwright 2009)')
plt.tight_layout()
plt.savefig("results/figures/exact_phase_transition_bybeta_min_wainwright2009.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,4))
for b in df['beta_min'].unique():
    subset = df[df['beta_min'] == b].sort_values(by='theta')
    plt.plot(subset['theta'], subset['unsigned_support_recovery_rate'], 'o-', label=f'beta_min={b}')
plt.xlabel(r'$\theta = \frac{n b^2}{\sigma^2 \log(p - k)}$')
plt.ylabel('Unsigned Support Recovery Probability')
plt.legend()
plt.title('Unsigned Support Recovery Rate Phase Transition By Beta_min(Wainwright 2009)')
plt.tight_layout()
plt.savefig("results/figures/unsigned_phase_transition_bybeta_min_wainwright2009.png", bbox_inches='tight')
plt.show()