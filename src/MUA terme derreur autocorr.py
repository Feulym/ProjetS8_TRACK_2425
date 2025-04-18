import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import pandas as pd

# --- MUA Functions ---
def simulate_MUA(N, T, sigma2, sigma_b2):
    A = np.array([
        [1, T, T**2 / 2],
        [0, 1, T],
        [0, 0, 1]
    ])
    Q = sigma2 * np.array([
        [T**5 / 20, T**4 / 8, T**3 / 6],
        [T**4 / 8, T**3 / 3, T**2 / 2],
        [T**3 / 6, T**2 / 2, T]
    ])
    L = np.linalg.cholesky(Q)
    X = np.zeros((3, N))
    for k in range(1, N):
        w = L @ randn(3)
        X[:, k] = A @ X[:, k-1] + w
    Y = X[0] + np.sqrt(sigma_b2) * randn(N)
    return Y, X

def compute_jerk(X, T):
    return (X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]) / T**3

def autocorrelation(signal):
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N + 1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:]

def compute_std_normalized_MUA(N, T, sigma2, sigma_b2=0, nb_realizations=1000, with_noise=False):
    Rjj0_values = []
    Rjj1_values = []
    for _ in range(nb_realizations):
        Y, X = simulate_MUA(N, T, sigma2, sigma_b2 if with_noise else 0)
        jerk = compute_jerk(Y, T)
        Rjj = autocorrelation(jerk)
        Rjj0_values.append(Rjj[0])
        Rjj1_values.append(Rjj[1])
    Rjj0_values = np.array(Rjj0_values)
    Rjj1_values = np.array(Rjj1_values)
    return (
        np.std(Rjj0_values), np.mean(Rjj0_values), np.std(Rjj0_values) / np.mean(Rjj0_values),
        np.std(Rjj1_values), np.mean(Rjj1_values), np.std(Rjj1_values) / abs(np.mean(Rjj1_values))
    )

# --- Paramètres ---
T = 1
sigma2 = 4
sigma_b2 = 0.5
nb_realizations = 1000
N_values = [5000, 2000, 1000, 500, 250, 100, 50, 30]

results_nb_0 = []
results_nb_1 = []
results_b_0 = []
results_b_1 = []

for N in N_values:
    std0, mean0, norm0, std1, mean1, norm1 = compute_std_normalized_MUA(N, T, sigma2, with_noise=False, nb_realizations=nb_realizations)
    results_nb_0.append((N, std0, mean0, norm0))
    results_nb_1.append((N, std1, mean1, norm1))

    std0_b, mean0_b, norm0_b, std1_b, mean1_b, norm1_b = compute_std_normalized_MUA(N, T, sigma2, sigma_b2, with_noise=True, nb_realizations=nb_realizations)
    results_b_0.append((N, std0_b, mean0_b, norm0_b))
    results_b_1.append((N, std1_b, mean1_b, norm1_b))
    
# -----------------------
# AFFICHAGE CONSOLE
# -----------------------
print("\n====== DONNÉES NON BRUITÉES (MUA) ======")
print("N\tRjj(0)_std\tRjj(0)_mean\tRjj(0)_norm\tRjj(1)_std\tRjj(1)_mean\tRjj(1)_norm\tRatio")
for i in range(len(N_values)):
    n = N_values[i]
    r0 = results_nb_0[i]
    r1 = results_nb_1[i]
    ratio = r1[3] / r0[3]
    print(f"{n}\t{r0[1]:.4f}\t\t{r0[2]:.4f}\t\t{r0[3]:.4f}\t\t{r1[1]:.4f}\t\t{r1[2]:.4f}\t\t{r1[3]:.4f}\t\t{ratio:.4f}")

print("\n====== DONNÉES BRUITÉES (MUA) ======")
print("N\tRjj(0)_std\tRjj(0)_mean\tRjj(0)_norm\tRjj(1)_std\tRjj(1)_mean\tRjj(1)_norm\tRatio")
for i in range(len(N_values)):
    n = N_values[i]
    r0 = results_b_0[i]
    r1 = results_b_1[i]
    ratio = r1[3] / r0[3]
    print(f"{n}\t{r0[1]:.4f}\t\t{r0[2]:.4f}\t\t{r0[3]:.4f}\t\t{r1[1]:.4f}\t\t{r1[2]:.4f}\t\t{r1[3]:.4f}\t\t{ratio:.4f}")


# --- Dataframes ---
df_nb = pd.DataFrame({
    "N": N_values,
    "Rjj(0)_norm": [r[3] for r in results_nb_0],
    "Rjj(1)_norm": [r[3] for r in results_nb_1],
    "Ratio": [r[3]/r0[3] for r0, r in zip(results_nb_0, results_nb_1)]
})

df_b = pd.DataFrame({
    "N": N_values,
    "Rjj(0)_norm": [r[3] for r in results_b_0],
    "Rjj(1)_norm": [r[3] for r in results_b_1],
    "Ratio": [r[3]/r0[3] for r0, r in zip(results_b_0, results_b_1)]
})

# --- Plots ---
# Écarts-types normalisés — sans bruit
plt.figure(figsize=(8, 5))
plt.plot(df_nb["N"], df_nb["Rjj(0)_norm"], 'o-', label='$R_{jj}(0)$', color='blue')
plt.plot(df_nb["N"], df_nb["Rjj(1)_norm"], 'o-', label='$R_{jj}(1)$', color='red')
plt.xscale('log')
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Écart-type normalisé")
plt.title("Écart-type normalisé en fonction de N (MUA sans bruit)")
plt.grid(True)
plt.legend()
plt.show()

# Écarts-types normalisés — bruité
plt.figure(figsize=(8, 5))
plt.plot(df_b["N"], df_b["Rjj(0)_norm"], 'o-', label='$R_{jj}(0)$', color='blue')
plt.plot(df_b["N"], df_b["Rjj(1)_norm"], 'o-', label='$R_{jj}(1)$', color='red')
plt.xscale('log')
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Écart-type normalisé")
plt.title("Écart-type normalisé en fonction de N (MUA avec bruit $\sigma_b^2 = 0.5$)")
plt.grid(True)
plt.legend()
plt.show()

# Rapport des écarts-types normalisés
mean_ratio_nb = df_nb["Ratio"].mean()
mean_ratio_b = df_b["Ratio"].mean()

plt.figure(figsize=(10, 6))
plt.plot(df_nb["N"], df_nb["Ratio"], 'o-', label='Non bruité', color='purple')
plt.plot(df_b["N"], df_b["Ratio"], 's--', label='Avec bruit', color='orange')
plt.axhline(mean_ratio_nb, color='purple', linestyle='--', alpha=0.5, label=f'Moyenne non bruitée ≈ {mean_ratio_nb:.4f}')
plt.axhline(mean_ratio_b, color='orange', linestyle='--', alpha=0.5, label=f'Moyenne bruitée ≈ {mean_ratio_b:.4f}')
plt.xscale('log')
plt.ylim(0, 2 * max(mean_ratio_nb, mean_ratio_b))
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Rapport $\\frac{R_{jj}(1)_{norm}}{R_{jj}(0)_{norm}}$")
plt.title("Rapport des écarts-types normalisés (MUA)")
plt.grid(True)
plt.legend()
plt.show()
