import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def trajec_MRU(N, T, sigma2):
    A = np.array([[1, T], [0, 1]])
    Q = sigma2 * np.array([[T**3/3, T**2/2], [T**2/2, T]])
    D = np.linalg.cholesky(Q)
    X = np.zeros((2, N))
    for k in range(N-1):
        w = D @ randn(2, 1)
        X[:, k+1] = A @ X[:, k] + w[:, 0]
    return X

def bruitage_distance(x, distance):
    bsigma = distance / 3
    bruit = bsigma * np.random.randn(*x.shape)
    return x + bruit, bsigma**2

def compute_acceleration(X, T):
    return (X[2:] - 2*X[1:-1] + X[:-2]) / T**2

def autocorrelation(signal):
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N + 1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:], lags[N-1:]

def compute_std_normalized(N, T, sigma2, nb_realizations=1000):
    Raa0_values = []
    Raa1_values = []
    for _ in range(nb_realizations):
        X = trajec_MRU(N, T, sigma2)
        a = compute_acceleration(X[0], T)
        Raa, _ = autocorrelation(a)
        Raa0_values.append(Raa[0])
        Raa1_values.append(Raa[1])
    Raa0_values = np.array(Raa0_values)
    Raa1_values = np.array(Raa1_values)
    return (
        np.std(Raa0_values), np.mean(Raa0_values), np.std(Raa0_values) / np.mean(Raa0_values),
        np.std(Raa1_values), np.mean(Raa1_values), np.std(Raa1_values) / abs(np.mean(Raa1_values))
    )

def compute_std_normalized_with_noise(N, T, sigma2, sigma_b2, nb_realizations=1000):
    Raa0_values = []
    Raa1_values = []
    for _ in range(nb_realizations):
        X = trajec_MRU(N, T, sigma2)
        Y, _ = bruitage_distance(X[0], distance=3*np.sqrt(sigma_b2))
        a = compute_acceleration(Y, T)
        Raa, _ = autocorrelation(a)
        Raa0_values.append(Raa[0])
        Raa1_values.append(Raa[1])
    Raa0_values = np.array(Raa0_values)
    Raa1_values = np.array(Raa1_values)
    return (
        np.std(Raa0_values), np.mean(Raa0_values), np.std(Raa0_values) / np.mean(Raa0_values),
        np.std(Raa1_values), np.mean(Raa1_values), np.std(Raa1_values) / abs(np.mean(Raa1_values))
    )

# ---------------------
# PARAMÈTRES
# ---------------------
T = 1
speed = 7.7
taux = 0.01
sigma = (speed * taux) / 3
sigma2 = sigma**2
sigma_b2 = 0.5
nb_realizations = 10
N_values = [5000, 2000, 1000, 500, 250, 100, 50, 30]

results_nb_0 = []
results_nb_1 = []
results_b_0 = []
results_b_1 = []

for N in N_values:
    std0, mean0, norm0, std1, mean1, norm1 = compute_std_normalized(N, T, sigma2, nb_realizations)
    results_nb_0.append((N, std0, mean0, norm0))
    results_nb_1.append((N, std1, mean1, norm1))

    std0_b, mean0_b, norm0_b, std1_b, mean1_b, norm1_b = compute_std_normalized_with_noise(N, T, sigma2, sigma_b2, nb_realizations)
    results_b_0.append((N, std0_b, mean0_b, norm0_b))
    results_b_1.append((N, std1_b, mean1_b, norm1_b))
    
# -----------------------
# AFFICHAGE CONSOLE
# -----------------------
print("\n====== DONNÉES NON BRUITÉES ======")
print("N\tRaa(0)_std\tRaa(0)_mean\tRaa(0)_norm\tRaa(1)_std\tRaa(1)_mean\tRaa(1)_norm\tRatio")
for i in range(len(N_values)):
    n = N_values[i]
    r0 = results_nb_0[i]
    r1 = results_nb_1[i]
    ratio = r1[3] / r0[3]
    print(f"{n}\t{r0[1]:.4f}\t\t{r0[2]:.4f}\t\t{r0[3]:.4f}\t\t{r1[1]:.4f}\t\t{r1[2]:.4f}\t\t{r1[3]:.4f}\t\t{ratio:.4f}")

print("\n====== DONNÉES BRUITÉES ======")
print("N\tRaa(0)_std\tRaa(0)_mean\tRaa(0)_norm\tRaa(1)_std\tRaa(1)_mean\tRaa(1)_norm\tRatio")
for i in range(len(N_values)):
    n = N_values[i]
    r0 = results_b_0[i]
    r1 = results_b_1[i]
    ratio = r1[3] / r0[3]
    print(f"{n}\t{r0[1]:.4f}\t\t{r0[2]:.4f}\t\t{r0[3]:.4f}\t\t{r1[1]:.4f}\t\t{r1[2]:.4f}\t\t{r1[3]:.4f}\t\t{ratio:.4f}")


# -----------------------
# PLOT DES ÉCARTS-TYPES NORMALISÉS — NON BRUITÉ
# -----------------------
Ns = [r[0] for r in results_nb_0]
norm_std_nb_0 = [r[3] for r in results_nb_0]
norm_std_nb_1 = [r[3] for r in results_nb_1]

plt.figure(figsize=(8, 5))
plt.plot(Ns, norm_std_nb_0, 'o-', color='blue', label='$R_{aa}(0)$')
plt.plot(Ns, norm_std_nb_1, 'o-', color='red', label='$R_{aa}(1)$')
plt.xscale('log')
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Écart-type normalisé")
plt.title("Écart-type normalisé en fonction de N (MRU sans bruit)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------
# PLOT DES ÉCARTS-TYPES NORMALISÉS — BRUITÉ
# -----------------------
norm_std_b_0 = [r[3] for r in results_b_0]
norm_std_b_1 = [r[3] for r in results_b_1]

plt.figure(figsize=(8, 5))
plt.plot(Ns, norm_std_b_0, 'o-', color='blue', label='$R_{aa}(0)$')
plt.plot(Ns, norm_std_b_1, 'o-', color='red', label='$R_{aa}(1)$')
plt.xscale('log')
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Écart-type normalisé")
plt.title("Écart-type normalisé en fonction de N (MRU avec bruit $\sigma_b^2 = 0.5$)")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------
# PLOT DES RAPPORTS Raa(1)/Raa(0)
# -----------------------
ratios_nb = [r1[3] / r0[3] for r0, r1 in zip(results_nb_0, results_nb_1)]
ratios_b = [r1[3] / r0[3] for r0, r1 in zip(results_b_0, results_b_1)]
mean_nb = np.mean(ratios_nb)
mean_b = np.mean(ratios_b)

plt.figure(figsize=(10, 6))
plt.plot(Ns, ratios_nb, 'o-', label='Non bruité', color='purple')
plt.plot(Ns, ratios_b, 's--', label='Avec bruit', color='orange')
plt.axhline(mean_nb, color='purple', linestyle='--', alpha=0.5, label=f'Moyenne non bruitée ≈ {mean_nb:.4f}')
plt.axhline(mean_b, color='orange', linestyle='--', alpha=0.5, label=f'Moyenne bruitée ≈ {mean_b:.4f}')
plt.xscale('log')
plt.ylim(0, 2 * max(mean_nb, mean_b))
plt.xlabel("Nombre d'échantillons (N)")
plt.ylabel("Rapport $\\frac{R_{aa}(1)_{norm}}{R_{aa}(0)_{norm}}$")
plt.title("Rapport des écarts-types normalisés (MRU)")
plt.grid(True)
plt.legend()
plt.show()
