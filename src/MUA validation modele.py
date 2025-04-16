import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def simulate_MUA(N, T, sigma2, sigma_b2):
    """Simule une trajectoire MUA bruitée."""
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
    """Estime le jerk à partir des positions."""
    return (X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]) / T**3

def autocorrelation(signal):
    """Calcule la fonction d'autocorrélation non biaisée."""
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N + 1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:]

def estimate_variances(Y, T, sigma2, sigma_b2):
    """Estime sigma^2 et sigma_b^2 à partir des autocorrélations et de la matrice M (que l'on inverse)."""
    jerk = compute_jerk(Y, T)
    R_j = autocorrelation(jerk)
    R_jyjy = np.array([R_j[0], R_j[1]])
    
    M = np.array([
        [11 / (20 * T), 20 / T**6],
        [13 / (60 * T), - 15 / T**6]
    ])
    
    M_inv = np.linalg.inv(M)
    variances = M_inv @ R_jyjy
    return variances, R_j[0], R_j[1]

def compute_precision(estimated_values, true_value):
    mean_estimation = np.mean(estimated_values)
    if true_value == 0:
        precision = 0  # Évite la division par zéro
    else:
        precision = abs((mean_estimation - true_value) / true_value) * 100
    return mean_estimation, precision

N = 50
T = 1
sigma2 = 4
sigma_b2 = 0
iterations = 1000

sigma2_estimates = []
sigma_b2_estimates = []
R_j0_estimates = []
R_j1_estimates = []

for _ in range(iterations):
    Y, X = simulate_MUA(N, T, sigma2, sigma_b2)
    estimated_variances, R_j0, R_j1 = estimate_variances(Y, T, sigma2, sigma_b2)
    sigma2_estimates.append(estimated_variances[0])
    sigma_b2_estimates.append(estimated_variances[1])
    R_j0_estimates.append(R_j0)
    R_j1_estimates.append(R_j1)

mean_sigma2, precision_sigma2 = compute_precision(sigma2_estimates, sigma2)
mean_Rj0, precision_Rj0 = compute_precision(R_j0_estimates, (11 * sigma2) / (20 * T))
mean_Rj1, precision_Rj1 = compute_precision(R_j1_estimates, (13 * sigma2) / (60 * T))

print(f"\nMoyenne des Variances estimées après {iterations} simulations :")
print(f"Sigma^2 estimé en moyenne : {mean_sigma2} avec une précision de {precision_sigma2:.5f}%")
print(f"Sigma^2 réel : {sigma2}")
print(f"R_j(0) estimé en moyenne : {mean_Rj0} avec une précision de {precision_Rj0:.5f}%")
print(f"Rjyjy(0) theorique : {(11 * sigma2) / (20 * T) + (20 * sigma_b2) / (T**6)}")
print(f"R_j(1) estimé en moyenne : {mean_Rj1} avec une précision de {precision_Rj1:.5f}%")
print(f"Rjyjy(1) theorique : {(13 * sigma2) / (60 * T) - (15 * sigma_b2) / (T**6)}")

plt.figure(figsize=(10, 6))
plt.hist(R_j0_estimates, bins=30, alpha=0.7, label='R_j(0) estimé')
plt.axvline((11 * sigma2) / (20 * T), color='red', linestyle='dashed', linewidth=2, label='R_j(0) théorique')
plt.title("Distribution des estimations de R_j(0)")
plt.xlabel("Valeur estimée de R_j(0)")
plt.ylabel("Fréquence")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(R_j1_estimates, bins=30, alpha=0.7, label='R_j(1) estimé')
plt.axvline((13 * sigma2) / (60 * T), color='red', linestyle='dashed', linewidth=2, label='R_j(1) théorique')
plt.title("Distribution des estimations de R_j(1)")
plt.xlabel("Valeur estimée de R_j(1)")
plt.ylabel("Fréquence")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(sigma2_estimates, bins=30, alpha=0.7, label='Sigma^2 estimé')
plt.axvline(sigma2, color='red', linestyle='dashed', linewidth=2, label='Sigma^2 réel')
plt.title("Distribution des estimations de Sigma^2")
plt.xlabel("Valeur estimée de Sigma^2")
plt.ylabel("Fréquence")
plt.legend()
plt.grid()
plt.show()



"""
# Simulation de la trajectoire MUA
Y, X = simulate_MUA(N, T, sigma2, sigma_b2)

# Estimation des variances
estimated_variances = estimate_variances(Y, T)

print(f"\nVariances estimées :")
print(f"Sigma^2 estimé : {estimated_variances[0]}")
print("Sigma^2 :", sigma2)

print(f"Sigma_b^2 estimé : {estimated_variances[1]}")
print("Sigma_b^2 :", sigma_b2)

##############################
# VISUALISATION
##############################
jerk = compute_jerk(Y, T)
R_j = autocorrelation(jerk)

plt.figure(figsize=(10, 6))
plt.plot(R_j, label='Autocorrélation du jerk')
plt.title("Autocorrélation du jerk pour une trajectoire MUA")
plt.xlabel("Décalage temporel")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()
"""