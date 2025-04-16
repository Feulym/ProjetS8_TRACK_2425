import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def simulate_MRU(N, T, sigma2, sigma_b2):
    """Simule une trajectoire MRU bruitée."""
    A = np.array([
        [1, T],
        [0, 1]
    ])
    
    Q = sigma2 * np.array([
        [T**3/3, T**2/2],
        [T**2/2, T]
    ])
    
    L = np.linalg.cholesky(Q)
    X = np.zeros((2, N))

    for k in range(1, N):
        w = L @ randn(2)
        X[:, k] = A @ X[:, k-1] + w

    Y = X[0] + np.sqrt(sigma_b2) * randn(N)
    return Y, X

def compute_acceleration(X, T):
    """Estime l'accélération à partir des positions."""
    return (X[2:] - 2*X[1:-1] + X[:-2]) / T**2

def autocorrelation(signal):
    """Calcule la fonction d'autocorrélation non biaisée."""
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N + 1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:]

def estimate_variances(Y, T, sigma2, sigma_b2):
    """Estime sigma^2 et sigma_b^2 à partir des autocorrélations et de la matrice M (que l'on inverse)."""
    acc = compute_acceleration(Y, T)
    R_a = autocorrelation(acc)
    R_aa = np.array([R_a[0], R_a[1]])
    
    M = np.array([
        [2/(3*T), 6/(T**4)],
        [1/(6*T), -4/(T**4)]
    ])
    
    M_inv = np.linalg.inv(M)
    variances = M_inv @ R_aa
    return variances, R_a[0], R_a[1]

def compute_precision(estimated_values, true_value):
    mean_estimation = np.mean(estimated_values)
    if true_value == 0:
        precision = 0  # Évite la division par zéro
    else:
        precision = abs((mean_estimation - true_value) / true_value) * 100
    return mean_estimation, precision

N = 1500
T = 1
sigma2 = 4
sigma_b2 = 0
iterations = 500

sigma2_estimates = []
sigma_b2_estimates = []
R_a0_estimates = []
R_a1_estimates = []

for _ in range(iterations):
    Y, X = simulate_MRU(N, T, sigma2, sigma_b2)
    estimated_variances, R_a0, R_a1 = estimate_variances(Y, T, sigma2, sigma_b2)
    sigma2_estimates.append(estimated_variances[0])
    sigma_b2_estimates.append(estimated_variances[1])
    R_a0_estimates.append(R_a0)
    R_a1_estimates.append(R_a1)

mean_sigma2, precision_sigma2 = compute_precision(sigma2_estimates, sigma2)
mean_Ra0, precision_Ra0 = compute_precision(R_a0_estimates, 2*sigma2/(3*T) + 6*sigma_b2/T**4)
mean_Ra1, precision_Ra1 = compute_precision(R_a1_estimates, sigma2/(6*T) - 4*sigma_b2/T**4)

print(f"\nMoyenne des Variances estimées après {iterations} simulations :")
print(f"Sigma^2 estimé en moyenne : {mean_sigma2} avec une précision de {precision_sigma2:.5f}%")
print(f"Sigma^2 réel : {sigma2}")
print(f"R_a(0) estimé en moyenne : {mean_Ra0} avec une précision de {precision_Ra0:.5f}%")
print(f"R_a(0) théorique : {2*sigma2/(3*T) + 6*sigma_b2/T**4}")
print(f"R_a(1) estimé en moyenne : {mean_Ra1} avec une précision de {precision_Ra1:.5f}%")
print(f"R_a(1) théorique : {sigma2/(6*T) - 4*sigma_b2/T**4}")

plt.figure(figsize=(10, 6))
plt.hist(sigma2_estimates, bins=30, alpha=0.7, label='Sigma^2 estimé')
plt.axvline(sigma2, color='red', linestyle='dashed', linewidth=2, label='Sigma^2 réel')
plt.title("Distribution des estimations de Sigma^2")
plt.xlabel("Valeur estimée de Sigma^2")
plt.ylabel("Nombre d'occurences")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(R_a0_estimates, bins=30, alpha=0.7, label='R_a(0) estimé')
plt.axvline(2*sigma2/(3*T) + 6*sigma_b2/T**4, color='red', linestyle='dashed', linewidth=2, label='R_a(0) théorique')
plt.title("Distribution des estimations de R_a(0)")
plt.xlabel("Valeur estimée de R_a(0)")
plt.ylabel("Nombre d'occurences")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(R_a1_estimates, bins=30, alpha=0.7, label='R_a(1) estimé')
plt.axvline(sigma2/(6*T) - 4*sigma_b2/T**4, color='red', linestyle='dashed', linewidth=2, label='R_a(1) théorique')
plt.title("Distribution des estimations de R_a(1)")
plt.xlabel("Valeur estimée de R_a(1)")
plt.ylabel("Nombre d'occurences")
plt.legend()
plt.grid()
plt.show()