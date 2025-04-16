import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

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
    return unbiased_corr[N-1:], lags[N-1:]

def theoretical_autocorr_values(T, sigma2, sigma_b2):
    R_theo = [
        (11 * sigma2) / (20 * T) + (20 * sigma_b2) / (T**6),   # lag 0
        (13 * sigma2) / (60 * T) - (15 * sigma_b2) / (T**6),   # lag ±1
        (sigma2) / (120 * T) + (6 * sigma_b2) / (T**6),        # lag ±2
        - (sigma_b2) / (T**6)                                  # lag ±3
    ]
    return R_theo

def plot_autocorr(Y, T, sigma2, sigma_b2, title_suffix, zoom=False):
    jerk = compute_jerk(Y, T)
    R_exp, lags = autocorrelation(jerk)
    R_theo_vals = theoretical_autocorr_values(T, sigma2, sigma_b2)

    # Theoretical autocorr full (non zero only at 0,1,2,3)
    R_theo_full = np.zeros_like(R_exp)
    R_theo_full[0] = R_theo_vals[0]     # lag 0
    R_theo_full[1] = R_theo_vals[1]     # lag 1
    R_theo_full[2] = R_theo_vals[2]     # lag 2
    R_theo_full[3] = R_theo_vals[3]     # lag 3

    if zoom:
        zoom_range = 10
        plt.figure(figsize=(10, 5))
        plt.plot(range(zoom_range+1), R_exp[:zoom_range+1], 'o-', label='Expérimental')
        plt.plot(range(zoom_range+1), R_theo_full[:zoom_range+1], 's--', label='Théorique', color='red')
        plt.title(f'Autocorrélation {title_suffix} (zoomée)')
        plt.xlabel('Décalage (lag)')
        plt.ylabel('R_jj')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(R_exp, label='Expérimental')
        plt.plot(R_theo_full, '--', color='red', label='Théorique (|τ| ≤ 3)')
        plt.title(f'Autocorrélation {title_suffix} (complète)')
        plt.xlabel('Décalage (lag)')
        plt.ylabel('R_jj')
        plt.legend()
        plt.grid()
        plt.show()

# --------------------
# PARAMÈTRES
# --------------------
N = 5000
T = 1
sigma2 = 4

# ----- SANS BRUIT -----
sigma_b2 = 0
Y, X = simulate_MUA(N, T, sigma2, sigma_b2)
plot_autocorr(Y, T, sigma2, sigma_b2, title_suffix='sans bruit', zoom=False)
plot_autocorr(Y, T, sigma2, sigma_b2, title_suffix='sans bruit', zoom=True)

# ----- AVEC BRUIT -----
sigma_b2 = 0.5
Y, X = simulate_MUA(N, T, sigma2, sigma_b2)
plot_autocorr(Y, T, sigma2, sigma_b2, title_suffix='avec bruit σ²_b = 0.5', zoom=False)
plot_autocorr(Y, T, sigma2, sigma_b2, title_suffix='avec bruit σ²_b = 0.5', zoom=True)