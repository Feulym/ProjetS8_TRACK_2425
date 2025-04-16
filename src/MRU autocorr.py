import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def trajec_MRU(N, T, sigma2):
    t = np.arange(0, N*T, T)
    A = np.array([[1, T], [0, 1]])
    Q = sigma2 * np.array([[T**3/3, T**2/2], [T**2/2, T]])
    D = np.linalg.cholesky(Q)
    X = np.zeros((2, N))
    for k in range(N-1):
        w = D @ randn(2, 1)
        X[:, k+1] = A @ X[:, k] + w[:, 0]
    return t, X

def compute_acceleration(X, T):
    return (X[2:] - 2*X[1:-1] + X[:-2]) / T**2

def autocorrelation(signal):
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N + 1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:], lags[N-1:]

def bruitage_distance(x, distance):
    bsigma = distance / 3
    bruit = bsigma * np.random.randn(*x.shape)
    x_bruite = x + bruit
    return x_bruite, bsigma**2

def theoretical_autocorr_MRU(T, sigma2, sigma_b2):
    R_theo = [
        (2 * sigma2) / (3 * T) + (6 * sigma_b2) / (T**4),   # lag 0
        (sigma2) / (6 * T) - (4 * sigma_b2) / (T**4),       # lag ±1
        sigma_b2 / (T**4)                                   # lag ±2
    ]
    return R_theo

def plot_autocorr_MRU(Y, T, sigma2, sigma_b2, title_suffix, zoom=False):
    a = compute_acceleration(Y, T)
    R_exp, lags = autocorrelation(a)

    # Autocorrélation théorique non nulle pour lags 0 à ±2
    R_theo_vals = theoretical_autocorr_MRU(T, sigma2, sigma_b2)
    R_theo_full = np.zeros_like(R_exp)
    R_theo_full[0] = R_theo_vals[0]
    R_theo_full[1] = R_theo_vals[1]
    R_theo_full[2] = R_theo_vals[2]

    if zoom:
        zoom_range = 10
        plt.figure(figsize=(10, 5))
        plt.plot(range(zoom_range+1), R_exp[:zoom_range+1], 'o-', label='Expérimental')
        plt.plot(range(zoom_range+1), R_theo_full[:zoom_range+1], 's--', color='red', label='Théorique')
        plt.title(f'Autocorrélation MRU {title_suffix} (zoomée)')
        plt.xlabel('Décalage (lag)')
        plt.ylabel('R_aa')
        plt.legend()
        plt.grid()
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(R_exp, label='Expérimental')
        plt.plot(R_theo_full, '--', color='red', label='Théorique (|τ| ≤ 2)')
        plt.title(f'Autocorrélation MRU {title_suffix} (complète)')
        plt.xlabel('Décalage (lag)')
        plt.ylabel('R_aa')
        plt.legend()
        plt.grid()
        plt.show()

# ---------------------
# PARAMÈTRES
# ---------------------
N = 5000
T = 1
speed = 7.7
taux = 0.01
distance = 0  # pas de bruit initialement
sigma = (speed * taux) / 3
sigma2 = sigma**2

# ------ SANS BRUIT ------
t, X = trajec_MRU(N, T, sigma2)
Y_no_noise, bsigma2_no = bruitage_distance(X[0], distance)
plot_autocorr_MRU(Y_no_noise, T, sigma2, bsigma2_no, title_suffix='sans bruit', zoom=False)
plot_autocorr_MRU(Y_no_noise, T, sigma2, bsigma2_no, title_suffix='sans bruit', zoom=True)

# ------ AVEC BRUIT σ²_b = 0.5 ------
bsigma2_target = 0.5
distance_bruit = 3 * np.sqrt(bsigma2_target)
Y_bruite, bsigma2 = bruitage_distance(X[0], distance_bruit)
plot_autocorr_MRU(Y_bruite, T, sigma2, bsigma2, title_suffix='avec bruit σ²_b = 0.5', zoom=False)
plot_autocorr_MRU(Y_bruite, T, sigma2, bsigma2, title_suffix='avec bruit σ²_b = 0.5', zoom=True)


