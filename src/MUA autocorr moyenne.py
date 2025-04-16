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
        (11 * sigma2) / (20 * T) + (20 * sigma_b2) / (T**6),
        (13 * sigma2) / (60 * T) - (15 * sigma_b2) / (T**6),
        (sigma2) / (120 * T) + (6 * sigma_b2) / (T**6),
        - (sigma_b2) / (T**6)
    ]
    return R_theo

def collect_autocorrs_MUA(N, T, sigma2, sigma_b2, nb_realizations=100, with_noise=False):
    autocorrs = []
    for _ in range(nb_realizations):
        sigma_b2_eff = sigma_b2 if with_noise else 0
        Y, X = simulate_MUA(N, T, sigma2, sigma_b2_eff)
        jerk = compute_jerk(Y, T)
        R_exp, _ = autocorrelation(jerk)
        autocorrs.append(R_exp)
    min_len = min(len(r) for r in autocorrs)
    autocorrs = np.array([r[:min_len] for r in autocorrs])
    return autocorrs

def plot_mean_with_variance_MUA(autocorrs, T, sigma2, sigma_b2, title_suffix, zoom=False):
    R_mean = np.mean(autocorrs, axis=0)
    R_std = np.std(autocorrs, axis=0)
    R_theo_vals = theoretical_autocorr_values(T, sigma2, sigma_b2)
    R_theo_full = np.zeros_like(R_mean)
    R_theo_full[0] = R_theo_vals[0]
    R_theo_full[1] = R_theo_vals[1]
    R_theo_full[2] = R_theo_vals[2]
    R_theo_full[3] = R_theo_vals[3]

    if zoom:
        zoom_range = 10
        x = range(zoom_range + 1)
        R_mean = R_mean[:zoom_range+1]
        R_std = R_std[:zoom_range+1]
        R_theo_full = R_theo_full[:zoom_range+1]
    else:
        x = range(len(R_mean))

    plt.figure(figsize=(10, 5))
    plt.fill_between(x, R_mean - 3*R_std, R_mean + 3*R_std, color='green', alpha=0.1, label='±3σ')
    plt.fill_between(x, R_mean - 2*R_std, R_mean + 2*R_std, color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(x, R_mean - R_std, R_mean + R_std, color='blue', alpha=0.3, label='±1σ')
    plt.plot(x, R_mean, label='Expérimental (moyenne)', color='blue')
    plt.plot(x, R_theo_full, 'r--', label='Théorique')
    plt.title(f"Autocorrélation de l'accélération estimée pour un MUA {title_suffix} ({'zoomée' if zoom else 'complète'})")
    plt.xlabel('Décalage (lag)')
    plt.ylabel('R_jj')
    plt.legend()
    plt.grid()
    plt.show()

# -------------------------
# PARAMÈTRES ET EXÉCUTION
# -------------------------
N = 50
T = 1
sigma2 = 4
nb_realizations = 1000

# ----- SANS BRUIT -----
sigma_b2_no = 0
autocorrs_mua_no_noise = collect_autocorrs_MUA(N, T, sigma2, sigma_b2_no, nb_realizations, with_noise=False)
plot_mean_with_variance_MUA(autocorrs_mua_no_noise, T, sigma2, sigma_b2_no, title_suffix='sans bruit', zoom=False)
plot_mean_with_variance_MUA(autocorrs_mua_no_noise, T, sigma2, sigma_b2_no, title_suffix='sans bruit', zoom=True)

# ----- AVEC BRUIT -----
sigma_b2 = 10
autocorrs_mua_with_noise = collect_autocorrs_MUA(N, T, sigma2, sigma_b2, nb_realizations, with_noise=True)
plot_mean_with_variance_MUA(autocorrs_mua_with_noise, T, sigma2, sigma_b2, title_suffix='avec bruit additif σ²_b = 0.5', zoom=False)
plot_mean_with_variance_MUA(autocorrs_mua_with_noise, T, sigma2, sigma_b2, title_suffix='avec bruit additif σ²_b = 0.5', zoom=True)
