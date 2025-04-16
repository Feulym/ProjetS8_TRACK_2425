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

def collect_autocorrs(N, T, sigma2, sigma_b2, nb_realizations=100, with_noise=False):
    autocorrs = []

    for _ in range(nb_realizations):
        t, X = trajec_MRU(N, T, sigma2)

        if with_noise:
            distance = 3 * np.sqrt(sigma_b2)
            Y, _ = bruitage_distance(X[0], distance)
        else:
            Y = X[0]

        a = compute_acceleration(Y, T)
        R_exp, _ = autocorrelation(a)
        autocorrs.append(R_exp)

    min_len = min(len(r) for r in autocorrs)
    autocorrs = np.array([r[:min_len] for r in autocorrs])
    return autocorrs

def plot_mean_with_variance(autocorrs, T, sigma2, sigma_b2, title_suffix, zoom=False):
    R_mean = np.mean(autocorrs, axis=0)
    R_std = np.std(autocorrs, axis=0)

    R_theo_vals = theoretical_autocorr_MRU(T, sigma2, sigma_b2)
    R_theo_full = np.zeros_like(R_mean)
    R_theo_full[0] = R_theo_vals[0]
    R_theo_full[1] = R_theo_vals[1]
    R_theo_full[2] = R_theo_vals[2]

    if zoom:
        zoom_range = 10
        x = range(zoom_range + 1)
        R_mean = R_mean[:zoom_range+1]
        R_std = R_std[:zoom_range+1]
        R_theo_full = R_theo_full[:zoom_range+1]
    else:
        x = range(len(R_mean))

    plt.figure(figsize=(10, 5))
    
    # Zones d'écart-type
    plt.fill_between(x, R_mean - 3*R_std, R_mean + 3*R_std, color='green', alpha=0.1, label='±3σ')
    plt.fill_between(x, R_mean - 2*R_std, R_mean + 2*R_std, color='orange', alpha=0.2, label='±2σ')
    plt.fill_between(x, R_mean - R_std,   R_mean + R_std,   color='blue', alpha=0.3, label='±1σ')

    # Moyenne expérimentale
    plt.plot(x, R_mean, label='Expérimental (moyenne)', color='blue')

    # Théorique
    plt.plot(x, R_theo_full, 'r--', label='Théorique')

    plt.title(f'Autocorrélation de l\'accélération estimée pour un MRU {title_suffix} ({"zoomée" if zoom else "complète"})')
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
nb_realizations = 10

distance = 0
sigma = (speed * taux) / 3
sigma2 = sigma**2

# ------ SANS BRUIT ------
Y_dummy, bsigma2_no = bruitage_distance(np.zeros(N), distance)
autocorrs_no_noise = collect_autocorrs(N, T, sigma2, bsigma2_no, nb_realizations, with_noise=False)
plot_mean_with_variance(autocorrs_no_noise, T, sigma2, bsigma2_no, title_suffix='sans bruit', zoom=False)
plot_mean_with_variance(autocorrs_no_noise, T, sigma2, bsigma2_no, title_suffix='sans bruit', zoom=True)

# ------ AVEC BRUIT σ²_b = 0.5 ------
bsigma2_target = 20
autocorrs_with_noise = collect_autocorrs(N, T, sigma2, bsigma2_target, nb_realizations, with_noise=True)
plot_mean_with_variance(autocorrs_with_noise, T, sigma2, bsigma2_target, title_suffix='avec bruit additif σ²_b = 0.5', zoom=False)
plot_mean_with_variance(autocorrs_with_noise, T, sigma2, bsigma2_target, title_suffix='avec bruit additif σ²_b = 0.5', zoom=True)
