import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

# Fonctions MRU de base
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
    return unbiased_corr[N-1:]

def theoretical_autocorr_MRU(T, sigma2, sigma_b2):
    return [
        (2 * sigma2) / (3 * T) + (6 * sigma_b2) / (T**4),
        (sigma2) / (6 * T) - (4 * sigma_b2) / (T**4)
    ]

def estimate_variances_from_R(R_hat, T):
    M = np.array([
        [2/(3*T), 6/(T**4)],
        [1/(6*T), -4/(T**4)]
    ])
    return np.linalg.inv(M) @ R_hat

# -----------------------
# Plot double axe
# -----------------------
def plot_with_dual_xaxis(N_values, std_norms, mean_values, std_values, true_value, ylabel, title):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(N_values, mean_values, 'o-', color='blue', label='Moyenne')
    ax1.fill_between(N_values, np.array(mean_values) - np.array(std_values),
                     np.array(mean_values) + np.array(std_values), color='blue', alpha=0.3, label='±1σ')
    ax1.fill_between(N_values, np.array(mean_values) - 2*np.array(std_values),
                     np.array(mean_values) + 2*np.array(std_values), color='orange', alpha=0.2, label='±2σ')
    ax1.fill_between(N_values, np.array(mean_values) - 3*np.array(std_values),
                     np.array(mean_values) + 3*np.array(std_values), color='green', alpha=0.1, label='±3σ')
    ax1.axhline(true_value, color='black', linestyle='--', label=f'Vraie valeur {ylabel}={true_value:.4f}')
    ax1.set_xscale('log')
    ax1.set_xlabel("Nombre d'échantillons (N)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.get_xlim())

    labels = [f"{s:.2f}" for s in std_norms]
    ax2.set_xticks(N_values)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Écart-type normalisé de $R_{aa}(0)$")

    plt.show()

# -----------------------
# Paramètres
# -----------------------
T = 1
speed = 7.7
taux = 0.01
sigma = (speed * taux) / 3
sigma2 = sigma**2
sigma_b2 = 0.5

N_values = [5000, 2000, 1000, 500, 250, 100, 50, 30]
nb_realizations_autocorr = 1000
nb_simulations_error = 1000

# Initialisation
std_norms_nb = []
std_norms_b = []

# -----------------------
# Calcul des écarts-types normalisés
# -----------------------
for N in N_values:
    autocorrs_nb = []
    autocorrs_b = []
    
    for _ in range(nb_realizations_autocorr):
        X = trajec_MRU(N, T, sigma2)
        Y_no_noise = X[0]
        Y_bruite, _ = bruitage_distance(X[0], distance=3*np.sqrt(sigma_b2))

        acc_no = compute_acceleration(Y_no_noise, T)
        acc_b = compute_acceleration(Y_bruite, T)

        R_no = autocorrelation(acc_no)
        R_b = autocorrelation(acc_b)

        autocorrs_nb.append(R_no)
        autocorrs_b.append(R_b)

    autocorrs_nb = np.array(autocorrs_nb)
    autocorrs_b = np.array(autocorrs_b)

    std_norm_nb = np.std(autocorrs_nb[:,0]) / np.mean(autocorrs_nb[:,0]), np.std(autocorrs_nb[:,1]) / np.mean(np.abs(autocorrs_nb[:,1]))
    std_norm_b = np.std(autocorrs_b[:,0]) / np.mean(autocorrs_b[:,0]), np.std(autocorrs_b[:,1]) / np.mean(np.abs(autocorrs_b[:,1]))

    std_norms_nb.append(std_norm_nb)
    std_norms_b.append(std_norm_b)

# -----------------------
# Simulation des erreurs
# -----------------------
mean_sigma2_nb, mean_sigma2_b = [], []
mean_sigmab2_nb, mean_sigmab2_b = [], []
std_sigma2_nb, std_sigma2_b = [], []
std_sigmab2_nb, std_sigmab2_b = [], []

for mode, std_norms in zip(["NON BRUITÉ", "BRUITÉ"], [std_norms_nb, std_norms_b]):
    for idx, N in enumerate(N_values):
        if mode == "NON BRUITÉ":
            R_theo = theoretical_autocorr_MRU(T, sigma2, 0)
        else:
            R_theo = theoretical_autocorr_MRU(T, sigma2, sigma_b2)

        R_theo = np.array(R_theo)

        std_err_0 = std_norms[idx][0] * R_theo[0]
        std_err_1 = std_norms[idx][1] * R_theo[1]

        sigma2_estimates = []
        sigmab2_estimates = []

        for _ in range(nb_simulations_error):
            error = np.array([
                std_err_0 * np.random.randn(),
                std_err_1 * np.random.randn()
            ])
            R_hat = R_theo + error
            sigma2_hat, sigmab2_hat = estimate_variances_from_R(R_hat, T)

            sigma2_estimates.append(sigma2_hat)
            sigmab2_estimates.append(sigmab2_hat)

        sigma2_estimates = np.array(sigma2_estimates)
        sigmab2_estimates = np.array(sigmab2_estimates)

        if mode == "NON BRUITÉ":
            mean_sigma2_nb.append(np.mean(sigma2_estimates))
            mean_sigmab2_nb.append(np.mean(sigmab2_estimates))
            std_sigma2_nb.append(np.std(sigma2_estimates))
            std_sigmab2_nb.append(np.std(sigmab2_estimates))
        else:
            mean_sigma2_b.append(np.mean(sigma2_estimates))
            mean_sigmab2_b.append(np.mean(sigmab2_estimates))
            std_sigma2_b.append(np.std(sigma2_estimates))
            std_sigmab2_b.append(np.std(sigmab2_estimates))

# -----------------------
# PLOTS
# -----------------------

# NON BRUITÉ
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_nb], mean_sigma2_nb, std_sigma2_nb, sigma2, "$\sigma^2$", "Évolution de $\sigma^2$ (MRU sans bruit)")
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_nb], mean_sigmab2_nb, std_sigmab2_nb, 0.0, "$\sigma_b^2$", "Évolution de $\sigma_b^2$ (MRU sans bruit)")

# BRUITÉ
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_b], mean_sigma2_b, std_sigma2_b, sigma2, "$\sigma^2$", "Évolution de $\sigma^2$ (MRU avec bruit)")
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_b], mean_sigmab2_b, std_sigmab2_b, sigma_b2, "$\sigma_b^2$", "Évolution de $\sigma_b^2$ (MRU avec bruit)")
