import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

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

def compute_jerk(Y, T):
    return (Y[3:] - 3*Y[2:-1] + 3*Y[1:-2] - Y[:-3]) / T**3

def autocorrelation(signal):
    N = len(signal)
    result = np.correlate(signal, signal, mode='full')
    lags = np.arange(-N+1, N)
    unbiased_corr = result / (N - np.abs(lags))
    return unbiased_corr[N-1:]

def theoretical_autocorr_values(T, sigma2, sigma_b2):
    return [
        (11*sigma2)/(20*T) + (20*sigma_b2)/(T**6),
        (13*sigma2)/(60*T) - (15*sigma_b2)/(T**6),
        (sigma2)/(120*T) + (6*sigma_b2)/(T**6),
        -(sigma_b2)/(T**6)
    ]

def estimate_variances_MUA(R_hat, T):
    M = np.array([
        [11/(20*T), 20/(T**6)],
        [13/(60*T), -15/(T**6)]
    ])
    return np.linalg.inv(M) @ R_hat

# --- Plotting with two x-axis ---
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
    ax2.set_xlabel("Écart-type normalisé de $R_{jj}(0)$")
    plt.show()

# --- Paramètres
T = 1
sigma2 = 4
sigma_b2 = 0.5
nb_realizations_autocorr = 1000
nb_simulations_error = 1000
N_values = [5000, 2000, 1000, 500, 250, 100, 50, 30]

# --- Collecte des écarts-types normalisés
std_norms_nb = []
std_norms_b = []

for N in N_values:
    Rjj0_nb = []
    Rjj1_nb = []
    Rjj0_b = []
    Rjj1_b = []
    for _ in range(nb_realizations_autocorr):
        Y_nb, _ = simulate_MUA(N, T, sigma2, 0)
        jerk_nb = compute_jerk(Y_nb, T)
        R_nb = autocorrelation(jerk_nb)
        Rjj0_nb.append(R_nb[0])
        Rjj1_nb.append(R_nb[1])

        Y_b, _ = simulate_MUA(N, T, sigma2, sigma_b2)
        jerk_b = compute_jerk(Y_b, T)
        R_b = autocorrelation(jerk_b)
        Rjj0_b.append(R_b[0])
        Rjj1_b.append(R_b[1])

    std_norms_nb.append((np.std(Rjj0_nb)/np.mean(Rjj0_nb), np.std(Rjj1_nb)/np.mean(np.abs(Rjj1_nb))))
    std_norms_b.append((np.std(Rjj0_b)/np.mean(Rjj0_b), np.std(Rjj1_b)/np.mean(np.abs(Rjj1_b))))

# --- Simulation des erreurs pour estimer sigma² et sigma_b²
mean_sigma2_nb, mean_sigma2_b = [], []
mean_sigmab2_nb, mean_sigmab2_b = [], []
std_sigma2_nb, std_sigma2_b = [], []
std_sigmab2_nb, std_sigmab2_b = [], []

for mode, std_norms in zip(["NON BRUITÉ", "BRUITÉ"], [std_norms_nb, std_norms_b]):
    for idx, N in enumerate(N_values):
        if mode == "NON BRUITÉ":
            R_theo = theoretical_autocorr_values(T, sigma2, 0)
        else:
            R_theo = theoretical_autocorr_values(T, sigma2, sigma_b2)

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
            R_hat = R_theo[:2] + error
            sigma2_hat, sigmab2_hat = estimate_variances_MUA(R_hat, T)

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

# --- PLOTS FINAUX

# NON BRUITÉ
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_nb], mean_sigma2_nb, std_sigma2_nb, sigma2, "$\sigma^2$", "Évolution de $\sigma^2$ (MUA sans bruit)")
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_nb], mean_sigmab2_nb, std_sigmab2_nb, 0.0, "$\sigma_b^2$", "Évolution de $\sigma_b^2$ (MUA sans bruit)")

# BRUITÉ
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_b], mean_sigma2_b, std_sigma2_b, sigma2, "$\sigma^2$", "Évolution de $\sigma^2$ (MUA avec bruit)")
plot_with_dual_xaxis(N_values, [s[0] for s in std_norms_b], mean_sigmab2_b, std_sigmab2_b, sigma_b2, "$\sigma_b^2$", "Évolution de $\sigma_b^2$ (MUA avec bruit)")
