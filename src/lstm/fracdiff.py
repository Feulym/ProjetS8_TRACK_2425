import numpy as np
from scipy.signal import lfilter
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import gamma
from scipy.linalg import cholesky, toeplitz

# ----------------------------
# Fractional Differentiation
# ----------------------------
def _frac_weights(N, d):
    """
    Calculate the fractional differencing weights using recursion.

    The weights are computed recursively as follows:
        w[0] = 1, for k == 1,
        w[k] = w[k-1] * ((-d + k - 1) / k) for k >= 1.

    Parameters:
        N (int): Number of weights to be computed.
        d (float): The fractional differencing order.

    Returns:
        np.ndarray: The fractional differencing weights, with shape (N,).
    """
    k = np.arange(1, N)
    w = np.concatenate(([1], np.cumprod((k - d - 1) / k)))
    return w

def _acv_frac_diff(k, d):
    """
    Compute the autocovariance for fractional differencing.

    For small lags (< 50) the autocovariance is computed directly using:
        acv[k] = gamma(1+2*d) / (gamma(1+d)**2) * (gamma(k-d) / gamma(k+1+d))
    For larger lags, a stable recursive formula is applied:
        acv[k] = acv[k-1] * ((k-1-d)/(k))

    Parameters:
        k (int or np.ndarray): Lag value(s).
        d (float): Fractional differencing order.

    Returns:
        float or np.ndarray: Autocovariance value(s) corresponding to k.
    """
    scalar_input = np.isscalar(k)
    k_arr = np.atleast_1d(k).astype(int)
    max_k = int(np.max(k_arr))
    treshold = min(50, max_k)

    # Allocate an array from 0 to max_k lags (max_k + 1)
    acv = np.empty(max_k + 1, dtype=np.float32)

    const = gamma(1 + 2*d) / (gamma(1+d)**2)

    # Small lags, direct computation with gamma
    k_small = np.arange(treshold + 1)
    acv[:treshold + 1] = const * (gamma(k_small - d) / gamma(k_small + 1 + d))
    
    # Larger lags, recursion
    if max_k > treshold:
        k_large = np.arange(treshold + 1, max_k + 1)
        factors = (k_large - 1 - d) / (k_large)
        acv[treshold + 1:] = acv[treshold] * np.cumprod(factors)
    
    acv = acv[k_arr]
    if scalar_input:
        return acv[0]

    return acv

def _chol_frac_diff(N, d):
    """
    Compute the Cholesky decomposition of the autocovariance matrix for fractional differencing.
    
    Implements numerical stabilization for large N.

    Parameters:
        N (int): The size of the autocovariance matrix.
        d (float): The fractional differencing order.

    Returns:
        np.ndarray: The Cholesky decomposition of the autocovariance matrix.
    """
    k = np.arange(N)
    r_vals = _acv_frac_diff(k, d)
    R = toeplitz(r_vals)
    L = cholesky(R, lower=True)
    return L

def frac_diff(x, d):
    """
    Apply fractional differencing to a time series.

    Parameters:
        x (np.ndarray): The input time series.
        d (float): The fractional differencing order.

    Returns:
        np.ndarray: The fractional difference of the time series.
    """
    if d == 0: return x
    N = len(x)
    w = _frac_weights(N, d)
    # Note: conv or np.dot(w[:t+1], x[t::-1]) is possible
    return lfilter(w, [1], x)

def fast_frac_diff(x, d):
    """
    Apply fast fractional differencing to a time series using FFT.

    Parameters:
        x (np.ndarray): The input time series.
        d (float): The fractional differencing order.

    Returns:
        np.ndarray: The fractional difference of the time series.
    """
    N = len(x)
    w = _frac_weights(N, d)
    nfl = next_fast_len(2 * N - 1)
    y = ifft(fft(x, nfl) * fft(w, nfl)).real[:N]
    return y
    
def chol_frac_diff(x, d):
    """
    Apply fractional differencing to a time series using Cholesky decomposition.

    This function computes the Cholesky decomposition of the autocovariance matrix 
    for fractional differencing and applies it to the time series.

    Parameters:
        x (np.ndarray): The input time series.
        d (float): The fractional differencing order.

    Returns:
        np.ndarray: The fractional difference of the time series.

    Note: This implementation is numerically stable even for large N values.
    """
    N = len(x)
    L = _chol_frac_diff(N, d)
    return L @ x


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    from statsmodels.tsa.stattools import acf

    np.random.seed(42)
    max_lags = 50
    N = 6_000
    d = -0.4

    white_noise = np.random.normal(0, 1, N)

    # Computation of different frac diff methods
    start_time = time.time()
    frac_diff_x = frac_diff(white_noise, d)
    frac_diff_time = time.time() - start_time

    start_time = time.time()
    fast_frac_diff_x = fast_frac_diff(white_noise, d)
    fast_frac_diff_time = time.time() - start_time

    start_time = time.time()
    chol_frac_diff_x = chol_frac_diff(white_noise, d)
    chol_frac_diff_time = time.time() - start_time

    # Time taken
    print(f"Time taken for frac_diff: {frac_diff_time:.6f} seconds")
    print(f"Time taken for fast_frac_diff: {fast_frac_diff_time:.6f} seconds")
    print(f"Time taken for chol_frac_diff: {chol_frac_diff_time:.6f} seconds")

    # ACF
    acf_frac_diff = acf(frac_diff_x, nlags=max_lags)
    acf_fast_frac_diff = acf(fast_frac_diff_x, nlags=max_lags)
    acf_chol_frac_diff = acf(chol_frac_diff_x, nlags=max_lags)

    k = np.arange(max_lags+1)
    acf_theoretical = _acv_frac_diff(k, d)
    acf_theoretical /= acf_theoretical[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    markerline1, stemlines1, baseline1 = ax.stem(np.arange(len(acf_frac_diff)), acf_frac_diff, 
                                               linefmt='b-', markerfmt='bo', basefmt=" ", 
                                               label="ACF (Fractional Differencing)")
    markerline2, stemlines2, baseline2 = ax.stem(np.arange(len(acf_fast_frac_diff)), acf_fast_frac_diff, 
                                               linefmt='g-', markerfmt='go', basefmt=" ", 
                                               label="ACF (Fast Fractional Differencing)")
    markerline3, stemlines3, baseline3 = ax.stem(np.arange(len(acf_chol_frac_diff)), acf_chol_frac_diff, 
                                               linefmt='r-', markerfmt='ro', basefmt=" ", 
                                               label="ACF (Cholesky Fractional Differencing)")
    markerline4, stemlines4, baseline4 = ax.stem(np.arange(len(acf_theoretical)), acf_theoretical, 
                                               linefmt='k--', markerfmt='kx', basefmt=" ", 
                                               label="ACF Théorique")

    plt.setp(markerline1, markersize=7)
    plt.setp(markerline2, markersize=5)
    plt.setp(markerline3, markersize=5)
    plt.setp(markerline4, markersize=6)
    plt.setp(stemlines4, linewidth=0.7)
    
    ax.set_title(f"ACF pour bruit blanc différencié fractionnaire (d={d})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF Normalisée")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # PSD
    def psd_est_frac_diff(x, fs = 1, nperseg = 256):
        # Welch retourne le spectre one-sided (pour f ∈ [0, fs/2])
        f_est, psd_est = welch(x, fs=fs, nperseg=nperseg)
        psd_est /= np.max(psd_est)
        return f_est, psd_est

    def psd_theo_frac_diff(f_range):
        # Filtrage par différenciation fractionnaire avec un transfert de (1-L)^d
        # S(f) ∝ |1-e^{-j2πf}|^(2d) = 2|sin(πf)|^(2d).
        f_theo = f_range.copy()
        psd_theo = np.empty_like(f_theo)
        psd_theo[1:] = (2 * np.sin(np.pi * f_theo[1:]))**(2*d)
        psd_theo[0] = psd_theo[1]
        psd_theo /= np.max(psd_theo)
        return f_theo, psd_theo

    f_est, psd_est_fd = psd_est_frac_diff(frac_diff_x)
    _, psd_est_fdd = psd_est_frac_diff(fast_frac_diff_x)
    _, psd_est_cfd = psd_est_frac_diff(chol_frac_diff_x)

    f_theo, psd_theo_fd = psd_theo_frac_diff(f_est) 

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(f_est, psd_est_fd, 'b-', linewidth=2, label="DSP estimée (Frac diff)")
    ax.plot(f_est, psd_est_fdd, 'g-', label="DSP estimée (Fast frac diff)")
    ax.plot(f_est, psd_est_cfd, 'r-', label="DSP estimée (Chol frac diff)")
    ax.plot(f_theo, psd_theo_fd, 'k--', label="DSP théorique")
    ax.set_title("Densité Spectrale de Puissance")
    ax.set_xlabel("Fréquence normalisée")
    ax.set_ylabel("DSP Normalisée")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()