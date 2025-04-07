import numpy as np
from typing import Tuple
from scipy.signal import lfilter, welch
from numpy.fft import fft, ifft, fftshift
from fracdiff import frac_diff, _acv_frac_diff
from statsmodels.tsa.stattools import acf

class Arfima:
    def __init__(self, d: float, ar_order: int, ma_order: int):
        """
        Initialize an ARFIMA process with fractional differencing parameter d,
        AR order (p) and MA order (q).
        
        Parameters:
            d (float): The fractional differencing order.
            ar_order (int): Order of the AR component.
            ma_order (int): Order of the MA component.
        """

        self.d = d
        self.ar_order = ar_order
        self.ma_order = ma_order
        self.phi, self.arroots = Arfima._generate_poly(ar_order)
        self.theta, self.maroots = Arfima._generate_poly(ma_order)
    
    @staticmethod
    def _generate_poly(order: int, low: float = 0.01, high: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates polynomial coefficients with roots chosen in conjugate pairs.

        If order > 0,
            the roots have a modulus in [low, high] (inside the unit circle).
        If order < 0,
            the roots will then be outside the unit circle.
        
        Returns:
            (np.ndarray, np.ndarray): Coefficients and roots of the polynome.
        """
        if order == 0:
            return (np.array([1]), None)

        n = np.abs(order)
        invert = order < 0
        nb_pairs = n // 2

        roots = np.array([])

        if nb_pairs > 0:
            r = np.random.uniform(low, high, nb_pairs)
            phi = np.random.uniform(0, np.pi, nb_pairs)
            pairs = r * np.exp(1j * phi)
            roots = np.concatenate([pairs, np.conj(pairs)])

        if p % 2 == 1:
            real_root = np.random.uniform(-high, high)
            roots = np.append(roots, real_root)

        if invert:
            roots = 1 / roots
        
        coeffs = np.poly(roots).real
        return coeffs, roots

    @staticmethod
    def arma_filter(x, phi, theta):
        return lfilter(theta, phi, x)

    def simulate(self, n: int) -> np.ndarray:
        """
        Simulate an ARFIMA process.

        The simulation concists of:
            1. Generating a gaussian white noise.
            2. Applying fractional differencing to generate a fractional integrated noise.
            3. Filetering the results with the ARMA filter.
        
        Parameters:
            n (int): Number of observations to simulate.
        
        Returns:
            np.ndarray: Simulated ARFIMA time series.
        """
        # assert np.all(np.abs(self.arroots) < 1.0), "AR: outside unit circle (not stationnary)"
        # assert np.all(np.abs(self.maroots) < 1.0), "MA: inside unit circle (not invertible)"

        wn = np.random.randn(n)

        fdn = frac_diff(wn, -self.d)
        x = Arfima.arma_filter(fdn, self.phi, self.theta)
        return x

def estimate_psd(x, fs=1, nperseg=256, noverlap=None, nfft=1024):
    """
    Estime la densité spectrale de puissance (DSP) d'un signal x
    en utilisant la méthode de Welch.
    
    Paramètres :
      - x : signal (tableau 1D)
      - fs : fréquence d'échantillonnage (Hz), par défaut 1.
      - nperseg : taille de la fenêtre utilisée (par défaut 256)
      - noverlap : recouvrement entre fenêtres (par défaut 50% de nperseg)
      - nfft : taille de la FFT (par défaut 1024)
      
    Renvoie :
      - f_est : tableau des fréquences
      - psd_est : DSP estimée (peut être normalisée, ici optionnelle)
    """
    if noverlap is None:
        noverlap = nperseg // 2
        
    f_est, psd_est = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    # Normalisation par la valeur maximale
    psd_est = psd_est / np.max(psd_est)
    return f_est, psd_est


def theoretical_psd_arfima(f, d, ar_coeff, ma_coeff, noise_variance=1):
    """
    Calcule la DSP théorique d'un processus ARFIMA.
    
    Le processus est défini par :
         AR_poly(L) y_t = MA_poly(L) (1 - L)^(-d) ε_t,
    de sorte que la DSP est donnée (à un facteur constant près) par :
         PSD(f) = noise_variance * |MA(e^{-j2πf})|² / |AR(e^{-j2πf})|² * |2 sin(πf)|^(2d)
    
    Paramètres :
      - f : tableau de fréquences (en Hz, normalisées par rapport à fs=1)
      - d : paramètre de différentiation fractionnaire
      - ar_coeff : coefficients du polynôme AR (ordre décroissant, avec le coefficient constant égal à 1)
      - ma_coeff : coefficients du polynôme MA (ordre décroissant, avec le coefficient constant égal à 1)
      - noise_variance : variance du bruit (par défaut 1)
      
    Renvoie :
      - f : tableau des fréquences (inchangé)
      - psd_theo : DSP théorique calculée
    """
    z = np.exp(-1j * 2 * np.pi * f)
    H_ma = np.polyval(ma_coeff, z)
    H_ar = np.polyval(ar_coeff, z)

    frac_factor = np.empty_like(f)
    idx_zero = (f == 0)
    if np.any(~idx_zero):
        frac_factor[~idx_zero] = (2 * np.sin(np.pi * f[~idx_zero]))**(-2*d)
        frac_factor[idx_zero] = (2 * np.sin(np.pi * f[~idx_zero][0]))**(-2*d)
    else:
        frac_factor[:] = 1.0

    psd_theo = noise_variance * (np.abs(H_ma)**2 / np.abs(H_ar)**2) * frac_factor

    return f, psd_theo

def theoretical_acf_arfima(d, ar_coeff, ma_coeff, nlags=50, noise_variance=1, nfft=1024):
    """
    Calcule l'autocorrélation théorique (ACF) d'un processus ARFIMA en calculant la DSP théorique 
    puis en prenant l'IFFT. Seules les valeurs pour des lags de 0 à nlags sont retournées.
    
    Paramètres :
      - d : paramètre de différentiation fractionnaire
      - ar_coeff : coefficients du polynôme AR (ordre décroissant, avec le coefficient constant égal à 1)
      - ma_coeff : coefficients du polynôme MA (ordre décroissant, avec le coefficient constant égal à 1)
      - noise_variance : variance du bruit (par défaut 1)
      - nfft : nombre de points pour la FFT/IFFT (par défaut 1024)
      - nlags : nombre de lags positifs à retourner (par défaut 50)
      
    Renvoie :
      - lags : tableau des lags de 0 à nlags
      - acf_theo : ACF théorique (réelle) pour ces lags
    """
    f = np.linspace(0, 1, nfft, endpoint=False)
    _, psd_theo = theoretical_psd_arfima(f, d, ar_coeff, ma_coeff, noise_variance)
    
    # Calcul de l'IFFT pour obtenir l'ACF (les indices 0 à nfft-1 correspondent aux lags 0 à nfft-1)
    acf = ifft(psd_theo)
    acf = np.real(acf)
    # acf = fftshift(acf)

    # lags = np.arange(-nfft//2, nfft - nfft//2)
    lags = np.arange(0, nlags+1)
    acf = acf[:nlags+1] / acf[0]
    
    return lags, acf


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Paramètres de simulation
    n = 5000       # longueur du signal
    fs = 1         # fréquence d'échantillonnage
    nfft = 1024    # nombre de points pour la FFT/IFFT

    # Liste des modèles à simuler : (ar_order, d, ma_order)
    models = [
        (0, 0.4, 0),  # ARFIMA(0, 0.4, 0)
        (2, 0.0, 2),  # ARFIMA(2, 0, 2)
        (1, 0.4, 1)   # ARFIMA(1, 0.4, 1)
    ]

    # Création d'une figure pour afficher les résultats
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 4 * len(models)))

    for i, (p, d, q) in enumerate(models):
        model = Arfima(d, p, q)
        x = model.simulate(n)
        
        # DSP estimée
        f_est, psd_est = estimate_psd(x, fs=fs, nfft=nfft)
        
        # DSP théorique à partir des coefficients AR et MA et du paramètre fractionnaire
        f_theo, psd_theo = theoretical_psd_arfima(f_est, d, model.phi, model.theta)
        
        # ACF estimée (en prenant, par exemple, 50 lags)
        acf_est = acf(x, nlags=50, fft=True)
        lags_est = np.arange(0, len(acf_est))
        
        # ACF théorique par IFFT de la DSP théorique
        lags_theo, acf_theo = theoretical_acf_arfima(d, model.phi, model.theta, nfft=nfft)
        
        # Affichage de la DSP
        ax1 = axes[i, 0]
        ax1.plot(f_est, 10 * np.log10(psd_est), label='DSP estimée', lw=1.5)
        ax1.plot(f_theo, 10 * np.log10(np.abs(psd_theo)), label='DSP théorique', lw=1.5)
        ax1.set_title(f"Modèle ARFIMA({p}, {d}, {q}) - DSP")
        ax1.set_xlabel("Fréquence")
        ax1.set_ylabel("PSD (dB/Hz)")
        ax1.legend()
        ax1.grid(True)
        
        # Affichage de l'ACF
        ax2 = axes[i, 1]
        ax2.stem(lags_est, acf_est, linefmt='b-', markerfmt='bo', basefmt=" ", label='ACF estimée')
        ax2.plot(lags_theo, acf_theo, 'r', lw=1.5, label='ACF théorique')
        ax2.set_title(f"Modèle ARFIMA({p}, {d}, {q}) - ACF")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Autocorrélation")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()