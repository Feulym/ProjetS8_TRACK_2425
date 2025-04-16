import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

def simulate_MUA(N, T, sigma2):
    """ 
        Génération d'une trajectoire MUA 
        d'après un bruit blanc gaussien centré 
        comprenant un bruit de modèle
    """
    t = np.arange(0, N*T, T)
    
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

    for k in range(N-1):
        w = L @ randn(3, 1)
        X[:, k+1] = A @ X[:, k] + w[:, 0]
    
    return t, X

def compute_jerk(X, T):
    """Estime le jerk à partir des positions."""
    j = (X[3:] - 3 * X[2:-1] + 3 * X[1:-2] - X[:-3]) / T**3
    return j

def autocorrelation(signal):
    """Calcule la fonction d'autocorrélation normalisée."""
    N = len(signal)
    result = np.correlate(signal, signal, mode='full') / N
    return result[result.size // 2:]

def estimate_variances(Y, T):
    """Estime sigma^2 et sigma_b^2 à partir des autocorrélations et de la matrice M."""
    jerk = compute_jerk(Y, T)
    R_j = autocorrelation(jerk)

    R_jyjy = np.array([R_j[0], R_j[1]])

    M = np.array([
        [11 / (20 * T), 20 / T**6],
        [13 / (60 * T), - 15 / T**6]
    ])

    """
    print("Matrice M :")
    print(M)
    """


    M_inv = np.linalg.inv(M)
    variances = M_inv @ R_jyjy
    
    """
    print("\nAutocorrélations calculées :")
    print(f"Rjyjy(0) estimé : {R_j[0]}") 
    print("Rjyjy(0) theorique :", (11 * sigma2) / (20 * T) + (20 * sigma_b2) / (T**6))
    print(f"Rjyjy(1) estimé : {R_j[1]}")
    print("Rjyjy(0) theorique :", (13 * sigma2) / (60 * T) - (15 * sigma_b2) / (T**6))
    """

    return variances

def bruitage_snr(x, snr_db):
    """
    Ajoute un bruit gaussien à un signal en fonction du rapport signal à bruit (SNR) donné.
    
    :param x: Signal d'entrée (numpy array)
    :param snr_db: Liste ou valeur unique du SNR en dB
    :return: Liste des signaux bruités si snr_db est une liste, sinon un seul signal bruité
    """
    X = x - np.mean(x)
    # Puissance moyenne du signal
    p_signal = np.mean(X**2)
    
    # Conversion en liste si snr_db est une valeur unique (plus pratique pour la suite du programme)
    if isinstance(snr_db, (int, float)):
        snr_db = [snr_db]
    
    x_bruite = []
    
    for snr in snr_db:
        # Conversion du SNR en échelle linéaire
        snr_linear = 10**(snr / 10)
        
        # Calcul de la puissance du bruit
        p_bruit = p_signal / snr_linear
        
        # Génération du bruit gaussien
        bruit = np.sqrt(p_bruit) * np.random.randn(*x.shape)
        
        # Ajout du bruit au signal
        x_bruite.append(x + bruit)
    
    # Retourne une liste si plusieurs SNR sont fournis, sinon ne retourne que la valeur de la premiere case
    return x_bruite if len(x_bruite) > 1 else x_bruite[0], p_bruit

##############################
# SIMULATION ET ESTIMATION
##############################
N = 5000  # Nombre d'échantillons
T = 1     # Temps d'échantillonnage
snr_db = 10
sigma2 = 4
sigma_b2 = 0

# Simulation de la trajectoire MUA
Y, X = simulate_MUA(N, T, sigma2)

# Y, sigma_b2 = bruitage_snr(Y, 70)

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
