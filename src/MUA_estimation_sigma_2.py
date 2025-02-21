import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################
def simulate_MUA(N, T, sigma2, sigma_b2):
    """Simule une trajectoire MUA bruitée."""
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

##############################
# SIMULATION ET ESTIMATION
##############################
N = 5000  # Nombre d'échantillons
T = 1     # Temps d'échantillonnage
snr_db = 10
sigma2 = 4
sigma_b2 = sigma2 / (10 ** (snr_db / 10))    # Variance bruit d'observation

# Simulation de la trajectoire MUA
Y, X = simulate_MUA(N, T, sigma2, sigma_b2)

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

M = 150

data_sigma = []
data_bsigma = []
nbr_echantillons = [500, 2000, 5000, 8000, 10000, 20000]


data_sigma = []  # Liste pour stocker les estimations de sigma2
data_bsigma = []  # Liste pour stocker les estimations de bsigma2
std_sigma = []  # Stockage des écarts-types pour chaque N
std_bsigma = []  # Stockage des écarts-types pour chaque N

# Boucle sur les différentes tailles d'échantillons
for N in nbr_echantillons:  
    temp1 = []
    temp2 = []
    for _ in range(M):  
        Y, X = simulate_MUA(N, T, sigma2, sigma_b2)
        
        estimated_variances = estimate_variances(Y, T)
        sigma2_est=estimated_variances[0]
        bsigma2_est=estimated_variances[1]
        
        temp1.append(sigma2_est)  
        temp2.append(bsigma2_est)  

    data_sigma.append(temp1)  # Ajouter la liste des estimations pour ce N
    std_sigma.append(np.std(temp1))  # Calculer l'écart-type
    data_bsigma.append(temp2)  
    std_bsigma.append(np.std(temp2))  # Calculer l'écart-type
# Création des histogrammes
plt.figure(figsize=(12, 5))

colors = ['blue', 'red', 'yellow', 'green', 'pink','purple']
labels = [f"N={n}" for n in nbr_echantillons]

plt.subplot(1, 2, 1)
for i in range(len(nbr_echantillons)):
    plt.hist(data_bsigma[i], bins=45, color=colors[i], edgecolor='black', alpha=0.6, label=labels[i])
plt.title("Histogramme de $\sigma_b^2$")
plt.xlabel("$\sigma^2$ estimé")
plt.ylabel("Fréquence")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(len(nbr_echantillons)):
    plt.hist(data_sigma[i], bins=45, color=colors[i], edgecolor='black', alpha=0.6, label=labels[i])
plt.title("Histogramme de $\sigma^2$")
plt.xlabel("$\sigma_b^2$ estimé")
plt.ylabel("Fréquence")
plt.legend()

plt.tight_layout()
plt.show()

# Representation de la dispersion
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(nbr_echantillons, [x / sigma_b2 * 100 for x in std_bsigma], marker='o', linestyle='-', color='red')
plt.xlabel("Taille de l'échantillon N")
plt.ylabel("Écart-type de $\sigma_b^2$ estimé (en %)")
plt.title("Évolution de la précision de l'estimation")
plt.grid()

# Tracer l'évolution de l'écart-type pour voir où la précision se stabilise
plt.subplot(1, 2, 2)
plt.plot(nbr_echantillons, [x / sigma2 * 100 for x in std_sigma], marker='o', linestyle='-', color='red')
plt.xlabel("Taille de l'échantillon N")
plt.ylabel("Écart-type de $\sigma^2$ estimé (en %)")
plt.title("Évolution de la précision de l'estimation")
plt.grid()

plt.tight_layout()
plt.show()