###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import package.common as com

##############################
# FONCTIONS
##############################
def compute_Q_matrix(sigma_w2, alpha, Tech):
    """
    Calcule la matrice de covariance du bruit de processus pour un modèle d'accélération Singer.

    Référence :
    R. A. Singer, 
    "Estimating Optimal Tracking Filter Performance for Manned Maneuvering Targets," 
    in IEEE Transactions on Aerospace and Electronic Systems, 
    vol. AES-6, no. 4, pp. 473-483, July 1970, 
    doi: 10.1109/TAES.1970.310128.

    Paramètres :
    - sigma_w2 : Variance du bruit de manœuvre.
    - alpha : Coefficient d'atténuation.
    - Tech : Intervalle de temps d'échantillonnage.

    Retour :
    - Matrice de covariance Q(k) de taille 3x3.
    """

    at = alpha * Tech
    
    q11 = (1 / (2 * alpha ** 5)) * (
        1 - np.exp(-2 * at)
        + 2 * at
        + (2 * at ** 3) / 3
        - 2 * at ** 2
        - 4* at * np.exp(-at)
    )
    
    q12 = (1 / (2 * alpha ** 4)) * (
        1 + np.exp(-2 * at)
        - 2 * np.exp(-at)
        + 2 * at * np.exp(-at)
        - 2 * at 
        + at ** 2
    )
    
    q13 = (1 / (2 * alpha ** 3)) * (
        1 - np.exp(-2 * at) - 2 * at * np.exp(-at)
    )

    q22 = (1 / (2 * alpha ** 3)) * (
        4 * np.exp(-at) - 3 - np.exp(-2 * at) + 2 * at
    )
    
    q23 = (1 / (2 * alpha ** 2)) * (
        np.exp(-2 * at) + 1 - 2 * np.exp(-at)
    )

    q33 = (1 / (2 * alpha)) * (1 - np.exp(-2 * at))

    Q = sigma_w2 * np.array([
        [q11, q12, q13],
        [q12, q22, q23],
        [q13, q23, q33]
    ])
    
    return Q

def traj_singer(N, Tech, sigma2, alpha):
    """
    Calcule une trajectoire d'un modèle d'accélération Singer.

    Référence :
    R. A. Singer, 
    "Estimating Optimal Tracking Filter Performance for Manned Maneuvering Targets," 
    in IEEE Transactions on Aerospace and Electronic Systems, 
    vol. AES-6, no. 4, pp. 473-483, July 1970, 
    doi: 10.1109/TAES.1970.310128.
    """
    Q = compute_Q_matrix(sigma2, alpha, Tech)
    L = np.linalg.cholesky(Q)

    A = np.array([
        [1, Tech, 1 / (alpha ** 2) * (-1 + alpha * Tech + np.exp(-alpha * Tech))],
        [0, 1, 1 / alpha * (1 - np.exp(-alpha * Tech))],
        [0, 0, np.exp(-alpha * Tech)],
    ])

    X = np.zeros((3, N))
    for k in range(N - 1):
        w = L @ np.random.randn(3, 1)
        X[:, k+1] = A @ X[:, k] + w[:, 0]
    
    t = np.arange(0, N*Tech, Tech)
    return t, X

###############################
# CALCUL Singer
###############################
N = 300             # Taille échantillon
alpha = 1/300       # Coefficiant d'atténuation (inverse du temps de manoeuvre)
sigma_m2 = 1e-4     # Variance accélaration de manoeuvre
sigma_w2 = 2 * alpha * sigma_m2    # Variance bbgc
Tech = 1            # Temps d'échantillonnage en seconde
M = 20              # Nombre de réalisation

com.multi_trajectoire(M, traj_singer, N, Tech, sigma_w2, alpha)

###############################
# AFFICHAGE
###############################
# t, X = traj_singer(N, Tech, sigma_w2, alpha)
# fig, axs = plt.subplots(3, 1, figsize=(10, 8))
# labels = ['Position', 'Vitesse', 'Accélération']
# colors = ['r', 'b', 'g']
# for i in range(3):
#     axs[i].plot(t, X[i, :], color=colors[i])
#     axs[i].set_title(labels[i], fontsize=14)
#     axs[i].set_xlabel('Temps (s)', fontsize=12)
#     axs[i].grid(True)

# plt.tight_layout()
# plt.show()
