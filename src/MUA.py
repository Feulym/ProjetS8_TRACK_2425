###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################
def Trajec_MUA(N, Tech, sigma):
    A = np.array([
        [1, Tech, Tech**2/2],
        [0, 1, Tech],
        [0, 0, 1]
    ])
    
    Q = sigma**2 * np.array([
        [Tech**5/20, Tech**4/8, Tech**3/6],
        [Tech**4/8, Tech**3/3, Tech**2/2],
        [Tech**3/6, Tech**2/2, Tech]
    ])
    
    L = np.linalg.cholesky(Q)
    
    X = np.zeros((3, N+1))

    for k in range(N):
        w = L @ np.random.randn(3, 1)
        X[:, k+1] = A @ X[:, k] + w[:,0]
    
    t = np.arange(0, (N+1)*Tech, Tech)
    return t, X

###############################
# CALCUL MUA
###############################
N = 500     # Taille échantillon
sigma = 3   # écart-type bbgc
Tech = 1    # Temps d'échantillonnage en seconde

t, X = Trajec_MUA(N, Tech, sigma)

###############################
# AFFICHAGE
###############################
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
labels = ['Position', 'Vitesse', 'Accélération']
for i in range(3):
    axs[i].plot(t, X[i, :])
    axs[i].set_title(labels[i])
    axs[i].grid(True)

plt.tight_layout()
plt.show()
