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
def trajec_MRU(N, Tech, sigma) :
    """ 
        Génération d'une trajectoire MRU 
        d'après un bruit blanc gaussien centré 
        comprenant un bruit de modèle
    """
    t = np.arange(0, (N+1)*Tech, Tech)
    
    A = np.array(([1, Tech],
                  [0, 1]))

    Q = sigma**2 * np.array([
        [Tech**3/3, Tech**2/2],
        [Tech**2/2, Tech     ]
    ])
    
    D = np.linalg.cholesky(Q)
    
    X = np.zeros((2, N+1))

    for k in range(N):
        w = D @ randn(2, 1)
        X[:, k+1] = A @ X[:, k] + w[:,0]
        print("W =", w, "\nX_", k, " = ", X[:, k+1])
    
    t = np.arange(0, (N+1)*Tech, Tech)
    return t, X

###############################
# CALCUL MRU
###############################
N = 50        # Taille échantillon
sigma = 3     # écart-type bbgc
Tech = 1      # Temps d'échantillonnage en seconde
M = 20        # Nombre de réalisation
com.multi_trajectoire(M, trajec_MRU, N, Tech, sigma)
