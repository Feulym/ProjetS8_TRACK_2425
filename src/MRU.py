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
def trajec_MRU(N, Tech, sigma, DEBUGSIGMA=False) :
    """ 
        Génération d'une trajectoire MRU 
        d'après un bruit blanc gaussien centré 
        comprenant un bruit de modèle
    """
    t = np.arange(0, N*Tech, Tech)
    
    A = np.array(([1, Tech],
                  [0, 1]))

    Q = sigma * np.array([
        [Tech**3/3, Tech**2/2],
        [Tech**2/2, Tech     ]
    ])
    
    D = np.linalg.cholesky(Q)
    
    X = np.zeros((2, N))
    sum_ = 0
    
    for k in range(N-1):
        w = D @ randn(2, 1)
        sum_ += w @ w.T
        X[:, k+1] = A @ X[:, k] + w[:,0]
    
    sum_ /= N-1;
    if DEBUGSIGMA:
        print("Q : ", Q)
        print("Variance W : ", sum_)
        print("Différence :", Q - sum_)
    
    return t, X

###############################
# CALCUL MRU
###############################
N = 5000        # Taille échantillon
sigma2 = 3     # Variance bbgc
Tech = 1      # Temps d'échantillonnage en seconde
M = 20        # Nombre de réalisation
X_mat, Y_mat = com.multi_trajectoire(M, trajec_MRU, N, Tech, sigma2)

