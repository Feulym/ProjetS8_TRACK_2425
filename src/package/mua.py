import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
import package.common as com


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
    
    X = np.zeros((3, N))

    for k in range(N-1):
        w = L @ np.random.randn(3, 1)
        X[:, k+1] = A @ X[:, k] + w[:,0]
    
    t = np.arange(0, (N+1)*Tech, Tech)
    return t, X


if __name__ == "__main__":
    N = 50        # Taille échantillon
    sigma2 = 3    # Variance bbgc
    Tech = 1      # Temps d'échantillonnage en seconde
    M = 20        # Nombre de réalisation

    X_mat, Y_mat = com.multi_trajectoire(M, Trajec_MUA, N, Tech, sigma2)
