###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################
def Trajec_MRU(N, Tech, sigma) :
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
        X[:, k+1] = (A @ X[:, k] + w)[:,0]
        print("W =", w, "\nX_", k, " = ", X[:, k+1])
    
    t = np.arange(0, (N+1)*Tech, Tech)
    return t, X

###############################
# CALCUL MRU
###############################
N = 101      # Taille échantillon
sigma = 3    # Variance bbgc
Tech = 1/10  # Temps d'échantillonnage en seconde

t1, X = Trajec_MRU(N, Tech, sigma)
t2, Y = Trajec_MRU(N, Tech, sigma)
###############################
# AFFICHAGE
###############################
fig, axs = plt.subplots(2, 2, figsize=(20, 8))
labels = ['Position', 'Vitesse', 'Accélération']
for i in range(2):
    axs[i,0].plot(t1, X[i, :])
    axs[i,0].set_title(labels[i])
    axs[i,0].grid(True)
    axs[i,1].plot(t2, Y[i, :])
    axs[i,1].set_title(labels[i])
    axs[i,1].grid(True)

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(X, Y, 'bo')
plt.title("Trajectoire")
plt.grid()