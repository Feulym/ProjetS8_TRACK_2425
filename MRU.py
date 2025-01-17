###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################

def w_mru(sigma, T_ech, n1, n2) :
    """ 
        Génération d'un bruit blanc gaussien centré 
        représentant un bruit de modèle
    """
    Q = sigma*np.array(([T_ech**3/3, T_ech**2/2],
                        [T_ech**2/2, T_ech]))
    return Q@randn(n1, n2) 

###############################
# CALCUL MRU
###############################
N = 500 # taille échantillon
n1 = 2; n2 = 1
sigma = 1 # Variance bbgc
T_ech = 1 # Temps d'échantillonnage en seconde
w = np.zeros([n1,N])
for i in range(N) :
    w[:,i] = w_mru(sigma, T_ech, n1, n2)
    print(w[:,i])

print(np.mean(w[0, :]))
print(np.mean(w[1, :]))
###############################
# AFFICHAGE
###############################
plt.figure,
plt.plot(w[0, :],w[1, :])
plt.show()