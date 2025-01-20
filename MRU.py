###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

##############################
# FONCTIONS
##############################
def w_mru(sigma, Tech, size) :
    """ 
        Génération d'un bruit blanc gaussien centré 
        représentant un bruit de modèle
    """
    Q = sigma * np.array(([Tech**3/3, Tech**2/2],
                        [Tech**2/2, Tech]))

    return Q @ randn(*size) 

###############################
# CALCUL MRU
###############################
N = 500         # Taille échantillon
n1 = 2; n2 = 1  # Dimensions d'un échantillon
sigma = 1       # Variance bbgc
Tech = 1        # Temps d'échantillonnage en seconde

w = np.zeros((n1,N))
for i in range(N) :
    w[:,i] = w_mru(sigma, Tech, (n1, n2))[:, 0]
    print(w[:,i])

print(np.mean(w[0, :]))
print(np.mean(w[1, :]))

###############################
# AFFICHAGE
###############################
plt.figure,
plt.plot(w[0, :],w[1, :])
plt.show()