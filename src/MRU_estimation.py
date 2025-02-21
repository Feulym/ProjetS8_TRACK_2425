###############################
# IMPORTS
###############################
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn
from mpl_toolkits.mplot3d import Axes3D

##############################
# FONCTIONS
##############################

def trajec_MRU(N, Tech, sigma2) :
    """ 
        Génération d'une trajectoire MRU 
        d'après un bruit blanc gaussien centré 
        comprenant un bruit de modèle
    """
    t = np.arange(0, N*Tech, Tech)
    
    A = np.array(([1, Tech],
                  [0, 1]))

    Q = sigma2 * np.array([
        [Tech**3/3, Tech**2/2],
        [Tech**2/2, Tech     ]
    ])
    
    D = np.linalg.cholesky(Q)

    X = np.zeros((2, N))

    for k in range(N-1):
        w = D @ randn(2, 1)
        X[:, k+1] = A @ X[:, k] + w[:,0]
           
    return t, X

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


def compute_acceleration(X, Tech):
    """ Estime l'accélération à partir des positions """
    a = (X[2:] - 2*X[1:-1] + X[:-2]) / Tech**2
    return a

def autocorrelation(signal):
    """ Calcule la fonction d'autocorrélation normalisée """
    N = len(signal)
    result = np.correlate(signal, signal, mode='full') / N
    return result[result.size//2:]



def autocorr_aa(signal, Tech):
    a = compute_acceleration(signal, Tech)
    R_a = autocorrelation(a)
    sigma2 = ((3*Tech*R_a[0])/2 + (9*Tech*R_a[1]/4)) * (24/33)
    bsigma2 = (Tech**4/22)*R_a[0] + ((3*Tech**4)/44 - Tech**4/4)*R_a[1]
    return sigma2, bsigma2

###############################
# CALCUL MRU
###############################
N = 5000       # Taille échantillon
sigma2 = 12     # Variance bruit de modèle
Tech = 1       # Temps d'échantillonnage en seconde
snr_db = 3    # SNR en dB (valeur cible)
M = 100        #réalisation

# Calcul de la variance du bruit d'observation
bsigma2 = sigma2 / (10 ** (snr_db / 10))    # Variance bruit d'observation


t, X1 = trajec_MRU(N, Tech, sigma2)
t, X2 = Trajec_MUA(N, Tech, sigma2)

# Génération du bruit blanc gaussien de même taille que X1[0]
Y = X1[0] + np.sqrt(bsigma2) * randn(N)

a = compute_acceleration(Y, Tech)
mean_a = np.mean(a)
#print("Moyenne de l'accélération :", mean_a)


R_a = autocorrelation(a)
indices = np.arange(0, 7)
valeurs = [R_a[i] for i in indices]  # Extraction des valeurs correspondantes

# Calcul de sigma2 et bsigma2
B = np.array([
    [2/(3*Tech), 6/(Tech**4)],
    [1/(6*Tech), -4/(Tech**4)],
])


S = np.array([
    R_a[0],
    R_a[1],
])


Y = np.linalg.inv(B) @ S

# Affichage des points sélectionnés uniquement
plt.figure(figsize=(6, 4))
plt.plot(indices, valeurs, marker='o', linestyle='-', color='blue', markersize=8)
plt.title("Autocorrélation de l’accélération (0 à 6 Tech)")
plt.xlabel("Décalage temporel")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


print("Raa(0) = ", R_a[0])
print("Raa(1) = ", R_a[1])
print("Raa(2) = ", R_a[2])
print("Raa(0) théorique avec un signal bruité :", 2*sigma2/(3*Tech) + 6*bsigma2/Tech**4)
print("Raa(1) théorique avec un signal bruité :", sigma2/(6*Tech) - 4*bsigma2/Tech**4)
print("Raa(2) théorique avec un signal bruité :", bsigma2/(Tech**4) )

#print("sigma2 =", ((3*Tech*R_a[0])/2 + (9*Tech*R_a[1]/4)) * (24/33))
#print("bsigma2 =", (Tech**4/22)*R_a[0] + ((3*Tech**4)/44 - Tech**4/4)*R_a[1])

print("sigma2 =", Y[0])
print("bsigma2 =", Y[1])
# print("bsigma2 =", R_a[2]*Tech**4)

#%%
data_sigma = []
data_bsigma = []
#nbr_echantillons = [100000,120000,130000]
nbr_echantillons = [1000,2500, 5000,10000,20000]


data_sigma = []  # Liste pour stocker les estimations de sigma2
data_bsigma = []  # Liste pour stocker les estimations de bsigma2
std_sigma = []  # Stockage des écarts-types pour chaque N
std_bsigma = []  # Stockage des écarts-types pour chaque N

# Boucle sur les différentes tailles d'échantillons
for N in nbr_echantillons:
    print(N)
    temp1 = []
    temp2 = []
    for _ in range(M):  
        t, X1 = trajec_MRU(N, Tech, sigma2)
        Y = X1[0] + np.sqrt(bsigma2) * np.random.randn(N)
        sigma2_est, bsigma2_est = autocorr_aa(Y, Tech)  
        
        temp1.append(sigma2_est)  
        temp2.append(bsigma2_est)  

    data_sigma.append(temp1)  # Ajouter la liste des estimations pour ce N
    std_sigma.append(np.std(temp1))  # Calculer l'écart-type
    data_bsigma.append(temp2)  
    std_bsigma.append(np.std(temp2))  # Calculer l'écart-type
# Création de la figure 3D
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Paramètres
colors = ['blue', 'red', 'yellow', 'green', 'pink', 'purple']
labels = [f"N={n}" for n in nbr_echantillons]
z_offsets = np.arange(len(nbr_echantillons))  # Décalage en Z pour chaque histogramme

# Création des histogrammes en 3D
for i, N in enumerate(nbr_echantillons):
    hist, bins = np.histogram(data_sigma[i], bins=30)  # Histogramme
    xpos = (bins[:-1] + bins[1:]) / 2  # Position des barres (centre des bins)
    ypos = np.full_like(xpos, z_offsets[i])  # Décalage selon Z (plans)
    zpos = np.zeros_like(xpos)  # Base des barres

    dx = dy = (bins[1] - bins[0])  # Largeur des barres
    dz = hist  # Hauteur des barres (fréquence)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors[i], alpha=0.6, label=labels[i])

# Paramètres des axes
ax.set_xlabel("$\sigma^2$ estimé")
ax.set_ylabel("Taille d'échantillon N")
ax.set_zlabel("Fréquence")
ax.set_title("Histogramme 3D de $\sigma^2$ estimé pour différentes tailles d'échantillon")
ax.set_yticks(z_offsets)  # Ajuste les ticks de l'axe Y
ax.set_yticklabels(labels)  # Associe les labels

ax.legend()
plt.show()

# Representation de la dispersion
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(nbr_echantillons, [x / bsigma2 * 100 for x in std_bsigma], marker='o', linestyle='-', color='red')
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

