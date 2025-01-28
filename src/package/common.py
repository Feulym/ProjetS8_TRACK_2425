###########################################
# Fonctions en commun pour les différentes 
# méthodes de génération de trajectoire
###########################################

import matplotlib.pyplot as plt
import numpy as np

def trajectoire_XY(fonction_type, N, Tech, sigma2, alpha=None):
    """
    Parameters
    ----------
    fonction_type : Fonction(N, Tech, sigma, (optional) alpha)
        Fonction générant un vecteur de point selon un type prédéterminé 
        (MRU, MUA, Singer).
    N : INT
        Taille des échantillons.
    Tech : FLOAT
        Temps d'échantillonnage du modèle.
    sigma2 : FLOAT
        Variance du bruit blanc de génération.
    alpha : FLOAT, optional
        UNIQUEMENT POUR SINGER
        Inverse du temps de manoeuvre. 
        The default is None (not used).

    Returns
    -------
    t1 : Vector size N of float
        Vecteur temps des points de coordonée X.
    X : Vector size d*N of float
        Vecteur d'état des coordonées X. 
        Si méthode de MRU : d=2, Position, Vitesse
        Sinon : d=3, Position, Vitesse, Accélération
    t2 : Vector size N of float
        Vecteur temps des points de coordonée Y.
    Y : Vector size d*N of float
        Vecteur d'état des coordonées Y. 
        Si méthode de MRU : d=2, Position, Vitesse
        Sinon : d=3, Position, Vitesse, Accélération.

    """
    if not alpha :
        t1, X = fonction_type(N, Tech, sigma2)
        t2, Y = fonction_type(N, Tech, sigma2)
    else :
        t1, X = fonction_type(N, Tech, sigma2, alpha)
        t2, Y = fonction_type(N, Tech, sigma2, alpha)
    return t1, X, t2, Y

def plot_trajectoire(X, Y, NEW_PLOT=True, SHOW=True):
    """
    Parameters
    ----------
    X : Vector size N of float
        Vecteur des coordonnées X.
    Y : Vector size N of float
        Vecteur des coordonnées Y.
    """
    if NEW_PLOT :
        plt.figure()
    plt.plot(X, Y, '-o')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectoires")
    
    if SHOW:
        plt.grid()
        plt.show()
    
def plot_vectetat(t1, X, t2=None, Y=None):
    nb_etat = np.size(X[:,0])
    fig, axs = plt.subplots(nb_etat, 1 if not Y.any() else 2)
    labels = ['Position', 'Vitesse', 'Accélération']
    colors = ['r', 'b', 'g']
    if not Y.any() :
        for i in range(nb_etat):
            axs[i].plot(t1, X[i, :], c=colors[i])
            axs[i].set_title(labels[i])
            axs[i].grid(True)
    else :
        for i in range(nb_etat):
            axs[i,0].plot(t1, X[i, :], c=colors[i])
            axs[i,0].set_title(labels[i])
            axs[i,0].grid(True)
            axs[i,1].plot(t2, Y[i, :], c=colors[i])
            axs[i,1].set_title(labels[i])
            axs[i,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def multi_trajectoire(M, fonction_type, N, Tech, sigma2, alpha=None, PLOT=True):
    """
    Parameters
    ----------
    M : INT
        Nombre de Trajectoire. 
    fonction_type : Fonction(N, Tech, sigma, (optional) alpha)
        Fonction générant un vecteur de point selon un type prédéterminé 
        (MRU, MUA, Singer).
    N : INT
        Taille des échantillons.
    Tech : FLOAT
        Temps d'échantillonnage du modèle.
    sigma2 : FLOAT
        Variance du bruit blanc de génération.
    alpha : FLOAT, optional
        UNIQUEMENT POUR SINGER
        Inverse du temps de manoeuvre. 
        The default is None (not used).

    Returns
    -------
    X_mat : Matrix size M*N of FLOAT
        Matrices des coordonées X de chaques itérations. 
    Y_mat : Matrix size M*N of FLOAT
        Matrices des coordonées Y de chaques itérations. 
        
    """
    X_mat = np.zeros((M,N))
    Y_mat = np.zeros((M,N))
    for m in range(M):
        t1, X, t2, Y = trajectoire_XY(fonction_type, N, Tech, sigma2, alpha)
        X_mat[m, :], Y_mat[m, :] = X[0, :], Y[0, :]
        if PLOT :
            plot_trajectoire(X[0, :], Y[0, :], 
                             True if m == 0 else False, 
                             True if m == M-1 else False)
    return X_mat, Y_mat
        
        