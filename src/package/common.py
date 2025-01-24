###########################################
# Fonctions en commun pour les différentes 
# méthodes de génération de trajectoire
###########################################

import matplotlib.pyplot as plt

def trajectoire_XY(fonction_type, N, Tech, sigma, alpha=None):
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
    sigma : FLOAT
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
        t1, X = fonction_type(N, Tech, sigma)
        t2, Y = fonction_type(N, Tech, sigma)
    else :
        t1, X = fonction_type(N, Tech, sigma, alpha)
        t2, Y = fonction_type(N, Tech, sigma, alpha)
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
    
def multi_trajectoire(M, fonction_type, N, Tech, sigma, alpha=None):
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
    sigma : FLOAT
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
    for m in range(M):
        t1, X, t2, Y = trajectoire_XY(fonction_type, N, Tech, sigma, alpha)
        plot_trajectoire(X[0, :], Y[0, :], 
                         True if m == 0 else False, 
                         True if m == M-1 else False)
    return t1, X, t2, Y
        
        