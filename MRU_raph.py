import numpy as np
import matplotlib.pyplot as plt

# Paramètres
sigma_carre = 1
Tech = 1
x_initial = np.array([0, 0])
target = "positions_vitesses.npy"

# Matrice Q et A
Q = sigma_carre * np.array([[Tech**3 / 3, Tech**2 / 2], [Tech**2 / 2, Tech]])
A = np.array([[1, Tech], [0, 1]])




def generate_bruit_model(Q, nbr_valeurs):
    """
    Génére nbr_valeurs aléatoires gaussiennes caractèrisées par une esperance nulle et une matrice de corrélation Q
    """

    # Vérifier si Q est définie positive
    if np.all(np.linalg.eigvals(Q) <= 0):
        raise ValueError("La matrice Q doit être définie positive.")

    # Décomposition de Cholesky : Q = D * D^T
    D = np.linalg.cholesky(Q)

    z = np.random.randn(nbr_valeurs)  # Bruit gaussien standard

    # Appliquer la transformation pour avoir la corrélation Q
    x = z @ D.T  # x suit une distribution N(0, Q)

    return x


def get_next_x(vecteur_x):
    """
    A partir d'un couple position, vitesse, générère aléatoirement la prochaine position 
    de façon aléatoire suivant le modèle indiqué par les matrices A et Q
    """
    
    return A@vecteur_x + generate_bruit_model(Q, 2).T


def generate_traj(x_initial, N, liste_positions=[]):
    """
    A partir d'un couple position, vitesse intial, génère une trajectoire de N valeurs 
    suivatnt le modèle indiqué par Q et A
    """
    if N==0:
        return liste_positions
    else:
        new_x = get_next_x(x_initial)
        liste_positions.append(x_initial)
        return generate_traj(new_x, N-1, liste_positions)    


def plot_data(liste_positions):
    """
    Affiche une trajectoire indiqué par des couples positions, vitesses
    """
    # Extraction des positions et des vitesses
    positions = [item[0] for item in liste_positions]
    vitesses = [item[1] for item in liste_positions]

    # Création des indices temporels (ou tout autre axe pertinent)
    temps = np.arange(len(liste_positions))

    # Création des sous-graphes
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Graphique de la position
    axs[0].plot(temps, positions, label="Position", color="blue")
    axs[0].set_title("Évolution de la position")
    axs[0].set_ylabel("Position")
    axs[0].grid()
    axs[0].legend()

    # Graphique de la vitesse
    axs[1].plot(temps, vitesses, label="Vitesse", color="red")
    axs[1].set_title("Évolution de la vitesse")
    axs[1].set_xlabel("Temps (ou indice)")
    axs[1].set_ylabel("Vitesse")
    axs[1].grid()
    axs[1].legend()

    # Ajuster les espaces entre les graphiques
    plt.tight_layout()

    # Affichage
    plt.show()
    
    
def save_data(liste_positions, target, format="npy"):
    """
    Sauvegarde une trajectoie dans un fichier
    """
    if format == "npy":
        np.save(target, liste_positions)
    else:
        raise ValueError("Format de sauvegarde des données non pris en charge")
    
    
def main_MRU(nbr_traj, max_vitesse, nbr_points, target):
    """
    Générère nbr_traj trajectoires à partir d'une position initiale et d'une vitesse intilae aléatoire
    composés de nbr_ponts chacunes suivant le modèle MRU
    """
    
    for i in range(nbr_traj):
        target_name = target + str(i)
        vitesse_intiale = random_float = np.random.uniform(0, max_vitesse)
        x_initial = np.array([0, vitesse_intiale])
        liste_pos = generate_traj(x_initial, nbr_points)
        save_data(liste_pos, target_name)
    


liste_positions = generate_traj(x_initial, 500)
plot_data(liste_positions)