import numpy as np
import pandas as pd
from numpy.random import randn
import os
import matplotlib.pyplot as plt
import seaborn as sns

##############################
# PARTIE 1: GÉNÉRATION DES TRAJECTOIRES
##############################

def compute_Q_matrix(sigma_w2, alpha, Tech):
    """
    Calcule la matrice de covariance du bruit de processus pour un modèle d'accélération Singer.
    
    Paramètres:
        sigma_w2 (float): Variance du bruit blanc
        alpha (float): Paramètre du modèle de manœuvre Singer
        Tech (float): Période d'échantillonnage
        
    Retourne:
        Q (ndarray): Matrice de covariance 3x3
    """
    at = alpha * Tech
    q11 = (1 / (2 * alpha ** 5)) * (
        1 - np.exp(-2 * at)
        + 2 * at
        + (2 * at ** 3) / 3
        - 2 * at ** 2
        - 4 * at * np.exp(-at)
    )
    
    q12 = (1 / (2 * alpha ** 4)) * (
        1 + np.exp(-2 * at)
        - 2 * np.exp(-at)
        + 2 * at * np.exp(-at)
        - 2 * at 
        + at ** 2
    )
    
    q13 = (1 / (2 * alpha ** 3)) * (
        1 - np.exp(-2 * at) - 2 * at * np.exp(-at)
    )

    q22 = (1 / (2 * alpha ** 3)) * (
        4 * np.exp(-at) - 3 - np.exp(-2 * at) + 2 * at
    )
    
    q23 = (1 / (2 * alpha ** 2)) * (
        np.exp(-2 * at) + 1 - 2 * np.exp(-at)
    )

    q33 = (1 / (2 * alpha)) * (1 - np.exp(-2 * at))

    Q = sigma_w2 * np.array([
        [q11, q12, q13],
        [q12, q22, q23],
        [q13, q23, q33]
    ])
    
    return Q

def traj_singer(N, Tech, sigma_w2, alpha):
    """
    Génère une trajectoire Singer pour une coordonnée (X ou Y)
    
    Paramètres:
        N (int): Nombre de points de la trajectoire
        Tech (float): Période d'échantillonnage
        sigma_w2 (float): Variance du bruit blanc
        alpha (float): Paramètre du modèle Singer
        
    Retourne:
        X[0] (ndarray): Séquence des positions
    """
    # Calcul de la matrice de covariance du bruit
    Q = compute_Q_matrix(sigma_w2, alpha, Tech)
    L = np.linalg.cholesky(Q)

    # Matrice de transition d'état pour le modèle Singer
    A = np.array([
        [1, Tech, 1/(alpha**2)*(-1 + alpha*Tech + np.exp(-alpha*Tech))],
        [0, 1, 1/alpha*(1 - np.exp(-alpha*Tech))],
        [0, 0, np.exp(-alpha*Tech)]
    ])

    # Génération de la trajectoire par simulation du système dynamique
    X = np.zeros((3, N))
    for k in range(N-1):
        w = L @ np.random.randn(3, 1)  # Bruit de processus
        X[:, k+1] = A @ X[:, k] + w[:,0]  # Propagation d'état

    return X[0]  # Retourne uniquement les positions

def Trajec_MUA(N, Tech, sigma):
    """
    Génère une trajectoire MUA (Mouvement Uniformément Accéléré) pour une coordonnée (X ou Y)
    
    Paramètres:
        N (int): Nombre de points de la trajectoire
        Tech (float): Période d'échantillonnage
        sigma (float): Écart-type du bruit
        
    Retourne:
        X[0] (ndarray): Séquence des positions
    """
    # Matrice de transition d'état pour le modèle MUA
    A = np.array([[1, Tech, Tech**2/2],
                  [0, 1, Tech],
                  [0, 0, 1]])
    
    # Matrice de covariance du bruit de processus
    Q = sigma**2 * np.array([[Tech**5/20, Tech**4/8, Tech**3/6],
                            [Tech**4/8, Tech**3/3, Tech**2/2],
                            [Tech**3/6, Tech**2/2, Tech]])
    
    L = np.linalg.cholesky(Q)
    
    # Génération de la trajectoire
    X = np.zeros((3, N))
    for k in range(N-1):
        w = L @ np.random.randn(3, 1)  # Bruit de processus
        X[:, k+1] = A @ X[:, k] + w[:,0]  # Propagation d'état
    
    return X[0]  # Retourne uniquement les positions

def trajec_MRU(N, Tech, sigma):
    """
    Génère une trajectoire MRU (Mouvement Rectiligne Uniforme) pour une coordonnée (X ou Y)
    
    Paramètres:
        N (int): Nombre de points de la trajectoire
        Tech (float): Période d'échantillonnage
        sigma (float): Écart-type du bruit
        
    Retourne:
        X[0] (ndarray): Séquence des positions
    """
    # Matrice de transition d'état pour le modèle MRU
    A = np.array([[1, Tech],
                  [0, 1]])
    
    # Matrice de covariance du bruit de processus
    Q = sigma * np.array([[Tech**3/3, Tech**2/2],
                         [Tech**2/2, Tech]])
    
    D = np.linalg.cholesky(Q)
    
    # Génération de la trajectoire
    X = np.zeros((2, N))
    for k in range(N-1):
        w = D @ randn(2, 1)  # Bruit de processus
        X[:, k+1] = A @ X[:, k] + w[:,0]  # Propagation d'état
    
    return X[0]  # Retourne uniquement les positions

def format_trajectory_for_csv(trajectory):
    """
    Formate une trajectoire numpy pour qu'elle soit correctement interprétée
    par la fonction string_to_numpy_array.
    
    Paramètres:
        trajectory (ndarray): Trajectoire à formater (points 2D)
        
    Retourne:
        formatted (str): Chaîne formatée représentant la trajectoire
    """
    formatted = "["
    for i, point in enumerate(trajectory):
        formatted += f"[{point[0]} {point[1]}]"
        if i < len(trajectory) - 1:
            formatted += ", "
    formatted += "]"
    return formatted

def plot_parameter_distributions(data):
    """
    Génère des graphiques de distribution pour les paramètres sigma et alpha des trajectoires
    
    Paramètres:
        data (dict): Dictionnaire contenant les paramètres des trajectoires
    """
    # Créer un dossier pour les graphiques
    output_dir = "trajectoires_data/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame(data)
    
    # Définir une palette de couleurs par type de trajectoire
    palette = {"MRU": "blue", "MUA": "green", "Singer": "red"}
    
    # 1. Distribution des paramètres sigma pour MRU et MUA
    plt.figure(figsize=(10, 6))
    sigma_data = df[df['param_sigma'].notna()].copy()
    sns.histplot(data=sigma_data, x='param_sigma', hue='type', kde=True, palette=palette)
    plt.title('Distribution du paramètre sigma par type de trajectoire')
    plt.xlabel('Sigma')
    plt.ylabel('Fréquence')
    plt.savefig(os.path.join(output_dir, 'sigma_distribution.png'))
    
    # 2. Distribution du paramètre alpha pour Singer
    plt.figure(figsize=(10, 6))
    singer_data = df[df['type'] == 'Singer'].copy()
    sns.histplot(data=singer_data, x='param_alpha', kde=True, color=palette['Singer'])
    plt.title('Distribution du paramètre alpha pour les trajectoires Singer')
    plt.xlabel('Alpha')
    plt.ylabel('Fréquence')
    plt.savefig(os.path.join(output_dir, 'alpha_distribution.png'))
    
    print(f"Graphiques sauvegardés dans le dossier '{output_dir}'")

def generate_trajectories(nb_traj=15000):
    """
    Génère les trajectoires avec paramètres variables
    
    Paramètres:
        nb_traj (int): Nombre total de trajectoires à générer
        
    Retourne:
        output_dir (str): Chemin vers le dossier contenant les fichiers générés
    """
    N = 50  # Nombre de points par trajectoire
    Tech = 1  # Période d'échantillonnage
    nb_par_type = nb_traj // 3  # Nombre de trajectoires par type

    # Structure pour stocker les trajectoires et leurs caractéristiques
    data = {
        'type': [],
        'param_sigma': [],
        'param_alpha': [],
        'param_sigma_w2': [],
        'noise_std': []  # paramètre pour stocker le bruit ajouté
    }

    # Liste pour stocker les trajectoires bruitées formatées
    trajectoires_bruitees = []

    # Génération MRU (Mouvement Rectiligne Uniforme)
    print("Génération des trajectoires MRU...")
    for i in range(nb_par_type):
        # Générer les trajectoires avec sigma variable
        sigma = np.random.uniform(2, 4)  # Varier sigma entre 2 et 4
        
        # Générer un bruit variable pour chaque trajectoire
        noise_std = np.random.uniform(0.1, 1.5)  # Varier l'écart-type du bruit
        
        # Générer la trajectoire
        X = trajec_MRU(N, Tech, sigma)
        Y = trajec_MRU(N, Tech, sigma)
        trajectory = np.column_stack((X, Y))
        
        # Ajouter du bruit
        noisy_trajectory = trajectory + np.random.normal(0, noise_std, (N, 2))
        
        # Formater et stocker la trajectoire bruitée
        formatted_traj = format_trajectory_for_csv(noisy_trajectory)
        trajectoires_bruitees.append(formatted_traj)
        
        # Stocker les métadonnées
        data['type'].append('MRU')
        data['param_sigma'].append(sigma)
        data['param_alpha'].append(None)
        data['param_sigma_w2'].append(None)
        data['noise_std'].append(noise_std)

    # Génération MUA (Mouvement Uniformément Accéléré)
    print("Génération des trajectoires MUA...")
    for i in range(nb_par_type):
        # Générer les trajectoires avec sigma variable
        sigma = np.random.uniform(2, 4)  # Varier sigma entre 2 et 4
        
        # Générer un bruit variable pour chaque trajectoire
        noise_std = np.random.uniform(0.1, 1.5)  # Varier l'écart-type du bruit 
        
        # Générer la trajectoire
        X = Trajec_MUA(N, Tech, sigma)
        Y = Trajec_MUA(N, Tech, sigma)
        trajectory = np.column_stack((X, Y))
        
        # Ajouter du bruit
        noisy_trajectory = trajectory + np.random.normal(0, noise_std, (N, 2))
        
        # Formater et stocker la trajectoire bruitée
        formatted_traj = format_trajectory_for_csv(noisy_trajectory)
        trajectoires_bruitees.append(formatted_traj)
        
        # Stocker les métadonnées
        data['type'].append('MUA')
        data['param_sigma'].append(sigma)
        data['param_alpha'].append(None)
        data['param_sigma_w2'].append(None)
        data['noise_std'].append(noise_std)

    # Génération Singer avec variation des paramètres
    print("Génération des trajectoires Singer...")
    for i in range(nb_par_type):
        # Utiliser différentes valeurs pour alpha et sigma_m2
        alpha = np.random.uniform(1/500, 1/100)
        sigma_m2 = np.random.uniform(2e-5, 5e-5)
        sigma_w2 = 2 * alpha * sigma_m2
        
        # Générer un bruit variable pour chaque trajectoire
        noise_std = np.random.uniform(0.1, 1.5)  # Varier l'écart-type du bruit
        
        # Générer la trajectoire
        X = traj_singer(N, Tech, sigma_w2, alpha)
        Y = traj_singer(N, Tech, sigma_w2, alpha)
        trajectory = np.column_stack((X, Y))
        
        # Ajouter du bruit
        noisy_trajectory = trajectory + np.random.normal(0, noise_std, (N, 2))
        
        # Formater et stocker la trajectoire bruitée
        formatted_traj = format_trajectory_for_csv(noisy_trajectory)
        trajectoires_bruitees.append(formatted_traj)
        
        # Stocker les métadonnées
        data['type'].append('Singer')
        data['param_sigma'].append(None)
        data['param_alpha'].append(alpha)
        data['param_sigma_w2'].append(sigma_w2)
        data['noise_std'].append(noise_std)

    # Créer le DataFrame pour les métadonnées
    df_meta = pd.DataFrame(data)

    # Créer le DataFrame pour les trajectoires bruitées
    df_bruitee = pd.DataFrame({
        'trajectoire_bruitées': trajectoires_bruitees
    })

    # Définir les noms de fichiers et créer le dossier de sortie
    output_dir = "trajectoires_data"
    os.makedirs(output_dir, exist_ok=True)

    meta_file = os.path.join(output_dir, "trajectoires_meta.csv")
    bruitees_file = os.path.join(output_dir, "trajectoires_bruitees.csv")
    labels_file = os.path.join(output_dir, "labels.csv")

    # Sauvegarde des fichiers
    print(f"Sauvegarde des données dans le dossier '{output_dir}'...")

    try:
        df_meta.to_csv(meta_file, index=False)
        print(f"✓ Fichier sauvegardé: {meta_file}")
    except Exception as e:
        print(f"⚠ Erreur lors de la sauvegarde de {meta_file}: {str(e)}")

    try:
        df_bruitee.to_csv(bruitees_file, index=False)
        print(f"✓ Fichier sauvegardé: {bruitees_file}")
    except Exception as e:
        print(f"⚠ Erreur lors de la sauvegarde de {bruitees_file}: {str(e)}")

    # Créer un fichier avec les labels numériques pour faciliter l'entrainement
    try:
        type_to_num = {'MRU': 0, 'MUA': 1, 'Singer': 2}
        df_meta['label_num'] = df_meta['type'].map(type_to_num)
        df_meta[['type', 'label_num']].to_csv(labels_file, index=False)
        print(f"✓ Fichier sauvegardé: {labels_file}")
    except Exception as e:
        print(f"⚠ Erreur lors de la sauvegarde de {labels_file}: {str(e)}")

    # Afficher les statistiques des données générées
    print(f"\nTotal des échantillons: {len(df_meta)}")
    print(f"Répartition des classes: MRU: {df_meta['type'].value_counts()['MRU']}, "
          f"MUA: {df_meta['type'].value_counts()['MUA']}, "
          f"Singer: {df_meta['type'].value_counts()['Singer']}")

    # Vérifier le premier exemple
    print("\nVérification du format des trajectoires bruitées:")
    print(f"Exemple: {trajectoires_bruitees[0][:100]}...")
    
    # Générer des graphiques de distribution
    plot_parameter_distributions(data)
    
    return output_dir


##############################
# PARTIE 2: CALCUL DES AUTOCORRÉLATIONS
##############################

def string_to_numpy_array(s):
    """
    Convertit la chaîne de caractères représentant un array numpy en array numpy
    Format attendu : [[x1 y1], [x2 y2], ..., [xn yn]]
    
    Paramètres:
        s (str): Chaîne de caractères à convertir
        
    Retourne:
        points (ndarray): Tableau numpy des points 2D
    """
    # Nettoyer la chaîne
    s = s.strip('[]')
    # Diviser en lignes
    rows = s.split(']')
    # Convertir chaque ligne en array de deux nombres
    points = []
    for row in rows:
        # Nettoyer la ligne
        row = row.strip('[] \n,')
        if row:  # Ignorer les lignes vides
            # Extraire les deux nombres de la ligne
            nums = [float(x) for x in row.split()]
            if len(nums) == 2:  # S'assurer qu'on a bien deux coordonnées
                points.append(nums)
    return np.array(points)

def compute_acceleration(X, Tech=1):
    """
    Estime l'accélération à partir des positions
    
    Paramètres:
        X (ndarray): Tableau des positions
        Tech (float): Période d'échantillonnage
        
    Retourne:
        a (ndarray): Tableau des accélérations estimées
    """
    a = (X[2:] - 2*X[1:-1] + X[:-2]) / Tech**2
    return a

def compute_jerk(X, Tech=1):
    """
    Estime le jerk (dérivée de l'accélération) à partir des positions
    
    Paramètres:
        X (ndarray): Tableau des positions
        Tech (float): Période d'échantillonnage
        
    Retourne:
        j (ndarray): Tableau des jerks estimés
    """
    a = compute_acceleration(X, Tech)  # Calcul de l'accélération
    j = (a[1:] - a[:-1]) / Tech  # Dérivée de l'accélération
    return j

def autocorrelation(signal):
    """
    Calcule la fonction d'autocorrélation normalisée
    
    Paramètres:
        signal (ndarray): Signal à analyser
        
    Retourne:
        result (ndarray): Fonction d'autocorrélation (partie positive des décalages)
    """
    N = len(signal)
    result = np.correlate(signal, signal, mode='full') / N
    return result[result.size//2:]  # Retourne seulement la partie positive (lags >= 0)



def process_trajectories(filepath):
    """
    Charge les trajectoires et calcule les autocorrélations
    
    Paramètres:
        filepath (str): Chemin vers le fichier CSV contenant les trajectoires
        
    Retourne:
        features_matrices (list): Liste des matrices de caractéristiques d'autocorrélation
    """
    # Charger les données
    df = pd.read_csv(filepath)
    
    # Convertir les trajectoires en arrays numpy
    features_matrices = []
    errors = 0
    
    print(f"Traitement de {len(df)} trajectoires...")
    
    # Traiter chaque trajectoire
    for i, row in enumerate(df.iterrows()):
        # Afficher une indication de progression tous les 1000 échantillons
        if i % 1000 == 0:
            print(f"Progression: {i}/{len(df)} trajectoires traitées")
            
        try:
            # Convertir la trajectoire en array numpy
            _, row_data = row  # Désempaqueter le tuple (index, données)
            traj = string_to_numpy_array(row_data['trajectoire_bruitées'])
            
            # Pour X et Y
            acc_autocorr = []
            jerk_autocorr = []
            
            for dim in [0, 1]:  # 0 pour X, 1 pour Y
                # Calcul de l'accélération et son autocorrélation
                acc = compute_acceleration(traj[:, dim])
                acc_corr = autocorrelation(acc)
                acc_autocorr.append(acc_corr[:4])
                
                # Calcul du jerk et son autocorrélation
                jerk = compute_jerk(traj[:, dim])
                jerk_corr = autocorrelation(jerk)
                jerk_autocorr.append(jerk_corr[:4])
            
            # Moyenner les résultats X et Y
            mean_acc_autocorr = np.mean(acc_autocorr, axis=0)
            mean_jerk_autocorr = np.mean(jerk_autocorr, axis=0)
            
            # Créer la matrice 2x4 pour cette trajectoire
            feature_matrix = np.vstack((mean_acc_autocorr, mean_jerk_autocorr))
            features_matrices.append(feature_matrix)
            
        except Exception as e:    # ne sert a rien 
            continue
    
    print(f"Traitement terminé. {len(features_matrices)} trajectoires traitées avec succès.")
    
    if errors > 0:
        print(f"Total des erreurs: {errors} sur {len(df)} trajectoires")
    
    return np.array(features_matrices)

def calculate_autocorrelations(data_dir):
    """
    Calcule les autocorrélations pour les trajectoires
    
    Paramètres:
        data_dir (str): Chemin vers le dossier contenant les données
        
    Retourne:
        (int): Nombre de trajectoires traitées avec succès
    """
    input_file = os.path.join(data_dir, "trajectoires_bruitees.csv")
    output_file = os.path.join(data_dir, "autocorrelations_matrices.npy")
    
    print(f"Chargement des trajectoires depuis {input_file}...")
    
    # Charger et traiter les données
    features_matrices = process_trajectories(input_file)
    
    print("\nShape des résultats:", features_matrices.shape)
    print("\nPour chaque trajectoire, nous avons une matrice 2x4:")
    print("- Ligne 1: Autocorrélations de l'accélération [Raa(0), Raa(1), Raa(2), Raa(3)]")
    print("- Ligne 2: Autocorrélations du jerk [Rjj(0), Rjj(1), Rjj(2), Rjj(3)]")
    
    if len(features_matrices) > 0:
        print("\nExemple pour la première trajectoire:")
        print(features_matrices[0])
    
    
    # Sauvegarder les résultats
    np.save(output_file, features_matrices)
    print(f"\nRésultats sauvegardés dans '{output_file}'")
    
    return features_matrices.shape[0]


##############################
# MAIN
##############################

if __name__ == "__main__":
    # Étape 1: Générer les trajectoires
    print("=== ÉTAPE 1: GÉNÉRATION DES TRAJECTOIRES ===")
    data_dir = generate_trajectories(nb_traj=15000)  # 5000 par type = 15000 total
    
    # Étape 2: Calculer les autocorrélations
    print("\n=== ÉTAPE 2: CALCUL DES AUTOCORRÉLATIONS ===")
    nb_processed = calculate_autocorrelations(data_dir)
    
    print(f"\n=== PROCESSUS TERMINÉ ===")
    print(f"Nombre total de trajectoires traitées avec succès: {nb_processed}")
    print(f"Les données sont prêtes pour l'entraînement!")