import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_file(filepath):
    """
    Charge les données d'un fichier HDF5 et retourne 
    les trajectoires, labels et longueurs.
    """
    with h5py.File(filepath, 'r') as f:
        trajectories = f['trajectories'][:]  # forme: (n_samples, max_seq_len, 2)
        labels = f['labels'][:]              # forme: (n_samples,)
        lengths = f['lengths'][:]            # forme: (n_samples,)
    return trajectories, labels, lengths

def analyze_file(filepath):
    """
    Analyse et visualise les données d'un fichier HDF5.
    
    Pour chaque fichier, le script :
      - Extrait un exemple de trajectoire par classe (MRU, MUA, Singer),
      - Affiche ces trajectoires dans une figure,
      - Affiche un histogramme des longueurs et un histogramme de la distribution des classes.
    """
    trajectories, labels, lengths = load_data_from_file(filepath)
    
    # Dictionnaire pour les noms de classes
    class_names = {0: 'MRU', 1: 'MUA', 2: 'Singer'}
    
    # Dictionnaire pour stocker un exemple de trajectoire par classe
    traj_examples = {}
    for idx, label in enumerate(labels):
        # Si on n'a pas encore d'exemple pour cette classe, on le récupère.
        if label not in traj_examples:
            L = lengths[idx]
            # Retirer le padding en utilisant la longueur effective.
            traj = trajectories[idx][:L]
            traj_examples[label] = traj
        # Si nous avons un exemple pour chaque classe, on peut arrêter la recherche.
        if len(traj_examples) == 3:
            break

    # ------------------------------
    # Affichage des trajectoires exemples
    # ------------------------------
    n_examples = len(traj_examples)
    plt.figure(figsize=(15, 5))
    for i, (label, traj) in enumerate(traj_examples.items()):
        plt.subplot(1, n_examples, i + 1)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', linestyle='-')
        plt.title(f"Trajectoire - {class_names.get(label, str(label))}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
    plt.suptitle(f"Exemples de trajectoires - {os.path.basename(filepath)}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ------------------------------
    # Affichage des histogrammes
    # ------------------------------
    plt.figure(figsize=(12, 5))
    
    # Histogramme des longueurs des trajectoires
    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=30, edgecolor='black')
    plt.title("Histogramme des longueurs")
    plt.xlabel("Longueur")
    plt.ylabel("Fréquence")
    
    # Histogramme de la distribution des classes
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(labels, return_counts=True)
    # Récupération des noms de classes pour l'affichage
    names = [class_names.get(lbl, str(lbl)) for lbl in unique]
    plt.bar(names, counts, edgecolor='black')
    plt.title("Distribution des classes")
    plt.xlabel("Classe")
    plt.ylabel("Nombre de trajectoires")
    
    plt.suptitle(f"Histogrammes - {os.path.basename(filepath)}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    # Dossier contenant les fichiers .h5
    data_folder = 'data'
    h5_files = glob.glob(os.path.join(data_folder, '*.h5'))
    print(f"Nombre de fichiers HDF5 trouvés dans '{data_folder}': {len(h5_files)}")
    
    # Parcours de chaque fichier et analyse
    for file in h5_files:
        print(f"Analyse du fichier : {file}")
        analyze_file(file)

if __name__ == "__main__":
    main()
