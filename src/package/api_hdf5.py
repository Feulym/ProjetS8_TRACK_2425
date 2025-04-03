import h5py
import os
import numpy as np


"""
Bibliothèque de gestion des données de trajectoires pour l'apprentissage automatique
====================================================================================

Ce module permet de gérer des trajectoires (listes de positions 2D) associées à des classes 
pour les sauvegarder, les charger et les manipuler dans un fichier HDF5. 
Il est conçu pour servir de base à des projets d'apprentissage automatique ou de classification.

Chaque trajectoire est une séquence de points (x, y) représentée par un tableau NumPy 2D. 
Les classes associées sont des chaînes de caractères.

import api_hdf5 as hdf

Fonctionnalités disponibles :
-----------------------------
1. **Création d'une base de donnée de trajectoires HDF5**
    hdf.create_hdf5(filename, liste_trajectoires, liste_classes)
2. **Ajout de trajectoires à une base de données existante**
    hdf.add_trajectories(filename, liste_trajectoires, liste_classes)
3. **Récupération des trajectoires depuis un fichier HDF5**
    hdf.read_hdf5(filename)
4. **Obtention de statistiques variées sur la base de données**
    hdf.get_statistiques(filename, do_print=False)
"""



def create_hdf5(filename, trajectoires, classes):
    """
    Crée un fichier HDF5 contenant des trajectoires et leurs classes associées.
    
    :param filepath: Nom du fichier HDF5 à créer.
    :param liste_trajectoires: Liste de tableaux numpy, chaque tableau représentant une trajectoire.
    :param liste_classes: Liste des classes correspondantes, une par trajectoire.
    """
    
    # Vérifier la validité des entrées
    if len(trajectoires) != len(classes):
        raise ValueError("Le nombre de trajectoires doit être égal au nombre de classes.")
    
    compteur = 0

    # Création et écriture dans le fichier HDF5
    with h5py.File(filename, "w") as hdf:
        
        for traj in trajectoires:
            
            compteur += 1
            traj_name = "traj" + str(compteur)
            
            # Créer un dataset pour la trajectoire
            dset = hdf.create_dataset(traj_name, data=traj)
            # Ajouter un attribut pour stocker la classe associée
            dset.attrs["labels"] = classes[compteur-1]

    print(f"Fichier HDF5 '{filename}' contenant '{compteur}' trajectoires créé avec succès.")
    
    
    
def add_trajectories(filename, trajectoires, classes):
    """
    Ajoute des trajectoires et leurs classes associées à un fichier HDF5 existant
    
    :param filepath: Nom du fichier HDF5 à modifier.
    :param liste_trajectoires: Liste de tableaux numpy, chaque tableau représentant une trajectoire.
    :param liste_classes: Liste des classes correspondantes, une par trajectoire.
    """
    
    # Vérifier la validité des entrées
    if len(trajectoires) != len(classes):
        raise ValueError("Le nombre de trajectoires doit être égal au nombre de classes.")
    
    
    done = 0

    # Ecriture dans le fichier HDF5
    with h5py.File(filename, "a") as hdf:
        
        compteur = len(hdf.keys())  # Compte les clés à la racine
        
        for traj in trajectoires:
            
            compteur += 1
            traj_name = "traj" + str(compteur)
            
            # Créer un dataset pour la trajectoire
            dset = hdf.create_dataset(traj_name, data=traj)
            # Ajouter un attribut pour stocker la classe associée
            dset.attrs["labels"] = classes[done]
            
            done += 1

    print(f"'{done}' trajectoires ajoutées avec succès au fichier '{filename}'")
    
    
    
def read_hdf5(filename):
    """
    Charge toutes les trajectoires et leurs classes depuis un fichier HDF5.
    
    :param filepath: Chemin vers le fichier HDF5.
    :return: traj, lengths, labels
    """
    
    liste_traj = []
    liste_classes = []
    
    with h5py.File(filename, "r") as hdf:
        # Access a specific dataset
        labels = hdf["labels"][:]
        lengths = hdf["lengths"][:]
        traj = hdf["trajectories"][:]
            
    print(f"'{traj.shape[0]}' trajectoires du fichier '{filename}' lues avec succès")
            
    return traj, lengths, labels



def format_bytes(nombre_bytes):
    if nombre_bytes < 1024:
        return f"{nombre_bytes} bytes"
    elif nombre_bytes < 1024 ** 2:
        return f"{nombre_bytes / 1024:.2f} KB"
    elif nombre_bytes < 1024 ** 3:
        return f"{nombre_bytes / (1024 ** 2):.2f} MB"
    elif nombre_bytes < 1024 ** 4:
        return f"{nombre_bytes / (1024 ** 3):.2f} GB"
    else:
        return f"{nombre_bytes / (1024 ** 4):.2f} TB"



def get_statistiques(filename, do_print=False):
    """
    Analyse un fichier HDF5 et retourne des statistiques sur son contenu, incluant la taille du fichier,
    le nombre total de datasets et le nombre de datasets par classe. Si `do_print` est défini sur True,
    les statistiques seront affichées dans la console.

    Parameters:
    - filename (str): Le chemin du fichier HDF5 à analyser.
    - do_print (bool, optional): Si True, les statistiques seront affichées dans la console. Par défaut, False.

    Returns:
    - taille_fichier (int): La taille du fichier en bytes.
    - total_datasets (int): Le nombre total de datasets dans le fichier.
    - classes (dict): Un dictionnaire où les clés sont les classes trouvées dans les datasets, et les valeurs sont 
        le nombre de datasets correspondant à chaque classe.
    """
    
    
    with h5py.File(filename, 'r') as f:
        
        # Etude du poids de cette base de données
        taille_fichier = os.path.getsize(filename)
        
        # Nombre total de datasets
        total_datasets = len(f)
        
        # Compter les datasets par classe
        classes = {}
        for key in f.keys():
            dataset = f[key]
            # Supposons que la classe soit un attribut du dataset, sinon à adapter
            if 'classe' in dataset.attrs:
                classe = dataset.attrs['classe']
                if classe not in classes:
                    classes[classe] = 0
                classes[classe] += 1
        
        
    if do_print:
        print("=========STATISTIQUES========")
        print(f"Nom du fichier: {filename}")
        print("Taille du fichier: ", format_bytes(taille_fichier))
        print(f"Nombre total de trajectoires : {total_datasets}")
        # Affichage des résultats pour chaque classe
        print("Nombre de datasets par classe :")
        for classe, count in classes.items():
            print(f"  {classe} : {count}")
        print("=============================")
            
    return taille_fichier, total_datasets, classes
        




if __name__ == "__main__":
    
    print("Exemple d'utilisation")
    
    # Nom du fichier HDF5
    filename = "trajectoires.h5"
    
    # Exemple de données
    trajectoires = [np.random.rand(30, 2), np.random.rand(35, 2), np.random.rand(28, 2)]
    classes = ["MRU", "MUA", "Singer"]
    
    # Création du fichier avec des trajtcoires dedans
    create_hdf5(filename, trajectoires, classes)
    
    # Ajout de trajectoires à un fichier existant
    traj2 = [np.random.rand(30, 2), np.random.rand(35, 2), np.random.rand(28, 2)]
    classes2 = ["MRU", "MUA", "Singer"]
    add_trajectories(filename, traj2, classes2)
    
    # Lecture des données
    liste_traj, liste_class = read_hdf5(filename)
    #print(liste_traj)
    #print(liste_class)
    
    # Affichage des statistiques du dataset
    get_statistiques(filename, True)