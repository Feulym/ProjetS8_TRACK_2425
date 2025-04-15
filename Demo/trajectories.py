import random
import numpy as np
import math


ECHELLE = 1 # 1 pixel sur l'image correspond à combien de km
TE = 1  # Période d'échantillonage (temps entre 2 échantillons) (nécessaire pour claucler la vitesse)


def generate_mru_trajectory(start, initial_velocity, num_points, sigma=0.5):
    """
    Génère une trajectoire MRU perturbée par du bruit blanc gaussien sur l'accélération.
    
    - `start`: (x, y) point de départ
    - `initial_velocity`: (vx, vy) vitesse de départ
    - `sigma`: écart-type du bruit blanc gaussien
    """
    trajectory = [start]
    velocity = np.array(initial_velocity, dtype=float)

    for _ in range(num_points - 1):
        # bruit blanc gaussien sur l'accélération
        acceleration = np.random.normal(loc=0.0, scale=sigma, size=2)
        velocity += acceleration
        next_point = trajectory[-1] + velocity
        trajectory.append(next_point)
    
    return np.array(trajectory)




def generate_mua_trajectory(start, velocity, num_points, acceleration=(0, 0), jerk_std=0.05):
    """Génère une trajectoire MUA en 2D avec un jerk gaussien."""
    trajectory = [start]
    velocities = [velocity]
    accelerations = [acceleration]
    jerk_std = jerk_std/20

    for i in range(num_points - 1):
        # Générer un jerk gaussien pour chaque composante (x, y)
        jerk = np.random.normal(0, jerk_std, size=2)
        
        if i % 50 == 0:
            new_acceleration = (0, 0)
        
        # Mettre à jour l'accélération
        new_acceleration = (accelerations[-1][0] + jerk[0], accelerations[-1][1] + jerk[1])
        accelerations.append(new_acceleration)
        
        # Mettre à jour la vitesse
        new_velocity = (velocities[-1][0] + new_acceleration[0], velocities[-1][1] + new_acceleration[1])
        velocities.append(new_velocity)
        
        # Mettre à jour la position
        next_point = (trajectory[-1][0] + new_velocity[0], trajectory[-1][1] + new_velocity[1])
        trajectory.append(next_point)

    return np.array(trajectory)



def generate_singer_trajectory(start, velocity, num_points, damping=0.5, noise_std=0.5):
    """Génère une trajectoire Singer en 2D avec accélérations aléatoires amorties."""
    trajectory = [start]
    acceleration = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])  # Accélération initiale

    for _ in range(num_points - 1):
        # Mise à jour de l'accélération avec amortissement et bruit blanc
        jerk = np.random.normal(0, noise_std, size=2)
        acceleration = damping * acceleration + jerk

        # Mise à jour de la vitesse
        velocity = velocity + acceleration

        # Mise à jour de la position
        next_point = trajectory[-1] + velocity
        trajectory.append(next_point)

    return np.array(trajectory)



def generate_random_trajectory(start, velocity, num_points, sigma=0.5):
    """Génère une trajectoire dont le modèle est choisi de façon aléatoire

    Args:
        start (_type_): _description_
        velocity (_type_): _description_
        num_points (_type_): _description_
        sigma (float, optional): _description_. Defaults to 0.5.
    """
    x = random.randint(1, 3)
    
    if x==1:
        traj = generate_mru_trajectory(start, velocity, num_points, sigma)
    
    elif x==2:
        traj = generate_mua_trajectory(start, velocity, num_points, jerk_std=sigma)
        
    else:
        traj = generate_singer_trajectory(start, velocity, num_points, noise_std=sigma)
        
    return traj


def allinone(start, velocity, num_points, sigma, nbr_sous_traj=4):
    """Génère de manière récursive une trajectoire de num_points composée de nbr_sous_traj sous-trajectoires qui suivent 
        chacun un des 3 modèles définit de façon aléatoire

    Args:
        start (_type_): _description_
        velocity (_type_): _description_
        num_points (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if nbr_sous_traj == 0:
        return np.array([])
    
    # Choix aléatoiresdes indices ou on changera de type de trajectoire
    indices = random.sample(range(2, num_points), nbr_sous_traj)
    indices.sort()
    if(len(indices) == 1):
        indices.append(num_points - 1)
    nbr_points = indices[1] - indices[0]
    traj = generate_random_trajectory(start, velocity, nbr_points, sigma)
    
    new_velocity = calc_vitesse(traj, get_2dvit=True)
    new_start = traj[-1]
    
    # Génération récursive de la prochaine trajectoire
    new_traj = allinone(new_start, new_velocity, num_points-nbr_points, sigma, nbr_sous_traj - 1)
    
    # Terminaison
    if(new_traj.size == 0):
        return traj
    else:
        return np.vstack((traj, new_traj))





# Calcule la vitesse à chaque instant en noeuds à partir d'une trajectoire
def calc_vitesse(trajectory, batch_size=1, get_2dvit=False):
    
    liste_vitesses = [0 for _ in range(len(trajectory))]
    
    for ii in range(1, len(trajectory)):
        x1, y1 = trajectory[ii - 1]
        x2, y2 = trajectory[ii]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)   # Calcul de la distance entre les 2 points successifs (en pixels)
        vitesse = distance / TE                                 # Calcul de la vitesse en m/s
        vitesse_noeuds = vitesse * 3600 / 1852                  # Conversion en noeuds
        liste_vitesses[ii] = round(vitesse_noeuds)
        
    if get_2dvit:
        return (x2 - x1, y2-y1)
        
    return liste_vitesses


def calc_vitesse_moyenne(liste_vitesses):
     
    moyennes = []
    somme = 0
    for i, v in enumerate(liste_vitesses):
        somme += v
        moyennes.append(somme / (i + 1))
        
    return moyennes



if __name__ == "__main__":

    # Test avec un point de départ et des paramètres arbitraires
    start_point = (100, 100)
    velocity = (5, 2)
    acceleration = (0.2, 0.1)
    damping = 0.9
    num_points = 50

    mru_test = generate_mru_trajectory(start_point, velocity, num_points)
    mua_test = generate_mua_trajectory(start_point, velocity, num_points)
    singer_test = generate_singer_trajectory(start_point, velocity, num_points)
    allinone_test = allinone(start_point, velocity, num_points, 0.5, 4)

    print(mru_test.shape, mua_test.shape, singer_test.shape, allinone_test.shape)

