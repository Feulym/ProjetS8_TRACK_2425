import random
import numpy as np


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



def generate_mua_trajectory(start, velocity, acceleration, num_points):
    """Génère une trajectoire MUA en 2D."""
    trajectory = [start]
    for _ in range(num_points - 1):
        velocity = (velocity[0] + acceleration[0], velocity[1] + acceleration[1])
        next_point = (trajectory[-1][0] + velocity[0], trajectory[-1][1] + velocity[1])
        trajectory.append(next_point)
    return np.array(trajectory)


def generate_singer_trajectory(start, velocity, damping, num_points):
    """Génère une trajectoire Singer en 2D."""
    trajectory = [start]
    acceleration = (random.uniform(-1, 1), random.uniform(-1, 1))  # Accélération initiale aléatoire
    for _ in range(num_points - 1):
        acceleration = (acceleration[0] * damping, acceleration[1] * damping)
        velocity = (velocity[0] + acceleration[0], velocity[1] + acceleration[1])
        next_point = (trajectory[-1][0] + velocity[0], trajectory[-1][1] + velocity[1])
        trajectory.append(next_point)
    return np.array(trajectory)











if __name__ == "__main__":

    # Test avec un point de départ et des paramètres arbitraires
    start_point = (100, 100)
    velocity = (5, 2)
    acceleration = (0.2, 0.1)
    damping = 0.9
    num_points = 50

    mru_test = generate_mru_trajectory(start_point, velocity, num_points)
    mua_test = generate_mua_trajectory(start_point, velocity, acceleration, num_points)
    singer_test = generate_singer_trajectory(start_point, velocity, damping, num_points)

    mru_test.shape, mua_test.shape, singer_test.shape

