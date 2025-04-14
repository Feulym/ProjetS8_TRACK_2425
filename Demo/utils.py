import numpy as np

couleurs = [
    ("Rouge", (255, 0, 0)),
    ("Vert", (0, 255, 0)),
    ("Bleu", (0, 0, 255)),
    ("Jaune", (255, 255, 0)),
    ("Cyan", (0, 255, 255)),
    ("Magenta", (255, 0, 255)),
    ("Orange", (255, 165, 0)),
    ("Rose", (255, 192, 203)),
    ("Violet", (128, 0, 128)),
    ("Gris", (128, 128, 128))
]

BUTTON_COLOR = (0, 128, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Vérifie si une trajectoire reste bien dans la zone balnche (= La mer)          
def is_trajectory_in_white_area(trajectory: np.ndarray, image: np.ndarray) -> bool:
    """
    Vérifie si toutes les positions de la trajectoire sont dans des zones blanches de l'image.
    
    :param trajectory: Liste numpy de couples (x, y) représentant la trajectoire.
    :param image: Image en niveaux de gris où les pixels blancs sont les zones valides.
    :return: True si toutes les positions sont dans le blanc, False sinon.
    """
    height, width = image.shape
    
    for x, y in trajectory:
        # Vérifier si les coordonnées sont dans les limites de l'image
        if not (0 <= x < width and 0 <= y < height):
            return False  # Une position est hors de l'image
        
        # Vérifier si le pixel correspondant est blanc (niveau de gris élevé)
        if image[int(y), int(x)] < 200:  # Seuil arbitraire pour le blanc
            return False  # Une position est dans le gris

    return True







# # Exemple d'utilisation avec une trajectoire fictive
# sample_trajectory = np.array([[500, 500], [600, 600], [700, 700]])
# result = is_trajectory_in_white_area(sample_trajectory, image)
# print(result)
