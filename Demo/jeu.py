import pygame
from enum import Enum
from faker import Faker
import numpy as np
import cv2
import random
import trajectories


# Variables Globales
BACKGROUND_IMAGE = "background_english_channel.png"  # À remplacer par ton image
NBRBOAT = 3
COEFFNORM = 3

# Initialisation de Pygame et Faker
pygame.init()
fake = Faker('fr_FR')

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



# Chargement de l'image de fond
background = pygame.image.load(BACKGROUND_IMAGE)
image = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_GRAYSCALE)
WIDTH, HEIGHT = background.get_width(), background.get_height()

# Création de la fenêtre avec la taille de l'image
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Police
font = pygame.font.Font(None, 24)


class BateauType(Enum):
    CARGO = "Cargo"
    VOILIER = "Voilier"
    PECHE = "Bateau de pêche"

# Classe pour les infos d'un bateau
class Boat:
    def __init__(self, vitesse, boat_type, trajectoire, color):
        self.name = fake.first_name()
        self.vitesse = vitesse
        self.type = boat_type
        self.trajectoire = trajectoire
        self.color = color
        
    def toString(self):
        return self.name + "\nVitesse: " + str(self.vitesse) + "\nType: " + self.type.value
    
        

# Classe pour les cartes d'information
class InfoCard:
    def __init__(self, x, y, bateau):
        self.x = x
        self.y = y
        self.text = bateau.toString()
        self.width = 150
        self.height = 100

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        lines = self.text.split("\n")
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (self.x + 5, self.y + 5 + i * 20))
            
            
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


liste_bateaux = []
info_cards = []


# Générer les bateaux et leurs trajectoires à afficher
for i in range(NBRBOAT):
    
    # Génération de la trajectoire
    start_point = (500/COEFFNORM, 400/COEFFNORM)
    velocity = (3, 1)
    num_points = 500//COEFFNORM
    traj = trajectories.generate_mru_trajectory(start_point, velocity, num_points)
    traj_norm = traj*COEFFNORM
    
    # Génération du bateau
    nom_couleur, rgb = couleurs[i]
    bateau = Boat(30, BateauType.CARGO, traj_norm, rgb)
    liste_bateaux.append(bateau)
    
    # Génération de la carte d'infos
    card = InfoCard(200*i, 100, bateau)
    info_cards.append(card)
    

# Boucle principale
running = True
while running:
    screen.blit(background, (0, 0))
    
    # Dessiner la trajectoire (lignes entre les points)
    i = 0
    for bateau in liste_bateaux:
        i += 1
        
        # Dessiner la trajectoire
        trajectory = bateau.trajectoire
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                pygame.draw.line(screen, bateau.color, trajectory[i], trajectory[i + 1], 5)  # rouge


    # Dessiner les cartes d'info
    for card in info_cards:
        card.draw(screen)

    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
