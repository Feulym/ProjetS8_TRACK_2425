import pygame
from enum import Enum
from faker import Faker
import numpy as np
import cv2
import random
import math

from utils import *
import trajectories


# Variables Globales
BACKGROUND_IMAGE = "background_english_channel.png"
NBRBOAT = 3
COEFFNORM = 3
VELOCITY = 5
DELAY = 50  # Délai entre 2 images

# Initialisation de Pygame et Faker
pygame.init()
pygame.display.set_caption("Démonstration TRACK!")
fake = Faker('fr_FR')


# Chargement de l'image de fond
background = pygame.image.load(BACKGROUND_IMAGE)
image = cv2.imread(BACKGROUND_IMAGE, cv2.IMREAD_GRAYSCALE)
WIDTH, HEIGHT = background.get_width(), background.get_height()


# Création de la fenêtre avec la taille de l'image et la police
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.Font(None, 24)   # Police


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
        self.vitesse_moyenne = 0
        
    def toString(self):
        return self.name + "\nVitesse: " + str(self.vitesse) + "\nType: " + self.type.value
      

class InfoCard:
    def __init__(self, x, y, bateau, couleur):
        self.x = x
        self.y = y
        self.bateau = bateau  # ← on garde le bateau, pas juste du texte
        self.width = 150
        self.height = 100
        self.couleur = couleur

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, self.couleur, (self.x, self.y, self.width, self.height), 2)
        
        # Texte dynamique : on relit les infos à chaque fois
        lines = [
            self.bateau.name,
            f"Vitesse: {self.bateau.vitesse:.2f} nds",
            f"Vmoy: {self.bateau.vitesse_moyenne:.2f} nds",
            f"Type: {self.bateau.type.value}"
        ]
        
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (self.x + 5, self.y + 5 + i * 20))

            
                     
# Créer la classe Button
class Button:
    def __init__(self, x, y, width, height, text, font_size=36):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.Font(None, font_size)
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER_COLOR

    def draw(self, screen):
        # Dessiner le bouton
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_x, mouse_y):
            pygame.draw.rect(screen, self.hover_color, self.rect)  # Couleur quand on survole
        else:
            pygame.draw.rect(screen, self.color, self.rect)  # Couleur par défaut

        # Ajouter du texte sur le bouton
        text = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(text, (self.rect.centerx - text.get_width() // 2, self.rect.centery - text.get_height() // 2))

    def is_clicked(self, event):
        # Vérifier si le bouton a été cliqué
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Clic gauche
            if self.rect.collidepoint(event.pos):  # Si on clique sur le bouton
                return True
        return False
            



liste_bateaux = []
liste_vitesses = [[] for _ in range(NBRBOAT)]
liste_vitesses_moyenne = [[] for _ in range(NBRBOAT)]
info_cards = []

button = Button(WIDTH - 170, HEIGHT - 70, 150, 50, "Rejouer")

# Générer les bateaux et leurs trajectoires à afficher
for i in range(NBRBOAT):
    
    # Génération de la trajectoire
    start_point = (500/COEFFNORM, 400/COEFFNORM)
    velocity = (random.randint(-VELOCITY, VELOCITY), random.randint(-VELOCITY, VELOCITY))
    num_points = 500//COEFFNORM
    if i%3 == 0:
        traj = trajectories.generate_mru_trajectory(start_point, velocity, num_points)
    elif i%3 == 1:
        traj = trajectories.generate_mua_trajectory(start_point, velocity, num_points)
    else:
        traj = trajectories.generate_singer_trajectory(start_point, velocity, num_points)
    traj_norm = traj*COEFFNORM
    
    # Génération du bateau
    nom_couleur, rgb = couleurs[i]
    bateau = Boat(30, BateauType.CARGO, traj_norm, rgb)
    liste_bateaux.append(bateau)
    
    # Calcul des vitesses
    liste_vitesses[i] = trajectories.calc_vitesse(traj)
    liste_vitesses_moyenne[i] = trajectories.calc_vitesse_moyenne(liste_vitesses[i])
    
    # Génération de la carte d'infos initiale
    card = InfoCard(200*i, 100, bateau, rgb)
    info_cards.append(card)
    
    
    
###############################################
###                                          ##
### ---------- BOUCLE PRINCIPALE ------------##
###                                          ##
###############################################


running = True
trajectory_index = 1  # On commence à afficher à partir du 2e point (index 1)

while running:
    screen.blit(background, (0, 0))
    
    # Dessiner la trajectoire progressivement
    for ii in range(len(liste_bateaux)):
        
        bateau = liste_bateaux[ii]
        trajectory = bateau.trajectoire
        if trajectory_index < len(liste_vitesses[ii]):
            bateau.vitesse = liste_vitesses[ii][trajectory_index]
            bateau.vitesse_moyenne = liste_vitesses_moyenne[ii][trajectory_index]
        else:
            bateau.vitesse = liste_vitesses[ii][-1]  # On garde la dernière valeur connue
            bateau.vitesse_moyenne = liste_vitesses_moyenne[ii][-1]  # On garde la dernière valeur connue

        print(bateau.vitesse)
        
        # Si la trajectoire a plus d'un point
        if len(trajectory) > 1:
            for i in range(trajectory_index):
                if i < len(trajectory) - 1:  # S'assurer qu'on ne dépasse pas la fin
                    pygame.draw.line(screen, bateau.color, trajectory[i], trajectory[i + 1], 5)  # Dessiner la ligne

        


    # Dessiner les cartes d'info
    for card in info_cards:
        card.draw(screen)
        
    # Dessiner le/les boutons
    button.draw(screen)

    pygame.display.flip()
    
    # Augmenter l'index pour dessiner la trajectoire progressivement
    if trajectory_index < len(trajectory) - 1:
        trajectory_index += 1
        
    pygame.time.delay(DELAY)
    
    # Gérer les événements (Quitter le jeu)
    for event in pygame.event.get():
        
        if button.is_clicked(event):
            trajectory_index = 1  # Réinitialiser l'animation pour recommencer
            
        if event.type == pygame.QUIT:
            running = False


pygame.quit()
