from enum import Enum
from faker import Faker
import cv2
import random
from utils import *
import trajectories
import sys
import os

# Import Pygame sans afficher son message
# On détourne temporairement stdout
stdout_backup = sys.stdout
sys.stdout = open(os.devnull, 'w')
import pygame       
# On réactive stdout
sys.stdout.close()
sys.stdout = stdout_backup



# Variables Globales
BACKGROUND_IMAGE = "background_taiwan.png"
NBRBOAT = 4
NBRWRONGBOAT = 1
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

# Génération de la trajectoire
start_point = (1200/COEFFNORM, 400/COEFFNORM)
num_points = 200//COEFFNORM


class BateauType(Enum):
    CARGO = "Cargo"
    CORVETTE = "Corvette"
    PECHE = "Bateau de pêche"
    
BoatTypesEnum = [BateauType.CARGO, BateauType.CORVETTE, BateauType.PECHE]
MouvementsTypeEnum = ["MRU", "MUA", "Singer", "Combinaison"]

# Vitesse de Base || Variance de Base de l'acceleration/jerk pour chaque type de bateau 
# CARGO, CORVETTE, PECHE
# Rapide|Non Maniable   TrèsRApide|Maniable     Lent|Maniable
params = [[8, 0.05], [8, 1], [3, 0.5], [6, 1]]
def randomize_params(param, variation_sigma_pourcentage=10.0):
    """Fais varier entre + et - variation_sigma_pourcentage les valeurs de 
        chaque paramètres en entrée de manière aléatoire

    Args:
        param (tuple): Les paramètres à modifier de façon aléatoire
        variation_sigma_pourcentage (float, optional): . Defaults to 10.0.

    Returns:
        tuple: Les paramètres d'entrée modifiés 
    """
    velo = param[0]
    var = param[1]
    velocity = (random.randint(-velo, velo), random.randint(-velo, velo))
    variation = random.uniform(var*(1 - variation_sigma_pourcentage/100), var*(1+ variation_sigma_pourcentage/100))
    variance = var + variation
    return velocity, variance


# Classe pour les infos d'un bateau
class Boat:
    def __init__(self, vitesse, boat_type, trajectoire, color, real=True):
        self.name = fake.first_name()
        self.vitesse = vitesse
        self.type = boat_type
        self.trajectoire = trajectoire
        self.color = color
        self.vitesse_moyenne = 0
        self.real = real
        self.mvmt_type = ""
        
    def toString(self):
        return self.name + "\nVitesse: " + str(self.vitesse) + "\nType: " + self.type.value
    
    def getInfos(self):
        separateur = "--------------\n"
        string = self.name + "\nType: " + self.type.value + "\n" + str(self.real) + " Bateau"
        print(separateur + string)
      

class InfoCard:
    def __init__(self, x, y, bateau, couleur):
        self.x = x
        self.y = y
        self.bateau = bateau  # ← on garde le bateau, pas juste du texte
        self.width = 150
        self.height = 100
        self.rect = pygame.Rect(x, y, self.width, self.height)
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
            
    def is_clicked(self, event):
        # Vérifier si le bouton a été cliqué
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Clic gauche
            if self.rect.collidepoint(event.pos):  # Si on clique sur le bouton
                return True
        return False

                     
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
            


def random_boat(real=True):
    """Génère les paramètres et caractéristiques d'un bateau de façon procédurale

    Args:
        real (bool, optional): Permet de choisir si les paramètres et caractéristiques du bateau sont cohérentes ou non. Defaults to True.
    """
    n = random.randint(0, 3)
    n2 = n
    while n2 == n:
        n2 = random.randint(0, 2)
    
    if n == 3:
        boatType = BoatTypesEnum[n2]
        mvmnt_type = MouvementsTypeEnum[n2]
        parametresF = params[n2]
    else:
        boatType = BoatTypesEnum[n]
        mvmnt_type = MouvementsTypeEnum[n]
        parametresF = params[n]
        
            
    if not real:
        parametres = randomize_params(parametresF, 20.0)
    else:
        parametres = randomize_params(parametresF)
        
        
    return boatType, parametres, mvmnt_type
        



liste_bateaux = []
liste_vitesses = [[] for _ in range(NBRBOAT)]
liste_vitesses_moyenne = [[] for _ in range(NBRBOAT)]
info_cards = []


def create_boat(i, start_point, num_points, parametres, typeTraj, boatType, real=True):
    
    if typeTraj == "MRU":
        traj = trajectories.generate_mru_trajectory(start_point, parametres[0], num_points, sigma=parametres[1])
    elif typeTraj == "MUA": 
        traj = trajectories.generate_mua_trajectory(start_point, parametres[0], num_points, jerk_std=parametres[1])
    elif typeTraj == "Singer":
        traj = trajectories.generate_singer_trajectory(start_point, parametres[0], num_points, noise_std=parametres[1])
    elif typeTraj == "Combinaison":
        traj = trajectories.allinone(start_point, parametres[0], num_points, parametres[1])
    
    # Normalisation de la trajectoire à la taille de la carte  
    traj_norm = traj*COEFFNORM
    
    # Génération du bateau
    _, rgb = couleurs[i]
    bateau = Boat(30, boatType, traj_norm, rgb, real)
    liste_bateaux.append(bateau)
    
    # Calcul des vitesses
    liste_vitesses[i] = trajectories.calc_vitesse(traj, bateau.real)
    liste_vitesses_moyenne[i] = trajectories.calc_vitesse_moyenne(liste_vitesses[i])
    
    # Génération de la carte d'infos initiale
    card = InfoCard(200*i, 100, bateau, rgb)
    info_cards.append(card)



# Déterminer l'indice du/des faux bateaux
indices_wrong_boats = random.sample(range(NBRBOAT), NBRWRONGBOAT) 


# Générer les bateaux et leurs trajectoires à afficher
for i in range(NBRBOAT):
    
    boatType, parametres, mvmt_type = random_boat(i not in indices_wrong_boats)
    create_boat(i, start_point, num_points, parametres, mvmt_type, boatType, i not in indices_wrong_boats)
    
    # match i%NBRBOAT:
    #     case 0:
    #         create_boat(i, start_point, num_points, params[i], "MRU", BateauType.CARGO)
    #     case 1:
    #         create_boat(i, start_point, num_points, params[i], "MUA", BateauType.CORVETTE)
    #     case 2:
    #         create_boat(i, start_point, num_points, params[i], "Singer", BateauType.PECHE)
    #     case 3:
    #         create_boat(i, start_point, num_points, params[i], "Combinaison", BateauType.CORVETTE)
    
   
def affichage_boats():
    for ii in range(liste_bateaux):
        bateau = liste_bateaux[ii]
        
    
    
    
    
###############################################
###                                          ##
### ---------- BOUCLE PRINCIPALE ------------##
###                                          ##
###############################################


running = True
trajectory_index = 1  # On commence à afficher à partir du 2e point (index 1)

button = Button(WIDTH - 170, HEIGHT - 70, 150, 50, "Rejouer")

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
    
    # Augmenter l'index pour dessiner les trajectoires progressivement
    if trajectory_index < len(trajectory) - 1:
        trajectory_index += 1
        
    pygame.time.delay(DELAY)
    
    # Gérer les événements (Quitter le jeu)
    for event in pygame.event.get():
        
        if button.is_clicked(event):
            trajectory_index = 1  # Réinitialiser l'animation pour recommencer
            
        for card in info_cards:
            
            if card.is_clicked(event):
                print("Click détecté sur la carte " + card.bateau.name)
                if not card.bateau.real:
                    # Afficher écran de victoire
                    print("victoire")
                    screen.fill(WHITE)
                    text = font.render("VICTOIRE !", True, couleurs[1][1])
                else:
                    # Afficher écran de défaite
                    screen.fill(WHITE)
                    text = font.render("DEFAITE", True, couleurs[0][1])
                    
                screen.blit(text, (200, 250))
                pygame.display.flip()
                
                for bateau in liste_bateaux:
                    bateau.getInfos()
                
                # Attendre clic ou ESC
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                waiting = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            waiting = False
            
        if event.type == pygame.QUIT:
            running = False


pygame.quit()
