import pygame
from enum import Enum
from faker import Faker

# Initialisation de Pygame et Faker
pygame.init()
fake = Faker('fr_FR')

# Chargement de l'image de fond
BACKGROUND_IMAGE = "Demo/background_english_channel.png"  # À remplacer par ton image
background = pygame.image.load(BACKGROUND_IMAGE)
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
    
class BoatInfo:
    def __init__(self, name, vitesse, boat_type):
        self.name = fake.first_name()
        self.vitesse = vitesse
        self.type = boat_type
        
    def toString(self):
        ""

# Classe pour les cartes d'information
class InfoCard:
    def __init__(self, x, y, bateau):
        self.x = x
        self.y = y
        self.text = bateau.name
        self.width = 150
        self.height = 50

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        lines = self.text.split("\n")
        for i, line in enumerate(lines):
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (self.x + 5, self.y + 5 + i * 20))

# Liste des cartes
info_cards = [
    InfoCard(100, 100, "Bateau 1\nVitesse: 20kn"),
    InfoCard(400, 300, "Bateau 2\nVitesse: 15kn")
]

# Boucle principale
running = True
while running:
    screen.blit(background, (0, 0))
    
    for card in info_cards:
        card.draw(screen)
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
