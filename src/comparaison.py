# Soit une base de données de trajectoires (matrice MxN)
# On l'enregistre en npy et en hdf5
# On compare la taille des bases de données, le temps de calcul
import random as rd
import numpy as np

def generate(N, M):
    matrice = np.array(rd.randn((M, N)))
    return matrice

generate(50, 50)