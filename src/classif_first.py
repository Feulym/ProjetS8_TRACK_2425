import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score
from package.common import multi_trajectoire
from MRU import trajec_MRU
from MUA import Trajec_MUA
from Singer import traj_singer


# Création de la base de données (peu d'échantillons pour l'instant)
M = 1000
N = 50

X1, Y1 = multi_trajectoire(M, trajec_MRU, N, 1, 1)
X2, Y2 = multi_trajectoire(M, Trajec_MUA, N, 1, 1)
X3, Y3 = multi_trajectoire(M, traj_singer, N, 1, 1, 1)

traj1 = [[(X1[i, j], Y1[i, j]) for j in range(N)] for i in range(M)]
traj2 = [[(X2[i, j], Y2[i, j]) for j in range(N)] for i in range(M)]
traj3 = [[(X3[i, j], Y3[i, j]) for j in range(N)] for i in range(M)]

trajectories = np.vstack((traj1, traj2, traj3))
labels = np.array(["MRU" for _ in range(1000)] + ["MUA" for _ in range(1000)] + ["Singer" for _ in range(1000)])

print(trajectories)

# Conversion des trajectoires en une matrice NumPy (assure une taille fixe)
X = trajectories  # Matrice 100x10
y = np.array(labels)


# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle OLS
model_ols = LinearRegression()
model_ols.fit(X_train, y_train)

# Prédictions et classification (seuil 0.5)
y_pred_ols = (model_ols.predict(X_test) > 0.5).astype(int)

# Modèle Ridge
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train, y_train)

# Prédictions Ridge
y_pred_ridge = (model_ridge.predict(X_test) > 0.5).astype(int)

# Évaluation
accuracy_ols = accuracy_score(y_test, y_pred_ols)
accuracy_ridge = accuracy_score(y_test, y_pred_ridge)

print(f"Accuracy OLS: {accuracy_ols:.2f}")
print(f"Accuracy Ridge: {accuracy_ridge:.2f}")
