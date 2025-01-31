import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from package.common import trajectoire_XY 
from MRU import trajec_MRU
from MUA import Trajec_MUA
from Singer import traj_singer

# Définition du dataset
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels):
        self.trajectories = [torch.tensor(traj, dtype=torch.float32) for traj in trajectories]
        self.labels = torch.tensor(labels, dtype=torch.uint8)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.labels[idx]

# Fonction pour le padding des séquences
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0).to(device)
    lengths = torch.tensor([len(seq) for seq in sequences])
    return padded_sequences, torch.tensor(labels).to(device), lengths

# Définition du modèle LSTM bidirectionnel
class TrajectoryClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, num_classes=3):
        super(TrajectoryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, _) = self.lstm(packed_x)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)  # Concaténer les deux directions
        out = self.fc(self.dropout(h_n))
        return out

# Simulation de trajectoires synthétiques
def generate_synthetic_trajectories(n_samples=500, min_N=50, max_N=150, Tech=0.1, min_sigma2=0.001, max_sigma2=0.05, alpha=0.1):
    trajectories = []
    labels = []
    for _ in range(n_samples):
        traj_type = np.random.choice([0, 1, 2])  # 0=MRU, 1=MUA, 2=Singer
        N = np.random.randint(min_N, max_N)  # Longueur variable
        sigma2 = np.random.uniform(min_sigma2, max_sigma2)  # Bruit variable
        noise = np.random.normal(0, 0.1, (N, 2))  # Bruit gaussien

        if traj_type == 2:
            # TODO: Variable alpha
            t1, X, t2, Y = trajectoire_XY(traj_singer, N, Tech, sigma2, alpha)
        elif traj_type == 1:
            t1, X, t2, Y = trajectoire_XY(Trajec_MUA, N, Tech, sigma2)
        else:
            t1, X, t2, Y = trajectoire_XY(trajec_MRU, N, Tech, sigma2)
        
        traj = np.stack((X[0, :], Y[0, :]), axis=1) + noise
        trajectories.append(traj)
        labels.append(traj_type)
    
    return trajectories, labels

# Génération des données
trajectories, labels = generate_synthetic_trajectories()
dataset = TrajectoryDataset(trajectories, labels)

# Séparation des données en train et test (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Initialisation du modèle
model = TrajectoryClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraînement
model.train()
num_epochs = 300
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch in train_dataloader:
        inputs, labels, lengths = batch
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}, Accuracy: {correct/total:.4f}")

print("Entraînement terminé.")

model.eval()  # Mettre le modèle en mode évaluation
y_true = []
y_pred = []

# Evaluation sur l'ensemble de test
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels, lengths = batch
        outputs = model(inputs, lengths)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Génération de la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Visualisation avec Seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['MRU', 'MUA', 'Singer'], yticklabels=['MRU', 'MUA', 'Singer'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()