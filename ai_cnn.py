import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force l'utilisation du premier GPU

# Vérifier que CUDA est disponible
assert torch.cuda.is_available(), "CUDA n'est pas disponible. Vérifiez l'installation de PyTorch."
print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory Usage:")
print(f"Allocated: {torch.cuda.memory_allocated(0)//1024**2}MB")
print(f"Cached: {torch.cuda.memory_reserved(0)//1024**2}MB")

# Force l'utilisation du GPU
device = torch.device('cuda:0')
torch.cuda.set_device(device)

# Optimisations pour les performances
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# Vérifier GPU et configurer le device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    # Configurer pour plus de performance
    torch.backends.cudnn.benchmark = True

# Parameters
learning_rate = 1e-3
num_epochs = 20
batch_size = 32
num_class = 3  # MRU, MUA, Singer

# Dataset personnalisé
class TrajectoryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Définition du CNN modifié avec une seule couche de convolution
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Une seule couche de convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        self.max1 = nn.MaxPool2d(kernel_size=1, stride=1)
        
        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(64)
        
        # Dropout
        self.dropout1 = nn.Dropout(0.5)
        
        # Fully connected layers
        # La taille d'entrée sera calculée dynamiquement
        self.fc1 = nn.Linear(64 * 3 * 5, 512)  
        self.fc2 = nn.Linear(512, num_class)

    def get_conv_output_size(self, x):
        # Helper function pour obtenir la taille de sortie des convolutions
        x = self.conv1(x)
        x = self.max1(x)
        return x.size()[1:]
        
    def forward(self, x):
        # Couche de convolution
        x = self.conv1(x)
        x = F.relu(self.batch1(x))
        x = self.max1(x)
        x = self.dropout1(x)
        
        # Flatten et fully connected
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Chargement et préparation des données
def prepare_data():
    # Charger les données
    X = np.load("trajectoires_data/autocorrelations_matrices.npy")
    n_samples = len(X)
    n_per_class = n_samples // 3
    
    # Créer les labels
    y = np.concatenate([
        np.zeros(n_per_class),
        np.ones(n_per_class),
        2 * np.ones(n_samples - 2*n_per_class)
    ])
    
    # Reshape pour CNN
    X = X.reshape(-1, 1, 2, 4)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Créer les datasets
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device, '\n')

# Initialiser le modèle
net = SimpleCNN()

# Vérifier la taille de sortie des convolutions avec un exemple
sample_input = torch.randn(1, 1, 2, 4)
conv_output_size = net.get_conv_output_size(sample_input)
print(f"Taille de sortie des convolutions: {conv_output_size}")

# Ajuster la taille de la première couche fully connected si nécessaire
flattened_size = conv_output_size.numel()
print(f"Taille après flatten: {flattened_size}")
if flattened_size != 64 * 3 * 5:
    print(f"Ajustement de la taille du fc1: 64 * 3 * 5 -> {flattened_size}")
    net.fc1 = nn.Linear(flattened_size, 512)

net.to(device)

# Chargement des données
train_loader, test_loader = prepare_data()

# Loss & Optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train
print("Début de l'entraînement...")
train_accuracies = []
train_losses = []

for epoch in range(num_epochs):
    correct_train = 0
    total_train = 0
    running_loss = 0.0
    net.train()
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    epoch_accuracy = (100 * correct_train) / total_train
    epoch_loss = running_loss / len(train_loader)
    train_accuracies.append(epoch_accuracy)
    train_losses.append(epoch_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%')

# Test
print("\nÉvaluation sur l'ensemble de test...")
net.eval()
y_pred = []
y_true = []
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

test_accuracy = (100 * correct_test) / total_test
print(f'\nTest Accuracy: {test_accuracy:.3f}%')

# Visualisations
plt.figure(figsize=(15, 5))

# Courbe de perte
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Courbe de perte')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Courbe d'accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Matrice de confusion
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['MRU', 'MUA', 'Singer'],
            yticklabels=['MRU', 'MUA', 'Singer'])
plt.title('Matrice de Confusion')
plt.ylabel('Vraie classe')
plt.xlabel('Classe prédite')
plt.show()

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(y_true, y_pred, target_names=['MRU', 'MUA', 'Singer']))

# Sauvegarder le modèle
torch.save(net.state_dict(), "trajectory_classifier_single_conv.pth")
print("\nModèle sauvegardé sous 'trajectory_classifier_single_conv.pth'")