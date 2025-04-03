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
import src.package.api_hdf5 as hdf5
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force l'utilisation du premier GPU

# Vérifier que CUDA est disponible
if torch.cuda.is_available():
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
num_epochs = 10
batch_size = 128
num_class = 3  # MRU, MUA, Singer
N_corr = 25 # Nombre d'échantillon lors de l'autocorrelation
N_features = 4 # 2 channels par coordonnées, donc 2*2 = 4 features

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
        self.conv1 = nn.Conv1d(in_channels=N_features, out_channels=64, kernel_size=2, padding=1)
        self.batch1 = nn.BatchNorm1d(64)  # Remplacer BatchNorm2d par BatchNorm1d
        self.max1 = nn.MaxPool1d(kernel_size=1, stride=1)

        
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
        x = self.conv1(x)  

        x = F.relu(self.batch1(x))  

        x = self.max1(x)
        x = self.dropout1(x)

        x = x.reshape(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return x

def acceleration(X, Tech=1):
    """ Estime l'accélération à partir des positions """
    return (X[2:] - 2*X[1:-1] + X[:-2]) / Tech**2

def jerk(X, Tech=1):
    """Estime le jerk à partir des positions."""
    return (X[3:] - 3*X[2:-1] + 3*X[1:-2] - X[:-3]) / Tech**3

def autocorr(signal, N_corr):
    N = len(signal)
    lags = np.arange(-N + 1, N)
    result = np.correlate(signal, signal, mode='full')
    unbiased_corr = result / (N - np.abs(lags))  # Normalisation non biaisée
    indices = np.arange(0, N_corr)
    unbiased_corr = unbiased_corr[N-1:]
    return [unbiased_corr[i]/unbiased_corr[0] for i in indices]

def affich_autocorr(ipt,label):
    # Création de la figure et des sous-graphiques (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Tracer les 4 graphiques
    axes[0, 0].plot(ipt[0,:], 'r')
    axes[0, 0].set_title("Autocorrelation de l'acceleration de X")

    axes[0, 1].plot(ipt[1,:], 'b')
    axes[0, 1].set_title("Autocorrelation du Jerk de X")

    axes[1, 0].plot(ipt[2,:], 'g')
    axes[1, 0].set_title("Autocorrelation de l'acceleration est de Y")

    axes[1, 1].plot(ipt[3,:], 'm')
    axes[1, 1].set_title("Autocorrelation du Jerk de Y")

    # Ajustement de l'affichage
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("Autocorr")
        ax.grid(True)
    plt.title(label)
    plt.tight_layout()
    plt.show()

def affich_traj(ipt,label):
    # Création de la figure et des sous-graphiques (2x2)
    print(ipt.shape)
    plt.figure,
    plt.plot(ipt[:,0],ipt[:,1])
    plt.title("trajectoire X,Y")

    plt.title(label)
    plt.show()

def compute_input(list_traj, list_lengths, N):
    ipt = np.zeros((len(list_traj),4,N))
    list_traj_X = list_traj[:,:,0]
    list_traj_Y = list_traj[:,:,1]
    for idx, traj_x in enumerate(list_traj_X):
        traj_x = traj_x[:list_lengths[idx]]
        ipt[idx,0,:] = autocorr(acceleration(traj_x), N)
        ipt[idx,1,:] = autocorr(jerk(traj_x),N)
    for idx, traj_y in enumerate(list_traj_Y):
        traj_y = traj_y[:list_lengths[idx]]
        ipt[idx,2,:] = autocorr(acceleration(traj_y), N)
        ipt[idx,3,:] = autocorr(jerk(traj_y),N)

    return ipt

# Chargement et préparation des données
def prepare_data(filename, N=50):
    # Charger les données
    list_trajxy, list_lengths, list_labels = hdf5.read_hdf5(filename) # Shape of list_trajxy : (Nb_Tot_traj, Len_Max, Nb_Dim)
    
    # Compute the Vector to input in CNN
    list_input = compute_input(list_trajxy, list_lengths, N) # Shape of list_input : (Nb_Tot_traj, 4, N)
    print(list_input.shape)
    MRU = np.where(list_labels==0)[0]
    MUA = np.where(list_labels==1)[0]
    SIN = np.where(list_labels==2)[0]
    print(MRU)
    print(MUA)
    print(SIN)
    affich_traj(list_trajxy[MRU[0],:,:],"MRU")
    affich_traj(list_trajxy[MUA[0],:,:],"MUA")
    affich_traj(list_trajxy[SIN[0],:,:],"SIN")
    affich_autocorr(list_input[MRU[0],:,:],"MRU")
    affich_autocorr(list_input[MUA[0],:,:],"MUA")
    affich_autocorr(list_input[SIN[0],:,:],"SIN")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(list_input, list_labels, test_size=0.2)
    
    # Créer les datasets
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


# Initialiser le modèle
net = SimpleCNN()

# Vérifier la taille de sortie des convolutions avec un exemple
sample_input = torch.randn(128, N_features, N_corr)
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
filename="data_nonbruite/data_min120_max300_sigma0.1_tau5_sigmaM24.06.h5"
train_loader, test_loader = prepare_data(filename, N_corr)

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