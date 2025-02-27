import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# =============================================================================
# 1. Create a synthetic dataset with different memory properties
# =============================================================================

class MemoryDataset(Dataset):
    def __init__(self, n_samples, seq_length):
        """
        Three classes:
          0: No memory (white noise)
          1: Short memory (AR(1) with coefficient 0.5)
          2: Long memory (AR(1) with coefficient 0.99)
        """
        self.data = []
        self.labels = []
        samples_per_class = n_samples // 3  # equally balanced classes

        for label in [0, 1, 2]:
            for i in range(samples_per_class):
                if label == 0:
                    # White noise: no temporal dependency
                    seq = np.random.randn(seq_length)
                elif label == 1:
                    # AR(1) with moderate memory (short memory)
                    seq = np.zeros(seq_length)
                    for t in range(1, seq_length):
                        seq[t] = 0.5 * seq[t - 1] + np.random.randn()
                elif label == 2:
                    # AR(1) with near unit root: long memory
                    seq = np.zeros(seq_length)
                    for t in range(1, seq_length):
                        seq[t] = 0.99 * seq[t - 1] + np.random.randn()
                self.data.append(seq)
                self.labels.append(label)

        self.data = np.array(self.data).astype(np.float32)  # shape: (n_samples, seq_length)
        self.labels = np.array(self.labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.data[idx][:, None]
        label = self.labels[idx]
        return seq, label

# Hyperparameters for dataset
batch_size = 32
seq_length = 5 * 60
train_samples = 3 * 2000
test_samples = 3 * 200

# Create datasets and loaders
train_dataset = MemoryDataset(train_samples, seq_length)
test_dataset  = MemoryDataset(test_samples, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 2. Define a simple LSTM classifier
# =============================================================================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Use the final hidden state from the last LSTM layer for classification
        last_hidden = hn[-1]
        out = self.fc(last_hidden)
        return out

# Set device, model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =============================================================================
# 3. Train the LSTM model
# =============================================================================

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    for sequences, labels in train_loader:
        sequences = sequences.to(device)  # shape: (batch_size, seq_length, 1)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_dataset)
    accuracy = total_correct / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Evaluate on the test set
model.eval()
total_correct = 0
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

test_accuracy = total_correct / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

# =============================================================================
# 4. Plot samples for each class 
# =============================================================================

def plot_sample(dataset, class_label, num_samples=3):
    indices = np.where(dataset.labels == class_label)[0][:num_samples]
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        seq, label = dataset[idx]
        plt.subplot(1, num_samples, i+1)
        plt.plot(seq, marker='o', linestyle='-')
        plt.title(f"Class {label}")
        plt.xlabel("Time")
        plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

for class_label in [0, 1, 2]:
    plot_sample(train_dataset, class_label)

# =============================================================================
# 5. Extract hidden states from the test set for visualization
# =============================================================================

hidden_states = []
labels_list = []

model.eval()
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        h0 = torch.zeros(model.num_layers, sequences.size(0), model.hidden_size).to(device)
        c0 = torch.zeros(model.num_layers, sequences.size(0), model.hidden_size).to(device)

        out, (hn, cn) = model.lstm(sequences, (h0, c0))
        last_hidden = hn[-1]  # shape: (batch_size, hidden_size)
        hidden_states.append(last_hidden.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

hidden_states_all = np.concatenate(hidden_states, axis=0)
labels_all = np.concatenate(labels_list, axis=0)

# =============================================================================
# 6. Visualize hidden states using PCA and t-SNE
# =============================================================================

# --- PCA Visualization ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(hidden_states_all)

plt.figure(figsize=(8, 6))
for label in np.unique(labels_all):
    indices = labels_all == label
    plt.scatter(pca_result[indices, 0], pca_result[indices, 1],
                label=f"Class {label}", alpha=0.7)
plt.title("PCA of LSTM Hidden States")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# --- Cercle des corrélations ---
components = pca.components_[:2]    # shape: (2, n_features)
eigenvalues = pca.explained_variance_[:2]   # shape: (2,)

loadings = components.T * np.sqrt(eigenvalues)

plt.figure(figsize=(8, 8))
for i, (x, y) in enumerate(loadings):
    plt.arrow(0, 0, x, y, color='r', alpha=0.5, head_width=0.03, head_length=0.03)
    plt.text(x * 1.15, y * 1.15, f"Var {i+1}", color='g', ha='center', va='center')
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Correlation circle")
plt.grid()
plt.show()

# --- t-SNE Visualization ---
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=0)
tsne_result = tsne.fit_transform(hidden_states_all)

plt.figure(figsize=(8, 6))
for label in np.unique(labels_all):
    indices = labels_all == label
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                label=f"Class {label}", alpha=0.7)
plt.title("t-SNE of LSTM Hidden States")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.show()
