import os
import csv
import datetime
import argparse
from typing import Tuple, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from data_gen import generate_and_save_data


# ----------------------------
# Global Configurations and Logging
# ----------------------------
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset and DataLoader Utilities
# ----------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, h5_path: str, mode: str = 'train'):
        """
        Dataset class that loads data from HDF5 file, ensuring correct sequence lengths.
        
        Args:
            h5_path: Path to HDF5 file.
            mode: 'train' or 'test'.
        """
        self.h5_path = h5_path
        self.mode = mode

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[f'{mode}/labels'])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a single trajectory and its label, trimming the sequence to its actual length.
        """
        # TODO: Use length
        with h5py.File(self.h5_path, 'r') as f:
            trajectory = f[f'{self.mode}/trajectories'][idx]
            label = f[f'{self.mode}/labels'][idx]
            length = f[f'{self.mode}/lengths'][idx]

        return torch.tensor(trajectory[:length], dtype=torch.float32), torch.tensor(label, dtype=torch.uint8)

def collate_fn(batch):
    """Collation for variable-length sequences"""
    # TODO: Use lengths
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels, dtype=torch.uint8), lengths


# ----------------------------
# Model Architecture
# ----------------------------
class TrajectoryClassifier(nn.Module):
    """
    LSTM-based classifier for trajectory data.
    Uses a bidirectional LSTM with dropout followed by a fully connected layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack the padded sequence for efficiency
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        # Concatenate the final hidden states from both directions
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc(self.dropout(h_n))


# ----------------------------
# Training, Evaluation, and Metrics Plotting
# ----------------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                num_epochs: int = 10, learning_rate: float = 1e-3, metrics_file: str = "metrics.csv",
                best_model_file: str = "best_model.pt", device: torch.device = DEFAULT_DEVICE) -> None:
    """
    Train the model with validation and save metrics to a CSV file
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Initialize CSV file
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    best_loss = float('inf')
    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for (inputs, labels, lengths) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total

        # --- Validation phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels, lengths in val_loader:
                inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Log metrics to CSV
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy])

        # Save the best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_file)
            print(f"Best model saved to {best_model_file}")


def eval_model(model: nn.Module, test_loader: DataLoader, class_names: List = None,
               save_path: str = "confusion_matrix.png", show_figure: bool = False,
               device: torch.device = DEFAULT_DEVICE) -> None:
    """
    Evaluate the model on the test set and save a confusion matrix.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        class_names (list): List of class names for the confusion matrix labels.
        save_path (str): File path to save the confusion matrix image.
        show_figure (bool): Whether to display the figure.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="Evaluating"):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)

            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate the confusion matrix in percentage
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100   # TODO: Check: cm.sum(axis=1, keepdims=True)

    # Load default class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Percentage')
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    # Annotate each cell with the corresponding percentage
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha='center', va='center',
                     color="black" if cm_percent[i, j] < 50 else "white")

    plt.savefig(save_path, bbox_inches="tight")
    print(f"Confusion matrix saved to {save_path}")
    
    if show_figure:
        plt.show()
    else:
        plt.close()


def plot_metrics(metrics_file: str = "metrics.csv", show_plot: bool = False) -> None:
    """
    Plot the training and validation loss and accuracy over epochs from a CSV metrics file.
    
    Args:
        metrics_file (str): Path to the CSV file containing the metrics.
        save_path (str): Path to save the plot as a PNG file.
        show_plot (bool): If True, the plot will be displayed. If False, it will only be saved.
    """
    save_path = os.path.splitext(metrics_file)[0] + ".png"

    epochs = []
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []

    with open(metrics_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_accuracy.append(float(row["train_accuracy"]))
            val_loss.append(float(row["val_loss"]))
            val_accuracy.append(float(row["val_accuracy"]))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

    if show_plot:
        plt.show()
    else:
        plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate trajectory classification model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of hidden dimensions for the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--regen_data", action="store_true", help="Regenerate dataset if it exists")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the pretrained model without training")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save metrics and evaluation files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(args.output_dir, f"metrics_hd{args.hidden_dim}_n{args.num_layers}_lr{args.learning_rate}_{timestamp}.csv")
    best_model_file = os.path.join(args.output_dir, f"best_model_hd{args.hidden_dim}_n{args.num_layers}_lr{args.learning_rate}_{timestamp}.pt")
    conf_matrix_file = os.path.join(args.output_dir, f"confusion_matrix_hd{args.hidden_dim}_n{args.num_layers}_lr{args.learning_rate}_{timestamp}.png")

    h5_path = 'trajectory_data.h5'
    if args.regen_data or not os.path.exists(h5_path):
        print("Generating and saving new data...")
        # TODO: Add kwargs
        generate_and_save_data(
            filepath=h5_path,
            n_samples=15_000,           # Number of trajectories
            traj_length=[15, 10 * 60]   # seconds
        )
    
    # Prepare dataset and data loaders
    full_dataset = TrajectoryDataset(h5_path, 'train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = TrajectoryDataset(h5_path, 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Set device
    device = torch.device(args.device)

    # Initialize the model
    model = TrajectoryClassifier(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    # Optionally load a pretrained model if available
    if os.path.exists(best_model_file):
        print(f"Loading pretrained model from {best_model_file}...")
        model.load_state_dict(torch.load(best_model_file, map_location=device))

    # Training
    if not args.eval_only:
        train_model(model, train_loader, val_loader, num_epochs=args.epochs,
                    learning_rate=args.learning_rate, metrics_file=metrics_file,
                    best_model_file=best_model_file, device=device)

    # Saving training metric plot
    plot_metrics(metrics_file=metrics_file, show_plot=False)

    # Evaluation and confusion matrix
    eval_model(model, test_loader, class_names=['MRU', 'MUA', 'Singer'],
               save_path=conf_matrix_file, show_figure=False, device=device)


if __name__ == "__main__":
    main()