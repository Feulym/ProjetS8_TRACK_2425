import os
import sys
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
from tqdm import tqdm

# Fix package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from package.viz import plot_metrics, plot_cm


# ----------------------------
# Global Configurations
# ----------------------------
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Dataset and DataLoader
# ----------------------------
class TrajectoryDataset(Dataset):
    def __init__(self, h5_path: str):
        """
        Loads the entire trajectory dataset from an HDF5 file.
        Assumes the file contains the keys:
          - "trajectories": the trajectories data,
          - "labels": the labels,
          - "lengths": actual sequence lengths.
          
        Args:
            h5_path: Path to the HDF5 file.
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f["labels"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single trajectory and its label, trimmed to its actual length.
        """
        with h5py.File(self.h5_path, 'r') as f:
            trajectory = f["trajectories"][idx]
            label = f["labels"][idx]
            length = f["lengths"][idx]
        return torch.tensor(trajectory[:length], dtype=torch.float32), torch.tensor(label, dtype=torch.uint8)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of variable-length sequences by padding them.
    Returns the padded sequences, labels, and original lengths.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels, dtype=torch.uint8), lengths


def prepare_dataloaders(h5_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the full dataset from the given file and split it into train, validation,
    and test DataLoaders.
    
    Split ratio: 80% train, 10% validation, 10% test.
    """
    full_dataset = TrajectoryDataset(h5_path)
    train_size = int(0.8 * len(full_dataset))
    remaining = len(full_dataset) - train_size
    val_size = remaining // 2
    test_size = remaining - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, val_loader, test_loader


# ----------------------------
# Model Architecture
# ----------------------------
class TrajectoryClassifier(nn.Module):
    """
    LSTM-based classifier for trajectory data.
    Uses a bidirectional LSTM with dropout followed by a fully connected layer.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2,
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
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        # Concatenate the final hidden states from both directions (forward and backward)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc(self.dropout(h_n))


# ----------------------------
# Training, Validation and Evaluation
# ----------------------------
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Returns:
        A tuple of (average loss, accuracy).
    """
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0

    for inputs, labels, lengths in tqdm(loader, desc="Training phase", leave=False):
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
    
    return epoch_loss / len(loader), correct / total if total > 0 else 0.0


def validate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   device: torch.device) -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Returns:
        A tuple of (average loss, accuracy).
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(loader, desc="Validation phase", leave=False):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(loader), correct / total if total > 0 else 0.0


def log_metrics(metrics_file: str, epoch: int, train_loss: float, train_acc: float,
                val_loss: float, val_acc: float) -> None:
    """
    Append epoch metrics to a CSV file.
    """
    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int, learning_rate: float, metrics_file: str,
                best_model_file: str, device: torch.device) -> None:
    """
    Train the model with periodic validation and save the best model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize CSV file
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save metrics
        log_metrics(metrics_file, epoch, train_loss, train_acc, val_loss, val_acc)

        # Save best model on validation
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_file)
            print(f"Best model saved to {best_model_file}")


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: torch.device) -> Tuple[List[int], List[int]]:
    """
    Evaluate the model on the test dataset.
    
    Returns:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and/or evaluate trajectory classification model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset file (HDF5).")
    parser.add_argument("--model_file", type=str, default=None,
                        help="Path to a pretrained model file for evaluation (required in eval_only mode).")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the pretrained model.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for the LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save metrics and evaluation files")
    args = parser.parse_args()

    # Verify if the Dataset file exists
    if not os.path.exists(args.dataset_file):
        raise FileNotFoundError(f"Dataset file {args.dataset_file} does not exist.")

    # Create the output directory (if it not already the case)
    os.makedirs(args.output_dir, exist_ok=True)

    # Unique name for training generated files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    marker = f"hd{args.hidden_dim}_n{args.num_layers}_lr{args.learning_rate}_{timestamp}"

    metrics_file = os.path.join(args.output_dir, f"metrics_{marker}.csv")
    best_model_file = os.path.join(args.output_dir, f"best_model_{marker}.pt")
    conf_matrix_file = os.path.join(args.output_dir, f"confusion_matrix_{marker}.png")

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(args.dataset_file, args.batch_size)

    # Instantiate the Trajectory LSTM model
    device = torch.device(args.device)
    model = TrajectoryClassifier(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)

    # If evaluation-only mode is selected, require a pretrained model file
    if args.eval_only:
        if args.model_file is None:
            raise ValueError("In eval_only mode, --model_file must be provided.")
        if not os.path.exists(args.model_file):
            raise FileNotFoundError(f"Model file {args.model_file} does not exist.")
        print(f"Loading pretrained model from {args.model_file}...")
        model.load_state_dict(torch.load(args.model_file, map_location=device))
    else:
        # Training a fresh model
        train_model(model, train_loader, val_loader, num_epochs=args.epochs,
                    learning_rate=args.learning_rate, metrics_file=metrics_file,
                    best_model_file=best_model_file, device=device)

        # Plot training metrics
        plot_metrics(metrics_file=metrics_file, show_plot=False)

    # Evaluate the model and plot the confusion matrix
    evaluations = evaluate_model(model, test_loader, device=device)
    plot_cm(evaluations, class_names=['MRU', 'MUA', 'Singer'], save_path=conf_matrix_file, show_figure=False)


if __name__ == "__main__":
    main()
