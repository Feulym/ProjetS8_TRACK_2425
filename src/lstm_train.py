import os
import argparse
import h5py
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix
from typing import Tuple, List
from tqdm import tqdm

from package.common import trajectoire_XY 
from MRU import trajec_MRU
from MUA import Trajec_MUA
from Singer import traj_singer


# TODO: Agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0).to(device)
    return padded_sequences, torch.tensor(labels, dtype=torch.uint8, device=device), lengths

class TrajectoryClassifier(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.fc(self.dropout(h_n))
        return out

def generate_synthetic_trajectories(n_samples=500, min_N=50, max_N=150, Tech=0.1, min_sigma2=0.001, 
                                    max_sigma2=0.1, min_tau=1, max_tau=300, min_sigma_m2=1e-4, max_sigma_m2=1):
    """
    Generate synthetic trajectories more efficiently using vectorization where possible.
    
    Args:
        n_samples: Number of trajectories to generate
        min_N/max_N: Min/max trajectory length
        Tech: Sampling period
        min_sigma2/max_sigma2: Min/max noise variance
        min_tau/max_tau: Min/max maneuver time (Singer model)
        min_sigma_m2/max_sigma_m2: Min/max acceleration magnitude (Singer model)
    
    Returns:
        tuple: (trajectories, labels)
    """
    trajectories = []
    labels = []
    
    # Pre-generate random parameters for all samples
    traj_types = np.random.choice([0, 1, 2], size=n_samples)  # 0=MRU, 1=MUA, 2=Singer
    N_values = np.random.randint(min_N, max_N, size=n_samples)
    sigma2_values = np.random.uniform(min_sigma2, max_sigma2, size=n_samples)
    
    # Pre-generate Singer model parameters
    singer_mask = (traj_types == 2)
    n_singer = np.sum(singer_mask)
    if n_singer > 0:
        tau_values = np.random.uniform(min_tau, max_tau, size=n_singer)
        sigma_m2_values = np.random.uniform(min_sigma_m2, max_sigma_m2, size=n_singer)
    
    singer_idx = 0
    for i in range(n_samples):
        N = N_values[i]
        sigma2 = sigma2_values[i]
        noise = np.random.normal(0, 12, (N, 2))  # GPS noise: +/-10m
        
        if traj_types[i] == 2:  # Singer
            tau = tau_values[singer_idx]
            sigma_m2 = sigma_m2_values[singer_idx]
            singer_idx += 1
            
            alpha = 1 / tau
            new_sigma2 = 2 * alpha * sigma_m2
            
            t1, X, t2, Y = trajectoire_XY(traj_singer, N, Tech, new_sigma2, alpha)
            
        elif traj_types[i] == 1:  # MUA
            t1, X, t2, Y = trajectoire_XY(Trajec_MUA, N, Tech, sigma2)
            
        else:  # MRU
            t1, X, t2, Y = trajectoire_XY(trajec_MRU, N, Tech, sigma2)
        
        # Stack and add noise in one operation
        traj = np.stack((X[0, :], Y[0, :]), axis=1) + noise
        trajectories.append(traj)
        labels.append(traj_types[i])
    
    return trajectories, labels

def generate_and_save_data(filepath: str, n_samples: int = 200_000, traj_length: List[int] = [15, 3600], batch_size=1_000):
    """Generate synthetic data and save to HDF5"""
    
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    
    min_seq_len = traj_length[0]
    max_seq_len = traj_length[1]
    
    with h5py.File(filepath, 'w') as f:
        train_group = f.create_group('train')
        test_group = f.create_group('test')

        train_trajectories = train_group.create_dataset(
            'trajectories', 
            shape=(train_size, max_seq_len, 2),  
            dtype='float32'
        )
        train_labels = train_group.create_dataset('labels', (train_size,), dtype='uint8')
        train_lengths = train_group.create_dataset('lengths', (train_size,), dtype='int32')

        test_trajectories = test_group.create_dataset(
            'trajectories',
            shape=(test_size, max_seq_len, 2),
            dtype='float32'
        )
        test_labels = test_group.create_dataset('labels', (test_size,), dtype='uint8')
        test_lengths = test_group.create_dataset('lengths', (test_size,), dtype='int32')

        for i in tqdm(range(0, n_samples, batch_size), desc="Generating data"):
            current_batch_size = min(batch_size, n_samples - i)
            trajectories, labels = generate_synthetic_trajectories(
                # TODO: kwargs
                n_samples=current_batch_size,
                min_N=min_seq_len,
                max_N=max_seq_len,
                Tech=1
            )

            lengths = np.array([len(traj) for traj in trajectories], dtype=np.int32)
            
            padded_trajectories = np.array([
                np.pad(traj, ((0, max_seq_len - len(traj)), (0, 0)), mode='constant')
                for traj in trajectories
            ], dtype='float32')

            if i + current_batch_size <= train_size:  # If still in the training set
                dataset_traj, dataset_labels, dataset_lengths = train_trajectories, train_labels, train_lengths
                idx_start, idx_end = i, i + current_batch_size
            elif i >= train_size:  # If in the test set
                dataset_traj, dataset_labels, dataset_lengths = test_trajectories, test_labels, test_lengths
                idx_start, idx_end = i - train_size, i - train_size + current_batch_size
            else:  # If batch spans training and test set
                train_end = train_size - i
                test_start = 0
                test_end = current_batch_size - train_end

                # Split batch between train and test
                train_trajectories[i:i + train_end] = padded_trajectories[:train_end]
                train_labels[i:i + train_end] = labels[:train_end]
                train_lengths[i:i + train_end] = lengths[:train_end]

                test_trajectories[test_start:test_end] = padded_trajectories[train_end:]
                test_labels[test_start:test_end] = labels[train_end:]
                test_lengths[test_start:test_end] = lengths[train_end:]
                continue  # Skip rest of loop for this iteration

            dataset_traj[idx_start:idx_end] = padded_trajectories
            dataset_labels[idx_start:idx_end] = labels
            dataset_lengths[idx_start:idx_end] = lengths

def train_model(model: nn.Module, train_loader: DataLoader, 
                num_epochs: int = 10, learning_rate: float = 1e-3):
    """Train the model with mixed precision"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # scaler = torch.GradScaler(device=device.type)
    
    model.train()
    best_loss = float('inf')    # TODO: Save and load last loss
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct, total = 0, 0
        
        for (inputs, labels, lengths) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            # with torch.autocast(device_type=device.type):
            #     outputs = model(inputs, lengths)
            #     if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            #         print("NaN or Inf detected in outputs!")
            #         return

            #     loss = criterion(outputs, labels)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # Note: If NaN loss: activate this part and
            # deactivate scaler and autocast part
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # TODO: Add validation

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'best_model.pt')

def eval_model(model: nn.Module, test_loader: DataLoader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc="Evaluating"):
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['MRU', 'MUA', 'Singer'],
                yticklabels=['MRU', 'MUA', 'Singer'], 
                cbar_kws={'label': 'Percentage'})
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

def main():
    # Args
    parser = argparse.ArgumentParser(description="Train and evaluate trajectory classification model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--regen_data", action="store_true", help="Regenerate dataset if it exists")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate the pretrained model without training")
    args = parser.parse_args()

    # Set random seed
    # set_seed()

    # Device setup
    # device = torch.device(args.device)

    # Data generation and storage
    h5_path = 'trajectory_data.h5'
    if args.regen_data or not os.path.exists(h5_path):
        print("Generating and saving new data...")
        # TODO: Fix samples or lengths
        generate_and_save_data(
            filepath=h5_path,
            n_samples=11_000,
            traj_length=[15, 60 * 10]
        )
    
    # Create datasets
    train_dataset = TrajectoryDataset(h5_path, 'train')
    test_dataset = TrajectoryDataset(h5_path, 'test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Initialize the model
    model = TrajectoryClassifier().to(device)

    # Check for pre-trained model
    model_path = 'best_model.pt'
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Training
    if not args.eval_only:
        train_model(model, train_loader, num_epochs=50)

    # Evaluation and confusion matrix
    eval_model(model, test_loader)


if __name__ == "__main__":
    main()