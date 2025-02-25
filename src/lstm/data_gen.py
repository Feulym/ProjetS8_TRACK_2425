import os
import datetime
import argparse
import h5py
import numpy as np

from typing import Tuple, List
from package.common import trajectoire_XY
from MRU import trajec_MRU
from MUA import Trajec_MUA
from Singer import traj_singer


# ----------------------------
# Data Generation Utilities
# ----------------------------
def generate_synthetic_trajectories(n_samples: int = 500, min_N: int = 50, max_N: int = 150,
                                    Tech: float = 0.1, min_sigma2: float = 0.001, max_sigma2: float = 0.1,
                                    min_tau:float = 1, max_tau: float = 300,
                                    min_sigma_m2:float = 1e-4, max_sigma_m2:float = 1) \
                                        -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    Generate synthetic trajectories.
    
    Args:
        n_samples: Number of trajectories to generate
        min_N/max_N: Min/max trajectory length
        Tech: Sampling period
        min_sigma2/max_sigma2: Min/max noise variance
        min_tau/max_tau: Min/max maneuver time (Singer model)
        min_sigma_m2/max_sigma_m2: Min/max acceleration magnitude (Singer model)
    
    Returns:
        tuple: (trajectories, labels, lengths)
    """
    trajectories = []
    labels = []
    lengths = np.random.randint(min_N, max_N, size=n_samples, dtype=np.int32)

    traj_types = np.random.choice([0, 1, 2], size=n_samples)  # 0=MRU, 1=MUA, 2=Singer
    sigma2_values = np.random.uniform(min_sigma2, max_sigma2, size=n_samples)
    
    # Pre-generate Singer model parameters
    singer_mask = (traj_types == 2)
    n_singer = np.sum(singer_mask)
    if n_singer > 0:
        tau_values = np.random.uniform(min_tau, max_tau, size=n_singer)
        sigma_m2_values = np.random.uniform(min_sigma_m2, max_sigma_m2, size=n_singer)  
    singer_idx = 0

    for i in range(n_samples):
        L = lengths[i]
        sigma2 = sigma2_values[i]
        noise = np.random.normal(0, 12, (L, 2))  # GPS noise: +/-10m
        
        if traj_types[i] == 2:  # Singer
            tau = tau_values[singer_idx]
            sigma_m2 = sigma_m2_values[singer_idx]
            singer_idx += 1
            
            alpha = 1 / tau
            new_sigma2 = 2 * alpha * sigma_m2
            
            _, X, _, Y = trajectoire_XY(traj_singer, L, Tech, new_sigma2, alpha)
            
        elif traj_types[i] == 1:    # MUA
            _, X, _, Y = trajectoire_XY(Trajec_MUA, L, Tech, sigma2)
            
        else:   # MRU
            _, X, _, Y = trajectoire_XY(trajec_MRU, L, Tech, sigma2)

        traj = np.stack((X[0, :], Y[0, :]), axis=1) + noise
        trajectories.append(traj)
        labels.append(traj_types[i])
    
    return trajectories, labels, lengths


def generate_and_save_data(filepath: str, n_samples: int = 200_000, traj_length: List[int] = [15, 3600], batch_size=1_000):
    """
    Generate synthetic trajectory data and save it to an HDF5 file.
    
    The data is split into training (80%) and testing (20%) sets.
    """
    
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    min_seq_len, max_seq_len = traj_length
    
    with h5py.File(filepath, 'w') as f:
        train_group = f.create_group('train')
        test_group = f.create_group('test')

        train_trajectories = train_group.create_dataset('trajectories', shape=(train_size, max_seq_len, 2), dtype='float32')
        train_labels = train_group.create_dataset('labels', (train_size,), dtype='uint8')
        train_lengths = train_group.create_dataset('lengths', (train_size,), dtype='int32')

        test_trajectories = test_group.create_dataset('trajectories', shape=(test_size, max_seq_len, 2), dtype='float32')
        test_labels = test_group.create_dataset('labels', (test_size,), dtype='uint8')
        test_lengths = test_group.create_dataset('lengths', (test_size,), dtype='int32')

        for i in tqdm(range(0, n_samples, batch_size), desc="Generating data"):
            current_batch_size = min(batch_size, n_samples - i)
            trajectories, labels, lengths = generate_synthetic_trajectories(
                # TODO: kwargs
                n_samples=current_batch_size,
                min_N=min_seq_len,
                max_N=max_seq_len,
                Tech=1
            )
            
            padded_trajectories = np.array([
                np.pad(traj, ((0, max_seq_len - len(traj)), (0, 0)), mode='constant')
                for traj in trajectories
            ], dtype='float32')

            if i + current_batch_size <= train_size:    # If still in the training set
                dataset_traj, dataset_labels, dataset_lengths = train_trajectories, train_labels, train_lengths
                idx_start, idx_end = i, i + current_batch_size
            elif i >= train_size:   # If in the test set
                dataset_traj, dataset_labels, dataset_lengths = test_trajectories, test_labels, test_lengths
                idx_start, idx_end = i - train_size, i - train_size + current_batch_size
            else:   # If batch spans training and test set
                train_end = train_size - i
                test_end = current_batch_size - train_end

                train_trajectories[i:i + train_end] = padded_trajectories[:train_end]
                train_labels[i:i + train_end] = labels[:train_end]
                train_lengths[i:i + train_end] = lengths[:train_end]

                test_trajectories[0:test_end] = padded_trajectories[train_end:]
                test_labels[0:test_end] = labels[train_end:]
                test_lengths[0:test_end] = lengths[train_end:]
                continue  # Skip rest of loop for this iteration

            dataset_traj[idx_start:idx_end] = padded_trajectories
            dataset_labels[idx_start:idx_end] = labels
            dataset_lengths[idx_start:idx_end] = lengths

