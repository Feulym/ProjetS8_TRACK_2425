import os
import sys
import argparse
import h5py
import numpy as np
from typing import Tuple, List, TypedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
# Fix package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from package.common import trajectoire_XY
from package.mru import trajec_MRU
from package.mua import Trajec_MUA
from package.singer import traj_singer


class TrajectoryGenParams(TypedDict):
    n_samples: int
    min_N: int
    max_N: int
    Tech: float
    min_sigma2: float
    max_sigma2: float
    min_tau: float
    max_tau: float
    min_sigma_m2: float
    max_sigma_m2: float
    noised: bool

def mean_22(X):
    """
    Compute the mean of the gap between two consecutive values 
    Args:
        X : A Vector off which we want to get the gap mean
    
    Returns:
        Mean of the gaps
    """
    X_2 = np.array([0])
    X_2 = np.concatenate((X_2, X[:-1]))
    return np.mean(abs(X-X_2))

def add_Noise(traj, mu_X=0, var_X=None, mu_Y=0, var_Y=None):
    if var_X == None :
        var_X=mean_22(traj[:,0])
    if var_Y == None :
        var_Y=mean_22(traj[:,1])

    print(var_X,var_Y)

    noise_X = np.random.normal(mu_X, var_X, (traj[:,0].shape[0], 1))  # GPS noise linked to the trajectory magnitude
    noise_Y = np.random.normal(mu_Y, var_Y, (traj[:,1].shape[0], 1))
    noise = np.stack((noise_X[:, 0], noise_Y[:, 0]), axis=1)
    return traj + noise

def generate_synthetic_trajectories(params: TrajectoryGenParams) -> Tuple[List[np.ndarray], List[int], np.ndarray]:
    """
    Generate synthetic trajectories based on provided parameters.

    Args:
        params: A dictionary containing trajectory generation parameters.

    Returns:
        trajectories: List of trajectory arrays.
        labels: List of trajectory labels.
        lengths: Array of trajectory lengths.
    """
    n_samples = params["n_samples"]
    min_N = params["min_N"]
    max_N = params["max_N"]
    Tech = params["Tech"]
    min_sigma2 = params["min_sigma2"]
    max_sigma2 = params["max_sigma2"]
    min_tau = params["min_tau"]
    max_tau = params["max_tau"]
    min_sigma_m2 = params["min_sigma_m2"]
    max_sigma_m2 = params["max_sigma_m2"]

    trajectories = []
    labels = []

    lengths = np.random.randint(min_N, max_N + 1, size=n_samples, dtype=np.int32)
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
        if traj_types[i] == 2:  # Singer model
            tau = tau_values[singer_idx]
            sigma_m2 = sigma_m2_values[singer_idx]
            singer_idx += 1

            alpha = 1 / tau
            new_sigma2 = 2 * alpha * sigma_m2

            _, X, _, Y = trajectoire_XY(traj_singer, L, Tech, new_sigma2, alpha)
            print("Singer mean", np.mean(X), np.mean(Y))
        elif traj_types[i] == 1:  # MUA model
            _, X, _, Y = trajectoire_XY(Trajec_MUA, L, Tech, sigma2)
            print("MUA mean", np.mean(X), np.mean(Y))
        else:  # MRU model
            _, X, _, Y = trajectoire_XY(trajec_MRU, L, Tech, sigma2)
            print("MRU mean", np.mean(X), np.mean(Y))

        traj = np.stack((X[0, :], Y[0, :]), axis=1)
        if params["noised"]:
            traj = add_Noise(traj)

        trajectories.append(traj)
        labels.append(traj_types[i])
    return trajectories, labels, lengths


def generate_and_save_data(filepath: str, params: TrajectoryGenParams, batch_size: int = 1000) -> None:
    """
    Generate synthetic trajectory data and save it to an HDF5 file.

    The dataset is saved as a unified dataset with keys:
      - "trajectories": Padded trajectory data.
      - "labels": Trajectory labels.
      - "lengths": Original lengths of each trajectory.
      
    Args:
        filepath: Path to save the HDF5 file.
        params: A dictionary of generation parameters.
        batch_size: Number of samples to generate per batch.
    """
    total_samples = params["n_samples"]
    max_seq_len = params["max_N"]

    with h5py.File(filepath, 'w') as f:
        ds_trajectories = f.create_dataset("trajectories", shape=(total_samples, max_seq_len, 2), dtype="float32")
        ds_labels = f.create_dataset("labels", shape=(total_samples,), dtype="uint8")
        ds_lengths = f.create_dataset("lengths", shape=(total_samples,), dtype="int32")
        
        num_batches = (total_samples + batch_size - 1) // batch_size
        sample_counter = 0
        
        # Create a copy of the parameters to modify the current batch size.
        batch_params = params.copy()

        for _ in tqdm(range(num_batches), desc="Generating data"):
            current_batch_size = min(batch_size, total_samples - sample_counter)
            batch_params["n_samples"] = current_batch_size
            
            trajectories, labels, lengths = generate_synthetic_trajectories(batch_params)
            
            # Pad each trajectory to have the same length (max_seq_len)
            padded_trajectories = np.array([
                np.pad(traj, ((0, max_seq_len - len(traj)), (0, 0)), mode='constant')
                for traj in trajectories
            ], dtype="float32")
            
            ds_trajectories[sample_counter:sample_counter + current_batch_size] = padded_trajectories
            ds_labels[sample_counter:sample_counter + current_batch_size] = labels
            ds_lengths[sample_counter:sample_counter + current_batch_size] = lengths
            
            sample_counter += current_batch_size


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic trajectory data.")
    parser.add_argument("--output_file", type=str, required=True, help="Output HDF5 file path.")
    parser.add_argument("--n_samples", type=int, default=200_000, help="Total number of trajectories.")
    parser.add_argument("--min_N", type=int, default=15, help="Minimum trajectory length.")
    parser.add_argument("--max_N", type=int, default=3600, help="Maximum trajectory length.")
    parser.add_argument("--Tech", type=float, default=1.0, help="Sampling period.")
    parser.add_argument("--min_sigma2", type=float, default=0.001, help="Minimum noise variance.")
    parser.add_argument("--max_sigma2", type=float, default=0.1, help="Maximum noise variance.")
    parser.add_argument("--min_tau", type=float, default=1, help="Minimum maneuver time (Singer model).")
    parser.add_argument("--max_tau", type=float, default=300, help="Maximum maneuver time (Singer model).")
    parser.add_argument("--min_sigma_m2", type=float, default=1e-4, help="Minimum acceleration magnitude (Singer model).")
    parser.add_argument("--max_sigma_m2", type=float, default=1, help="Maximum acceleration magnitude (Singer model).")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for generation.")
    parser.add_argument("--noised", type=int, default=0, help="Generate Noised or Unnoised trajectories")
    args = parser.parse_args()
    
    # Build generation parameters using the TypedDict.
    gen_params: TrajectoryGenParams = {
        "n_samples": args.n_samples,
        "min_N": args.min_N,
        "max_N": args.max_N,
        "Tech": args.Tech,
        "min_sigma2": args.min_sigma2,
        "max_sigma2": args.max_sigma2,
        "min_tau": args.min_tau,
        "max_tau": args.max_tau,
        "min_sigma_m2": args.min_sigma_m2,
        "max_sigma_m2": args.max_sigma_m2,
        "noised": True if args.noised == 1 else False
    }
    
    generate_and_save_data(args.output_file, gen_params, batch_size=args.batch_size)
    print(f"Data successfully saved to {args.output_file}")


if __name__ == "__main__":
    main()
