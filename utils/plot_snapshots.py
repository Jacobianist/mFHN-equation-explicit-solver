#!/usr/bin/env python3
"""
Plot snapshots from mFHN simulation HDF5 output files.

This script loads an HDF5 file produced by the mFHN explicit solver,
extracts the 'u' dataset, and plots the first and last time snapshots.
Supports both 1D and 2D simulation data.

Usage:
    python plot_snapshots.py <path_to_result.h5>

Examples:
    python plot_snapshots.py results/2026-03-16/120000_dim1_N1024/result.h5
    python plot_snapshots.py results/2026-03-16/120000_dim2_N1024/result.h5
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_u_dataset(filepath: str) -> np.ndarray:
    """
    Load the 'u' dataset from an HDF5 file.

    Args:
        filepath: Path to the HDF5 file

    Returns:
        numpy array with shape (num_snapshots, N) for 1D
        or (num_snapshots, N, N) for 2D

    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If 'u' dataset is not found
    """
    with h5py.File(filepath, 'r') as f:
        u_data = f['u'][:]
    return u_data


def plot_1d(u_data: np.ndarray, filepath: str) -> None:
    """
    Plot 1D simulation snapshots.

    Creates a figure with two subplots showing the first and last
    time snapshots of the activator variable u(x).

    Args:
        u_data: Array of shape (num_snapshots, N)
        filepath: Path to the source HDF5 file (for title)
    """
    num_snapshots = u_data.shape[0]
    x = np.arange(u_data.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # First snapshot (t=0)
    axes[0].plot(x, u_data[0], 'b-', linewidth=1.5, label='u(x)')
    axes[0].set_xlabel('Grid point index')
    axes[0].set_ylabel('u')
    axes[0].set_title(f'Initial State (snapshot 0)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim([u_data.min() - 0.1, u_data.max() + 0.1])

    # Last snapshot
    axes[1].plot(x, u_data[-1], 'ro--', linewidth=0.5, label='u(x)')
    axes[1].set_xlabel('Grid point index')
    axes[1].set_ylabel('u')
    axes[1].set_title(f'Final State (snapshot {num_snapshots - 1})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([u_data.min() - 0.1, u_data.max() + 0.1])

    plt.suptitle(f'mFHN 1D Simulation - Activator u\nFile: {filepath}', fontsize=10)
    plt.tight_layout()


def plot_2d(u_data: np.ndarray, filepath: str) -> None:
    """
    Plot 2D simulation snapshots as heatmaps.

    Creates a figure with two subplots showing the first and last
    time snapshots of the activator variable u(x, y).

    Args:
        u_data: Array of shape (num_snapshots, N, N)
        filepath: Path to the source HDF5 file (for title)
    """
    num_snapshots = u_data.shape[0]
    N = u_data.shape[1]
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = u_data.min()
    vmax = u_data.max()

    # First snapshot (t=0)
    im0 = axes[0].pcolormesh(X, Y, u_data[0], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('x index')
    axes[0].set_ylabel('y index')
    axes[0].set_title(f'Initial State (snapshot 0)')
    axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], label='u')

    # Last snapshot
    im1 = axes[1].pcolormesh(X, Y, u_data[-1], shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('x index')
    axes[1].set_ylabel('y index')
    axes[1].set_title(f'Final State (snapshot {num_snapshots - 1})')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], label='u')

    plt.suptitle(f'mFHN 2D Simulation - Activator u\nFile: {filepath}', fontsize=10)
    plt.tight_layout()


def main():
    """Main entry point for the plot script."""
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"Error: Expected 1 argument, got {len(sys.argv) - 1}")
        print(f"Usage: python {sys.argv[0]} <path_to_result.h5>")
        sys.exit(1)

    filepath = sys.argv[1]

    # Load data
    print(f"Loading data from: {filepath}")
    try:
        u_data = load_u_dataset(filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Dataset 'u' not found in file. Available datasets: {list(h5py.File(filepath, 'r').keys())}")
        sys.exit(1)

    print(f"Data shape: {u_data.shape}")
    print(f"Number of snapshots: {u_data.shape[0]}")
    print(f"Grid size: {u_data.shape[1]}x{u_data.shape[2] if len(u_data.shape) == 3 else 1}")

    # Determine dimensionality and plot
    if len(u_data.shape) == 2:
        print("Detected: 1D simulation")
        plot_1d(u_data, filepath)
    elif len(u_data.shape) == 3:
        print("Detected: 2D simulation")
        plot_2d(u_data, filepath)
    else:
        print(f"Error: Unexpected data shape: {u_data.shape}")
        sys.exit(1)

    plt.show()


if __name__ == '__main__':
    main()
