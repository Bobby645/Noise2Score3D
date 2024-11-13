import os
import torch
import numpy as np
import re

def load_point_cloud(file_path):
    """
    Read a point cloud file (.txt or .xyz) and return the point cloud data as a NumPy array.
    Supports coordinates separated by spaces or commas.
    
    Args:
        file_path (str): Path to the point cloud file.
        
    Returns:
        np.ndarray: Array of point coordinates with shape (N, 3).
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Split using commas or spaces
                parts = re.split(r'[,\s]+', line.strip())
                # Extract the first 3 values as coordinates
                if len(parts) >= 3:
                    coords = [float(x) for x in parts[:3]]
                    points.append(coords)
    return np.array(points)

def add_gaussian_noise(points, std_dev):
    """
    Add Gaussian noise to a point cloud.
    
    Args:
        points (np.ndarray): Original point cloud with shape (N, 3).
        std_dev (float): Standard deviation of the Gaussian noise.
        
    Returns:
        np.ndarray: Noisy point cloud.
    """
    noise = np.random.normal(0, std_dev, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def process_point_clouds(input_dir, output_dir, noise_levels):
    """
    Add Gaussian noise to point cloud files in the input directory and save the results to the output directory.
    
    Args:
        input_dir (str): Directory containing original point cloud files.
        output_dir (str): Directory to save the noisy point cloud files.
        noise_levels (list of float): List of noise levels as a proportion of the bounding box diagonal length.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all point cloud files (.txt or .xyz) in the input directory
    point_cloud_files = [
        f for f in os.listdir(input_dir) 
        if f.lower().endswith('.txt') or f.lower().endswith('.xyz')
    ]

    for file_name in point_cloud_files:
        file_path = os.path.join(input_dir, file_name)
        points = load_point_cloud(file_path)

        # Calculate the diagonal length of the bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)

        base_name = os.path.splitext(file_name)[0]

        for noise_level in noise_levels:
            std_dev = diagonal * noise_level
            noisy_points = add_gaussian_noise(points, std_dev)
            noise_suffix = f"_{int(noise_level * 1000)}"  # e.g., 0.005 -> _5

            noisy_file_name = f"{base_name}{noise_suffix}_10k.txt"
            noisy_file_path = os.path.join(output_dir, noisy_file_name)

            # Save the noisy point cloud to a file with space as the delimiter
            np.savetxt(noisy_file_path, noisy_points, fmt='%.6f', delimiter=' ')

        print(f"Processed file: {file_name}")

if __name__ == "__main__":
    # Set input and output directories
    input_dir = "./data/pointclouds/test/10000_poisson"     # Replace with your point cloud directory path
    output_dir = "./data/NoisyDataSets/PUNet_test"          # Replace with your desired output directory path

    # Define noise levels (0.5%, 1%, 1.5%, 2%, 3%)
    noise_levels = [0.005, 0.01, 0.015, 0.02, 0.03]

    process_point_clouds(input_dir, output_dir, noise_levels)
