import os
import numpy as np
import trimesh
import random
import multiprocessing
from functools import partial

def poisson_disk_sampling(mesh, num_points):
    """
    Perform Poisson disk sampling on a given mesh to generate a specified number of points.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        num_points (int): The number of points to sample.

    Returns:
        np.ndarray: Sampled point cloud of shape (num_points, 3).
    """
    area = mesh.area_faces
    areasum = np.sum(area)
    sample_prob = area / areasum
    num_faces = len(mesh.faces)

    # Calculate the number of points to sample per face based on face area
    num_samples_per_face = np.ceil(sample_prob * num_points).astype(int)

    # Generate sampled points
    sampled_points = []
    for i in range(num_faces):
        face = mesh.faces[i]
        vert_indices = face
        verts = mesh.vertices[vert_indices]

        # Generate random points on the current face
        for _ in range(num_samples_per_face[i]):
            u = random.random()
            v = random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            w = 1 - u - v
            point = u * verts[0] + v * verts[1] + w * verts[2]
            sampled_points.append(point)

    # If more points are sampled than needed, randomly select num_points
    if len(sampled_points) > num_points:
        sampled_points = random.sample(sampled_points, num_points)
    else:
        # If fewer points are sampled, randomly duplicate some points
        while len(sampled_points) < num_points:
            sampled_points.append(random.choice(sampled_points))

    return np.array(sampled_points)

def add_gaussian_noise(points, std_dev):
    """
    Add Gaussian noise to a point cloud.

    Args:
        points (np.ndarray): Original point cloud of shape (N, 3).
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy point cloud.
    """
    noise = np.random.normal(0, std_dev, size=points.shape)
    noisy_points = points + noise
    return noisy_points

def process_mesh(mesh_path, original_output_dir, noisy_output_dir, noise_levels):
    """
    Process a single mesh file: perform Poisson disk sampling, generate point cloud, add noise, and save results.

    Args:
        mesh_path (str): Path to the mesh file.
        original_output_dir (str): Directory to save the original point cloud.
        noisy_output_dir (str): Directory to save the noisy point clouds.
        noise_levels (list of float): List of noise levels as a proportion of the bounding box diagonal length.
    """
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh):
            # If the loaded object is not a Trimesh, attempt to concatenate into a single Trimesh
            mesh = trimesh.util.concatenate(mesh.dump())
        
        # Perform Poisson disk sampling to generate a point cloud with 10,000 points
        points = poisson_disk_sampling(mesh, 10000)

        # Calculate the diagonal length of the bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)

        # Save the original point cloud to original_output_dir
        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        original_output_path = os.path.join(original_output_dir, f"{mesh_name}.txt")
        np.savetxt(original_output_path, points, fmt='%.6f')

        # Add Gaussian noise for each noise level and save to noisy_output_dir
        for noise_level in noise_levels:
            std_dev = diagonal * noise_level
            noisy_points = add_gaussian_noise(points, std_dev)
            noise_suffix = f"_{int(noise_level * 1000)}"  # e.g., 0.005 -> _5
            noisy_output_path = os.path.join(noisy_output_dir, f"{mesh_name}{noise_suffix}_10k.txt")  # Point cloud count
            np.savetxt(noisy_output_path, noisy_points, fmt='%.6f')

        print(f"Processed: {mesh_name}")

    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")

def main(mesh_root_dir, original_output_dir, noisy_output_dir):
    """
    Process all mesh files in the root directory by performing sampling and adding noise.

    Args:
        mesh_root_dir (str): Root directory containing mesh files.
        original_output_dir (str): Directory to save original point clouds.
        noisy_output_dir (str): Directory to save noisy point clouds.
    """
    # Define noise levels
    noise_levels = [0.005, 0.01, 0.015, 0.02, 0.03]  # Corresponding to 0.5%, 1%, 1.5%, 2%, 3%

    # Create output directories if they don't exist
    if not os.path.exists(original_output_dir):
        os.makedirs(original_output_dir)
    if not os.path.exists(noisy_output_dir):
        os.makedirs(noisy_output_dir)

    # Traverse all subdirectories in the mesh root directory to find mesh files
    mesh_paths = []
    for root, dirs, files in os.walk(mesh_root_dir):
        for file in files:
            if file.lower().endswith('.off'):
                mesh_path = os.path.join(root, file)
                # Only process meshes in the 'train' subdirectory
                if 'train' in mesh_path.lower():
                    mesh_paths.append(mesh_path)

    print(f"Total meshes to process: {len(mesh_paths)}")

    # Use multiprocessing to accelerate processing
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    func = partial(
        process_mesh, 
        original_output_dir=original_output_dir, 
        noisy_output_dir=noisy_output_dir, 
        noise_levels=noise_levels
    )
    pool.map(func, mesh_paths)
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Set mesh root directory and output directories
    mesh_root_dir = "./data/ModelNet40_mesh"                  # Replace with your mesh root directory path
    original_output_dir = "./data/ModelNet40_train_gtpc_10k"  # Replace with the output directory for original point clouds
    noisy_output_dir = "./data/NoisyDataSets/ModelNet40_noisy" # Replace with the output directory for noisy point clouds

    main(mesh_root_dir, original_output_dir, noisy_output_dir)
