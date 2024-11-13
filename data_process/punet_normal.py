import os
import numpy as np
import trimesh
import random
import multiprocessing
from functools import partial
import open3d as o3d

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

def process_mesh_and_pointcloud(mesh_filepath, xyz_filepath, output_dir):
    """
    Process a single .off mesh file and corresponding .xyz point cloud file,
    compute normals, and save a new .xyz file with normals.

    Args:
        mesh_filepath (str): Path to the .off mesh file.
        xyz_filepath (str): Path to the .xyz point cloud file.
        output_dir (str): Directory to save the processed .xyz file with normals.
    """
    try:
        # Load mesh file (.off)
        mesh = trimesh.load_mesh(mesh_filepath)

        if not isinstance(mesh, trimesh.Trimesh):
            # If the loaded object is not a Trimesh, attempt to concatenate into a single Trimesh
            mesh = trimesh.util.concatenate(mesh.dump())
        
        # Get vertices and face indices
        vertices = mesh.vertices
        faces = mesh.faces

        # Compute face normals
        mesh_normals = mesh.face_normals

        # Initialize vertex normals to zero
        vertex_normals = np.zeros(vertices.shape, dtype=np.float32)

        # Accumulate face normals to vertex normals
        for i, face in enumerate(faces):
            for vertex in face:
                vertex_normals[vertex] += mesh_normals[i]

        # Normalize vertex normals
        norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norm[norm == 0] = 1  # Prevent division by zero
        vertex_normals = vertex_normals / norm

        # Load corresponding point cloud file (.xyz)
        point_cloud = np.loadtxt(xyz_filepath)

        # Build KDTree using Open3D for nearest neighbor search
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        # Assign normals to each point in the point cloud
        point_normals = np.zeros_like(point_cloud)
        
        for i, point in enumerate(point_cloud):
            _, idx, _ = kd_tree.search_knn_vector_3d(point, 1)  # Find nearest vertex

            # Boundary check to ensure index is within range
            nearest_idx = min(max(idx[0], 0), len(vertex_normals) - 1)
            point_normals[i] = vertex_normals[nearest_idx]

        # Save the point cloud with normals to a new file
        output_filepath = os.path.join(output_dir, os.path.basename(xyz_filepath))
        np.savetxt(output_filepath, np.hstack((point_cloud, point_normals)), fmt='%.6f')
        print(f'Successfully processed and saved: {output_filepath}')

    except Exception as e:
        print(f"Error processing {mesh_filepath}: {e}")

def batch_process(off_dir, xyz_dir, output_dir):
    """
    Batch process mesh and point cloud directories to generate point clouds with normals.

    Args:
        off_dir (str): Directory containing .off mesh files.
        xyz_dir (str): Directory containing .xyz point cloud files.
        output_dir (str): Directory to save the processed .xyz files with normals.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the .off directory to find mesh files
    mesh_paths = []
    for root, dirs, files in os.walk(off_dir):
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
        process_mesh_and_pointcloud, 
        original_output_dir=output_dir, 
        noisy_output_dir=None, 
        noise_levels=None
    )
    pool.map(
        lambda mesh_path: process_mesh_and_pointcloud(
            mesh_path, 
            os.path.join(xyz_dir, os.path.splitext(os.path.basename(mesh_path))[0] + '.xyz'), 
            output_dir
        ), 
        mesh_paths
    )
    pool.close()
    pool.join()

if __name__ == "__main__":
    # Define directories
    off_dir = "./data/meshes/test"  # Replace with your .off directory path
    xyz_dir = "./data/pointclouds/test/10000_poisson"  # Replace with your .xyz directory path
    output_dir = "./data/NoisyDataSets/PUNet_normal/test/10k"  # Replace with your desired output directory path

    # Start batch processing
    batch_process(off_dir, xyz_dir, output_dir)
