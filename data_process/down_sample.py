import os
import open3d as o3d
import numpy as np

def downsample_point_cloud_xyz(input_folder, output_folder, target_points=10000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.xyz'):
            file_path = os.path.join(input_folder, filename)
            print(f'Processing {file_path}...')
            
            # Load point cloud from .xyz file
            pcd = o3d.io.read_point_cloud(file_path, format='xyz')

            # Get current number of points
            current_points = np.asarray(pcd.points).shape[0]

            # Uniform downsampling
            if current_points > target_points:
                downsample_ratio = target_points / current_points
                pcd_downsampled = pcd.uniform_down_sample(int(1 / downsample_ratio))
                
                # Ensure exactly `target_points`
                sampled_points = pcd_downsampled.points[:target_points]
                pcd_downsampled.points = o3d.utility.Vector3dVector(sampled_points)

            else:
                pcd_downsampled = pcd  # If points are already below or at the target, no resampling.

            # Save the downsampled point cloud to .xyz
            output_path = os.path.join(output_folder, filename)
            o3d.io.write_point_cloud(output_path, pcd_downsampled, write_ascii=True)
            print(f'Saved downsampled point cloud to {output_path}')

input_folder = 'modelnet-40_subtest/gt_points_50k'
output_folder = 'modelnet-40_subtest/gt_points_10k'
downsample_point_cloud_xyz(input_folder, output_folder)
