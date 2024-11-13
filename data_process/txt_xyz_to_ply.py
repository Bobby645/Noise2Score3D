import os
import numpy as np

def load_xyz(file_path):
    """
    Load point cloud data from a .xyz or .txt file, each line contains x, y, z coordinates.
    """
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = map(float, parts[:3])
                    points.append([x, y, z])
    return np.array(points)

def save_ply(file_path, points):
    """
    Save point cloud data to a .ply file.

    Parameters:
    - file_path: Path to save the .ply file
    - points: Point cloud data, shape (N, 3)
    """
    num_points = points.shape[0]
    ply_header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
end_header
"""
    with open(file_path, 'w') as f:
        f.write(ply_header)
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

def xyz_to_ply(input_file, output_file):
    """
    Convert a .xyz or .txt point cloud file to .ply format.

    Parameters:
    - input_file: Path to the input .xyz or .txt point cloud file
    - output_file: Path to the output .ply file
    """
    points = load_xyz(input_file)
    save_ply(output_file, points)
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    # Path to the input .xyz or .txt point cloud file
    input_file = "path_to_input_file.txt"  # Replace with your input file path

    # Path to save the .ply file
    output_file = "path_to_output_file.ply"  # Replace with your desired output .ply file path

    # Execute the conversion
    xyz_to_ply(input_file, output_file)
