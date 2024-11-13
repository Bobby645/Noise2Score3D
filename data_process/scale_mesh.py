import os
import numpy as np
import re  
 # Import regular expressions module
 
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
                coords = [float(x) for x in parts[:3]]
                points.append(coords)
    return np.array(points)

def load_off(file_path):
    """
    Load a mesh file in OFF format and return vertices and faces.
    
    Args:
        file_path (str): Path to the OFF file.
        
    Returns:
        tuple: (vertices, faces) as NumPy arrays.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Use regular expressions to match 'OFF' and the following numbers
    first_line = lines[0].strip()
    match = re.match(r'(OFF)(.*)', first_line)
    if not match:
        raise ValueError(f"Invalid OFF file format: {file_path}")
    
    # Extract 'OFF' and the remaining content
    header, rest = match.groups()
    
    if header != 'OFF':
        raise ValueError(f"Invalid OFF file header: {file_path}")
    
    # Initialize variables
    n_verts = n_faces = n_edges = None
    vertex_start = 0  # Line number where vertex data starts
    
    # Check if there are numbers following 'OFF'
    if rest.strip():
        # There are numbers after 'OFF', possibly without spaces, handle accordingly
        line_data = rest.strip() + ' ' + lines[1].strip()
        nums = line_data.strip().split()
        n_verts, n_faces, n_edges = map(int, nums[:3])
        vertex_start = 1
    else:
        # No content after 'OFF', read the next line
        second_line = lines[1].strip()
        nums = second_line.strip().split()
        if len(nums) >= 3:
            n_verts, n_faces, n_edges = map(int, nums[:3])
            vertex_start = 2
        else:
            # Not enough information in the second line, read the third line
            third_line = lines[2].strip()
            nums += third_line.strip().split()
            n_verts, n_faces, n_edges = map(int, nums[:3])
            vertex_start = 3
    
    # Read vertices
    verts = []
    for i in range(vertex_start, vertex_start + n_verts):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            verts.append([float(x) for x in parts[:3]])
        else:
            raise ValueError(f"Invalid vertex data at line {i+1} in {file_path}")
    
    # Read faces
    faces = []
    for i in range(vertex_start + n_verts, vertex_start + n_verts + n_faces):
        face_data = lines[i].strip().split()
        if not face_data:
            continue  # Skip empty lines
        face_vertex_count = int(face_data[0])
        face_indices = [int(idx) for idx in face_data[1:1 + face_vertex_count]]
        if face_vertex_count == 3:
            faces.append(face_indices)
        elif face_vertex_count == 4:
            # Split quadrilateral into two triangles
            faces.append([face_indices[0], face_indices[1], face_indices[2]])
            faces.append([face_indices[0], face_indices[2], face_indices[3]])
        else:
            raise ValueError(f"Unsupported face with {face_vertex_count} vertices in {file_path}")
    
    return np.array(verts), np.array(faces)


def save_off(file_path, verts, faces):
    with open(file_path, 'w') as f:
        f.write('OFF\n')
        f.write(f"{len(verts)} {len(faces)} 0\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")  

def scale_and_align_mesh(verts, points):
    
    mesh_min = np.min(verts, axis=0)
    mesh_max = np.max(verts, axis=0)
    mesh_scale = mesh_max - mesh_min
    mesh_center = (mesh_max + mesh_min) / 2.0

    pointcloud_min = np.min(points, axis=0)
    pointcloud_max = np.max(points, axis=0)
    pointcloud_scale = pointcloud_max - pointcloud_min
    pointcloud_center = (pointcloud_max + pointcloud_min) / 2.0

    scale_factor = np.min(pointcloud_scale / mesh_scale)

    verts_scaled = verts * scale_factor

    mesh_scaled_min = np.min(verts_scaled, axis=0)
    mesh_scaled_max = np.max(verts_scaled, axis=0)
    mesh_scaled_center = (mesh_scaled_max + mesh_scaled_min) / 2.0

    translation = pointcloud_center - mesh_scaled_center

    verts_aligned = verts_scaled + translation

    return verts_aligned

def process_files(pointcloud_root_dir, off_root_dir, output_root_dir):
    
    for category in os.listdir(pointcloud_root_dir):
        category_pointcloud_dir = os.path.join(pointcloud_root_dir, category)
        if not os.path.isdir(category_pointcloud_dir):
            continue  

        print(f"Processing category: {category}")

        category_off_dir = os.path.join(off_root_dir, category)
        if not os.path.isdir(category_off_dir):
            print(f"OFF category directory not found: {category_off_dir}")
            continue

        category_output_dir = os.path.join(output_root_dir, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)

        for root, dirs, files in os.walk(category_pointcloud_dir):
            for file in files:
                if file.lower().endswith('.txt') or file.lower().endswith('.xyz'):
                    pointcloud_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(pointcloud_file_path, category_pointcloud_dir)
                    pointcloud_subdir = os.path.dirname(relative_path)
                    pointcloud_name = os.path.splitext(file)[0]

                    found_off = False
                    for split in ['train', 'test']:
                        off_dir_split = os.path.join(category_off_dir, split, pointcloud_subdir)
                        off_file_path = os.path.join(off_dir_split, pointcloud_name + '.off')

                        if os.path.exists(off_file_path):
                            found_off = True
                            output_dir_split = os.path.join(category_output_dir, split, pointcloud_subdir)
                            if not os.path.exists(output_dir_split):
                                os.makedirs(output_dir_split)
                            output_off_path = os.path.join(output_dir_split, pointcloud_name + '.off')

                            try:
                                points = load_point_cloud(pointcloud_file_path)
                                verts, faces = load_off(off_file_path)

                                verts_aligned = scale_and_align_mesh(verts, points)

                                save_off(output_off_path, verts_aligned, faces)

                                print(f"Processed and saved: {output_off_path}")

                            except Exception as e:
                                print(f"Error processing file {pointcloud_name}: {e}")

                            break  

                    if not found_off:
                        print(f"Corresponding OFF file not found for point cloud: {pointcloud_file_path}")

# 使用示例
if __name__ == "__main__":
    pointcloud_root_dir = "/modelnet40_normal_resampled"  
    off_root_dir = "Documents/ModelNet40"             
    output_root_dir = "ModelNet40_mesh"         

    process_files(pointcloud_root_dir, off_root_dir, output_root_dir)
