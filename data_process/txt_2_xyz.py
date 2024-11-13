import os

def convert_txt_to_xyz(src_folder, dest_folder):
    """
    Recursively traverse all folders and files in src_folder,
    convert all .txt files to .xyz files keeping only the first three columns,
    and save them to dest_folder.
    Preserve the original directory structure.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        
        if os.path.isdir(src_path):
            convert_txt_to_xyz(src_path, dest_path)
        else:
            if item.endswith('.txt'):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                dest_path = os.path.splitext(dest_path)[0] + '.xyz'
                
                with open(src_path, 'r') as src_file, \
                     open(dest_path, 'w') as dest_file:
                    for line in src_file:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            dest_file.write(' '.join(parts[:3]) + '\n')

# Specify source and destination folder paths
source_folder = 'path_to_source_folder'
destination_folder = 'path_to_destination_folder'

# Start conversion
convert_txt_to_xyz(source_folder, destination_folder)
