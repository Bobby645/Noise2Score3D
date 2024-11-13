import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

class PointCloudTVLoss(nn.Module):
    def __init__(self, k=16):
        super(PointCloudTVLoss, self).__init__()
        self.k = k
        self.e = 1e-12  # Small constant to avoid division by zero

    def forward(self, point_cloud):
        # point_cloud: B x N x 3
        batch_size, num_points, _ = point_cloud.size()
        tv_loss = 0.0
        
        for b in range(batch_size):
            # Compute neighborhood for each point
            nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(point_cloud[b].cpu().detach().numpy())
            distances, indices = nbrs.kneighbors(point_cloud[b].cpu().detach().numpy())
            
            # Calculate total variation
            pc = point_cloud[b]
            h_tv = 0.0
            for i in range(num_points):
                neighbors = pc[indices[i]]  # Neighboring points of the current point
                diff = pc[i] - neighbors  # Difference between current point and neighbors
                h_tv += torch.sum(torch.sqrt(torch.sum(diff ** 2, dim=1) + self.e))  # Charbonnier norm
            
            tv_loss += h_tv / num_points
        
        return tv_loss / batch_size

def get_all_xyz_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.xyz')]


if __name__ == '__main__':

    # Initialize loss function
    tv_loss = PointCloudTVLoss(k=6)

    # Define the directory containing .xyz files
    file_directory = './log/denoise/KPconv/test'
    
    # Retrieve all .xyz file paths
    file_paths = get_all_xyz_files(file_directory)
    
    # Load all point cloud files
    point_clouds = [np.loadtxt(file) for file in file_paths]
    point_cloud_tensor = torch.tensor(np.array(point_clouds), dtype=torch.float32)

    # Compute total variation loss
    loss = tv_loss(point_cloud_tensor)
    print("Point Cloud Total Variation Loss:", loss.item())
