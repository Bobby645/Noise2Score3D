import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import numpy as np
import random
import torch.nn.functional as F
from models.easy_kpconv.layers.kpconv_blocks import KPConvBlock, KPResidualBlock
from models.easy_kpconv.layers.unary_block import UnaryBlockPackMode
from models.easy_kpconv.ops.graph_pyramid import build_grid_and_radius_graph_pyramid
from models.easy_kpconv.ops.nearest_interpolate import nearest_interpolate_pack_mode
from typing import List, Tuple, Dict

def pack_points(points_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a list of points.

    Args:
        points_list (List[torch.Tensor]): The list of points to pack with a length of B, each in the shape of (Ni, 3).

    Returns:
        A tensor of the packed points in the shape of (N', 3), where N' = sum(Ni).
        A tensor of the lengths in the batch in the shape of (B).
    """
    points = torch.cat(points_list, dim=0)
    lengths = torch.tensor([p.shape[0] for p in points_list], dtype=torch.int64).cuda()
    
    return points, lengths

def dataloader(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Process input data to meet the KPConv network's input requirements.

    Args:
        data (torch.Tensor): Input data, point cloud of shape [B, N, 3].

    Returns:
        Dict[str, torch.Tensor]: Processed data dictionary containing merged point clouds and length arrays.
    """
    batch_size, num_points, _ = data.shape

    # Split the point cloud data into a list of individual point clouds
    points_list = [data[i] for i in range(batch_size)]
    
    if batch_size == 1:
        # If batch_size is 1, directly process the single point cloud
        collated_dict = {"points": points_list[0]}
        collated_dict["lengths"] = torch.tensor([collated_dict["points"].shape[0]], dtype=torch.int64).cuda()
    else:
        # Process multiple point clouds by packing them into a single large point cloud and generating length arrays
        points, lengths = pack_points(points_list)
        
        collated_dict = {
            "points": points,
            "lengths": lengths
        }
    
    collated_dict["batch_size"] = batch_size
    return collated_dict

def dataloader1(data: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Process input data to meet the KPConv network's input requirements.

    Args:
        data (torch.Tensor): Input data, point cloud of shape [B, N, 3].

    Returns:
        Dict[str, torch.Tensor]: Processed data dictionary containing original point clouds and length arrays.
    """
    batch_size, num_points, _ = data.shape

    collated_dict = {
        "points": data,  # Maintain the original input shape of the point cloud
        "lengths": torch.tensor([num_points] * batch_size, dtype=torch.int64),  # Length of each point cloud
        "batch_size": batch_size
    }

    return collated_dict

class Config:
    def __init__(self):
        self.first_subsampling_dl = 0.1
        self.kpconv_radius = 2.5
        self.input_dim = 3
        self.init_dim = 64
        self.num_kernel_points = 15
        self.kpconv_architecture = ['simple', 'simple', 'pool', 'simple', 'upsample']
        self.kernel_size = 15
        self.basic_voxel_size = 0.04
        self.num_stages = 5
        self.kpconv_sigma = 2.0
        self.neighbor_limits = [24, 40, 34, 35, 34] 


class get_model(nn.Module):
    def __init__(self, cfg, normal_channel=None):
        super().__init__()

        self.num_stages = cfg.num_stages
        self.voxel_size = cfg.basic_voxel_size
        self.kpconv_radius = cfg.kpconv_radius
        self.kpconv_sigma = cfg.kpconv_sigma
        self.neighbor_limits = cfg.neighbor_limits
        self.first_radius = self.voxel_size * self.kpconv_radius
        self.first_sigma = self.voxel_size * self.kpconv_sigma

        input_dim = cfg.input_dim
        first_dim = cfg.init_dim
        kernel_size = cfg.kernel_size
        first_radius = self.first_radius
        first_sigma = self.first_sigma

        self.encoder1_1 = KPConvBlock(input_dim, first_dim, kernel_size, first_radius, first_sigma)
        self.encoder1_2 = KPResidualBlock(first_dim, first_dim * 2, kernel_size, first_radius, first_sigma)

        self.encoder2_1 = KPResidualBlock(
            first_dim * 2, first_dim * 2, kernel_size, first_radius, first_sigma, strided=True
        )
        self.encoder2_2 = KPResidualBlock(first_dim * 2, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2)
        self.encoder2_3 = KPResidualBlock(first_dim * 4, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2)

        self.encoder3_1 = KPResidualBlock(
            first_dim * 4, first_dim * 4, kernel_size, first_radius * 2, first_sigma * 2, strided=True
        )
        self.encoder3_2 = KPResidualBlock(first_dim * 4, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4)
        self.encoder3_3 = KPResidualBlock(first_dim * 8, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4)

        self.encoder4_1 = KPResidualBlock(
            first_dim * 8, first_dim * 8, kernel_size, first_radius * 4, first_sigma * 4, strided=True
        )
        self.encoder4_2 = KPResidualBlock(first_dim * 8, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8)
        self.encoder4_3 = KPResidualBlock(
            first_dim * 16, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8
        )

        self.encoder5_1 = KPResidualBlock(
            first_dim * 16, first_dim * 16, kernel_size, first_radius * 8, first_sigma * 8, strided=True
        )
        self.encoder5_2 = KPResidualBlock(
            first_dim * 16, first_dim * 32, kernel_size, first_radius * 16, first_sigma * 16
        )
        self.encoder5_3 = KPResidualBlock(
            first_dim * 32, first_dim * 32, kernel_size, first_radius * 16, first_sigma * 16
        )

        self.decoder4 = UnaryBlockPackMode(first_dim * 48, first_dim * 16)
        self.decoder3 = UnaryBlockPackMode(first_dim * 24, first_dim * 8)
        self.decoder2 = UnaryBlockPackMode(first_dim * 12, first_dim * 4)
        self.decoder1 = UnaryBlockPackMode(first_dim * 6, first_dim * 2)

        self.regressor = nn.Sequential(
            nn.Linear(first_dim * 2, first_dim),
            nn.GroupNorm(8, first_dim),
            nn.ReLU(),
            nn.Linear(first_dim, cfg.input_dim),
        )
        self.sigma_min = 0.004
        self.sigma_max = 0.034
        self.sigma_annealing = 800000

    def set_sigma(self, iter):
        perc = min((iter + 1) / float(self.sigma_annealing), 1.0)
        self.sigma = self.sigma_max * (1 - perc) + self.sigma_min * perc
        self.loss_sigma = self.sigma
        # self.sigma = random.uniform(self.sigma_min, self.sigma_max)

    def add_noise(self, input, sigma):
        mu = torch.randn_like(input)
        return input + sigma * mu, mu

    def forward(self, data_dict, iter):
        data_dict = dataloader(data_dict)
        output_dict = {}
        
        points = data_dict["points"]
        lengths = data_dict["lengths"]
        # Add noise
        # self.sigma = 0  # Inference
        self.set_sigma(iter)  # Set sigma for training iterations
        points, mu = self.add_noise(points, self.sigma)

        graph_pyramid = build_grid_and_radius_graph_pyramid(
            points, lengths, self.num_stages, self.voxel_size, self.first_radius, self.neighbor_limits
        )

        points_list = graph_pyramid["points"]
        neighbors_list = graph_pyramid["neighbors"]
        subsampling_list = graph_pyramid["subsampling"]
        upsampling_list = graph_pyramid["upsampling"]

        # feats_s1 = torch.cat([torch.ones_like(feats[:, :1]), feats], dim=1)
        feats_s1 = points_list[0]
        feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder1_2(points_list[0], points_list[0], feats_s1, neighbors_list[0])

        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        feats_s5 = self.encoder5_1(points_list[4], points_list[3], feats_s4, subsampling_list[3])
        feats_s5 = self.encoder5_2(points_list[4], points_list[4], feats_s5, neighbors_list[4])
        feats_s5 = self.encoder5_3(points_list[4], points_list[4], feats_s5, neighbors_list[4])

        latent_s5 = feats_s5

        latent_s4 = nearest_interpolate_pack_mode(latent_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)

        latent_s3 = nearest_interpolate_pack_mode(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)

        latent_s2 = nearest_interpolate_pack_mode(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        latent_s1 = nearest_interpolate_pack_mode(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        scores = self.regressor(latent_s1)

        output_dict = scores

        return output_dict, self.sigma, mu

def create_model(cfg):
    return get_model(cfg)

def run_test():
    cfg = Config()
    model = create_model(cfg).cuda()
    # Example input data, shape [B, N, 3]
    data = torch.rand(2, 8000, 3).cuda()
    # Perform forward pass through the model
    outputs, _ = model(data, iter=0)
    # print(outputs["outputs"].shape)  # Shape of the output
    # print(model.state_dict().keys())
    # print(model)

def knn_point_2(k, ref, query):
    """
    k: number of neighbors
    ref: (B, N, 3) reference points
    query: (B, M, 3) query points
    
    Returns:
    dists: (B, M, k) square distances of the k nearest neighbors
    idx: (B, M, k) indices of the k nearest neighbors
    """
    # Calculate squared distances
    B, N, _ = ref.shape
    _, M, _ = query.shape
    
    dists = torch.sum((query.unsqueeze(2) - ref.unsqueeze(1)) ** 2, dim=-1)
    # Get k nearest neighbors
    dists, idx = torch.topk(dists, k=k, dim=-1, largest=False, sorted=True)
    return dists, idx

def get_repulsion_loss(pred, pred2):
    """
    Calculate the repulsion loss between predicted points.

    Args:
        pred: Predicted points, shape (B, N, 3)
        pred2: Another set of predicted points, shape (B, N, 3)

    Returns:
        uniform_loss: The repulsion loss value
    """
    k = 7  # number of neighbors
    dist_square, idx = knn_point_2(k, pred, pred2)
    dist_square = dist_square[:, :, 1:]  # Remove self-distance
    target_dist = 0.0792 * 2  # 0.00363
    dist_square = torch.abs(target_dist - dist_square)
    uniform_loss = torch.mean(dist_square)
    return uniform_loss

class getLoss(nn.Module):
    def __init__(self):
        super(getLoss, self).__init__()

    def forward(self, pred, noise_std, mu, pcl_noisy):
        """
        Calculate the total loss.

        Args:
            pred: Predicted likelihood parameters, shape [B*N, 3] --> [B, 3, N]
            noise_std: Noise standard deviation, float
            mu: Noise mean, shape [B*N, 3]
            pcl_noisy: Noisy point cloud, shape [B, N, 3]
                   
        Returns:
            total_loss: Total loss
        """
        device = pred.device
        mu = mu.to(device).float()

        # noise_std is a float, needs to be broadcasted to the shape of pred
        noise_std = torch.tensor(noise_std, device=device).float()
        noise_std = noise_std.view(1, 1).expand_as(pred)
        # Loss
        loss1 = F.mse_loss(noise_std * pred, -mu)

        return loss1

if __name__ == '__main__':
    run_test()
