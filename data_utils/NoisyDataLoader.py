import os
import torch
import numpy as np
import warnings
import pickle
import random
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def add_noises(input):
    noise_std=0.010
    input = torch.tensor(input, dtype=torch.float32)
    mu = torch.randn_like(input)
    return input + noise_std * mu.numpy(),noise_std,mu.numpy()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class NoisyDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False,transform= None):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.transform =transform

        self.file_path = 'if u want to load a file'
        if self.file_path == 'path/to/specific/file.txt':
            self.datapath = [self.file_path]
            self.file_list = [os.path.basename(self.file_path)]
        else:
            if split == 'train':
                file_list = [line.rstrip() for line in open(os.path.join(root, '../noisy_dataset.txt'))]
                # valid_suffixes = ['_15_10k', '_10_10k', '_5_10k' ,'_20_10k' ,'_30_10k']
            elif split == 'test':
                file_list = [line.rstrip() for line in open(os.path.join(root, '../test_dataset.txt'))]
                # valid_suffixes = ['_20_10k']
            else:
                raise ValueError("Invalid split parameter. Must be 'train' or 'test'.")

            valid_suffixes = ['_005','_01','015']
            self.datapath = [os.path.join(root, f"{file}{suffix}.txt") for file in file_list for suffix in valid_suffixes if os.path.exists(os.path.join(root, f"{file}{suffix}.txt"))]
            self.file_list = [f"{file}{suffix}" for file in file_list for suffix in valid_suffixes if os.path.exists(os.path.join(root, f"{file}{suffix}.txt"))]

            # self.datapath = [os.path.join(root, f"{file}.txt") for file in file_list if os.path.exists(os.path.join(root, f"{file}.txt"))]
            # self.file_list = [file for file in file_list if os.path.exists(os.path.join(root, f"{file}.txt"))]

        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    point_set = np.loadtxt(self.datapath[index], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set

                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.list_of_points, f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points = pickle.load(f) 

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set = self.list_of_points[index]
        else:
            point_set = np.loadtxt(self.datapath[index], delimiter=',').astype(np.float32)
            # point_set = np.loadtxt(self.datapath[index]).astype(np.float32)
            
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]  # Sequential sampling points
                # point_set = point_set
                
        if not self.use_normals:
            point_set = point_set[:, 0:3]
    
        return point_set

    def __getitem__(self, index):
        point_set = self._get_item(index)
        file_name = self.file_list[index]

        point_set = torch.tensor(point_set, dtype=torch.float32)  

        # 归一化
        centroid = torch.mean(point_set[:, 0:3], dim=0)
        point_set[:, 0:3] -= centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(point_set[:, 0:3] ** 2, dim=1)))
        point_set[:, 0:3] /= furthest_distance
        
        return point_set, centroid, furthest_distance, file_name

if __name__ == '__main__':
    data = NoisyDataLoader('/data/NoisyDataSets/NoisyColoredGaussianDataSet/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    for point in DataLoader:
        print(point.shape) # Print the shape of the point set tensor
