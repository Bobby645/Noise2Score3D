import os
import sys
import torch
import pytorch3d
import random
import time
import point_cloud_utils as pcu
import numpy as np
import argparse
import logging
import importlib
from pathlib import Path
from tqdm import tqdm
from data_utils.NoisyDataLoader import NoisyDataLoader
from chamferdistance import chamfer_distance_unit_sphere
from models.KPconv_test import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--model', default='KPconv_test', help='model name [default: KPconv_test]')
    parser.add_argument('--log_dir', type=str, default='KPconv', help='experiment root')
    parser.add_argument('--ckpt', type=str, default='./log/denoise/KPconv')
    parser.add_argument('--input_root', type=str, default='./data/input_full_test_50k_0.030')
    parser.add_argument('--gt_root', type=str, default='./data/gt_points_50k')
    parser.add_argument('--gts_mesh_dir', type=str, default='./data/gt_meshes')
    parser.add_argument('--sigma', type=float, default=0.03)
    
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Additional parameters
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def input_iter(input_dir, gt_dir):
    input_files = sorted(os.listdir(input_dir))
    gt_files = sorted(os.listdir(gt_dir))
    for fn in input_files:
        if not fn.endswith('.xyz'):
            continue
        # Check if the corresponding ground truth file exists
        if fn not in gt_files:
            print(f"Warning: Ground truth file for {fn} not found in {gt_dir}. Skipping...")
            continue

        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_gt = torch.FloatTensor(np.loadtxt(os.path.join(gt_dir, fn)))
        centroid = torch.mean(pcl_noisy[:, 0:3], dim=0)
        pcl_noisy[:, 0:3] -= centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(pcl_noisy**2, dim=1)))
        pcl_noisy /= furthest_distance
        yield {
            'pcl_noisy': pcl_noisy,
            'pcl_gt': pcl_gt,
            'name': fn[:-4],
            'center': centroid,
            'scale': furthest_distance
        }

def load_off(off_dir):
    all_meshes = {}
    input_files = sorted(os.listdir(off_dir))
    for fn in input_files:
        if not fn.endswith('.off'):
            continue
        name = fn[:-4]
        path = os.path.join(off_dir, fn)
        verts, faces = pcu.load_mesh_vf(path)
        verts = torch.FloatTensor(verts)
        faces = torch.LongTensor(faces)
        all_meshes[name] = {'verts': verts, 'faces': faces}
    return all_meshes

def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts = verts.unsqueeze(0)
    p_max = verts.max(dim=-2, keepdim=True)[0]
    p_min = verts.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    verts = verts - center
    # Scale
    radius = 1.0
    scale = (verts ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    verts = verts / scale
    verts = verts[0]

    # Normalize pcl
    pcl = (pcl.unsqueeze(0) - center) / scale
    pcl = pcl[0]

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])

    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls, min_triangle_area=0) # p2m setting

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    def log_string(str):
        logger.info(str)
        print(str)
    
    '''CREATE DIR'''
    # Input/Output
    input_dir = args.input_root
    gt_dir = args.gt_root

    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('denoise')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        log_dir = exp_dir.joinpath('test')
    else:
        log_dir = exp_dir.joinpath(args.log_dir).joinpath('test/')
    log_dir.mkdir(exist_ok=True)
    
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{args.model}_test.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    '''MODEL LOADING'''
    model_module = importlib.import_module(args.model)
    config = Config()
    model = model_module.get_model(config, normal_channel=None)
    model = model.cuda()
    checkpoint_path = os.path.join(args.ckpt, 'checkpoints', 'model_step_4500.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    log_string(f'Loaded model from checkpoint: {checkpoint_path}')

    num_samples = 0
    total_cd_loss = 0.0
    total_p2f_loss = 0.0
    '''DATA LOADING'''
    log_string('Start testing...')
    log_string('Load data pointcloud...')
    meshes = load_off(args.gts_mesh_dir)
    # Initialize time variables
    inference_times = []
    for data in input_iter(input_dir, gt_dir):
        log_string(data['name'])
        pcl_noisy = data['pcl_noisy'].cuda() # N 3
        pcl_noisy = pcl_noisy.unsqueeze(0)   # Add batch dimension, 1 N 3
        pcl_gt = data['pcl_gt'].cpu().numpy()
        '''TESTING'''
        '''Start inference timing'''
        start_time = time.time()
        model = model.eval()
        with torch.no_grad():
            pred, noise_std, mu = model(pcl_noisy, None) # N 3 cuda 
            pred = pred.unsqueeze(0).repeat(1, 1, 1) ########### KPconv activation
            # Denormalize
            pcl_noisy = pcl_noisy.cpu()
            pcl_noisy = pcl_noisy * data['scale'] + data['center']  # BN3
            # Tweedie
            variance = args.sigma ** 2
            denoised_points = pred.cpu() * variance + pcl_noisy
            denoised_points = denoised_points[0].numpy() # N3
            '''End inference timing'''
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            log_string(f'Inference Time: {inference_time:.2f} s')

            save_path = os.path.join(log_dir, f"{data['name']}.xyz")
            np.savetxt(save_path, denoised_points, fmt='%.8f')

            # Chamfer Distance
            chamfer = chamfer_distance_unit_sphere(denoised_points, pcl_gt, batch_reduction='mean')[0].item()
            total_cd_loss += chamfer
            num_samples += 1

            # Point-to-Face Distance
            mesh_data = meshes.get(data['name'])
            verts = mesh_data['verts'].cuda()
            faces = mesh_data['faces'].cuda()
            p2f = point_mesh_bidir_distance_single_unit_sphere(
                pcl=torch.FloatTensor(denoised_points).cuda(),
                verts=verts,
                faces=faces
            ).item()
            total_p2f_loss += p2f
            log_string(f'Save xyz [Val] CD: {chamfer * 10000:.6f} | p2f: {p2f * 10000:.6f}')

    avg_chamfer = total_cd_loss / num_samples
    avg_p2f = total_p2f_loss / num_samples
    log_string(f'Average Chamfer Distance (CD) Loss: {avg_chamfer * 10000:.6f} | P2F loss: {avg_p2f * 10000:.6f}')
    # Total inference time and average inference time
    total_inference_time = sum(inference_times)
    average_inference_time = total_inference_time / len(inference_times)
    log_string(f'Total Inference Time: {total_inference_time:.2f} seconds')

    return avg_chamfer, avg_p2f
            
if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    main(args)
