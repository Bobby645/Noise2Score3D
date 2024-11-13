import point_cloud_utils as pcu
import numpy as np
import pytorch3d.loss
import torch

# Normalize point cloud
def normalize_pcl(pc, center, scale):
    return (pc - center) / scale

def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
     # Center the point cloud
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale
def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean',single_directional=False):
    if isinstance(gen, np.ndarray):
        gen = torch.tensor(gen)
        gen = gen.unsqueeze(0)  # Add batch dimension
        gen = gen.to('cuda')
        ref = torch.tensor(ref)
        ref = ref.unsqueeze(0)  # Add batch dimension
        ref = ref.to('cuda')
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)

    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction,single_directional = single_directional)

if __name__ == '__main__':
    
    gt_path = './data/ground_truth/Icosahedron.xyz'
    pred_path = './data/predicted/Icosahedron_denoised.xyz'
    
    # Load ground truth point cloud
    pc_gt = np.loadtxt(gt_path, dtype=np.float32)  # [N, 3]
     # Load predicted point cloud
    pc_pred = np.loadtxt(pred_path, dtype=np.float32)   # [N, 3]
    # Ensure point clouds have only XYZ coordinates
    pc_pred = pc_pred[:, :3]
    pc_gt = pc_gt[:, :3]

    # chamfer_dist = pcu.chamfer_distance(pc_pred, pc_gt)
    loss_cd ,_ = chamfer_distance_unit_sphere(pc_pred, pc_gt)
    print(loss_cd.item()*10000)