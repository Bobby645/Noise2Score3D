import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import shutil
import argparse
import random
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from tqdm import tqdm
from data_utils.NoisyDataLoader import NoisyDataLoader
from models.KPconv import Config
from chamferdistance import chamfer_distance_unit_sphere
from PCTV import PointCloudTVLoss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='Use CPU mode')
    parser.add_argument('--gpu', type=str, default='1', help='Specify GPU device')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size in training')
    parser.add_argument('--model', default='KPconv', help='Model name [default: KPconv]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='Training on ModelNet10/40')
    parser.add_argument('--epoch', default=800, type=int, help='Number of epochs in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate in training')
    parser.add_argument('--num_point', type=int, default=8000, help='Number of points')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training')
    parser.add_argument('--log_dir', type=str, default='KPconv', help='Experiment root directory')
    parser.add_argument('--decay_rate', type=float, default=1e-6, help='Decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='Use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='Save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='Use uniform sampling')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval steps')  # Added save interval
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def main(args):
    def log_string(s):
        logger.info(s)
        print(s)

    """HYPER PARAMETERS"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """CREATE DIR"""
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('denoise')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{args.model}.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETERS ...')
    log_string(args)
    model = importlib.import_module(args.model)

    """DATA LOADING"""
    log_string('Loading dataset ...')
    data_path = './data/NoisyColoredGaussianDataSet'
    test_path = './data/NoisyColoredGaussianDataSet'
    train_dataset = NoisyDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = NoisyDataLoader(root=test_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)

    """MODEL LOADING"""
    num_class = args.num_category

    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('./train_denoise.py', str(exp_dir))
    shutil.copy('./data_utils/NoisyDataLoader.py', str(exp_dir))
    config = Config()  # KPconv configuration
    denoiser = model.get_model(config, normal_channel=args.use_normals)
    criterion = model.get_loss()

    denoiser.apply(inplace_relu)  # Ensure all ReLU layers perform in-place operations to save memory and improve computational efficiency

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            denoiser.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(denoiser.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    if not args.use_cpu:
        denoiser = denoiser.cuda()
        criterion = criterion.cuda()
    checkpoint_path = os.path.join(str(exp_dir), 'checkpoints', 'best_model.pth')

    try:
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        denoiser.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_string('Loaded pretrained model')
    except FileNotFoundError:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    except Exception as e:
        log_string(f'Error loading checkpoint: {e}')
        start_epoch = 0

    global_epoch = 0
    global_step = 0
    best_loss = float('inf')  # Initialize variable to positive infinity

    """TRAINING"""
    logger.info('Start training......')
    for epoch in range(start_epoch, args.epoch):
        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch}):')
        total_loss = 0.0
        denoiser = denoiser.train()
        scheduler.step()
        # Data loading
        for batch_id, (points, centroid, furthest_distance, file_name) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            if not args.use_cpu:
                points = points.cuda()

            # Model output
            pred, noise_std, mu = denoiser(points, global_step)

            # Calculate self-supervised loss
            self_supervised_loss = criterion(pred, noise_std, mu, points)
            
            # Calculate total loss
            loss = self_supervised_loss

            loss.backward()
            orig_grad_norm = clip_grad_norm_(denoiser.parameters(), "inf")  # Calculate gradient norm
            optimizer.step()
            total_loss += loss.item()
            global_step += 1  # Record training progress, save model every `save_interval` steps
            if global_step % args.save_interval == 0:
                savepath = str(checkpoints_dir) + f'/model_step_{global_step}.pth'
                log_string(f'Saving model to {savepath}')
                state = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }
                torch.save(state, savepath)
        print(noise_std)
        train_loss = total_loss / len(trainDataLoader)  # Calculate average training loss
        log_string(f'Train Instance Loss: {train_loss} | Grad {orig_grad_norm:.6f}')
        
        # Test set loss
        PCTVloss = PointCloudTVLoss(k=6)
        with torch.no_grad():
            total_loss = 0.0
            denoiser = denoiser.eval()
            for batch_id, (points, centroid, furthest_distance, file_name) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
                if not args.use_cpu:
                    points = points.cuda()  # B N 3
                
                pred, noise_std, mu = denoiser(points, global_step)  # Get predictions pred B*N 3 
                # Reshape pred
                B, N, _ = points.size()
                pred = pred.view(B, N, 3)
                pred = pred.to(points.device)
                
                denoised_points = pred * (0.02**2) + points  # B N 3
                # Compute PCTV loss
                loss = PCTVloss(denoised_points)
                total_loss += loss.item()
            
            test_loss = total_loss / len(testDataLoader)
            log_string(f'Test Loss: {test_loss}')
            
            # Save the best model
            if test_loss < best_loss:
                best_loss = test_loss
                logger.info('Saving best model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string(f'Saving to {savepath}')
                state = {
                    'epoch': epoch,
                    'model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }
                torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
