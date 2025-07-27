# Noise2Score3D: Unsupervised Tweedie's Approach for Point Cloud Denoising
# ICCV 2025 has accepted this paper!
## Installation
The code has been tested on and NVIDIA RTX 3080ti GPU with the following settings:

```
Python 3.9
Ubuntu 20.04
CUDA 11.1
PyTorch 1.10.1+cu111
PyTorch3D 0.6.2
pykeops 2.2.3
```

# Dataset
## training data
The training dataset [ModelNet-40] is the same as ``Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning`` by Pedro Hermosilla, Tobias Ritschel, Timo Ropinski. Please check their GitHub repo [here](https://github.com/phermosilla/TotalDenoising).
## testing data
Our test dataset [PU-Net] is the same as ``Score-Based Point Cloud Denoising`` by Shitong Luo and Wei Hu. Please check their GitHub repo [here](https://github.com/luost26/score-denoise). 

Our test dataset [ModelNet-40] is the same as ``DMRDenoise`` by Shitong Luo and Wei Hu. Please check their GitHub repo [here](https://github.com/luost26/DMRDenoise). 
Note that the 10k datasets in it is the one we got by downsampling by 50k

Our test dataset [ModelNet-40 Simulated LiDAR noise] is the same as ``Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning`` by Pedro Hermosilla, Tobias Ritschel, Timo Ropinski. Please check their GitHub repo [here](https://github.com/phermosilla/TotalDenoising).

## Pretrained models
Pre-training weights will be released upon receipt of the paper.............................
We can't put any other links..............
## Inference only

```
# Tests where the noise level is known
python test.py --input_root <Path to the folder of the noise point cloud you want to test> --gt_root --gts_mesh_dir --sigma <Sigma for inference, depending on noise level>

# Testing for unknown noise levels
python test_tv.py --input_root <Path to the folder of the noise point cloud you want to test> --gt_root --gts_mesh_dir --sigma_min <Minimum sigma value to try> --sigma_ma  --iterations <Number of iterations for sigma search>
```

## Train 

```
python train_denoise.py --batch_size 8 --save_interval 500
```
