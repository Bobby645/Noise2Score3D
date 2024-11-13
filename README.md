# Noise2Score3D: Unsupervised Tweedie's Approach for Point Cloud Denoising

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
The training dataset [ModelNet-40] is the same as ``Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning`` by Pedro Hermosilla, Tobias Ritschel, Timo Ropinski. Please check their GitHub repo [here](https://github.com/phermosilla/TotalDenoising).

Our test dataset [PU-Net] is the same as ``Score-Based Point Cloud Denoising`` by Shitong Luo and Wei Hu. Please check their GitHub repo [here](https://github.com/luost26/score-denoise). 

Our test dataset [ModelNet-40] is the same as ``DMRDenoise`` by Shitong Luo and Wei Hu. Please check their GitHub repo [here](https://github.com/luost26/DMRDenoise). 

Our test dataset [ModelNet-40 Simulated LiDAR noise] is the same as ``Total Denoising: Unsupervised Learning of 3D Point Cloud Cleaning`` by Pedro Hermosilla, Tobias Ritschel, Timo Ropinski. Please check their GitHub repo [here](https://github.com/phermosilla/TotalDenoising).

## Pretrained models


## Inference only

```
python test.py 
python test_tv.py 
```

## Train 

```
python train_denoise.py
```
