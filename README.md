# Modular Scenery/Style Transfer CycleGAN in PyTorch

## Overview
This repository contains a modular, organized PyTorch port of the CycleGAN architecture for high-resolution photo-to-anime scenery style transfer. It is optimized to run on a 16GB VRAM GPU.

## Codebase Structure
```
human-anime-gan/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # PyTorch Dataset/DataLoader & online augmentations
│   ├── models.py       # PyTorch definitions of Generator & Discriminator networks
│   ├── train.py        # Core CycleGANTrainer class containing train/test step logic
│   └── utils.py        # Helper utilities: configuration parser, plotting, saving predictions
├── train.py            # Main training entry point
├── preprocess.py       # Standalone script for image border cleaning and square cropping
├── live_cam.py         # Real-time webcam style transfer using trained models
├── config.yaml         # Training & data configuration parameters
└── requirements.txt    # Python package dependencies
```

## Installation
Set up your virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Preprocessing Custom Scenery Images (Optional)
If you add more custom scenery photos to your source dataset, you can crop/resize them:
```bash
python preprocess.py --input /path/to/raw/photos --output data/custom_preprocessed
```

### 2. Training the CycleGAN Model
Edit configuration settings inside `config.yaml` to specify correct dataset directories and hyperparameter values. Then start training:
```bash
python train.py config.yaml
```

To resume training from a specific epoch checkpoint:
```bash
python train.py config.yaml --checkpoint models/checkpoint_epoch_10.pt
```

### 3. Live Webcam Stylization
To test your trained model in real-time using a webcam:
```bash
python live_cam.py --checkpoint models/checkpoint_epoch_100.pt --image-size 256
```
*Press `q` inside the windows to quit the camera stream.*

## SOTA Architecture Specifications (PyTorch Port)
* **Instance Normalization**: Uses `InstanceNorm2d` (with affine parameters) for high-fidelity style transfer.
* **ResNet Generator (SE & SA Attention)**: Switch from simple U-Net to a ResNet-9 backbone with **Squeeze-and-Excitation (SE)** channel attention in all residual blocks, and **Self-Attention (SA)** in the bottleneck to coordinate global structures.
* **Multi-Scale Discriminators**: Evaluates images at both full-resolution (128x128 or 256x256) and 2x downsampled scale for better shape and texture control.
* **Spectral Normalization**: Applied to all convolutional layers of the discriminators to stabilize GAN training and prevent mode collapse.
* **Noise Injection**: Added gaussian noise with standard deviation decaying over epochs (`0.1 / (epoch + 1)`) to real inputs during training.
* **Mixed Precision (AMP)**: Supported PyTorch Automatic Mixed Precision (`autocast`) for high-throughput GPU training.