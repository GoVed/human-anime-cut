import os
import sys
import argparse
import random
import torch
from src.models import Generator, MultiScaleDiscriminator, PatchSampleF
from src.dataset import get_dataloader
from src.train import CycleGANTrainer, CUTTrainer
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser(description="Train modular scenery/style transfer CycleGAN/CUT in PyTorch")
    # Positional argument with optional default for backwards compatibility
    parser.add_argument("config", type=str, nargs="?", default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint (.pt) to resume training")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--method", type=str, choices=["cyclegan", "cut"], default="cyclegan", help="Training method to use: 'cyclegan' or 'cut'")
    args = parser.parse_args()

    # 1. Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
        
    config = load_config(args.config)
    print(f"Loaded configuration parameters successfully from: {args.config}")

    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Extract configuration values
    photo_dir = config.get('human_data_dir', 'data/dataset/train_photo')
    test_photo_dir = config.get('test_human_data_dir', 'data/dataset/test/HR_photo')
    style_dir = config.get('anime_data_dir', 'data/dataset/combined_style')
    
    epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 4)
    image_size = config.get('image_size', 256)
    learning_rate = config.get('learning_rate', 0.0002)
    checkpoint_dir = config.get('model_save_dir', 'models')
    sample_dir = config.get('output_dir', 'sample')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 3. Process image paths and perform train/test split for target domain (style)
    # Reserve 520 files for testing, remaining for training
    if not os.path.exists(style_dir):
        print(f"Error: Style target directory '{style_dir}' does not exist.")
        sys.exit(1)
        
    style_paths = sorted([
        os.path.join(style_dir, f) for f in os.listdir(style_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # Shuffle style paths to ensure random test split
    random.shuffle(style_paths)
    test_style_paths = style_paths[:520]
    train_style_paths = style_paths[520:]

    print(f"Loaded {len(style_paths)} style images. Split: {len(train_style_paths)} train, {len(test_style_paths)} test.")

    # Process image paths for source domain (real photo)
    if not os.path.exists(photo_dir):
        print(f"Error: Photo training directory '{photo_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(test_photo_dir):
        print(f"Warning: Photo test directory '{test_photo_dir}' not found. Using training images for testing.")
        test_photo_dir = photo_dir

    train_photo_paths = sorted([
        os.path.join(photo_dir, f) for f in os.listdir(photo_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    test_photo_paths = sorted([
        os.path.join(test_photo_dir, f) for f in os.listdir(test_photo_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Loaded {len(train_photo_paths)} training photos and {len(test_photo_paths)} test photos.")

    # 4. Construct DataLoaders
    print("Building DataLoaders...")
    train_loader = get_dataloader(
        photo_dir=train_photo_paths,
        style_dir=train_style_paths,
        size=image_size,
        batch_size=batch_size,
        is_train=True,
        num_workers=config.get('num_workers', 4)
    )

    test_loader = get_dataloader(
        photo_dir=test_photo_paths,
        style_dir=test_style_paths,
        size=image_size,
        batch_size=batch_size,
        is_train=False,
        num_workers=config.get('num_workers', 4)
    )

    # 5. Build models and optimizers
    if args.method == "cut":
        print("Initializing CUT models (single-direction Generator + Projector + Discriminator)...")
        G_XtoY = Generator()
        Di_Y = MultiScaleDiscriminator()
        # Projector for layers: raw input (3), Relu1 (64), Relu2 (128), Relu3 (256), Bottleneck1 (256), Bottleneck5 (256)
        F = PatchSampleF(in_channels_list=[3, 64, 128, 256, 256, 256])
        
        optimizer_G = torch.optim.Adam(list(G_XtoY.parameters()) + list(F.parameters()), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(Di_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=10000, gamma=0.95)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=10000, gamma=0.95)
        
        trainer = CUTTrainer(
            G=G_XtoY,
            D=Di_Y,
            F=F,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            scheduler_G=scheduler_G,
            scheduler_D=scheduler_D,
            use_amp=not args.no_amp,
            device=device
        )
    else:
        print("Initializing CycleGAN models (bidirectional Generators + Discriminators)...")
        G_XtoY = Generator()
        G_YtoX = Generator()
        Di_X = MultiScaleDiscriminator()
        Di_Y = MultiScaleDiscriminator()

        optimizer_G_XtoY = torch.optim.Adam(G_XtoY.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_G_YtoX = torch.optim.Adam(G_YtoX.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_Di_X = torch.optim.Adam(Di_X.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_Di_Y = torch.optim.Adam(Di_Y.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        scheduler_G_XtoY = torch.optim.lr_scheduler.StepLR(optimizer_G_XtoY, step_size=10000, gamma=0.95)
        scheduler_G_YtoX = torch.optim.lr_scheduler.StepLR(optimizer_G_YtoX, step_size=10000, gamma=0.95)
        scheduler_Di_X = torch.optim.lr_scheduler.StepLR(optimizer_Di_X, step_size=10000, gamma=0.95)
        scheduler_Di_Y = torch.optim.lr_scheduler.StepLR(optimizer_Di_Y, step_size=10000, gamma=0.95)

        trainer = CycleGANTrainer(
            G_XtoY=G_XtoY,
            G_YtoX=G_YtoX,
            Di_X=Di_X,
            Di_Y=Di_Y,
            optimizer_G_XtoY=optimizer_G_XtoY,
            optimizer_G_YtoX=optimizer_G_YtoX,
            optimizer_Di_X=optimizer_Di_X,
            optimizer_Di_Y=optimizer_Di_Y,
            scheduler_G_XtoY=scheduler_G_XtoY,
            scheduler_G_YtoX=scheduler_G_YtoX,
            scheduler_Di_X=scheduler_Di_X,
            scheduler_Di_Y=scheduler_Di_Y,
            use_amp=not args.no_amp,
            device=device
        )

    # 8. Optionally load checkpoint to resume
    start_epoch = 0
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if checkpoint_path.lower() == 'latest':
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
        elif checkpoint_path.isdigit():
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{checkpoint_path}.pt')
            
        if os.path.exists(checkpoint_path):
            start_epoch = trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Starting from scratch.")

    # 9. Fit the models
    print(f"Starting training for {epochs} epochs...")
    trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        sample_step_interval=config.get('sample_step_interval', 500),
        checkpoint_interval=config.get('checkpoint_interval', 10),
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    main()