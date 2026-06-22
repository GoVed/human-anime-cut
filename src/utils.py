import os
import torch
import yaml
import numpy as np
import torchvision.transforms.functional as TF

def load_config(config_path):
    """Load hyperparameters and paths from config.yaml"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def denormalize(tensor):
    """Denormalize a [-1, 1] tensor to [0, 1]"""
    return (tensor + 1.0) / 2.0

def save_training_samples(real_x, real_y, G_XtoY, G_YtoX, epoch, step, output_dir="sample"):
    """
    Saves predictions during training, following the exact naming convention from the Keras notebook:
    - human_{epoch}_{step}.png, human_to_anime_{epoch}_{step}.png, etc.
    """
    os.makedirs(output_dir, exist_ok=True)
    G_XtoY.eval()
    G_YtoX.eval()
    
    with torch.no_grad():
        fake_y = G_XtoY(real_x)
        fake_x = G_YtoX(real_y)
        
        cycled_x = G_YtoX(fake_y)
        cycled_y = G_XtoY(fake_x)
        
        same_x = G_YtoX(real_x)
        same_y = G_XtoY(real_y)
        
    display_list = [real_x[0], fake_y[0], real_y[0], fake_x[0], cycled_x[0], cycled_y[0], same_x[0], same_y[0]]
    names = [
        "human", "human_to_anime", "anime", "anime_to_human",
        "human_cycle", "anime_cycle", "human_identity", "anime_identity"
    ]
    
    for img, name in zip(display_list, names):
        img_denorm = denormalize(img.cpu())
        img_clipped = torch.clamp(img_denorm, 0.0, 1.0)
        img_pil = TF.to_pil_image(img_clipped)
        img_pil.save(os.path.join(output_dir, f"{name}_{epoch}_{step}.png"))

def save_test_samples(real_x, real_y, G_XtoY, G_YtoX, epoch, output_dir="sample/test"):
    """
    Saves predictions during test evaluation, following the exact naming convention from the Keras notebook:
    - epoch_{epoch}_{title}.png
    """
    os.makedirs(output_dir, exist_ok=True)
    G_XtoY.eval()
    G_YtoX.eval()
    
    with torch.no_grad():
        fake_y = G_XtoY(real_x)
        fake_x = G_YtoX(real_y)
        
        cycled_x = G_YtoX(fake_y)
        cycled_y = G_XtoY(fake_x)
        
        same_x = G_YtoX(real_x)
        same_y = G_XtoY(real_y)
        
    display_list = [real_x[0], fake_y[0], cycled_x[0], same_x[0], real_y[0], fake_x[0], cycled_y[0], same_y[0]]
    titles = ["Real_Human", "Fake_Anime", "Cycled_Human", "Same_Human", "Real_Anime", "Fake_Human", "Cycled_Anime", "Same_Anime"]
    
    for img, title in zip(display_list, titles):
        img_denorm = denormalize(img.cpu())
        img_clipped = torch.clamp(img_denorm, 0.0, 1.0)
        img_pil = TF.to_pil_image(img_clipped)
        img_pil.save(os.path.join(output_dir, f"epoch_{epoch}_{title}.png"))
