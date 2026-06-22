import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

class UnpairedDataset(Dataset):
    """
    Unpaired dataset for real photo and anime style images.
    Returns random pairs of images from each dataset for CycleGAN style transfer.
    """
    def __init__(self, photo_dir, style_dir, size=128, is_train=True):
        self.size = size
        self.is_train = is_train

        if isinstance(photo_dir, list):
            self.photo_paths = photo_dir
        else:
            self.photo_paths = sorted([
                os.path.join(photo_dir, f) for f in os.listdir(photo_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]) if (photo_dir and os.path.exists(photo_dir)) else []

        if isinstance(style_dir, list):
            self.style_paths = style_dir
        else:
            self.style_paths = sorted([
                os.path.join(style_dir, f) for f in os.listdir(style_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]) if (style_dir and os.path.exists(style_dir)) else []

        self.photo_size = len(self.photo_paths)
        self.style_size = len(self.style_paths)

    def crop_to_square(self, img, horizontal_random_pixel=10):
        w, h = img.size
        crop_size = min(w - horizontal_random_pixel, h)
        if self.is_train:
            left = random.randint(0, w - crop_size)
            top = random.randint(0, h - crop_size)
        else:
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
        return TF.crop(img, top, left, crop_size, crop_size)

    def preprocess_style(self, img):
        # 1. Crop to square
        img = self.crop_to_square(img, horizontal_random_pixel=10)
        
        # 2. Random horizontal flip
        if self.is_train and random.random() > 0.5:
            img = TF.hflip(img)
        
        # 3. Resize to self.size x self.size
        img = TF.resize(img, [self.size, self.size])
        
        # 4. Convert to Tensor
        img_t = TF.to_tensor(img)
        
        if self.is_train:
            # 5. Random Contrast
            contrast_factor = random.uniform(0.8, 1.2)
            mean = torch.mean(img_t, dim=(1, 2), keepdim=True)
            img_t = (img_t - mean) * contrast_factor + mean
            
            # 6. Random Brightness (additive in range [-0.1, 0.1])
            brightness_factor = random.uniform(-0.1, 0.1)
            img_t = img_t + brightness_factor
            
            # Clip to valid range [0.0, 1.0]
            img_t = torch.clamp(img_t, 0.0, 1.0)
            
        # 7. Normalize to [-1, 1]
        img_t = (img_t - 0.5) / 0.5
        return img_t

    def preprocess_photo(self, img):
        # 1. Crop to square
        img = self.crop_to_square(img, horizontal_random_pixel=16)
        
        # 2. Resize to self.size x self.size first (HUGE speedup: rotates a small image instead of raw high-res)
        img = TF.resize(img, [self.size, self.size])
        
        if self.is_train:
            # 3. Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                
        # 5. Convert to Tensor
        img_t = TF.to_tensor(img)
        
        if self.is_train:
            # 6. Random Contrast
            contrast_factor = random.uniform(0.8, 1.2)
            mean = torch.mean(img_t, dim=(1, 2), keepdim=True)
            img_t = (img_t - mean) * contrast_factor + mean
            
            # 7. Random Brightness (additive in range [-0.1, 0.1])
            brightness_factor = random.uniform(-0.1, 0.1)
            img_t = img_t + brightness_factor
            
            # Clip to valid range [0.0, 1.0]
            img_t = torch.clamp(img_t, 0.0, 1.0)
            
        # 8. Normalize to [-1, 1]
        img_t = (img_t - 0.5) / 0.5
        return img_t

    def __len__(self):
        return max(self.photo_size, self.style_size)

    def __getitem__(self, index):
        # Load real photo image
        if self.photo_size > 0:
            photo_path = self.photo_paths[index % self.photo_size]
            photo_img = Image.open(photo_path).convert('RGB')
            photo_tensor = self.preprocess_photo(photo_img)
        else:
            photo_tensor = torch.zeros(3, self.size, self.size)

        # Load anime style image
        if self.style_size > 0:
            if self.is_train:
                style_idx = random.randint(0, self.style_size - 1)
            else:
                style_idx = index % self.style_size
            style_path = self.style_paths[style_idx]
            style_img = Image.open(style_path).convert('RGB')
            style_tensor = self.preprocess_style(style_img)
        else:
            style_tensor = torch.zeros(3, self.size, self.size)

        return photo_tensor, style_tensor


def get_dataloader(photo_dir, style_dir, size=128, batch_size=2, is_train=True, num_workers=4):
    """
    Helper function to construct DataLoader for CycleGAN datasets.
    """
    dataset = UnpairedDataset(photo_dir, style_dir, size=size, is_train=is_train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=(num_workers > 0)
    )
    return dataloader
