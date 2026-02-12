import os
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Tuple, List, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    """Dataset for segmentation tasks"""
    
    def __init__(self, 
                 image_dir: str, 
                 mask_dir: str, 
                 transform=None, 
                 is_train: bool = True,
                 image_size: int = 256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.is_train = is_train
        self.image_size = image_size
        
        # Get all image files
        self.image_files = []
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for ext in extensions:
            self.image_files.extend(list(self.image_dir.glob(f"*{ext}")))
            self.image_files.extend(list(self.image_dir.glob(f"*{ext.upper()}")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Find corresponding masks
        self.mask_files = []
        for img_path in self.image_files:
            mask_path = self.find_mask(img_path)
            if mask_path:
                self.mask_files.append(mask_path)
            else:
                if is_train:
                    print(f"Warning: No mask found for {img_path.name}")
                self.mask_files.append(None)
        
        # Filter out images without masks for training
        if is_train:
            valid_indices = [i for i, mask in enumerate(self.mask_files) if mask is not None]
            if len(valid_indices) == 0:
                raise ValueError(f"No valid training samples found with masks in {mask_dir}")
            self.image_files = [self.image_files[i] for i in valid_indices]
            self.mask_files = [self.mask_files[i] for i in valid_indices]
    
    def find_mask(self, img_path: Path) -> Optional[Path]:
        """Find mask file for given image"""
        stem = img_path.stem
        
        # Try common mask naming patterns
        patterns = [
            f"{stem}_mask",
            f"{stem}_seg",
            f"{stem}_label",
            stem,  # Same name as image
            stem.replace('image', 'mask'),
            stem.replace('img', 'mask'),
        ]
        
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        for pattern in patterns:
            for ext in extensions:
                mask_path = self.mask_dir / f"{pattern}{ext}"
                if mask_path.exists():
                    return mask_path
                # Try uppercase extension
                mask_path = self.mask_dir / f"{pattern}{ext.upper()}"
                if mask_path.exists():
                    return mask_path
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load mask if available
        mask_path = self.mask_files[idx]
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            # Normalize mask to 0-1
            if mask.max() > 1:
                mask = (mask > 128).astype(np.float32)
            else:
                mask = mask.astype(np.float32)
        else:
            # Create empty mask for test/validation
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            try:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            except Exception as e:
                print(f"Error transforming image {img_path.name}: {e}")
                # Fallback to simple resize
                image = transforms.ToTensor()(Image.fromarray(image).resize((self.image_size, self.image_size)))
                mask = transforms.ToTensor()(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))
        else:
            # Basic resize
            image = transforms.ToTensor()(Image.fromarray(image).resize((self.image_size, self.image_size)))
            mask = transforms.ToTensor()(Image.fromarray(mask).resize((self.image_size, self.image_size), Image.NEAREST))
        
        return {
            'image': image,
            'mask': mask.unsqueeze(0) if mask.dim() == 2 else mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path) if mask_path else ''
        }


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.Affine(
            translate_percent=(-0.2, 0.2),
            scale=(0.7, 1.3),
            rotate=(-90, 90),
            p=0.8
        ),
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.5),
        A.GaussianBlur(blur_limit=(5, 7), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.CoarseDropout(
            num_holes=12,
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            fill_value=0,
            p=0.5
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(image_size: int = 256) -> A.Compose:
    """Get validation data transforms"""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def create_data_loaders(config: Union[Dict, str]) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation"""
    # Load config if string
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Extract paths
    if 'paths' in config:
        paths = config['paths']
        train_image_dir = paths.get('train_images')
        train_mask_dir = paths.get('train_masks')
    else:
        train_image_dir = config.get('train_images')
        train_mask_dir = config.get('train_masks')
    
    # Get data parameters
    data_config = config.get('data', {})
    batch_size = data_config.get('batch_size', 2)
    image_size = data_config.get('image_size', 256)
    val_split = data_config.get('val_split', 0.2)
    num_workers = data_config.get('num_workers', 0)
    
    print(f"Creating data loaders...")
    print(f"  Train images: {train_image_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Create transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create full dataset
    try:
        full_dataset = SegmentationDataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            transform=train_transform,
            is_train=True,
            image_size=image_size
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise
    
    # Ensure we have enough data
    dataset_size = len(full_dataset)
    if dataset_size < 2:
        raise ValueError(f"Need at least 2 images for training, but found {dataset_size}")
    
    # Calculate split sizes
    val_size = max(1, int(val_split * dataset_size))  # At least 1
    train_size = dataset_size - val_size
    
    # If train_size becomes 0, adjust
    if train_size < 1:
        train_size = 1
        val_size = dataset_size - 1
    
    # Set random seed for reproducibility
    torch.manual_seed(config.get('experiment', {}).get('seed', 42))
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Set validation transforms
    val_dataset.dataset.transform = val_transform
    
    print(f"  Dataset: {dataset_size} total")
    print(f"  Split: {train_size} train, {val_size} validation")
    
    # Adjust batch size for small datasets
    train_batch_size = min(batch_size, train_size)
    val_batch_size = min(batch_size, val_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False  # Important for small datasets
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"  Train batches: {len(train_loader)} (batch size: {train_batch_size})")
    print(f"  Val batches: {len(val_loader)} (batch size: {val_batch_size})")
    
    return train_loader, val_loader