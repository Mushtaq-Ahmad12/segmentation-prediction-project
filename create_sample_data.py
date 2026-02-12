#!/usr/bin/env python3
"""
Script to create sample data structure for testing.
Run this if you don't have actual segmentation data yet.
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil

def create_sample_images(output_dir, num_images=10):
    """Create sample images for testing"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} sample images in {output_dir}...")
    
    for i in range(num_images):
        # Create random image
        height = np.random.randint(200, 500)
        width = np.random.randint(200, 500)
        
        # Create random color or grayscale image
        if np.random.random() > 0.5:
            # Color image
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        else:
            # Grayscale image
            img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        
        # Add some patterns
        cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), 2)
        cv2.circle(img, (250, 250), 50, (0, 255, 0), -1)
        
        # Save image
        img_path = output_path / f"sample_image_{i+1:03d}.png"
        cv2.imwrite(str(img_path), img)
    
    print(f"Created {num_images} sample images")

def main():
    """Create sample data structure"""
    
    # Create directories
    directories = [
        "data/raw/seg_pred",
        "data/raw/seg_train",
        "data/raw/seg_test",
        "data/processed/preprocessed",
        "data/processed/predictions",
        "outputs/predictions",
        "outputs/logs",
        "outputs/reports",
        "models",
        "models/model_checkpoints"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create sample images in seg_pred
    create_sample_images("data/raw/seg_pred", num_images=20)
    
    # Create sample training data (if needed)
    create_sample_images("data/raw/seg_train", num_images=5)
    create_sample_images("data/raw/seg_test", num_images=5)
    
    # Copy sample configs if they don't exist
    if not Path("configs/model_config.yaml").exists():
        shutil.copy("configs/model_config.yaml.example", 
                   "configs/model_config.yaml")
    
    print("\nSample data structure created successfully!")
    print("\nTo use the project:")
    print("1. Place your actual images in data/raw/seg_pred/")
    print("2. Run: python scripts/predict.py")
    print("3. Check results in outputs/predictions/")

if __name__ == "__main__":
    main()