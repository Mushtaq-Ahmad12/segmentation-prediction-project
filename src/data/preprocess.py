# src/data/preprocess.py
import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

class ImagePreprocessor:
    """Preprocess images for segmentation models"""
    
    def __init__(self, config_path: str = 'configs/preprocessing_config.yaml'):
        self.config = self._load_config(config_path)
        print(f"Loaded config from: {config_path}")
        print(f"Config: {self.config}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default config
            return {
                'image_size': [256, 256],
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'save_visualization': True
            }
    
    def find_images(self, input_dir: str, extensions: List[str]) -> List[Path]:
        """Find all image files in directory with given extensions"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            print(f"Absolute path: {input_path.absolute()}")
            return []
        
        print(f"Looking for images in: {input_path.absolute()}")
        
        image_files = []
        # Try multiple methods to find images
        for ext in extensions:
            # Lowercase
            image_files.extend(input_path.glob(f"*{ext}"))
            # Uppercase
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            # Mixed case
            image_files.extend(input_path.glob(f"*{ext}"))
        
        # Also try recursive search
        if len(image_files) == 0:
            print("Trying recursive search...")
            for ext in extensions:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        # Sort for consistent ordering
        image_files.sort()
        
        return image_files
    
    def preprocess_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: cv2 cannot read {image_path}")
                return None
            
            # Convert BGR to RGB
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                # Grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Resize
            target_size = tuple(self.config.get('image_size', [256, 256]))
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            if self.config.get('normalize', True):
                img = img.astype(np.float32) / 255.0
                mean = np.array(self.config.get('mean', [0.485, 0.456, 0.406]))
                std = np.array(self.config.get('std', [0.229, 0.224, 0.225]))
                img = (img - mean) / std
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def process_directory(self, 
                         input_dir: str, 
                         output_dir: str, 
                         extensions: List[str] = None):
        """Process all images in directory"""
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find images
        image_files = self.find_images(input_dir, extensions)
        
        if len(image_files) == 0:
            print(f"\nERROR: No images found in {input_dir}")
            print(f"Expected extensions: {extensions}")
            print(f"Please check:")
            print(f"1. Directory exists: {Path(input_dir).exists()}")
            print(f"2. Files are in the correct format")
            print(f"3. You have read permissions")
            
            # List what's actually in the directory
            input_path = Path(input_dir)
            if input_path.exists():
                all_items = list(input_path.iterdir())
                print(f"\nItems found in {input_dir}:")
                for item in all_items:
                    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
            
            return
        
        print(f"\nFound {len(image_files)} images to process")
        print("Images found:")
        for i, img_path in enumerate(image_files[:10]):  # Show first 10
            print(f"  {i+1}. {img_path.name}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files)-10} more")
        
        # Process images
        successful = 0
        for img_path in tqdm(image_files, desc="Preprocessing images"):
            # Preprocess image
            processed_img = self.preprocess_image(img_path)
            
            if processed_img is not None:
                # Save as numpy array
                np_filename = output_path / f"{img_path.stem}_processed.npy"
                np.save(np_filename, processed_img)
                
                # Save visualization if configured
                if self.config.get('save_visualization', True):
                    # Convert back to uint8 for saving as image
                    if processed_img.dtype == np.float32:
                        vis_img = (processed_img * 255).astype(np.uint8)
                    else:
                        vis_img = processed_img.astype(np.uint8)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(vis_img.shape) == 3:
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    
                    img_filename = output_path / f"{img_path.stem}_visualization.jpg"
                    cv2.imwrite(str(img_filename), vis_img)
                
                successful += 1
        
        print(f"\nPreprocessing complete!")
        print(f"Successfully processed: {successful}/{len(image_files)} images")
        print(f"Output saved to: {output_path.absolute()}")