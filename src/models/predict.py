import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from tqdm import tqdm
import torch.nn.functional as F

from .unet import UNet


class SegmentationPredictor:
    """Predictor for segmentation models"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "configs/config.yaml",
                 device: str = None):
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get image size from config
        self.image_size = tuple(self.config['data']['image_size'])
        
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model"""
        
        if not Path(model_path).exists():
            print(f"Model not found at {model_path}. Creating new model...")
            model_config = self.config['model']
            model = UNet(
                n_channels=model_config['in_channels'],
                n_classes=model_config['classes']
            )
        else:
            try:
                model = UNet.load(model_path, device=self.device)
                print(f"Loaded model from {model_path}")
            except:
                print(f"Failed to load model. Creating new model...")
                model_config = self.config['model']
                model = UNet(
                    n_channels=model_config['in_channels'],
                    n_classes=model_config['classes']
                )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        original_size = image.shape[:2]
        image = cv2.resize(image, self.image_size[::-1])  # cv2 uses (width, height)
        
        # Normalize
        mean = np.array(self.config['data']['mean'])
        std = np.array(self.config['data']['std'])
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0)  # Add batch dimension
        
        return image.to(self.device), original_size
    
    def postprocess_mask(self, 
                        mask: torch.Tensor,
                        original_size: tuple,
                        threshold: float = 0.5) -> np.ndarray:
        """Postprocess prediction mask"""
        
        # Remove batch dimension and move to CPU
        mask = mask.squeeze().cpu().numpy()
        
        # Threshold
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Resize to original size
        binary_mask = cv2.resize(binary_mask, original_size[::-1], 
                                interpolation=cv2.INTER_NEAREST)
        
        return binary_mask * 255  # Scale to 0-255
    
    def predict_single(self, 
                      image: np.ndarray,
                      threshold: float = 0.5) -> np.ndarray:
        """Predict segmentation mask for single image"""
        
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        mask = self.postprocess_mask(output, original_size, threshold)
        
        return mask
    
    def predict_batch(self, 
                     images: List[np.ndarray],
                     threshold: float = 0.5) -> List[np.ndarray]:
        """Predict segmentation masks for batch of images"""
        
        batch_tensors = []
        original_sizes = []
        
        # Preprocess all images
        for image in images:
            input_tensor, original_size = self.preprocess_image(image)
            batch_tensors.append(input_tensor)
            original_sizes.append(original_size)
        
        # Stack batch
        batch = torch.cat(batch_tensors, dim=0)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Postprocess each mask
        masks = []
        for i, output in enumerate(outputs):
            mask = self.postprocess_mask(output.unsqueeze(0), 
                                        original_sizes[i], 
                                        threshold)
            masks.append(mask)
        
        return masks
    
    def predict_from_path(self, 
                         image_path: str,
                         threshold: float = 0.5) -> np.ndarray:
        """Predict segmentation mask from image file"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Predict
        mask = self.predict_single(image, threshold)
        
        return mask
    
    def predict_directory(self, 
                         input_dir: str,
                         output_dir: str,
                         threshold: float = 0.5,
                         extensions: List[str] = None):
        """Predict masks for all images in directory"""
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        input_path = Path(input_dir)
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_file in tqdm(image_files, desc="Predicting masks"):
            try:
                # Predict mask
                mask = self.predict_from_path(str(img_file), threshold)
                
                # Save mask
                output_file = output_path / f"{img_file.stem}_mask.png"
                cv2.imwrite(str(output_file), mask)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        print(f"Predictions saved to {output_dir}")
    
    def create_overlay(self, 
                      image: np.ndarray,
                      mask: np.ndarray,
                      alpha: float = 0.5,
                      color: tuple = (0, 255, 0)) -> np.ndarray:
        """Create overlay of mask on original image"""
        
        # Ensure image is in BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay