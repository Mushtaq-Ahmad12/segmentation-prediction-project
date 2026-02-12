# quick_predict.py
import sys
from pathlib import Path
import cv2
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def simple_predict():
    """Simple prediction without using DataLoader"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import model
    from models.unet import UNet
    model = UNet(n_channels=3, n_classes=1)
    model.to(device)
    model.eval()
    
    # Load image
    image_path = Path("data/raw/seg_pred/3.jpg")
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img_rgb.shape[:2]
    
    # Resize
    img_resized = cv2.resize(img_rgb, (256, 256))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
    
    # Process output
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Resize to original size
    mask = cv2.resize(mask, original_size[::-1], interpolation=cv2.INTER_NEAREST)
    
    # Save results
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mask
    mask_path = output_dir / "3_mask.png"
    cv2.imwrite(str(mask_path), mask)
    print(f"Mask saved to: {mask_path}")
    
    # Create overlay
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green overlay
    
    overlay_path = output_dir / "3_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"Overlay saved to: {overlay_path}")
    
    return mask_path, overlay_path

if __name__ == "__main__":
    simple_predict()