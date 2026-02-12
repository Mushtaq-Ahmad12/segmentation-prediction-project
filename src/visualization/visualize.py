import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms


class SegmentationVisualizer:

    """Visualization utilities for segmentation"""
    
    @staticmethod
    def plot_metrics_history(train_history, val_history, save_path=None):
     """Plot training history"""
     epochs = range(1, len(train_history['loss']) + 1)
    
     fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 columns
    
    # Loss
     axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Train')
     axes[0, 0].plot(epochs, val_history['loss'], 'r-', label='Val')
     axes[0, 0].set_title('Loss')
     axes[0, 0].set_xlabel('Epoch')
     axes[0, 0].legend()
     axes[0, 0].grid(True, alpha=0.3)
    
    # IoU
     axes[0, 1].plot(epochs, train_history['iou'], 'b-', label='Train')
     axes[0, 1].plot(epochs, val_history['iou'], 'r-', label='Val')
     axes[0, 1].set_title('IoU Score')
     axes[0, 1].set_xlabel('Epoch')
     axes[0, 1].legend()
     axes[0, 1].grid(True, alpha=0.3)
    
    # Dice
     axes[0, 2].plot(epochs, train_history['dice'], 'b-', label='Train')
     axes[0, 2].plot(epochs, val_history['dice'], 'r-', label='Val')
     axes[0, 2].set_title('Dice Score')
     axes[0, 2].set_xlabel('Epoch')
     axes[0, 2].legend()
     axes[0, 2].grid(True, alpha=0.3)
    
    # F1
     axes[1, 0].plot(epochs, train_history['f1'], 'b-', label='Train')
     axes[1, 0].plot(epochs, val_history['f1'], 'r-', label='Val')
     axes[1, 0].set_title('F1 Score')
     axes[1, 0].set_xlabel('Epoch')
     axes[1, 0].legend()
     axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
     axes[1, 1].plot(epochs, train_history['accuracy'], 'b-', label='Train')
     axes[1, 1].plot(epochs, val_history['accuracy'], 'r-', label='Val')
     axes[1, 1].set_title('Accuracy')
     axes[1, 1].set_xlabel('Epoch')
     axes[1, 1].legend()
     axes[1, 1].grid(True, alpha=0.3)
    
    # Hide empty subplot (if using 2x3 grid, last one empty)
     axes[1, 2].axis('off')
    
     plt.tight_layout()
    
     if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
     else:
        plt.show()
    
    plt.close()