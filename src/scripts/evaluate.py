import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import yaml
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.predict import SegmentationPredictor
from utils.metrics import SegmentationMetrics
from visualization.visualize import SegmentationVisualizer


def evaluate_model(model_path: str,
                  test_image_dir: str,
                  test_mask_dir: str,
                  config_path: str = "configs/model_config.yaml",
                  threshold: float = 0.5,
                  device: str = None):
    """Evaluate model on test dataset"""
    
    # Create predictor
    predictor = SegmentationPredictor(
        model_path=model_path,
        config_path=config_path,
        device=device
    )
    
    # Get all image files
    test_path = Path(test_image_dir)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(test_path.glob(f'*{ext}'))
        image_files.extend(test_path.glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} test images")
    
    # Initialize metrics
    all_metrics = {
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Process each image
    for img_file in tqdm(image_files, desc="Evaluating"):
        try:
            # Load ground truth mask
            mask_file = Path(test_mask_dir) / f"{img_file.stem}_mask.png"
            if not mask_file.exists():
                # Try different naming convention
                mask_file = Path(test_mask_dir) / f"{img_file.stem}.png"
            
            if not mask_file.exists():
                print(f"Mask not found for {img_file.stem}")
                continue
            
            gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Could not load mask: {mask_file}")
                continue
            
            # Predict mask
            pred_mask = predictor.predict_from_path(str(img_file), threshold)
            
            # Calculate metrics
            metrics = SegmentationMetrics.calculate_all_metrics(
                gt_mask, pred_mask, threshold
            )
            
            # Store metrics
            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
            std_metrics = np.std(values)
            print(f"{key.upper()}: {avg_metrics[key]:.4f} ± {std_metrics:.4f}")
        else:
            avg_metrics[key] = 0.0
            print(f"{key.upper()}: No values")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--test_image_dir', type=str, default='data/raw/seg_test',
                       help='Directory containing test images')
    parser.add_argument('--test_mask_dir', type=str, default='data/raw/seg_test',
                       help='Directory containing test masks')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results to file')
    
    args = parser.parse_args()
    
    # Check if test data exists
    test_image_path = Path(args.test_image_dir)
    if not test_image_path.exists():
        print(f"Test image directory not found: {args.test_image_dir}")
        print("Please create the directory and add test images.")
        return
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        test_image_dir=args.test_image_dir,
        test_mask_dir=args.test_mask_dir,
        config_path=args.config,
        threshold=args.threshold,
        device=args.device
    )
    
    # Save results if requested
    if args.save_results:
        import json
        from datetime import datetime
        
        results = {
            'model_path': args.model_path,
            'test_image_dir': args.test_image_dir,
            'test_mask_dir': args.test_mask_dir,
            'threshold': args.threshold,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = Path('outputs/reports') / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()