import argparse
# import sys
# from pathlib import Path
import sys
from pathlib import Path

# add PROJECT ROOT to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


# Add src to path
# sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models.predict import SegmentationPredictor
from src.models.unet import UNet



def main():
    parser = argparse.ArgumentParser(description='Run segmentation predictions')
    parser.add_argument('--input_dir', type=str, default='data/raw/seg_pred',
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    parser.add_argument('--create_overlay', action='store_true',
                       help='Create overlay images')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SegmentationPredictor(
        model_path=args.model_path,
        config_path=args.config,
        device=args.device
    )
    
    # Run predictions
    predictor.predict_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )
    
    print(f"\nPredictions saved to: {args.output_dir}")
    
    # Create overlays if requested
    if args.create_overlay:
        from src.visualization.visualize import SegmentationVisualizer
        import cv2
        import numpy as np
        from tqdm import tqdm
        
        input_path = Path(args.input_dir)
        output_path = Path(args.output_dir)
        overlay_path = output_path / "overlays"
        overlay_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"\nCreating overlays for {len(image_files)} images...")
        
        for img_file in tqdm(image_files, desc="Creating overlays"):
            try:
                # Load original image
                image = cv2.imread(str(img_file))
                
                # Load corresponding mask
                mask_file = output_path / f"{img_file.stem}_mask.png"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    
                    # Create overlay
                    overlay = SegmentationVisualizer.create_overlay(
                        image, mask, alpha=0.5
                    )
                    
                    # Save overlay
                    overlay_file = overlay_path / f"{img_file.stem}_overlay.png"
                    cv2.imwrite(str(overlay_file), overlay)
                    
            except Exception as e:
                print(f"Error creating overlay for {img_file}: {e}")
        
        print(f"Overlays saved to: {overlay_path}")


if __name__ == '__main__':
    main()