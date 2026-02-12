# scripts/preprocess.py
import argparse
import sys
import os
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # from src.data.preprocess import ImagePreprocessor
    from src.data.preprocess import ImagePreprocessor
except ImportError as e:
    print(f"ERROR: Could not import ImagePreprocessor: {e}")
    print("Please make sure:")
    print("1. src/data/preprocess.py exists")
    print("2. It contains ImagePreprocessor class")
    print(f"3. Project root is: {PROJECT_ROOT}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for segmentation')
    parser.add_argument('--input_dir', type=str, default='data/raw/seg_pred',
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='data/processed/preprocessed',
                       help='Directory to save preprocessed images')
    parser.add_argument('--config', type=str, default='configs/preprocessing_config.yaml',
                       help='Path to preprocessing configuration file')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                       help='Image file extensions to process')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nERROR: Input directory does not exist: {input_dir}")
        print(f"Creating empty directory for testing...")
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created {input_dir}")
        print("Please add some images to this directory and run again.")
        return
    
    # Create preprocessor
    try:
        preprocessor = ImagePreprocessor(args.config)
    except Exception as e:
        print(f"ERROR: Failed to create ImagePreprocessor: {e}")
        return
    
    # Process directory
    print("\n" + "="*50)
    print("Starting preprocessing...")
    print("="*50)
    
    preprocessor.process_directory(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        extensions=args.extensions
    )
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()