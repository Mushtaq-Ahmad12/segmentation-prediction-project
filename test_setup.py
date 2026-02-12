# test_project.py
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Current directory: {os.getcwd()}")

# Check if config file exists
config_path = PROJECT_ROOT / "configs" / "config.yaml"
print(f"\nLooking for config at: {config_path}")
print(f"Config exists: {config_path.exists()}")

if config_path.exists():
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"\nConfig loaded successfully!")
    print(f"Config keys: {list(config.keys())}")
else:
    print("\nCreating default config...")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        'experiment': {
            'name': 'segmentation_experiment',
            'seed': 42,
            'use_gpu': False
        },
        'paths': {
            'train_images': 'data/raw/seg_train/',
            'train_masks': 'data/raw/seg_train/',
            'test_images': 'data/raw/seg_test/',
            'model_save': 'models/',
            'logs': 'outputs/logs/',
            'predictions': 'outputs/predictions/'
        },
        'data': {
            'batch_size': 2,
            'image_size': 256,
            'val_split': 0.2,
            'num_workers': 0
        },
        'model': {
            'name': 'unet',
            'in_channels': 3,
            'classes': 1
        },
        'training': {
            'epochs': 20,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'patience': 5,
            'optimizer': 'adam'
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default config created at {config_path}")

# Test imports
print("\nTesting imports...")
try:
    from src.models.unet import UNet
    from src.utils.metrics import LossFunctions
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import error: {e}")