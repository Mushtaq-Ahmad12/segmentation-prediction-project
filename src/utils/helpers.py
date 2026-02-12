import torch
import random
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_experiment_dir(base_dir: str = "experiments"):
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    
    # Create directories
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "configs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "predictions").mkdir(parents=True, exist_ok=True)
    
    return str(exp_dir)


def save_config(config: dict, save_path: Path):
    """Save configuration to file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(config_path: str):
    """Load configuration from file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config