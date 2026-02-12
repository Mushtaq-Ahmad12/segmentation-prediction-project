import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import sys
import os
import segmentation_models_pytorch as smp
import torch.nn.functional as F

import segmentation_models_pytorch as smp

import segmentation_models_pytorch as smp

def create_model(self):
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1   # binary segmentation
    )
    return model
    # Freeze encoder for first 5 epochs (optional but recommended)
    for param in model.encoder.parameters():
        param.requires_grad = False
    return model

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import UNet
from src.data.loader import create_data_loaders
from src.utils.metrics import SegmentationMetrics, LossFunctions
from src.utils.helpers import (
    set_seed,
    create_experiment_dir,
    save_config,
    get_device
)
from src.visualization.visualize import SegmentationVisualizer


class Trainer:
    """Training pipeline for segmentation models with accuracy tracking"""
    
    def __init__(self, config_path: str):
        # Convert to absolute path if relative
        if not os.path.isabs(config_path):
            config_path = str(PROJECT_ROOT / config_path)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = get_device()
        
        # Set seed for reproducibility
        seed = self.config.get('experiment', {}).get('seed', 42)
        set_seed(seed)
        
        # Create experiment directory
        self.exp_dir = create_experiment_dir("experiments")
        print(f"Experiment directory: {self.exp_dir}")
        
        # Save config
        save_config(self.config, Path(self.exp_dir) / "configs" / "config.yaml")
        
        # Initialize model
        self.model = self.create_model()
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self.create_optimizer()
        
        # Initialize learning rate scheduler (optional)
        self.scheduler = self.create_scheduler()
        
        # Initialize loss function
        self.criterion = self.create_criterion()
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(self.config)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(Path(self.exp_dir) / "logs")
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
        # Metrics history (including accuracy)
        self.train_history = {'loss': [], 'iou': [], 'dice': [], 'f1': [], 'accuracy': []}
        self.val_history = {'loss': [], 'iou': [], 'dice': [], 'f1': [], 'accuracy': []}
    
    def create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_config = self.config.get('model', {})
        
        if model_config.get('name') == 'unet':
            model = UNet(
                n_channels=model_config.get('in_channels', 3),
                n_classes=model_config.get('classes', 1)
            )
        else:
            raise ValueError(f"Unknown model: {model_config.get('name')}")
        
        print(f"Created {model_config.get('name', 'unet')} model")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        train_config = self.config.get('training', {})
        
        lr = train_config.get('learning_rate', 0.001)
        weight_decay = train_config.get('weight_decay', 0.0001)
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        return optimizer
    
    def create_scheduler(self):
     """Create learning rate scheduler"""
     train_config = self.config.get('training', {})
     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,
        patience=5
        # verbose=True   <-- REMOVE THIS LINE
    )
     return scheduler
    
    def create_criterion(self):
     def criterion(pred, target):
        pos_weight = torch.tensor([2.0]).to(self.device)  # adjust weight
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
        dice = LossFunctions.dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice
     return criterion
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Check if train loader is empty
        if len(self.train_loader) == 0:
            print("Warning: Train loader is empty!")
            return {'loss': 0.0, **epoch_metrics}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate batch metrics (now includes accuracy)
            with torch.no_grad():
                batch_metrics = SegmentationMetrics.calculate_batch_metrics(masks, outputs)
                for key in epoch_metrics.keys():
                    epoch_metrics[key] += batch_metrics.get(key, 0.0)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}: Loss: {loss.item():.4f}')
        
        # Average metrics
        if len(self.train_loader) > 0:
            avg_loss = epoch_loss / len(self.train_loader)
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= len(self.train_loader)
        else:
            avg_loss = 0.0
        
        return {'loss': avg_loss, **epoch_metrics}
    
    def validate_epoch(self) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                epoch_loss += loss.item()
                
                # Calculate batch metrics (includes accuracy)
                batch_metrics = SegmentationMetrics.calculate_batch_metrics(masks, outputs)
                for key in epoch_metrics.keys():
                    epoch_metrics[key] += batch_metrics.get(key, 0.0)
        
        # Average metrics
        if len(self.val_loader) > 0:
            avg_loss = epoch_loss / len(self.val_loader)
            for key in epoch_metrics.keys():
                epoch_metrics[key] /= len(self.val_loader)
        else:
            avg_loss = 0.0
        
        return {'loss': avg_loss, **epoch_metrics}
        print(f"    Output logits - mean: {outputs.mean().item():.4f}, "
           f"min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = Path(self.exp_dir) / "models" / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.exp_dir) / "models" / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = str(PROJECT_ROOT / checkpoint_path)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_iou = checkpoint['best_val_iou']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.val_history = checkpoint.get('val_history', self.val_history)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Resuming training from epoch {self.current_epoch}")
    
    def log_to_tensorboard(self, train_metrics: dict, val_metrics: dict):
        """Log metrics to tensorboard"""
        for key in train_metrics.keys():
            self.writer.add_scalar(f'Train/{key}', train_metrics[key], self.current_epoch)
            if key in val_metrics:
                self.writer.add_scalar(f'Val/{key}', val_metrics[key], self.current_epoch)
        
        # Log learning rate
        self.writer.add_scalar('Train/lr', 
                              self.optimizer.param_groups[0]['lr'], 
                              self.current_epoch)
    
    def train(self, num_epochs: int = None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.get('training', {}).get('epochs', 20)
        
        patience = self.config.get('training', {}).get('patience', 5)
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch()
            self.train_history['loss'].append(train_metrics['loss'])
            for key in ['iou', 'dice', 'f1', 'accuracy']:
                self.train_history[key].append(train_metrics.get(key, 0.0))
            
            # Validate
            val_metrics = self.validate_epoch()
            self.val_history['loss'].append(val_metrics['loss'])
            for key in ['iou', 'dice', 'f1', 'accuracy']:
                self.val_history[key].append(val_metrics.get(key, 0.0))
            
            # Print metrics with accuracy
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train IoU: {train_metrics.get('iou', 0):.4f}, "
                  f"Train Acc: {train_metrics.get('accuracy', 0):.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val IoU: {val_metrics.get('iou', 0):.4f}, "
                  f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            
            # Log to tensorboard
            self.log_to_tensorboard(train_metrics, val_metrics)
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # Check if this is the best model (based on validation loss)
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_iou = val_metrics.get('iou', 0)
                self.best_val_acc = val_metrics.get('accuracy', 0)
                self.save_checkpoint("checkpoint_best.pth", is_best=True)
                patience_counter = 0
                print(f"  New best model! Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val IoU: {val_metrics.get('iou', 0):.4f}, "
                      f"Val Acc: {val_metrics.get('accuracy', 0):.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Close tensorboard writer
        self.writer.close()
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        # Plot training history
        vis_path = Path(self.exp_dir) / "training_history.png"
        SegmentationVisualizer.plot_metrics_history(
            self.train_history, self.val_history, str(vis_path)
        )
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation IoU: {self.best_val_iou:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Results saved to: {self.exp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Convert config path to absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)
    
    print(f"Loading config from: {config_path}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please run: python setup.py")
        return
    
    # Create trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = str(PROJECT_ROOT / resume_path)
        
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            print(f"Warning: Checkpoint not found at {resume_path}")
    
    # Start training
    trainer.train(args.epochs)
    

if __name__ == '__main__':
    main()