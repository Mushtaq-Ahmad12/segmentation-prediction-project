import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.classifier import ResNetClassifier

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Training history saved to {save_path}")

def main():
    # ===== CONFIGURATION =====
    data_dir = "data/raw/seg_train/seg_train"  # Path to training data
    batch_size = 64
    epochs = 20  # Increased for better training
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==========================

    # Create output directory for logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"training_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # Print system info
    print("=" * 60)
    print("🚀 IMAGE CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path: {data_dir}")
    print(f"Absolute path: {os.path.abspath(data_dir)}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print("=" * 60)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory {data_dir} not found!")
        print("\n🔍 Let's see what's actually in data/raw/:")
        
        # Show what's in data/raw/
        if os.path.exists("data/raw"):
            print("Contents of data/raw:")
            for item in os.listdir("data/raw"):
                item_path = os.path.join("data/raw", item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    # Show contents of seg_train if it exists
                    if item == "seg_train":
                        seg_train_path = item_path
                        for sub in os.listdir(seg_train_path)[:5]:
                            sub_path = os.path.join(seg_train_path, sub)
                            if os.path.isdir(sub_path):
                                print(f"    📁 {sub}/")
        else:
            print("data/raw/ doesn't exist!")
            print("Creating data/raw/ directory...")
            os.makedirs("data/raw", exist_ok=True)
        return

    # ---- Transforms ----
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---- Load dataset ----
    try:
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
        print(f"\n✅ Found {len(full_dataset)} images")
        print(f"📂 Classes: {full_dataset.classes}")
        print(f"🔢 Number of classes: {len(full_dataset.classes)}")
        
        # Show class distribution
        print("\n📊 Class distribution:")
        class_counts = {}
        for _, label in full_dataset.samples:
            class_name = full_dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # ---- Train/Val Split (80/20) ----
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation
    val_dataset.dataset.transform = val_transform

    print(f"\n📊 Dataset split:")
    print(f"  Training: {train_size} images")
    print(f"  Validation: {val_size} images")

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    # ---- Model ----
    model = ResNetClassifier(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    print(f"\n🚀 Model Architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("\n" + "=" * 60)
    print("🏋️  STARTING TRAINING")
    print("=" * 60)

    # ---- Training Loop ----
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n📝 Epoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate batch accuracy
            batch_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)

            # Print progress
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx:3d}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | '
                      f'Acc: {batch_acc:.2f}%')

        # Calculate training metrics
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculate validation metrics
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save checkpoint with all information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'classes': full_dataset.classes
            }
            
            torch.save(checkpoint, f"{log_dir}/best_model.pth")
            print(f"  ✅ New best model saved! (Accuracy: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  ⚠️ Early stopping triggered after {epoch+1} epochs")
                break

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, f"{log_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"  💾 Checkpoint saved for epoch {epoch+1}")

    # ---- Training Complete ----
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Training logs saved in: {log_dir}/")
    
    # Plot and save training history
    plot_training_history(
        train_losses, val_losses, 
        train_accuracies, val_accuracies,
        save_path=f"{log_dir}/training_history.png"
    )
    
    # Save final model
    torch.save(model.state_dict(), f"{log_dir}/final_model.pth")
    print(f"✅ Final model saved as: {log_dir}/final_model.pth")
    print(f"✅ Best model saved as: {log_dir}/best_model.pth")
    
    # Print final summary
    print("\n📈 Final Performance:")
    print(f"  Best Train Accuracy: {max(train_accuracies):.2f}%")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Final Train Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"  Final Validation Accuracy: {val_accuracies[-1]:.2f}%")

if __name__ == '__main__':
    main()