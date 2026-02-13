import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.classifier import ResNetClassifier

def main():
    # ===== CONFIGURATION FOR COLAB =====
    # Use the nested path for Colab (where class folders are one level deeper)
    data_dir = "data/raw/seg_train/seg_train"  # ← CHANGED for Colab
    batch_size = 64
    epochs = 10
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ====================================

    # Print current directory and check path
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking path: {data_dir}")
    print(f"Absolute path: {os.path.abspath(data_dir)}")
    
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
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # ---- Train/Val Split (80/20) ----
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation
    val_dataset.dataset.transform = val_transform

    # ---- DataLoaders (num_workers=2 for Colab) ----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ---- Model ----
    model = ResNetClassifier(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n🚀 Starting training on {device}")
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print("-" * 60)

    # ---- Training Loop ----
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

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

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        # Validation
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

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'classes': full_dataset.classes
            }, "best_classifier.pth")
            print(f"  ✅ Saved best model (acc: {val_acc:.2f}%)")

    print(f"\n🎉 Training complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved as: best_classifier.pth")

if __name__ == '__main__':
    main()