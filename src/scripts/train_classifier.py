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
    # ===== CONFIGURATION =====
    data_dir = "data/raw/seg_train"  # Points to folder with class subfolders
    batch_size = 64
    epochs = 20
    lr = 0.001
    num_classes = 6  # buildings, forest, glacier, mountain, sea, street
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ==========================

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found!")
        print("Available directories in data/raw/:")
        if os.path.exists("data/raw"):
            print(os.listdir("data/raw"))
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
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    print(f"Found {len(full_dataset)} images")
    print(f"Classes: {full_dataset.classes}")
    print(f"Number of classes: {len(full_dataset.classes)}")

    # ---- Train/Val Split (80/20) ----
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation
    val_dataset.dataset.transform = val_transform

    # ---- DataLoaders (num_workers=0 for Windows) ----
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- Model ----
    model = ResNetClassifier(num_classes=len(full_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ---- Training Loop ----
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_classifier.pth")
            print(f"  -> Saved best model (acc: {val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()