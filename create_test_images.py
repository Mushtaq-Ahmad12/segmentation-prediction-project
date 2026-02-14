# create_test_images.py
import shutil
import random
from pathlib import Path

# Create test directory
test_dir = Path("data/raw/seg_test")
test_dir.mkdir(parents=True, exist_ok=True)

# Copy 5 random images from each class
train_dir = Path("data/raw/seg_train/seg_train")
for class_folder in train_dir.iterdir():
    if class_folder.is_dir():
        images = list(class_folder.glob("*.jpg"))
        selected = random.sample(images, min(5, len(images)))
        for img in selected:
            new_name = f"{class_folder.name}_{img.name}"
            shutil.copy(img, test_dir / new_name)
            print(f"✅ Copied: {new_name}")

print(f"\n🎉 Created {len(list(test_dir.glob('*.jpg')))} test images!")