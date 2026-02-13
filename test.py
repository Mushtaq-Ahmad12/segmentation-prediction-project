from pathlib import Path
import sys

data_dir = Path("data/raw/seg_train")
print(f"Absolute path: {data_dir.absolute()}")
print(f"Exists: {data_dir.exists()}")

if data_dir.exists():
    subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    print(f"Subfolders found: {subdirs[:6]}")
    print(f"Total subfolders: {len(subdirs)}")
else:
    print("❌ Path not found. Check the folder name and location.")