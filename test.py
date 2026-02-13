from pathlib import Path
import os

base = Path("data/raw")
print(f"Checking: {base.absolute()}")
print(f"Exists: {base.exists()}")

if base.exists():
    print("\nContents of data/raw:")
    for item in base.iterdir():
        print(f"  {item.name}/")
        
        # Look inside each folder
        if item.is_dir():
            sub_items = list(item.iterdir())[:3]  # First 3 items
            for sub in sub_items:
                print(f"    {sub.name}/")
                
                # If it's a directory with class folders, show them
                if sub.is_dir():
                    classes = list(sub.iterdir())[:3]
                    for cls in classes:
                        if cls.is_dir():
                            print(f"      {cls.name}/")