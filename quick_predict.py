# quick_test.py
import torch
from PIL import Image
from pathlib import Path

# Find the latest training log
logs = sorted(Path(".").glob("training_logs_*"))
if logs:
    latest_log = logs[-1]
    checkpoint = latest_log / "best_model.pth"
    
    # Find a test image
    test_images = list(Path("data/raw/seg_test").glob("*.jpg"))
    if test_images:
        print(f"Running: python src/scripts/predict_classifier.py {test_images[0]} --checkpoint {checkpoint}")
        import subprocess
        subprocess.run(["python", "src/scripts/predict_classifier.py", str(test_images[0]), "--checkpoint", str(checkpoint)])
    else:
        print("No test images found in data/raw/seg_test/")
else:
    print("No training logs found. Train the model first!")