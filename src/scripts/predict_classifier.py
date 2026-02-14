#!/usr/bin/env python
"""
Prediction script for trained image classifier
Usage: python predict_classifier.py <image_path> [--checkpoint CHECKPOINT_PATH]
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.classifier import ResNetClassifier

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"📂 Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = ResNetClassifier(num_classes=6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def predict_image(image_path, model, device):
    """Predict class for a single image"""
    # Define classes
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Define transforms (must match validation transforms used in training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None
    
    # Transform and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get top-5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    result = {
        'filename': Path(image_path).name,
        'predicted_class': classes[predicted.item()],
        'confidence': confidence.item() * 100,
        'top5_predictions': [
            {'class': classes[idx.item()], 'probability': prob.item() * 100}
            for idx, prob in zip(top5_idx[0], top5_prob[0])
        ]
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Predict image class using trained model')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--checkpoint', type=str, 
                       default='training_logs_20260214_043945/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, 
                       help='Show top K predictions')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"❌ Error: Image not found at {args.image_path}")
        return
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        print("\nAvailable training logs:")
        logs_dir = Path("training_logs_*")
        for log in sorted(Path(".").glob("training_logs_*")):
            if log.is_dir():
                best_model = log / "best_model.pth"
                if best_model.exists():
                    print(f"  📁 {log}/")
                    print(f"     └── best_model.pth")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Print model info
    if 'best_acc' in checkpoint:
        print(f"🏆 Model accuracy: {checkpoint['best_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"📊 Trained for {checkpoint['epoch'] + 1} epochs")
    
    # Predict
    print(f"\n🔍 Predicting: {args.image_path}")
    result = predict_image(args.image_path, model, device)
    
    if result:
        print("\n" + "="*50)
        print("✅ PREDICTION RESULT")
        print("="*50)
        print(f"📸 Image: {result['filename']}")
        print(f"🎯 Predicted class: \033[1;32m{result['predicted_class']}\033[0m")
        print(f"📊 Confidence: \033[1;36m{result['confidence']:.2f}%\033[0m")
        
        print("\n📈 Top 5 predictions:")
        print("-" * 40)
        for i, pred in enumerate(result['top5_predictions'], 1):
            bar_length = int(pred['probability'] / 2)  # Scale for display
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{i}. {pred['class']:12s} |{bar}| {pred['probability']:.2f}%")
        
        # Save result to JSON
        output_file = f"prediction_{Path(args.image_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Result saved to: {output_file}")

if __name__ == '__main__':
    main()