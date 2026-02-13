import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.classifier import ResNetClassifier

def predict_image(image_path, model_path="best_classifier.pth"):
    # Classes (Intel Image Classification)
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ResNetClassifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and predict
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = classes[predicted.item()]
    confidence = confidence.item()
    
    return class_name, confidence

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_classifier.py <image_path>")
        return
    
    image_path = sys.argv[1]
    class_name, confidence = predict_image(image_path)
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()