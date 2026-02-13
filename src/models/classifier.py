import torch.nn as nn
import torchvision.models as models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Use weights='DEFAULT' instead of pretrained=True
        self.backbone = models.resnet34(weights='DEFAULT')
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)