from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Global model
trained_model = None

# Class labels
class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]


# Model class
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Load model (only once)
def load_model():
    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet(num_classes=len(class_names))
        trained_model.load_state_dict(
            torch.load("model/saved_model.pth", map_location=torch.device('cpu'))
        )
        trained_model.eval()

    return trained_model


# Prediction function
def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = predicted.item()

    return class_names[predicted_class]