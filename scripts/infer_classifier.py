import sys
import os

# Добавляем корень проекта в путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
from models.classifier import get_classifier_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path, model_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    model = get_classifier_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    classes = ['apple', 'banana']
    print(f"Predicted: {classes[predicted.item()]}")

if __name__ == '__main__':
    predict(sys.argv[1], "weights/classifier.pth")
