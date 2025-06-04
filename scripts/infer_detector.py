import sys
import os

# Добавим корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from models.detector import get_detector_model

transform = T.Compose([
    T.ToTensor()
])

CLASSES = ['__background__', 'apple', 'banana']

def predict(image_path, model_path, threshold=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Загрузка модели
    model = get_detector_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Предсказание
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Визуализация результатов
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            box = box.cpu().numpy()
            label_name = CLASSES[label]
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="red")

    image.show()

    # Печать в консоль
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score >= threshold:
            print(f"{CLASSES[label]} at {box.tolist()} (score: {score:.2f})")

if __name__ == '__main__':
    image_path = sys.argv[1]  # Путь к изображению
    model_path = "weights/detector.pth"
    predict(image_path, model_path)
