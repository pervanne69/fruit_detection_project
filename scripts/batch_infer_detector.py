import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from glob import glob
from PIL import Image, ImageDraw
import torchvision.transforms as T
import pandas as pd
from models.detector import get_detector_model

# Параметры
input_dir = "data/val"
output_dir = "output_detector"
model_path = "weights/detector.pth"
threshold = 0.2
classes = {1: "apple", 2: "banana"}

# Подготовка модели
model = get_detector_model(num_classes=3)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# CSV-таблица
results = []

# Прогон по изображениям
os.makedirs(output_dir, exist_ok=True)
image_paths = glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)

for path in image_paths:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    boxes = predictions["boxes"]
    labels = predictions["labels"]
    scores = predictions["scores"]

    for box, label, score in zip(boxes, labels, scores):
        score = score.item()
        if score >= threshold:
            label_name = classes.get(label.item(), "unknown")
            box = box.tolist()
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label_name} ({score:.2f})", fill="red")

            results.append({
                "filename": os.path.relpath(path, input_dir),
                "label": label_name,
                "score": round(score, 4),
                "box": [round(x, 1) for x in box]
            })

    # Сохраняем картинку
    save_path = os.path.join(output_dir, os.path.basename(path))
    image.save(save_path)

# Сохраняем CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "detection_results.csv"), index=False)
print(f"✅ Saved {len(results)} detections to detection_results.csv")
