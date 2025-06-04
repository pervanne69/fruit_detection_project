import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import json
from glob import glob
from tqdm import tqdm
from models.detector import get_detector_model
from PIL import Image
import torchvision.transforms as T

# Названия классов по индексам
CLASSES = {1: "apple", 2: "banana"}


def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


if __name__ == "__main__":
    model_path = "weights/detector.pth"
    input_dir = "data/val"
    threshold = 0.05  # фиксируем очень низкий threshold для анализа

    model = get_detector_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    image_paths = glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True)
    print(f"📂 Found {len(image_paths)} images in val set")

    all_scores = []
    class_counts = {1: 0, 2: 0}
    empty_detections = 0

    for path in tqdm(image_paths, desc="🔍 Evaluating"):
        image_tensor = load_image(path)
        with torch.no_grad():
            output = model(image_tensor)[0]

        scores = output["scores"].tolist()
        labels = output["labels"].tolist()

        if not scores:
            empty_detections += 1
        else:
            all_scores.extend(scores)
            for label in labels:
                if label in class_counts:
                    class_counts[label] += 1

    print("\n📊 Detection Score Stats")
    if all_scores:
        print(f"  ➤ Total Detections: {len(all_scores)}")
        print(f"  ➤ Mean Score: {sum(all_scores) / len(all_scores):.4f}")
        print(f"  ➤ Max Score: {max(all_scores):.4f}")
        print(f"  ➤ Min Score: {min(all_scores):.4f}")
    else:
        print("  ⚠️ No detections with score above threshold")

    print(f"\n📈 Class Counts:")
    for cls_id, count in class_counts.items():
        print(f"  - {CLASSES[cls_id]}: {count} detections")

    print(f"\n🚫 No detections on {empty_detections} images")
