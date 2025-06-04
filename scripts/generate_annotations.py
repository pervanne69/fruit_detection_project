import os
import cv2
import json
from tqdm import tqdm

def create_annotation(image_path, label):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"⚠️ No contours found in {image_path} — fallback to full image")
        x1, y1, x2, y2 = 0, 0, w - 1, h - 1
    else:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x + bw)
        y2 = min(h - 1, y + bh)

        # Отбрасываем пустые боксы
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Skipping zero-area box in {image_path}")
            return None

    # Сохраняем адекватный бокс
    return {
        "filename": os.path.relpath(image_path, "data/train"),
        "boxes": [[x1, y1, x2, y2]],
        "labels": [label]
    }

if __name__ == '__main__':
    data_root = "data/train"
    classes = {
        "apples": 1,
        "bananas": 2
    }

    annotations = []

    # 🍏🍌 Обработка классов объектов
    for cls_name, cls_id in classes.items():
        cls_path = os.path.join(data_root, cls_name)
        for fname in tqdm(os.listdir(cls_path), desc=f"Processing {cls_name}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            full_path = os.path.join(cls_path, fname)
            ann = create_annotation(full_path, cls_id)
            if ann:
                annotations.append(ann)

    # 🌫️ Обработка пустых изображений
    empty_path = os.path.join(data_root, "empty")
    if os.path.exists(empty_path):
        print("Processing empty (background class)...")
        for fname in tqdm(os.listdir(empty_path), desc="Empty"):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                annotations.append({
                    "filename": os.path.join("empty", fname),
                    "boxes": [],
                    "labels": []
                })

    # 💾 Сохраняем аннотации
    os.makedirs("data", exist_ok=True)
    with open("data/annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n✅ Saved {len(annotations)} annotations to data/annotations.json")
