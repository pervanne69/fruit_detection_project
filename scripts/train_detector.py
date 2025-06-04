import sys
import os

# Добавим корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.detector import get_detector_model
from datasets.fruit_detection_dataset import FruitDetectionDataset
import utils.annotations as ann

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    annotations = ann.load_annotations("data/annotations.json")

    dataset = FruitDetectionDataset("data/train", annotations, transform=T.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = get_detector_model(num_classes=3)
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)


    print(f"🚀 Starting training on {len(dataset)} images")

    for epoch in range(3):
        print(f"\n🌀 Epoch {epoch+1}/3")
        total_loss = 0.0

        for i, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # отбрасываем пустые таргеты
            targets = [t for t in targets if len(t["boxes"]) > 0]
            if not targets:
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            print(f"[Batch {i+1}] Loss: {losses.item():.4f}")

        print(f"✅ Epoch {epoch+1} completed. Total loss: {total_loss:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/detector.pth")
    print("✅ Saved detector model to weights/detector.pth")
