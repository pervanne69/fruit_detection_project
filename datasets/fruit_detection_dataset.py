import os
import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T

class FruitDetectionDataset(Dataset):
    def __init__(self, root_dir, annotations, transform=None):
        self.root_dir = root_dir
        self.annotations = annotations
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],  # стандарт для ImageNet
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root_dir, ann['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if ann['boxes']:
            boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(ann['labels'], dtype=torch.int64)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)

        target = {"boxes": boxes, "labels": labels}
        return image, target
