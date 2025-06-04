import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

def get_classification_dataloaders(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    return {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    }
