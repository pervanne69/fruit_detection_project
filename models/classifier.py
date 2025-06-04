import torch.nn as nn
from torchvision import models

def get_classifier_model(num_classes=2):
    model = models.resnet18(weights="DEFAULT")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
