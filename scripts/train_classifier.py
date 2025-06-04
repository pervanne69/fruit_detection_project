import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.fruit_classification_dataset import get_classification_dataloaders
from models.classifier import get_classifier_model


# Добавляем корень проекта в sys.path
current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

# Устанавливаем рабочую директорию для корректной загрузки данных
os.chdir(root_dir)

def train(model, dataloaders, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Loss: {running_loss:.4f}")
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = get_classification_dataloaders("data")
    model = get_classifier_model()
    model = train(model, dataloaders, device)
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/classifier.pth")
