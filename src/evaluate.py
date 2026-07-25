import torch
from src.model import FashionMNISTModel
from src.data import test_loader

def evaluate(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            flattened_images = images.view(images.size(0), -1)
            logits = model(flattened_images)
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

