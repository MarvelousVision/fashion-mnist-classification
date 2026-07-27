import torch
from src.model import FashionMNISTModel
from src.data import test_loader

def evaluate(model, test_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            flattened_images = images.view(images.size(0), -1)
            logits = model(flattened_images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    average_loss = total_loss / num_batches
    return average_loss, accuracy

