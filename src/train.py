import torch
import torch.nn as nn

from src.model import FashionMNISTModel
from src.data import train_loader, test_loader
from src.evaluate import evaluate

model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model.train()
num_epochs=5
for epoch in range(num_epochs): 
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        flattened_images = images.view(images.size(0), -1)  
        logits = model(flattened_images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {average_loss:.4f}")

accuracy = evaluate(model, test_loader) 
print(f"Test accuracy: {accuracy:.2f}%")