import torch
import torch.nn as nn

from src.model import FashionMNISTModel
from src.data import train_loader

model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
model.train()
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
print("\n" + "="*50)
print(f"Total batches: {len(train_loader)}")
print(f"Total loss: {total_loss:.4f}")
print(f"Average loss: {average_loss:.4f}")
