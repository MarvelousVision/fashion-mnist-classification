import torch
import torch.nn as nn

from src.model import FashionMNISTModel
from src.data import train_loader, test_loader, val_loader
from src.evaluate import evaluate

model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

train_losses = []
val_losses = []
val_accuracies = []

num_epochs=5
for epoch in range(num_epochs): 
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

    avg_train_loss = total_loss / len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"  Train loss:     {avg_train_loss:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    print(f"  Validation accuracy: {val_acc:.2f}%")
    print("-" * 60)

    

train_loss, train_acc = evaluate(model, train_loader, criterion)
print(f"Train loss: {train_loss:.4f}")
print(f"Train accuracy: {train_acc:.2f}%")

val_loss, val_acc = evaluate(model, val_loader, criterion)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc:.2f}%")

import matplotlib.pyplot as plt

plt.switch_backend('Agg') 

epochs = range(1, num_epochs + 1)


plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.savefig('loss_curve.png')  
plt.close()  

plt.figure(figsize=(10, 5))
plt.plot(epochs, val_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.ylim(0, 100)
plt.legend()
plt.grid()
plt.savefig('accuracy_curve.png')  
plt.close()  

