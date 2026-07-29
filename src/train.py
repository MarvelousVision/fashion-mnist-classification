import torch
import torch.nn as nn

from src.model import FashionMNISTModel , FashionMNISTCNN
from src.data import train_loader, test_loader, val_loader
from src.evaluate import evaluate, collect_predictions, plot_misclassified_examples
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionMNISTCNN().to(device)
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
        # flattened_images = images.view(images.size(0), -1)  
        # logits = model(flattened_images)
        logits = model(images)
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

true_labels, predicted_labels = collect_predictions(model, val_loader)
cm = confusion_matrix(true_labels, predicted_labels)
print(f"Confusion matrix shape: {cm.shape}")  
print(f"Total samples: {cm.sum()}")

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=True,
    square=True
)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Confusion Matrix - Fashion MNIST', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()


plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
