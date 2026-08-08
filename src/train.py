import torch
import torch.nn as nn
import os

from src.model import  FashionMNISTCNN
from src.data import train_loader,  val_loader
from src.evaluate import evaluate
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

SEED = 42
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.0

torch.manual_seed(SEED)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True) 

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS): 
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"  Train loss:     {avg_train_loss:.4f}")
        print(f"  Validation loss: {val_loss:.4f}")
        print(f"  Validation accuracy: {val_acc:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch +1
            torch.save(
                model.state_dict(),
                "outputs/models/best_cnn_baseline.pth"
            )
            print(f"New best model saved (epoch {epoch+1})")

    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('outputs/figures/cnn_loss_curve.png')  
    plt.close()  

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, marker='o', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.ylim(80, 95)
    plt.legend()
    plt.grid()
    plt.savefig("outputs/figures/cnn_val_accuracy_curve.png")  
    plt.close()  

    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train()
