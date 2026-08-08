import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.data import val_loader
from src.evaluate import (
    calculate_per_class_accuracy,
    collect_predictions,
    plot_misclassified_examples,
)
from src.model import FashionMNISTCNN

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

def analyze():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FashionMNISTCNN().to(device)
    model.load_state_dict(
        torch.load(
            "outputs/models/best_cnn_baseline.pth",
            weights_only=True,
            map_location=device,
        )
    )
    model.eval()

    true_labels, predicted_labels = collect_predictions(model, val_loader)
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Confusion matrix shape: {cm.shape}")  
    print(f"Total samples: {cm.sum()}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )

    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix - Fashion MNIST", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        "outputs/figures/cnn_val_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    calculate_per_class_accuracy(cm, class_names)
        
    plot_misclassified_examples(
            model=model,
            data_loader=val_loader,
            class_names=class_names,
            num_examples=9,
            save_path="outputs/figures/cnn_val_misclassified_examples.png"
    )
if __name__ == "__main__":
    analyze()

