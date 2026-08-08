import torch
import torch.nn as nn

from src.model import FashionMNISTCNN
from src.data import test_loader
from src.evaluate import evaluate

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

def test():
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTCNN().to(device)

    state_dict = torch.load(
        "outputs/models/best_cnn_baseline.pth",
        weights_only=True,
        map_location=device,
    )

    model.load_state_dict(state_dict)

    model.eval()

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    test()