import torch
import torch.nn as nn

from src.model import FashionMNISTCNN
from src.data import test_loader
from src.evaluate import evaluate, collect_predictions, calculate_per_class_accuracy
from sklearn.metrics import confusion_matrix

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

criterion = nn.CrossEntropyLoss()
model = FashionMNISTCNN()

state_dict = torch.load(
    "outputs/models/best_cnn_baseline.pth",
    weights_only=True
)

model.load_state_dict(state_dict)

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion
)

print(f"Final test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2f}%")

true_labels, predicted_labels = collect_predictions(model, test_loader)
cm = confusion_matrix(true_labels, predicted_labels)
class_accuracies = calculate_per_class_accuracy(cm, class_names)