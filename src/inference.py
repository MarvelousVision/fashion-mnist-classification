import torch

from src.model import FashionMNISTCNN
from src.data import test_loader 

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionMNISTCNN().to(device)
    state_dict = torch.load(
            "outputs/models/best_cnn_baseline.pth",
            weights_only=True,
            map_location=device,
        )
    model.load_state_dict(state_dict)
    model.eval()
    return model , device


def predict(model, image, device):
    image = image.to(device)
    image = image.unsqueeze(0)
  
    with torch.no_grad():
        logits = model(image)
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1)
        confidence = probabilities[0, predicted_class].item()
    return CLASS_NAMES[predicted_class], confidence

if __name__ == "__main__":
    model, device = load_model()
    image, true_label = test_loader.dataset[0]
    true_class = CLASS_NAMES[true_label]
    predicted_class, confidence = predict(model, image, device)
    print(f"True class:      {true_class}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence:      {confidence:.2%}")