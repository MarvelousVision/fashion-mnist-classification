import torch
import os
import matplotlib.pyplot as plt
def evaluate(model, data_loader, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
#            flattened_images = images.view(images.size(0), -1)
#            logits = model(flattened_images)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    average_loss = total_loss / num_batches
    return average_loss, accuracy

def collect_predictions(model, data_loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
#            flattened_images = images.view(images.size(0), -1)
 #           logits = model(flattened_images)
            logits = model(images)
            predicted = torch.argmax(logits, dim=1)
            true_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(predicted.cpu().tolist())
    return true_labels, predicted_labels



def plot_misclassified_examples(
    model,
    data_loader,
    class_names,
    num_examples=9,
    save_path="outputs/figures/misclassified_examples.png",
):
    model.eval()
    device = next(model.parameters()).device
    mistakes = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            flattened_images = images.view(images.size(0), -1)
            logits = model(flattened_images)
            predictions = torch.argmax(logits, dim=1)

            incorrect_indices = torch.where(predictions != labels)[0]

            for index in incorrect_indices:
                mistakes.append(
                    (
                        images[index].cpu(),
                        labels[index].item(),
                        predictions[index].item(),
                    )
                )

                if len(mistakes) == num_examples:
                    break
            if len(mistakes) == num_examples:
                break

    if not mistakes:
        print("No misclassified examples found.")
        return

    cols = 3
    rows = (len(mistakes) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
    axes = axes.flatten()

    for axis, (image, true_label, predicted_label) in zip(axes, mistakes):
        axis.imshow(image.squeeze().numpy(), cmap="gray")
        axis.set_title(
            f"True: {class_names[true_label]}\n"
            f"Predicted: {class_names[predicted_label]}"
        )
        axis.axis("off")

    for axis in axes[len(mistakes):]:
        axis.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"Saved misclassified examples to: {save_path}")

def calculate_per_class_accuracy(cm, class_names):
    class_accuracies = []

    for i, class_name in enumerate(class_names):
        correct = cm[i, i]
        total = cm[i].sum()
        accuracy = 100 * correct / total

        class_accuracies.append(accuracy)
        print(f"{class_name}: {accuracy:.2f}%")

    return class_accuracies
