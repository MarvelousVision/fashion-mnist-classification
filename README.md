# Fashion-MNIST Classification with PyTorch

This project implements and compares two neural network approaches for classifying Fashion-MNIST images:

- a fully connected neural network baseline
- a convolutional neural network (CNN)

The goal was not only to achieve good classification performance, but also to build a clean and reproducible PyTorch pipeline with proper train/validation/test separation, checkpointing, model comparison, error analysis, and inference.

## Project Overview

Fashion-MNIST is a dataset of 28×28 grayscale clothing images across 10 classes.

The project follows this workflow:

data loading  
→ train / validation / test split  
→ baseline model  
→ CNN model  
→ validation-based model selection  
→ error analysis  
→ controlled regularization experiment  
→ final test evaluation  
→ inference

## Dataset

The project uses the Fashion-MNIST dataset from `torchvision`.

Dataset split:

- Training: 54,000 images
- Validation: 6,000 images
- Test: 10,000 images

Each image has shape:

`[1, 28, 28]`

where:

- `1` is the grayscale channel
- `28 × 28` is the image resolution

The dataset contains 10 classes:

0. T-shirt/top  
1. Trouser  
2. Pullover  
3. Dress  
4. Coat  
5. Sandal  
6. Shirt  
7. Sneaker  
8. Bag  
9. Ankle boot  

The training loader uses shuffling, while validation and test loaders do not.

A fixed random seed is used for reproducibility.

## Project Structure

```text
fashion-mnist-classification/
│
├── data/
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── analysis.py
│   ├── test.py
│   └── inference.py
│
├── .gitignore
├── requirements.txt
└── README.md

## Models

### Fully Connected Baseline

The baseline model uses a simple fully connected architecture.

Input images are flattened from:

`[1, 28, 28]`

to:

`[784]`

Architecture:

```text
784
→ Linear(784, 128)
→ ReLU
→ Linear(128, 10)
```

The model outputs raw logits for 10 classes.

The baseline achieved approximately:

`87–88% validation accuracy`

The main limitation of the baseline is that flattening removes the explicit spatial structure of the image.

## CNN Model

The CNN preserves the 2D structure of the input image and learns local visual patterns.

Architecture:

```text
Input: [1, 28, 28]

Conv2d(1 → 16, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2d(2)

Conv2d(16 → 32, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2d(2)

Flatten

Linear(32 × 7 × 7, 128)
→ ReLU
→ Linear(128, 10)
```

Tensor shapes:

```text
Input:         [batch, 1, 28, 28]
After conv1:   [batch, 16, 28, 28]
After pool1:   [batch, 16, 14, 14]
After conv2:   [batch, 32, 14, 14]
After pool2:   [batch, 32, 7, 7]
After flatten: [batch, 1568]
Output:        [batch, 10]
```

The final layer returns raw logits.

No Softmax is applied during training because `CrossEntropyLoss` expects raw logits.

## Training Configuration

Final selected configuration:

```text
Optimizer:      Adam
Learning rate:  0.001
Weight decay:   0
Epochs:         5
Batch size:     64
Random seed:    42
Loss function:  CrossEntropyLoss
```

The best model checkpoint is selected using the lowest validation loss.

Checkpoint:

`outputs/models/best_cnn_baseline.pth`

## Validation Results

The final CNN achieved:

```text
Best validation loss:     0.2583
Best validation accuracy: 90.40%
```

This clearly outperformed the fully connected baseline.

## Regularization Experiment

A controlled experiment was performed using Adam weight decay.

All other conditions were kept fixed:

* same architecture
* same train/validation split
* same random seed
* same learning rate
* same number of epochs
* same batch size

Comparison:

| Model                       | Validation Loss | Validation Accuracy |
| --------------------------- | --------------: | ------------------: |
| CNN baseline                |          0.2583 |              90.40% |
| CNN + weight decay (`1e-4`) |          0.2665 |              89.88% |

The weight-decay version performed slightly worse.

Therefore, the simpler CNN without weight decay was selected as the final model.

## Final Test Results

After model selection was complete, the test set was evaluated once.

Final test performance:

```text
Test loss:     0.2731
Test accuracy: 89.85%
```

The difference between validation and test accuracy was small:

```text
Validation accuracy: 90.40%
Test accuracy:       89.85%
Gap:                 0.55 percentage points
```

This indicates that the selected model generalized well to unseen data.

## Per-Class Performance

Final test per-class accuracy:

| Class       | Accuracy |
| ----------- | -------: |
| T-shirt/top |   86.10% |
| Trouser     |   97.90% |
| Pullover    |   85.00% |
| Dress       |   93.00% |
| Coat        |   82.10% |
| Sandal      |   97.60% |
| Shirt       |   67.90% |
| Sneaker     |   93.10% |
| Bag         |   97.80% |
| Ankle boot  |   98.00% |

The strongest classes were:

* Ankle boot
* Trouser
* Bag
* Sandal

The weakest class was:

* Shirt

The main remaining classification errors occurred between visually similar upper-body classes such as:

* Shirt
* Pullover
* Coat
* T-shirt/top

## Error Analysis

The project includes:

* confusion matrix
* per-class accuracy
* misclassified examples
* validation error analysis

These results are saved under:

`outputs/figures/`

Error analysis showed that the model performs very well on visually distinctive classes, while visually similar clothing categories remain more difficult.

This also explains why the CNN significantly outperformed the fully connected baseline: convolutional layers preserve and exploit spatial structure.

## Learning Curves

Training and validation curves are saved under:

```text
outputs/figures/cnn_loss_curve.png
outputs/figures/cnn_val_accuracy_curve.png
```

The validation loss decreased during training and validation accuracy improved steadily, with no severe overfitting observed.

## Inference

The project includes a simple inference pipeline in:

`src/inference.py`

The inference flow is:

```text
image
→ add batch dimension
→ model
→ logits
→ Softmax probabilities
→ predicted class
→ confidence
```

Example output:

```text
True class:      Ankle boot
Predicted class: Ankle boot
Confidence:      99.79%
```

Softmax is used during inference to convert logits into probabilities.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the CNN:

```bash
python -m src.train
```

Run validation error analysis:

```bash
python -m src.analysis
```

Run final test evaluation:

```bash
python -m src.test
```

Run inference:

```bash
python -m src.inference
```

## Requirements

Main dependencies:

```text
torch==2.13.0
torchvision==0.28.0
matplotlib==3.10.7
scikit-learn==1.7.2
seaborn==0.13.2
```

## Key PyTorch Concepts Demonstrated

This project demonstrates practical usage of:

* `torch.Tensor`
* `Dataset`
* `DataLoader`
* `random_split`
* `nn.Module`
* `nn.Linear`
* `nn.Conv2d`
* `nn.ReLU`
* `nn.MaxPool2d`
* `CrossEntropyLoss`
* Adam optimizer
* `loss.backward()`
* `optimizer.step()`
* `optimizer.zero_grad()`
* `model.train()`
* `model.eval()`
* `torch.no_grad()`
* checkpointing with `state_dict()`
* validation-based model selection
* reproducible training
* confusion matrices
* per-class accuracy
* inference with Softmax probabilities

## Main Takeaways

The fully connected baseline provided a useful reference point, but the CNN achieved better performance because it could learn local spatial patterns directly from the images.

A controlled weight-decay experiment showed that adding regularization did not improve validation performance, so the simpler baseline CNN was selected.

The final model achieved:

`89.85% test accuracy`

The main limitation remains distinguishing visually similar upper-body clothing categories, especially `Shirt`.

## Future Improvements

Possible future improvements include:

* deeper CNN architectures
* Batch Normalization
* Dropout
* data augmentation
* learning-rate scheduling
* more extensive hyperparameter tuning
* transfer learning on a more complex image dataset
* deployment through an API

The current version is intentionally kept compact and focused on demonstrating a clean end-to-end PyTorch classification workflow.

```
```
