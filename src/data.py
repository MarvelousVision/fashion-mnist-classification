from torchvision import datasets , transforms
import torch
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 64
SEED = 42
TRAIN_SIZE = 54_000
VAL_SIZE = 6_000

transform = transforms.ToTensor()

full_train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [TRAIN_SIZE, VAL_SIZE],
    generator=split_generator,
)

train_loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=train_loader_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)