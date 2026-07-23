from torchvision import datasets , transforms
import torch
from torch.utils.data import DataLoader

train_data = datasets.FashionMNIST(
    root='data',               
    train=True,                
    download=True,          
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',               
    train=False,                
    download=True,          
    transform=transforms.ToTensor()
)

image, label = train_data[0]

train_loader = DataLoader(
    train_data,          
    batch_size=64,       
    shuffle=True         
)
test_loader = DataLoader(
    test_data,          
    batch_size=64,       
    shuffle=False       
)
batch_images, batch_labels = next(iter(train_loader))

flattened_images = batch_images.view(batch_images.size(0), -1)

