from torchvision import datasets , transforms
import torch
from torch.utils.data import DataLoader, random_split

full_train_data = datasets.FashionMNIST(
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

train_size = 54000
val_size = 6000  


generator = torch.Generator().manual_seed(42)
train_data, val_data = random_split(
    full_train_data,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False 
)

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
