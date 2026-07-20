from torchvision import datasets , transforms

train_data = datasets.FashionMNIST(
    root='data',               
    train=True,                
    download=True,          
    transform=transforms.ToTensor()
)