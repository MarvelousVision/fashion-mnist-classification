import torch.nn as nn
import torch

class FashionMNISTBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2= nn.Linear(128,10)

    def forward(self, x):
        x = self.fc1(x)         
        x = self.relu(x)        
        x = self.fc2(x)          

        return x 

class FashionMNISTCNN(nn.Module):
   def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
   def forward(self, x):
        x = self.conv1(x)        
        x = self.relu(x)
        x = self.pool(x)          
        
        x = self.conv2(x)        
        x = self.relu(x)
        x = self.pool(x)           

        x = torch.flatten(x, start_dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)            
        return x