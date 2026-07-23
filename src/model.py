import torch.nn as nn
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(784, 128)
        self.fc2= nn.Linear(128,10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)         
        x = self.relu(x)        
        x = self.fc2(x)          
        return x 
