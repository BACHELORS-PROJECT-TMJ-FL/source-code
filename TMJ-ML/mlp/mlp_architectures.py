from torch import nn
import torch

class Net3layer(nn.Module):   

    def __init__(self):
        super(Net3layer, self).__init__()
        self.fc1 = nn.Linear(49, 40)
        self.fc2 = nn.Linear(40, 24)
        self.fc3 = nn.Linear(24, 6)
        self.fc4 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x) 
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
    
        return x
    
class Net2layer(nn.Module):   

    def __init__(self):
        super(Net2layer, self).__init__()
        self.fc1 = nn.Linear(49, 40)
        self.fc2 = nn.Linear(40, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x) 
        x = torch.sigmoid(x)
    
        return x
class Net1layer(nn.Module):   

    def __init__(self):
        super(Net1layer, self).__init__()
        self.fc1 = nn.Linear(49, 40)
        self.fc2 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
    
        return x
    
class Net1largelayer(nn.Module):   

    def __init__(self):
        super(Net1largelayer, self).__init__()
        self.fc1 = nn.Linear(49, 80)
        self.fc2 = nn.Linear(80, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
    
        return x