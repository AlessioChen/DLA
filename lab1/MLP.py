import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1) # Flatten the input 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x