import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()

        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))


        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if len(x.shape) > 2: 
            x = x.view(x.size(0), -1) # Flatten the input 
        
        x = self.input_layer(x)
        x = self.activation(x)

        for layer in self.hidden_layers: 
            x = layer(x)
            x = self.activation(x)
        
        x = self.output_layer(x)
        return x