import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None :
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x 
        
        out = self.linear1(x)
        out = self.activation(x)
        out = self.linear2(x)
        out += identity # skip connectiop
        out = self.activation(out)
        return out 

class RedidualMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_blocks: int = 1) -> None: 
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if len(x.shape) > 2: 
            x = x.view(x.size(0), -1) 
        
        x = self.input_layer(x)
        x = self.activation(x)

        for block in self.residual_blocks: 
            x = block(x)
        
        x = self.output_layer(x)
        return x

