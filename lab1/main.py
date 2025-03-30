import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optmin

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from MLP import MLP
from Trainer import Trainer
import os

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) #MNIST meand and std
])

trainining_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

# Split training data into train and validation sets (80/20)
train_size = int(0.8 * len(trainining_set))
validation_size = len(trainining_set) - train_size
trainining_set, validation_set = random_split(trainining_set, [train_size, validation_size])

# Create data loaders 
batch_size = 64 
train_loader = DataLoader(trainining_set, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


lr = 0.001
model = MLP(28*28, 512, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optmin.Adam(model.parameters(), lr=0.001)
    

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    use_comet_ml=True

)
    
trainer.train(num_epochs=10)

test_acc = trainer.evaluate()
trainer.plot_curves()

print(f"Final test accuracy: {test_acc:.4f}")