import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optmin

from MLP import MLP
from DataManager import DataManager
from Trainer import Trainer

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

data_manager = DataManager(
    batch_size= 64, 
    val_split=0.2
)

lr = 0.001
model = MLP(28*28, 512, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optmin.Adam(model.parameters(), lr=lr)
train_loader, validation_loader, test_loader = data_manager.load_data()   

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