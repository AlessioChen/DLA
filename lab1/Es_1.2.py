import torch 
import time 
import numpy as np
import torch.nn as nn 
import torch.optim as optmin
import matplotlib.pyplot as plt

from MLP import MLP
from Residual import RedidualMLP
from DataManager import DataManager
from Trainer import Trainer
from typing import List, Dict, Any

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

data_manager = DataManager(
    batch_size= 64, 
    val_split=0.2
)

lr = 0.01
train_loader, validation_loader, test_loader = data_manager.load_data()   

def run_experiment(model_type: str, 
                    depths: List[int], 
                    input_size: int = 28*28,
                    hidden_size: int = 512, 
                    output_size: int = 10, 
                    num_epochs: int = 10, 
                    lr: float = 0.01, 
                    momentum: float = 0.9, 
                    weight_decay: float = 0.0005,
                   use_comet: bool = False
                   ) -> List[Dict[str, Any]]:
    

    results: List[Dict[str, any]]  = []

    for depth in depths: 
        print(f"\n{'='*50}")
        print(f"Training {model_type} with depth {depth}")
        print(f"{'='*50}")

        if model_type == "MLP":
            model = MLP(input_size, hidden_size, output_size, depth)
        else: 
            model = RedidualMLP(input_size, hidden_size, output_size, num_blocks=depth - 2 )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optmin.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)

        trainer = Trainer(
            model = model, 
            train_loader=train_loader, 
            validation_loader=validation_loader, 
            test_loader=test_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            use_comet_ml=use_comet, 
            comet_project_name="MLA_lab_es_1_2", 
            depth=depth
        )

        trainer.train(num_epochs)
        test_acc = trainer.evaluate()

        results.append({
            'model_type': model_type,
            'depth': depth,
            'train_loss': trainer.train_losses[-1],
            'val_loss': trainer.validation_losses[-1],
            'train_acc': trainer.train_accuracies[-1],
            'val_acc': trainer.validation_accuracies[-1],
            'test_acc': test_acc,
            'all_train_losses': trainer.train_losses,
            'all_val_losses': trainer.validation_losses,
            'all_train_accs': trainer.train_accuracies,
            'all_val_accs': trainer.validation_accuracies
        })
    

    return results

depths = [2, 4, 8, 16, 32]

mlp_results = run_experiment("MLP", depths, use_comet=True)
res_mlp_results = run_experiment("ResidualMLP", depths, use_comet=True)
