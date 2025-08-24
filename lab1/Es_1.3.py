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
from CNN import CNN

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

data_manager = DataManager(
    batch_size= 64, 
    val_split=0.2,
    dataset_name="CIFAR10"
)

lr = 0.001
train_loader, validation_loader, test_loader = data_manager.load_data()   

def run_cnn_experiment(model_type: str, 
                    depths: List[int], 
                    in_channles: int = 3,
                    num_filters: int = 16, 
                    num_classes: int = 10, 
                    num_epochs: int = 20, 
                    lr: float = 0.001, 
                    use_comet: bool = False
                   ) -> List[Dict[str, Any]]:
    

    results: List[Dict[str, any]]  = []

    for depth in depths: 
        print(f"\n{'='*50}")
        print(f"Training {model_type} CNN with depth {depth}")
        print(f"{'='*50}")
        
        skip = True if model_type == "ResiaulCNN" else False 
        model = CNN(
            in_channels=in_channles,
            num_filters=num_filters,
            num_blocks=depth, 
            skip=skip,
            num_classes=num_classes
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optmin.SGD(model.parameters(), lr = lr)

        trainer = Trainer(
            model = model, 
            train_loader=train_loader, 
            validation_loader=validation_loader, 
            test_loader=test_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            use_comet_ml=use_comet, 
            comet_project_name="MLA_lab_es_1_3", 
            depth=depth
        )

        trainer.train(num_epochs)
        test_acc = trainer.evaluate()

        results.append({
            'model_type': model_type,
            'depth': depth,
            'train_loss': trainer.train_losses[-1],
            'val_loss': trainer.validation_losses[-1],
            'train_acc': trainer.train_accurracies[-1],
            'val_acc': trainer.validation_accurracies[-1],
            'test_acc': test_acc,
            'all_train_losses': trainer.train_losses,
            'all_val_losses': trainer.validation_losses,
            'all_train_accs': trainer.train_accurracies,
            'all_val_accs': trainer.validation_accurracies
        })
    

    return results

depths = [1, 2, 3,]

cnn_results = run_cnn_experiment("CNN", depths, use_comet=True)
res_cnn_results = run_cnn_experiment("ResidualCNN", depths, use_comet=True)
