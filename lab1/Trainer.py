import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optmin
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import time

class Trainer: 
    def __init__(self, 
                 model: nn.modules, 
                 train_loader: DataLoader, 
                 validation_loader: DataLoader, 
                 test_loader: DataLoader, 
                 criterion: nn.Module, 
                 optimizer: optmin.Optimizer, 
                 device: torch.device):
    
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device 

        self.train_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.train_accurracies: List[float] = []
        self.validation_accurracies: List[float] = []
        self.best_validation_accuracy = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0 
        correct = 0 
        total = 0 

        for inputs, labels in self.train_loader: 
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(inputs)    
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # prediction
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total 
        epoch_accuracy = correct / total 
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total= 0
        
        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        return val_epoch_loss, val_epoch_acc

    def train(self, num_epochs: int = 10) -> None: 

        for epoch in range(1 + num_epochs):
            _start_time = time.time()
            # training phase
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accurracies.append(train_acc)

            # validation phase 
            val_loss, val_acc = self.validate_epoch()

            self.validation_losses.append(val_loss)
            self.validation_accurracies.append(val_acc)

            _epoch_time = time.time() - _start_time
            print(f'Epoch {epoch}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}', 
                  f"Epoch time: {_epoch_time:.2f} sec")
        

    def evaluate(self) -> float:
        self.model.eval()

        correct = 0
        total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')
        
        return test_acc

    def plot_curves(self) -> Tuple[Figure, Tuple[Axes, Axes]]:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.validation_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(self.train_accurracies, label='Train Accuracy')
        ax2.plot(self.validation_accurracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        return fig, (ax1, ax2)