from comet_ml import start

from torch.utils.data import DataLoader
from typing import List, Tuple
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm 
from dotenv import load_dotenv

import torch 
import torch.nn as nn
import torch.optim as optmin
import matplotlib.pyplot as plt
import os


load_dotenv()

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 validation_loader: DataLoader, 
                 test_loader: DataLoader, 
                 criterion: nn.Module, 
                 optimizer: optmin.Optimizer, 
                 device: torch.device, 
                 use_comet_ml: bool = False, 
                 comet_project_name: str = 'general', 
                 comet_workspace: str = 'alessiochen', 
                 checkpoint_dir: str = './checkpoints',
                 save_checkpoint: bool = False,
                 depth: int = 2):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.use_comet_ml = use_comet_ml

        self.save_checkpoint = save_checkpoint
        self.checkpoint_dir = checkpoint_dir

        self.train_losses: List[float] = []
        self.validation_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.validation_accuracies: List[float] = []
        self.best_validation_accuracy = 0.0

        if self.save_checkpoint:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if use_comet_ml:
            self.experiment = start(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=comet_project_name,
                workspace=comet_workspace   
            )
            total_params = sum(p.numel() for p in model.parameters())
            self.experiment.log_parameters({
                "model_type": model.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_size": getattr(train_loader, 'batch_size', "unknown"),
                "device": str(device),
                "total_parameters": total_params
            })
            
            self.experiment.set_name(f"{model.__class__.__name__}_{depth}_layers")
            self.experiment.set_model_graph(str(model), overwrite=True)

    def train_epoch(self, epoch: int, num_epochs: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0

        process_bar = tqdm(enumerate(self.train_loader), desc=f"Training {epoch}/{num_epochs}")
        for i, (inputs, labels) in process_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

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
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.validation_loader, desc="Validating"):
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
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch, num_epochs)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validation
            val_loss, val_acc = self.validate_epoch()
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_acc)

            test_acc = self.evaluate()

            if self.use_comet_ml:
                self.experiment.log_metrics({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc
                }, step=epoch)

            print(f'Epoch {epoch}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Checkpoint
            if self.save_checkpoint and val_acc > self.best_validation_accuracy:
                self.best_validation_accuracy = val_acc
                checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        if self.use_comet_ml:
            self.experiment.end()

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')
        return test_acc

    def plot_curves(self) -> Tuple[Figure, Tuple[Axes, Axes]]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.validation_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.validation_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.show()
        return fig, (ax1, ax2)
