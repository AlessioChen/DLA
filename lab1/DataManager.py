from torchvision.transforms import transforms
from torchvision import datasets
from typing import Tuple
from torch.utils.data import DataLoader, random_split

class DataManager:
    def __init__(self, batch_size: int = 64, val_split: float = 0.2) -> None:
        self.batch_size = batch_size
        self.val_split = val_split
        
        # Define transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Load MNIST dataset
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=self.transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=self.transform)
        
        # Split training data into train and validation sets
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, test_loader
