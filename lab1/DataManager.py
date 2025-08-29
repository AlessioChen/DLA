from torchvision.transforms import transforms
from torchvision import datasets
from typing import Tuple
from torch.utils.data import DataLoader, random_split

class DataManager:
    def __init__(self, batch_size: int = 64, val_split: float = 0.2, dataset_name='MNIST') -> None:
        self.batch_size = batch_size
        self.val_split = val_split
        self.dataset_name = dataset_name
            
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if self.dataset_name == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR10':
            
            mean = (0.4914, 0.4822, 0.4465)
            std  = (0.247, 0.243, 0.261)

            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        elif self.dataset_name == "CIFAR100":
            
            mean = (0.5071, 0.4865, 0.4409)
            std = (0.2673, 0.2564, 0.2761)
            
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            
            train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
            
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
        
        # Split training data into train and validation sets
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        
        return train_loader, val_loader, test_loader