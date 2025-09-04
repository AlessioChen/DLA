
import torch 

from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from utils import fgmattack


class MyCIFAR10(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class MyCIFAR100(Dataset):
    def __init__(self, train=True, ood_set=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR100(root="./data", train=train, download=True, transform=transform)

        if ood_set is None:
            raise ValueError("You must provide a tuple of class names for the OOD set")

        ood_labels = [dataset.class_to_idx[k] for k in ood_set]
        indices = [i for i, label in enumerate(dataset.targets) if label in ood_labels]
        self.subset = Subset(dataset, indices)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]