import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import numpy as np
from tqdm import tqdm  

from torch.utils.data import DataLoader
from Cnn import CNN
from Mydata import MyCIFAR10
from utils import fgmattack

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42)
    parser.add_argument("--adv_train", action="store_true",
                    help="If set, use adversarially-augmented training")
    args = parser.parse_args()
    return args


def adversal_augmentation(model, images, labels, fraction=0.2):
    """
    Randomly attack a fraction of images in the batch.
    """
    batch_size = images.size(0)
    num_samples = max(1, int(fraction * batch_size))
    attack_indices = random.sample(range(batch_size), num_samples)
    
    for i in attack_indices:
        image = images[i].clone()  
        image.requires_grad = True 
        image, _, _ = fgmattack(labels[i], image, model, 1/255)
        image = image.detach()  
        images[i] = image
    
    return images

if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = MyCIFAR10(train=True)
    model = CNN().to(device)
    checkpoint_path = "checkpoints/cnn_cifar10.pth"
    
    if args.adv_train:      
        checkpoint_path = "checkpoints/cnn_cifar10_adv.pth"
    
    lr = 0.001
    batch_size = 128
    epochs = 20
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0

        # Wrap DataLoader with tqdm for batch progress
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100):
            images, labels = images.to(device), labels.to(device)

            if args.adv_train:
                images = adversal_augmentation(model, images, labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")
