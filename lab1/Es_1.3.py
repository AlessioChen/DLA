import torch 
import numpy as np
import torch.nn as nn 
import torch.optim as optmin
import argparse

from DataManager import DataManager
from Trainer import Trainer
from CNN import CNN


def run_cnn_experiment(model_type: str, 
                       num_blocks: int, 
                       in_channels: int = 3,
                       num_filters: int = 16, 
                       num_classes: int = 10, 
                       num_epochs: int = 20, 
                       lr: float = 0.01, 
                       momentum: float = 0.9, 
                       weight_decay: float = 0.0005,
                       use_comet: bool = False,
                       save_checkpoint: bool = False,
                       device: str = "cpu",
                       train_loader=None,
                       validation_loader=None,
                       test_loader=None):
    
    print(f"\n{'='*50}")
    print(f"Training {model_type} with {num_blocks} blocks")
    print(f"{'='*50}")
        
    skip = True if model_type == "ResidualCNN" else False 
    model = CNN(
        in_channels=in_channels,
        num_filters=num_filters,
        num_blocks=num_blocks, 
        skip=skip,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optmin.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        validation_loader=validation_loader, 
        test_loader=test_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device=device, 
        use_comet_ml=use_comet, 
        comet_project_name="MLA_lab_es_1_3", 
        depth=num_blocks,
        save_checkpoint=save_checkpoint
    )

    trainer.train(num_epochs)
    test_acc = trainer.evaluate()
    return test_acc

def parse_args(): 
    
    parser = argparse.ArgumentParser(description="Run CNN/ResidualCNN experiment")
    parser.add_argument(
        "--num_blocks", 
        type=int, 
        required=True, 
        help="Number of blocks in the CNN/ResidualCNN model"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="CNN", 
        choices=["CNN", "ResidualCNN"], 
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        help="Learning rate"
    )
    parser.add_argument(
        "--momentum", 
        type=float, 
        default=0.9, 
        help="SGD momentum"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0005, 
        help="Weight decay"
    )
    parser.add_argument(
        "--use_comet", 
        action="store_true", 
        help="Enable Comet logging"
    )
    parser.add_argument(
        "--save_checkpoint", 
        action="store_true", 
        help="Save model checkpoints"
    )
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    data_manager = DataManager(
        batch_size=128, 
        val_split=0.2,
        dataset_name="CIFAR10"
    )

    train_loader, validation_loader, test_loader = data_manager.load_data()   

    args = parse_args()
    
    run_cnn_experiment(
        model_type=args.model_type,
        num_blocks=args.num_blocks,
        num_epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        use_comet=args.use_comet,
        save_checkpoint=args.save_checkpoint,
        device=device,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader
    )
