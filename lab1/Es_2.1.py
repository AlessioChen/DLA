import argparse
import torch 
import numpy as np
import torch.nn as nn 

from DataManager import DataManager
from Trainer import Trainer
from CNN import CNN, CNN_FeatureExtractor

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def run_baseline(train_loader, test_loader, device):
    """Feature extraction + Linear SVM baseline"""
    checkpoint = torch.load("./checkpoints/residual_cnn_checkpoint.pt", map_location=device)

    feature_model = CNN_FeatureExtractor(
        in_channels=3, num_filters=16, num_blocks=10, skip=True
    )
    
    feature_model.load_state_dict(checkpoint["model_state_dict"])  
    feature_model.to(device)
    feature_model.eval()

    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feat = feature_model(x)
                features.append(feat.cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(features), np.concatenate(labels)

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_clf = SVC(kernel="linear", C=1.0)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    print("Linear SVM accuracy:", accuracy_score(y_test, y_pred))


def run_finetune(train_loader, validation_loader, test_loader, device, freezed_layers, optimizer_name):
    """Fine-tuning the CNN on CIFAR-100"""
    cnn_model = CNN(
        in_channels=3, num_filters=16, num_blocks=10, skip=True, num_classes=100
    )
    checkpoint = torch.load("./checkpoints/residual_cnn_checkpoint.pt", map_location=device)
    pretrained_dict = checkpoint["model_state_dict"]
    
    # Remove classifier weights from the checkpoint before loading 
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
    cnn_model.load_state_dict(pretrained_dict, strict=False)
    cnn_model.to(device)

    # freeze some layers 
    for name, param in cnn_model.named_parameters():
        if name in freezed_layers: 
            parse_args.requires_grad = False 
            
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.01)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD( cnn_model.parameters(), lr=0.01, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=cnn_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        comet_project_name="dla-lab1-es-2-1", 
        use_comet_ml=True,
    )

    trainer.train(num_epochs=20)
    test_acc = trainer.evaluate()
    print("Fine-tuned CNN accuracy on CIFAR-100:", test_acc)

def parse_args(): 
    
    parser = argparse.ArgumentParser(description="Run CNN experiments on CIFAR-100")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "finetune"],
        required=True,
        help="Choose experiment: 'baseline' for feature extractor + SVM, 'finetune' for fine-tuning CNN."
    )
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        required=True,
        help="Optimizer to use for fine-tuning."
    )
    
    parser.add_argument(
        "--freezed_layers",
        type=str, 
        default="layer1,layer2",
        help="Comma-separated list of layers to unfreeze (e.g., 'layer1,layer2')."
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    data_manager = DataManager(
        batch_size=128, 
        val_split=0.2,
        dataset_name="CIFAR100"
    )
    train_loader, validation_loader, test_loader = data_manager.load_data()   

    if args.mode == "baseline":
        run_baseline(train_loader, test_loader, device)
    elif args.mode == "finetune":
        freezed_layers = args.freezed_layers.split(",")
        run_finetune(train_loader, validation_loader, test_loader, device, freezed_layers, args.optimizer)


if __name__ == "__main__":
    main()
