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


def run_finetune(train_loader, validation_loader, test_loader, device):
    """Fine-tuning the CNN on CIFAR-100"""
    cnn_model = CNN(
        in_channels=3, num_filters=16, num_blocks=10, skip=True, num_classes=100
    )
    checkpoint = torch.load("./checkpoints/residual_cnn_checkpoint.pt", map_location=device)
    pretrained_dict = checkpoint["model_state_dict"]
    
    # Remove classifier weights from the checkpoint
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}
    
    # Load remaining weights
    cnn_model.load_state_dict(pretrained_dict, strict=False)
    cnn_model.to(device)

    # Freeze all layers except layer2 + classifier
    for name, param in cnn_model.named_parameters():
        if "layer2" not in name and "classifier" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.01)

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
        run_finetune(train_loader, validation_loader, test_loader, device)


if __name__ == "__main__":
    main()
