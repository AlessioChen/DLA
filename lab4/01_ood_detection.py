import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, average_precision_score

from Cnn import CNN
from Mydata import MyCIFAR10, MyCIFAR100

OOD_OBJECTS = ("clock", "keyboard", "telephone", "lamp", "wardrobe")
OOD_FLOWERS = ("orchid", "poppy", "rose", "sunflower", "tulip")
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(dataset, batch_size=128, num_workers=2, seed=42):
    set_seeds(seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, generator=generator, worker_init_fn=seed_worker)

def get_loaders(ood_set="objects", batch_size=128):
    id_dataset = MyCIFAR10(train=False)
    id_loader = make_loader(id_dataset, batch_size=batch_size)

    if ood_set == "objects":
        ood_dataset = MyCIFAR100(train=True, ood_set=OOD_OBJECTS)
    elif ood_set == "flowers":
        ood_dataset = MyCIFAR100(train=True, ood_set=OOD_FLOWERS)
    elif ood_set == "noise":
        ood_dataset = datasets.FakeData(2500, (3, 32, 32), transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown OOD set {ood_set}")

    ood_loader = make_loader(ood_dataset, batch_size=batch_size)
    return id_loader, ood_loader

def max_logit(logits):
    return logits.max(dim=1)[0]

def max_softmax(logits, T=1.0):
    probs = F.softmax(logits / T, dim=1)
    return probs.max(dim=1)[0]

def compute_scores(loader, model, score_fn, device):
    scores = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            output = model(x)
            scores.append(score_fn(output))
    return torch.cat(scores).cpu()

def save_image_grid(loader, path, nrow=8):
    os.makedirs(path, exist_ok=True)
    
    imgs, _ = next(iter(loader))
    grid = make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Images saved to {path}")

def plot_scores(id_scores, ood_scores, ood_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Sorted scores
    axes[0].plot(sorted(id_scores), label="ID (CIFAR-10)")
    axes[0].plot(sorted(ood_scores), label=f"OOD ({ood_name})")
    axes[0].set_xlabel("Ordered samples")
    axes[0].set_ylabel("Scores")
    axes[0].set_title("Sorted OOD Scores")
    axes[0].legend()

    # Right: Histogram
    axes[1].hist(id_scores, bins=25, density=True, alpha=0.5, label="ID (CIFAR-10)")
    axes[1].hist(ood_scores, bins=25, density=True, alpha=0.5, label=f"OOD ({ood_name})")
    axes[1].set_xlabel("Scores")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution (Max Softmax)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()
    print(f"Scores plot saved to {save_dir}")

def plot_roc_pr(y_true, scores_dict, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curves
    for label, y_scores in scores_dict.items():
        RocCurveDisplay.from_predictions(y_true, y_scores, ax=axes[0], name=label)
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # PR curves
    for label, y_scores in scores_dict.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_scores, ax=axes[1], name=label)
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC + PR plot saved to {save_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ood_set", type=str, default="noise",
                        choices=["objects", "flowers", "noise"],
                        help="Which OOD dataset to use")
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def compute_accuracy(loader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_base = CNN().to(device)
    checkpoint_base = torch.load("checkpoints/cnn_cifar10.pth", map_location=device)
    model_base.load_state_dict(checkpoint_base, strict=False)

    model_avd = CNN().to(device)
    checkpoint_avd = torch.load("checkpoints/cnn_cifar10_adv.pth", map_location=device)
    model_avd.load_state_dict(checkpoint_avd, strict=False)

    # --- Load datasets ---
    id_loader, ood_loader = get_loaders(args.ood_set, batch_size=args.batch_size)

    # --- Compute scores ---
    id_scores_base = compute_scores(id_loader, model_base, max_softmax, device)
    ood_scores_base = compute_scores(ood_loader, model_base, max_softmax, device)

    id_scores_avd = compute_scores(id_loader, model_avd, max_softmax, device)
    ood_scores_avd = compute_scores(ood_loader, model_avd, max_softmax, device)
    
    plot_scores(
        id_scores_base, 
        ood_scores_base,
        args.ood_set, 
        os.path.join(PLOTS_DIR, args.ood_set, f"scores_max_softmax_{args.ood_set}.png")
    )
    
    plot_scores(
        id_scores_avd, 
        ood_scores_avd,
        args.ood_set, 
        os.path.join(PLOTS_DIR, args.ood_set, f"scores_max_softmax_{args.ood_set}_finetuned.png")
    )

  


    # --- Metrics ---
    y_true = np.concatenate([np.ones(len(id_scores_base)), np.zeros(len(ood_scores_base))])
    y_scores_base = np.concatenate([id_scores_base.numpy(), ood_scores_base.numpy()])
    y_scores_avd = np.concatenate([id_scores_avd.numpy(), ood_scores_avd.numpy()])

    roc_auc_base = roc_auc_score(y_true, y_scores_base)
    pr_auc_base = average_precision_score(y_true, y_scores_base)
    roc_auc_avd = roc_auc_score(y_true, y_scores_avd)
    pr_auc_avd = average_precision_score(y_true, y_scores_avd)

    print(f"Baseline ROC AUC: {roc_auc_base:.4f}, PR AUC: {pr_auc_base:.4f}")
    print(f"Finetuned ROC AUC: {roc_auc_avd:.4f}, PR AUC: {pr_auc_avd:.4f}")

    scores_dict = {
        "Baseline": y_scores_base,
        "Fine-tuned": y_scores_avd
    }
    plot_roc_pr(y_true, scores_dict,
                os.path.join(PLOTS_DIR, args.ood_set, "roc_pr_max_softmax_compare.png"))