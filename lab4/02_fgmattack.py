import os
import torch

import matplotlib.pyplot as plt
import random
import numpy as np
import argparse

from torch.utils.data import DataLoader
from Cnn import CNN
from Mydata import MyCIFAR10

from utils import fgmattack


# CIFAR10 class names
CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attack', choices=['targeted', 'untargeted'], default='untargeted')
    parser.add_argument('--seed', default=27)
    parser.add_argument('--target', type=str, default='cat', choices=list(CIFAR10_CLASSES),
                        help="Target class for targeted attack")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples to attack and visualize")
    args = parser.parse_args()
    return args


def plot(args, images_orig, images_adv, preds_orig, preds_adv, budgets, model, device="cuda"):
    """Grid with attacked images evaluation"""
    fig, axs = plt.subplots(args.n_samples, 4, figsize=(15, 12))
    fig.suptitle(f"Attack type: {args.attack}", fontsize=16)

    model.eval()

    for i, (image_orig, image_adv) in enumerate(zip(images_orig, images_adv)):
        image_orig = image_orig.detach().cpu()
        image_adv = image_adv.detach().cpu()
        diff = image_orig - image_adv
        
        # original image
        axs[i, 0].imshow(image_orig.permute(1, 2, 0))
        axs[i, 0].set_title(f"pred: {CIFAR10_CLASSES[preds_orig[i]]}", fontsize=12)
        axs[i, 0].axis("off")

        # attacked image
        axs[i, 1].imshow(torch.clamp(image_adv.permute(1, 2, 0), 0., 1.))
          
        axs[i, 1].set_title(f"pred: {CIFAR10_CLASSES[preds_adv[i]]}", fontsize=12)
        axs[i, 1].axis("off")

       # Perturbation heatmap
        axs[i, 2].imshow(255 * torch.clamp(diff, 0., 1.).squeeze().mean(0), cmap=plt.get_cmap('PuOr'))
        axs[i, 2].set_title(f"Positive Perturbation heatmap", fontsize=12)
        axs[i, 2].axis("off")
    
        # attack magnitude distribution
        axs[i, 3].hist((255 * torch.clamp(diff, 0., 1.)).flatten().numpy(), density=True)
        axs[i, 3].set_xlabel("magnitude")
        axs[i, 3].set_title(f"steps: {budgets[i]}")

    plt.tight_layout()
    os.makedirs("plots/adversarial", exist_ok=True)
    output_path = os.path.join("plots/adversarial", f"{args.attack}.png")
    plt.savefig(output_path)
    print(f"Printed img={output_path}")






def adverarials(args, model, dataset, device='cuda'):
    # pick random samples
    samples = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(args.n_samples)]

    images_orig, preds_orig, images_adv, preds_adv, iters_list = [], [], [], [], []

    model.eval()

    for i, (image, label) in enumerate(samples):
        image, label = image.to(device), torch.tensor(label, device=device)

        # only use target if attack is targeted
        target = None
        if args.attack == "targeted":
            target = torch.tensor(CIFAR10_CLASSES.index(args.target), device=device)

        images_orig.append(image.cpu())
        output = model(image.unsqueeze(0))
        pred = output.argmax()
        preds_orig.append(pred.cpu())

        if pred.item() != label.item():
            print(f"Image {i} classifier is already wrong")
            iters = 0
        elif target is not None and label.item() == target.item():
            print(f"Image {i} target label same as GT")
            iters = 0
        else:
            image, pred, iters = fgmattack(
                label, image, model, 1 / 255, target
            )

        images_adv.append(image.cpu())
        preds_adv.append(pred.item())
        iters_list.append(iters)

    plot(args, images_orig, images_adv, preds_orig, preds_adv, iters_list, model)


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    test_dataset = MyCIFAR10(train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.n_samples, shuffle=False, num_workers=2)

    # load model
    model = CNN().to(device)
    checkpoint_path = "./checkpoints/cnn_cifar10.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # run adversarial attacks
    adverarials(args, model, test_dataset)

    print("Adversarial examples generated and saved successfully.")
