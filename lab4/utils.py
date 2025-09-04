import torch 
import torch.nn as nn 


def fgmattack(gt, image, model, eps, target=None, max_step=50):
    """Fast Gradient Method Attack (iterative until success)."""
    n = 0
    done = False
    loss_fn = nn.CrossEntropyLoss()

    image = image.clone().detach().requires_grad_(True)

    while not done and n < max_step:
        output = model(image.unsqueeze(0))  # [1, K]

        if target is None:
            yt = gt.unsqueeze(0)
        else:
            yt = target.unsqueeze(0)

        model.zero_grad()
        loss = loss_fn(output, yt)
        loss.backward()

        if target is None:
            # untargeted FGSM
            image = image + eps * torch.sign(image.grad)
        else:
            # targeted FGSM
            image = image - eps * torch.sign(image.grad)

        # prepare for next iteration
        image = image.detach().clone().requires_grad_(True)

        output = model(image.unsqueeze(0))
        pred = output.cpu().argmax()
        n += 1

        if target is None and pred != gt:
            done = True
        if target is not None and pred == target:
            done = True

    return image, pred, n
