# MNIST Classification with MLPs and Residual Connections

This project explores training Multilayer Perceptrons (MLPs) from scratch in PyTorch to classify handwritten digits from the [MNIST dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).
We start with a baseline MLP (Exercise 1.1) and extend it by introducing Residual Connections (Exercise 1.2), comparing their performance across increasing network depths.

# Project Structure

`MLP.py` – Baseline Multilayer Perceptron architecture

`Residual.py` – Residual MLP blocks & model

`DataManager.py` – Loads MNIST dataset and handles train/val/test splits

`Trainer.py` – Training, evaluation loop, and logging utilities

`Es_1.1.py` – Runs the baseline MLP (Exercise 1.1)

`Es_1.2.py` – Runs MLP vs ResidualMLP experiments (Exercise 1.2)`

## 1.1  Baseline MLP

It's a simple 2 layer MLP trained on MIST. 

### Implementation details 
| Component            | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Input**            | 28 × 28 pixel images, flattened into a 784-dimensional vector |
| **Hidden Layer**     | Fully connected layer with 512 units                          |
| **Output**           | 10 classes (digits 0–9)                                       |
| **Loss Function**    | CrossEntropyLoss                                              |
| **Optimizer**        | Stochastic Gradient Descent (SGD) with learning rate = 0.001  |
| **Batch Size**       | 64                                                            |
| **Train/Val Split**  | 80% training, 20% validation                                  |
| **Epochs (default)** | 10                                                            |


### How to run 

```
python Es_1.1.py
```

By default, it trains for 10 epochs. 

### Result (10 epochs)

| Metric         | Training Set | Validation Set   |
| -------------- | ------------ | ---------------- |
| **Loss**       | 0.3329       | 0.3450           |
| **Accuracy**   | 90.71%       | 90.30%           |

## 1.2 Adding Residual Connections
This exercise extends the baseline MLP by introducing [Residual Connections](https://arxiv.org/abs/1512.03385).
The goal is to study whether residual connections make training deeper networks easier and improve performance compared to plain MLPs.



### Implementation details 
| Component         | Description                                                                   |
| ----------------- | ----------------------------------------------------------------------------- |
| **Models**        | `MLP`, `ResidualMLP`                                                          |
| **Depths Tested** | \[2, 4, 8, 16, 32]                                                            |
| **Loss Function** | CrossEntropyLoss                                                              |
| **Optimizer**     | SGD with learning rate = 0.001                                                |
| **Batch Size**    | 64                                                                            |
| **Epochs**        | 10                                                                            |
| **Dataset**       | MNIST (80% train, 20% validation, test split included)                        |
| **Logging**       | [Comet ML](https://www.comet.com/alessiochen/mla-lab-es-1-2/view/new/panels)


### How to run 

```
python Es_1.2.py
```

### Results 
| Model           | Depth | Val Accuracy |
| --------------- | ----- | -------- |
| **MLP**         | 2     | 90.975%  |
| **ResidualMLP** | 2     | 90.733%  |
| **MLP**         | 4     | 89.275%  |
| **ResidualMLP** | 4     | 92.658%  |
| **MLP**         | 8     | 11.45%   |
| **ResidualMLP** | 8     | 94.983%  |
| **MLP**         | 16    | 11.45%   |
| **ResidualMLP** | 16    | 96.633%  |
| **MLP**         | 32    | 11.45%   |
| **ResidualMLP** | 32    | 96.733%  |

From the results we can see that adding residual connections significantly improves deep MLP performance. Vanilla MLPs suffer from vanishing gradients as depth increases, causing training to fail for networks with 8+ layers. Residual connections provide shortcut paths that let gradients bypass intermediate layers, enabling effective training and high accuracy even in very deep networks.