# LAB1: MLPs, Residual Networks, and Transfer Learning on MNIST & CIFAR

This project explores the design, training, and evaluation of deep learning architectures specifically Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs)—on image classification tasks using MNIST and CIFAR datasets. The study focuses on understanding how network depth and residual connections impact training stability and performance.

Key components of the project include:

- Building and training baseline MLPs on MNIST.
- Introducing Residual MLPs to overcome vanishing gradient issues in deeper networks.
- Comparing plain CNNs vs Residual CNNs on CIFAR-10 to analyze depth benefits in convolutional architectures.
- Exploring transfer learning and fine-tuning of pre-trained residual CNNs for CIFAR-100, including feature extraction with classical classifiers and end-to-end fine-tuning.


# Project Structure

`MLP.py` – Baseline Multilayer Perceptron architecture

`Residual.py` – Residual MLP blocks & model

`DataManager.py` – Loads MNIST dataset and handles train/val/test splits

`Trainer.py` – Training, evaluation loop, and logging utilities

`Es_1.1.py` – Runs the baseline MLP (Exercise 1.1)

`Es_1.2.py` – Runs MLP vs ResidualMLP experiments (Exercise 1.2)

`Es_1.3.py` - Runs CNN vs ResidualCNN depth study on CIFAR-10 (Exercise 1.3)

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
| **Logging**       | [Comet ML](https://www.comet.com/alessiochen/dla-lab1-es-1-2/view/new/panels)


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

From the results I can see that adding residual connections significantly improves deep MLP performance. Vanilla MLPs suffer from vanishing gradients as depth increases, causing training to fail for networks with 8+ layers. Residual connections provide shortcut paths that let gradients bypass intermediate layers, enabling effective training and high accuracy even in very deep networks.

## 1.3 Rinse & Repeat with CNNs (CIFAR-10)

I repeat the study using Convolutional Neural Networks on CIFAR-10, verifying that deeper ≠ better for plain CNNs and that residual connections fix this.

| Component        | Description                                                                  |
| ---------------- | ---------------------------------------------------------------------------- |
| Models           | `CNN` (plain stacked conv blocks), `ResidualCNN` (ResNet-style `BasicBlock`) |
| Blocks Tested    | \[1, 5, 10]                                                                  |
| Block (plain)    | (Conv3×3 → BN → ReLU) × 2, optional MaxPool between stages                   |
| Block (residual) | torchvision-style `BasicBlock` with identity/1×1 downsample when needed      |
| Channels         | Kept consistent across comparisons (per stage)                               |
| Dataset          | CIFAR-10 (32×32 RGB), 80/20 train/val, test split included                   |
| Loss             | CrossEntropyLoss                                                             |
| Optimizer        | SGD (lr = 0.01)                                                              |
| Batch Size       | 128                                                                          |
| Epochs           | 20                                                                           |
| Device           | CUDA                                                                         |
| Logging          | [Comet ML](https://www.comet.com/alessiochen/dla-lab1-es-1-3/view/new/panels) |

### How to run

```
python Es_1.3.py --model_type CNN --num_blocks 5 --epochs 20 --lr 0.01 -- --use_comet
```
### Results 

| Model       | Blocks |   Val Acc  | Val Loss |
| ----------- | :----: | :--------: | :------: |
| ResidualCNN |   10   |   79.02%   |  0.6146  |
| ResidualCNN |    5   |   77.38%  |  0.6528  |
| ResidualCNN |    1   |   68.77%   |  0.8869  |
| CNN (plain) |   10   |   53.93%   |  1.2873  |
| CNN (plain) |    5   |   70.67%   |  0.8346  |
| CNN (plain) |    1   |   66.04%   |  0.9499  |

ResidualCNNs benefit from depth: 1 → 5 → 10 blocks steadily improves (68.8% → 77.4% → 79.0%).


## Exercise 2.1: Fine-tune a Pre-trained Residual CNN on CIFAR-100

This exercise investigates the use of transfer learning from a pre-trained Residual CNN (trained on CIFAR-10 in Exercise 1.3) to CIFAR-100:

1. Feature Extraction + Classical Classifier: Using the pre-trained CNN as a fixed feature extractor, and training a Linear SVM on the extracted features.
2. Fine-tuning the CNN: Adapting the pre-trained CNN to the new CIFAR-100 task by replacing the 10-class classifier with a 100-class classifier and selectively unfreezing layers.

| Component                 | Description                                                            |
| ------------------------- | ---------------------------------------------------------------------- |
| **Pre-trained Model**     | ResidualCNN (10 residual blocks) trained on CIFAR-10                   |
| **Classifier (baseline)** | Linear SVM (scikit-learn)                                              |
| **Fine-tuning**           | Replace classifier with 100-class output, and trained with different freezed layers |
| **Loss Function**         | CrossEntropyLoss                                                       |
| **Optimizer**             | Adam (lr = 0.01) or SGD(lr =0.01)                                   |
| **Batch Size**            | 128                                                                    |
| **Epochs (fine-tuning)**  | 20                                                                     |
| Logging          | [Comet ML](https://www.comet.com/alessiochen/dla-lab1-es-2-1/view/new/panels) |

### How to run

```
python Es_2_1.py --mode baseline 
python3 Es_2_1.py --mode finetune --optimizer adam --freezed_layers layer1,layer2

```

### Results
| Approach                                  | Test Accuracy |
| ------------------------------            | -------- |
| Linear SVM (feature extractor)            | 23.25%   |
| Fine-tuned CNN ADAM + freezed layer1,2    | 45.64%   |
| Fine-tuned CNN ADAM + freezed layer2      | 47.72%   |
| Fine-tuned CNN SGD +  freezed layer1,2    | 45.18%   |
| Fine-tuned CNN SGS +  freezed layer2      | 46.34%   |