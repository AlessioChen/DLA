# Lab3: Sentiment Analysis with DistilBERT on Cornell Rotten Tomatoes Dataset

This project explores sentiment analysis using the [Cornell Rotten Tomatoes movie review dataset](https://huggingface.co/datasets/rotten_tomatoes).  

## Dataset

- Name: Rotten Tomatoes (via HuggingFace Datasets)
- Size: ~10,662 sentences
- Classes: Binary sentiment classification
    - 0 → Negative
    - 1 → Positive

Dataset splits:

- Train: 8,530 samples
- Validation: 1,066 samples
- Test: 1,066 samples

## Project Structure
- `baseline.py` - SVM baseline using frozen DistilBERT embeddings.
- `finetune.py` - Full DistilBERT fine-tuning and LoRA-efficient fine-tuning.

##  Stable Baseline with DistilBERT + SVM

I first build a stable baseline by using DistilBERT as a frozen feature extractor and training a Linear SVM classifier on top.

## How to run 

```
python baseline.py
```

### Implementation details 

| Component             | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| **Model**             | [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased)          |
| **Feature Extractor** | Use `[CLS]` embedding from `last_hidden_state[:, 0, :]`                |
| **Classifier**        | Linear SVM (`LinearSVC` from scikit-learn)                             |
| **Batch Size**        | 64 (for feature extraction)                                            |
| **Training**          | Only SVM is trained, DistilBERT remains frozen                         |
| **Evaluation**        | Accuracy on val and test                           |
| **logging**            | [comet ml](https://www.comet.com/alessiochen/dla-lab3/view/new/panels)


## Full Fine-tuning & LoRA-efficient Fine-tuning 
I also implemented full fine-tuning of DistilBERT and LoRA-efficient fine-tuning for comparison.

### model details 
| Method           | Trainable Parameters | Total Parameters |
| ---------------- | -------------------- | ---------------- |
| Full Fine-tuning | 68M (all)            | 68M              |
| LoRA Fine-tuning | 739,586              | 68M              |

### Training setup:

- `Batch size`: 64
- `Epochs`: 20
- `Optimizer`: AdmaW
- `Learning rate`: 1e-5
- `Weight decay` : 0.01
- `LoRA r`: 8 
- `LoRa alpha`: 32 

## How to run 

```
python finetune.py
python finetune.py --lora 
```

### Results 
| Method           | Validation Accuracy | Test Accuracy |
| ---------------- | ------------------- | ------------- |
| Base line SVM    | 82.22%              | 79,83%        |
| Full Fine-tuning | 85.27%              | 83.86%        |
| LoRA Fine-tuning | 82.74%              | 81.14%        |
