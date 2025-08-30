from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def extract_cls_embeddings(texts, tokenizer, model, batch_size=64):  
    """Helper function: encode texts -> [CLS] embeddings"""
    embeddings = []
    for i in range(0, len(texts), batch_size):  
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**inputs)
            # outputs.last_hidden_state shape: [batch, seq_len, hidden_size]
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  
            embeddings.append(cls_embeddings)
    
    return np.concatenate(embeddings, axis=0)  

if __name__ == '__main__':
    dataset = load_dataset('rotten_tomatoes')
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval() 
    
    # Extract features for train/val/test 
    X_train = extract_cls_embeddings(dataset["train"]["text"], tokenizer, model)
    y_train = dataset["train"]["label"]

    X_val = extract_cls_embeddings(dataset["validation"]["text"], tokenizer, model)
    y_val = dataset["validation"]["label"]

    X_test = extract_cls_embeddings(dataset["test"]["text"], tokenizer, model)
    y_test = dataset["test"]["label"]
    
    print("start training linear SVC")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    
    print("start evaluating on val split")
    y_val_pred = clf.predict(X_val)
    print("start evaluating on test split")
    y_test_pred = clf.predict(X_test)

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

