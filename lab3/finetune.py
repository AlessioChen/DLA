from comet_ml import Experiment
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import LoraConfig, TaskType, get_peft_model
from dotenv import load_dotenv

load_dotenv()

import numpy as np 
import argparse
import os 
def tokenize_function(sample, tokenizer):
    return tokenizer(sample['text'],truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="Use LoRA-efficient fine-tuning")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()
    
    return args
    

if __name__ == '__main__':  
    args = get_args()
    dataset = load_dataset("rotten_tomatoes")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') 
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    
    experiment = Experiment(
        os.getenv("COMET_API_KEY"),
        project_name="dla-lab3",
        workspace="alessiochen"
    )

    data_collator =  DataCollatorWithPadding(tokenizer=tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels= 2
    )
    
    if args.lora: 
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_lin", "v_lin"], # query and value adaptation
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    experiment.log_metric("total_parameters", total_params)
    experiment.log_metric("trainable_parameters", trainable_params)
        
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="comet_ml",
        run_name="distilbert-finetune-rotten-tomatoes",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    

    print("Evaluation on validation set:")
    val_results = trainer.evaluate(tokenized_datasets["validation"])
    print(val_results)
    experiment.log_metrics(val_results, prefix="validation")

    # Evaluate on test
    print("Evaluation on test set:")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(test_results)
    experiment.log_metrics(test_results, prefix="test")
    
    experiment.end()
    