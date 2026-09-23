# deep_learning_methods/transformer_main.py

import time
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels).item()
    return corrects / len(labels)

def run_epoch(model, dataloader, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0

    with torch.set_grad_enabled(is_train):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            total_acc += calculate_accuracy(logits, labels) * input_ids.size(0)

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_acc / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def save_results(experiment_name, epoch_data, best_eval, best_eval_ep, best_test, best_test_ep, best_induced):
    os.makedirs("results", exist_ok=True)
    
    # 1. Save detailed epochs CSV
    df_epochs = pd.DataFrame(epoch_data)
    df_epochs.to_csv(f"results/{experiment_name}_epochs.csv", index=False)
    
    # 2. Append to master summary
    summary_data = [{
        "Experiment": experiment_name,
        "Best Eval Acc": best_eval,
        "Best Eval Epoch": best_eval_ep,
        "Best Test Acc": best_test,
        "Best Test Epoch": best_test_ep,
        "Best Induced Test Acc": best_induced
    }]
    df_summary = pd.DataFrame(summary_data)
    master_file = "results/master_summary_results.csv"
    if os.path.exists(master_file):
        df_summary.to_csv(master_file, mode='a', header=False, index=False)
    else:
        df_summary.to_csv(master_file, index=False)

def save_graphs(experiment_name, epoch_data):
    os.makedirs("results", exist_ok=True)
    epochs = [d["Epoch"] for d in epoch_data]
    train_acc = [d["Train Acc"] for d in epoch_data]
    eval_acc = [d["Eval Acc"] for d in epoch_data]
    train_loss = [d["Train Loss"] for d in epoch_data]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
    plt.plot(epochs, eval_acc, label='Eval Accuracy', marker='o')
    plt.title(f'Accuracy: {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
    plt.title(f'Loss: {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/{experiment_name}_graph.png")
    plt.close()

def train_transformer(dataset_name: str, model_name: str = 'distilbert-base-uncased', batch_size=16, num_epochs=3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    safe_model_name = model_name.replace("/", "-")
    experiment_name = f"{dataset_name}_Transformer_{safe_model_name}"
    
    print(f"\n{'='*50}")
    print(f" TRAINING {model_name.upper()} ON {dataset_name.upper()}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'rotten_tomatoes':
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
        train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
        val_texts, val_labels = dataset['validation']['text'], dataset['validation']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']
    else:
        dataset = load_dataset("stanfordnlp/imdb")
        full_train = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_texts, train_labels = full_train['train']['text'], full_train['train']['label']
        val_texts, val_labels = full_train['test']['text'], full_train['test']['label']
        test_texts, test_labels = dataset['test']['text'], dataset['test']['label']

    train_dataset = TransformerDataset(train_texts, train_labels, tokenizer)
    val_dataset = TransformerDataset(val_texts, val_labels, tokenizer)
    test_dataset = TransformerDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_eval_acc, best_eval_epoch = 0.0, 0
    best_test_acc, best_test_epoch = 0.0, 0
    best_induced_test_acc = 0.0
    epoch_results_data = []

    print("\nStarting Training...")
    print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Acc':<10} | {'Eval Acc':<9} | {'Test Acc':<9} | {'Epoch Time':<11} | {'Best Induced Test'}")
    print("-" * 85)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, is_train=True)
        _, eval_acc = run_epoch(model, val_loader, optimizer, device, is_train=False)
        _, test_acc = run_epoch(model, test_loader, optimizer, device, is_train=False)
        
        epoch_time = time.time() - start_time
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch
            
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best_eval_epoch = epoch
            best_induced_test_acc = test_acc
            
        print(f"{epoch:<7} | {train_loss:<11.4f} | {train_acc:<10.4f} | {eval_acc:<9.4f} | {test_acc:<9.4f} | {epoch_time:<9.2f}s | {best_induced_test_acc:.4f}")
        
        epoch_results_data.append({
            "Epoch": epoch,
            "Train Loss": round(train_loss, 4),
            "Train Acc": round(train_acc, 4),
            "Eval Acc": round(eval_acc, 4),
            "Test Acc": round(test_acc, 4),
            "Epoch Time (s)": round(epoch_time, 2),
            "Best Induced Test": round(best_induced_test_acc, 4)
        })

    # Save to CSV and generate graphs
    save_results(experiment_name, epoch_results_data, best_eval_acc, best_eval_epoch, best_test_acc, best_test_epoch, best_induced_test_acc)
    save_graphs(experiment_name, epoch_results_data)

def run_all_transformers(model_name: str = 'distilbert-base-uncased', num_epochs: int = 3, batch_size: int = 16):
    """Runs the transformer experiment on both datasets automatically."""
    datasets = ['rotten_tomatoes', 'imdb']
    for ds in datasets:
        # 3 epochs is standard for fine-tuning transformers
        train_transformer(dataset_name=ds, model_name=model_name, batch_size=batch_size, num_epochs=num_epochs)
    
    print("\nTransformer experiments completed. Results and graphs saved in 'results' folder.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Transformer experiments")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="HuggingFace model name to use")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    args = parser.parse_args()
    
    run_all_transformers(model_name=args.model_name, num_epochs=args.epochs, batch_size=args.batch_size)