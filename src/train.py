# src/train.py

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import random
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEED, MODEL_NAME, NUM_CLASSES, MAX_LEN, BATCH_SIZE,
    GRAD_ACCUM_STEPS, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    WARMUP_RATIO, EARLY_STOPPING_PATIENCE, CLASSES,
    DATA_DIR, MODEL_DIR, LOGS_DIR, CHARTS_DIR
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


CLASS_WEIGHTS = [0.9765, 0.9765, 1.1066, 0.9765, 0.9765]


class MentalHealthDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text'].tolist()
        self.labels = df['label_id'].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)} GB")
    return device


def load_data():
    print("\nLoading preprocessed splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    train_df = train_df.dropna(subset=['text', 'label_id'])
    val_df = val_df.dropna(subset=['text', 'label_id'])
    test_df = test_df.dropna(subset=['text', 'label_id'])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_dataloaders(train_df, val_df, test_df, tokenizer):
    print("\nBuilding dataloaders...")
    train_dataset = MentalHealthDataset(train_df, tokenizer)
    val_dataset = MentalHealthDataset(val_df, tokenizer)
    test_dataset = MentalHealthDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def build_model(device, class_weights):
    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    model = model.to(device)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, criterion


def build_optimizer_scheduler(model, num_training_steps):
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"\nOptimizer: AdamW | lr={LEARNING_RATE} | weight_decay={WEIGHT_DECAY}")
    print(f"Scheduler: Linear warmup over {num_warmup_steps} steps")
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{len(loader)} | Loss: {total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, f1, acc, precision, recall, all_preds, all_labels


def save_checkpoint(model, tokenizer, epoch, val_f1, is_best=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_DIR, f'checkpoint_epoch_{epoch}')
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    meta = {'epoch': epoch, 'val_f1': val_f1}
    with open(os.path.join(checkpoint_path, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"  Saved checkpoint: checkpoint_epoch_{epoch} (val_f1={val_f1:.4f})")

    if is_best:
        best_path = os.path.join(MODEL_DIR, 'best_model')
        model.save_pretrained(best_path)
        tokenizer.save_pretrained(best_path)
        with open(os.path.join(best_path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"  *** New best model saved (val_f1={val_f1:.4f}) ***")


def save_training_curves(history):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_f1'], 'b-o', label='Train F1')
    ax2.plot(epochs, history['val_f1'], 'r-o', label='Val F1')
    ax2.set_title('Training and Validation Macro F1', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print("  Saved: training_curves.png")


def save_logs(history, test_metrics):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'model': MODEL_NAME,
            'batch_size': BATCH_SIZE,
            'grad_accum': GRAD_ACCUM_STEPS,
            'effective_batch_size': BATCH_SIZE * GRAD_ACCUM_STEPS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'max_len': MAX_LEN,
            'epochs_run': len(history['train_loss']),
            'seed': SEED
        },
        'history': history,
        'test_metrics': test_metrics
    }
    log_path = os.path.join(LOGS_DIR, 'runs.json')
    runs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            runs = json.load(f)
    runs.append(log)
    with open(log_path, 'w') as f:
        json.dump(runs, f, indent=2)
    print(f"  Saved: logs/runs.json")


def main():
    print("="*60)
    print("Mental Health Classification — RoBERTa Fine-tuning")
    print("="*60)

    train_df, val_df, test_df = load_data()
    device = set_device()

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    model, criterion = build_model(device, CLASS_WEIGHTS)

    num_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    optimizer, scheduler = build_optimizer_scheduler(model, num_training_steps)

    scaler = GradScaler('cuda')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    best_val_f1 = 0.0
    patience_counter = 0

    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-"*40)

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device
        )

        val_loss, val_f1, val_acc, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        history['train_f1'].append(round(train_f1, 4))
        history['val_f1'].append(round(val_f1, 4))

        print(f"\n  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")
        print(f"  Val Acc:    {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Rec: {val_rec:.4f}")

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        save_checkpoint(model, tokenizer, epoch, val_f1, is_best=is_best)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\n" + "="*60)
    print("Training complete! Running final evaluation on test set...")
    print("="*60)

    best_model_path = os.path.join(MODEL_DIR, 'best_model')
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    model = model.to(device)

    test_loss, test_f1, test_acc, test_prec, test_rec, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    test_metrics = {
        'test_loss': round(test_loss, 4),
        'test_f1_macro': round(test_f1, 4),
        'test_accuracy': round(test_acc, 4),
        'test_precision_macro': round(test_prec, 4),
        'test_recall_macro': round(test_rec, 4)
    }

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Macro F1:  {test_f1:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")

    save_training_curves(history)
    save_logs(history, test_metrics)

    print("\n" + "="*60)
    print("All done!")
    print(f"Best Val F1:  {best_val_f1:.4f}")
    print(f"Test F1:      {test_f1:.4f}")
    print(f"Model saved to: {best_model_path}")
    print("="*60)


if __name__ == "__main__":
    main()