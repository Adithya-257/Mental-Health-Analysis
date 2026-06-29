# src/baseline.py

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEED, CLASSES, DATA_DIR, CHARTS_DIR, LOGS_DIR

import random
random.seed(SEED)
np.random.seed(SEED)


def load_data():
    print("Loading splits...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).dropna(subset=['text', 'label_id'])
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv')).dropna(subset=['text', 'label_id'])
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')).dropna(subset=['text', 'label_id'])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def build_tfidf(train_df, val_df, test_df):
    print("\nBuilding TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    X_val = vectorizer.transform(val_df['text'].astype(str))
    X_test = vectorizer.transform(test_df['text'].astype(str))

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Train matrix: {X_train.shape}")
    return vectorizer, X_train, X_val, X_test


def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
    max_iter=1000,
    random_state=SEED,
    class_weight='balanced',
    C=1.0,
    solver='lbfgs'
        )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def evaluate_model(model, X, y_true, split_name):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n{split_name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\nClassification Report ({split_name}):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    return y_pred, {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4), 'precision_macro': round(precision, 4), 'recall_macro': round(recall, 4)}


def save_confusion_matrix(y_true, y_pred, split_name):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES
    )
    plt.title(f'Confusion Matrix — Baseline (TF-IDF + LR) — {split_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    fname = f'baseline_confusion_matrix_{split_name.lower()}.png'
    plt.savefig(os.path.join(CHARTS_DIR, fname), dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def save_logs(val_metrics, test_metrics):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'TF-IDF + Logistic Regression',
        'config': {
            'max_features': 50000,
            'ngram_range': '(1,2)',
            'min_df': 2,
            'sublinear_tf': True,
            'C': 1.0,
            'solver': 'lbfgs',
            'class_weight': 'balanced'
        },
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    log_path = os.path.join(LOGS_DIR, 'baseline_results.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Saved: logs/baseline_results.json")


def main():
    print("="*60)
    print("Baseline Model — TF-IDF + Logistic Regression")
    print("="*60)

    train_df, val_df, test_df = load_data()

    y_train = train_df['label_id'].values
    y_val = val_df['label_id'].values
    y_test = test_df['label_id'].values

    vectorizer, X_train, X_val, X_test = build_tfidf(train_df, val_df, test_df)
    model = train_logistic_regression(X_train, y_train)

    val_preds, val_metrics = evaluate_model(model, X_val, y_val, 'Validation')
    test_preds, test_metrics = evaluate_model(model, X_test, y_test, 'Test')

    save_confusion_matrix(y_val, val_preds, 'Validation')
    save_confusion_matrix(y_test, test_preds, 'Test')
    save_logs(val_metrics, test_metrics)

    print("\n" + "="*60)
    print("Baseline complete!")
    print(f"Baseline Test F1:  {test_metrics['f1_macro']:.4f}")
    print(f"RoBERTa Test F1:   0.8862")
    print(f"Improvement:       +{0.8862 - test_metrics['f1_macro']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()