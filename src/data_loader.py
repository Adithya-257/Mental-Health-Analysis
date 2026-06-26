# src/data_loader.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEED, CLASSES, DATA_DIR, CHARTS_DIR

random.seed(SEED)
np.random.seed(SEED)

FILE_LABEL_MAP = {
    "depression_2018_features_tfidf_256":    "depression",
    "depression_2019_features_tfidf_256":    "depression",
    "anxiety_2018_features_tfidf_256":       "anxiety",
    "anxiety_2019_features_tfidf_256":       "anxiety",
    "bipolarreddit_2018_features_tfidf_256": "bipolar",
    "bipolarreddit_2019_features_tfidf_256": "bipolar",
    "bipolarreddit_pre_features_tfidf_256": "bipolar",
    "ptsd_pre_features_tfidf_256":          "ptsd",
    "ptsd_2018_features_tfidf_256":          "ptsd",
    "ptsd_2019_features_tfidf_256":          "ptsd",
    "jokes_2018_features_tfidf_256":         "normal",
    "jokes_2019_features_tfidf_256":         "normal",
    "fitness_2018_features_tfidf_256":       "normal",
    "fitness_2019_features_tfidf_256":       "normal",
    "relationships_2018_features_tfidf_256": "normal",
    "relationships_2019_features_tfidf_256": "normal",
    
}


def load_raw_data():
    print("Loading datasets from /data...")
    all_dfs = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".csv"):
            continue

        file_key = os.path.splitext(filename)[0]
        if file_key not in FILE_LABEL_MAP:
            continue

        label = FILE_LABEL_MAP[file_key]
        filepath = os.path.join(DATA_DIR, filename)

        df = pd.read_csv(filepath, usecols=['post'])
        df['label'] = label
        df = df.dropna(subset=['post'])
        df = df[df['post'].str.strip() != ""]

        print(f"  Loaded {filename}: {len(df)} posts → label='{label}'")
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal posts loaded: {len(combined_df)}")
    print(f"Label distribution:\n{combined_df['label'].value_counts()}")

    return combined_df


def balance_classes(df):
    print("\nBalancing classes...")

    class_counts = df['label'].value_counts()
    print(f"Before balancing:\n{class_counts}")

    MAX_SAMPLES = 5000
    balanced_dfs = []

    for label in CLASSES:
        class_df = df[df['label'] == label]
        n = min(len(class_df), MAX_SAMPLES)
        sampled = class_df.sample(n=n, random_state=SEED)
        balanced_dfs.append(sampled)
        print(f"  {label}: {len(class_df)} → {n} samples")

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nAfter balancing: {len(balanced_df)} total posts")
    print(f"Label distribution:\n{balanced_df['label'].value_counts()}")

    return balanced_df


def add_label_ids(df):
    label2id = {label: idx for idx, label in enumerate(CLASSES)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['label_id'] = df['label'].map(label2id)
    print(f"\nLabel mapping: {label2id}")
    return df, label2id, id2label


def run_eda(df):
    print("\nRunning EDA and saving charts...")
    os.makedirs(CHARTS_DIR, exist_ok=True)

    df['post_length'] = df['post'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(10, 6))
    class_counts = df['label'].value_counts()
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
    bars = plt.bar(class_counts.index, class_counts.values, color=colors)
    plt.title('Class Distribution — Reddit Mental Health Corpus', fontsize=14, fontweight='bold')
    plt.xlabel('Mental Health Category', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(count), ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: class_distribution.png")

    plt.figure(figsize=(10, 6))
    plt.hist(df['post_length'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    plt.title('Post Length Distribution (Word Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    median_len = df['post_length'].median()
    plt.axvline(median_len, color='red', linestyle='--', label=f'Median: {median_len:.0f} words')
    plt.legend(fontsize=11)
    plt.xlim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'post_length_distribution.png'), dpi=150)
    plt.close()
    print("  Saved: post_length_distribution.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='label', y='post_length', order=CLASSES, hue='label', palette='viridis', legend=False)
    plt.title('Post Length Distribution per Class', fontsize=14, fontweight='bold')
    plt.xlabel('Mental Health Category', fontsize=12)
    plt.ylabel('Word Count', fontsize=12)
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'post_length_per_class.png'), dpi=150)
    plt.close()
    print("  Saved: post_length_per_class.png")

    print(f"\nPost length statistics:")
    print(f"  Mean:   {df['post_length'].mean():.1f} words")
    print(f"  Median: {df['post_length'].median():.1f} words")
    print(f"  Max:    {df['post_length'].max()} words")
    print(f"  Min:    {df['post_length'].min()} words")
    print(f"  Posts > 512 words: {(df['post_length'] > 512).sum()} ({(df['post_length'] > 512).mean()*100:.1f}%)")


def compute_class_weights(df):
    print("\nComputing class weights...")

    label_counts = df['label_id'].value_counts().sort_index()
    total = len(df)
    num_classes = len(CLASSES)
    weights = []

    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(round(weight, 4))
        print(f"  {CLASSES[i]:12s}: {count:5d} samples → weight {weight:.4f}")

    print(f"\nFinal class weights: {weights}")
    return weights


def split_and_save(df):
    print("\nSplitting dataset 80/10/10...")
    os.makedirs(DATA_DIR, exist_ok=True)

    df = df.rename(columns={'post': 'text'})
    df = df[['text', 'label', 'label_id']].copy()

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df['label_id']
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label_id']
    )

    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)

    print(f"  Train:      {len(train_df):5d} samples")
    print(f"  Validation: {len(val_df):5d} samples")
    print(f"  Test:       {len(test_df):5d} samples")
    print(f"  Saved to /data as train.csv, val.csv, test.csv")

    return train_df, val_df, test_df


def main():
    df = load_raw_data()
    df = balance_classes(df)
    df, label2id, id2label = add_label_ids(df)
    run_eda(df)
    class_weights = compute_class_weights(df)
    train_df, val_df, test_df = split_and_save(df)

    print("\n" + "="*50)
    print("Data loading pipeline complete!")
    print(f"label2id: {label2id}")
    print(f"id2label: {id2label}")
    print(f"Class weights: {class_weights}")
    print("="*50)

    return class_weights, label2id, id2label


if __name__ == "__main__":
    main()