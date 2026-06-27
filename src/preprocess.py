# src/preprocess.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SEED

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def basic_clean(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def roberta_clean(text):
    return basic_clean(text)


def vader_clean(text):
    text = basic_clean(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    text = ' '.join(tokens)
    return text


def preprocess_dataframe(df, branch='roberta'):
    df = df.copy()
    if branch == 'roberta':
        df['text'] = df['text'].apply(roberta_clean)
    elif branch == 'vader':
        df['text'] = df['text'].apply(vader_clean)
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.split().str.len() >= 5]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    import pandas as pd
    from config import DATA_DIR

    print("Testing preprocessor...")

    sample_texts = [
        "I've been feeling really depressed lately http://help.com check this out <b>bold</b>",
        "My anxiety is through the roof!!! Can't sleep at all 😭😭",
        "   Just had a great day at the gym   ",
    ]

    print("\nBranch 1 — RoBERTa clean:")
    for t in sample_texts:
        print(f"  IN:  {t[:60]}")
        print(f"  OUT: {roberta_clean(t)[:60]}")
        print()

    print("Branch 2 — VADER clean:")
    for t in sample_texts:
        print(f"  IN:  {t[:60]}")
        print(f"  OUT: {vader_clean(t)[:60]}")
        print()

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    print(f"Preprocessing train set ({len(train_df)} rows)...")
    train_clean = preprocess_dataframe(train_df, branch='roberta')
    print(f"After cleaning: {len(train_clean)} rows")
    print(f"Sample cleaned text:\n{train_clean['text'].iloc[0][:200]}")