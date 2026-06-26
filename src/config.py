# src/config.py

import os

# ── Reproducibility ───────────────────────────────────────────
# Every file that uses randomness imports this seed and sets it.
# Guarantees identical results every run.
SEED = 42

# ── Model ─────────────────────────────────────────────────────
# The pretrained model we're fine-tuning from HuggingFace Hub.
MODEL_NAME = "roberta-base"

# ── Classes ───────────────────────────────────────────────────
# Our 5 target mental health categories.
# Order matters — label_id 0 = depression, 1 = anxiety, etc.
# This must stay consistent across all files.
CLASSES = ["depression", "anxiety", "bipolar", "ptsd", "normal"]
NUM_CLASSES = len(CLASSES)

# ── Training hyperparameters ──────────────────────────────────
# BATCH_SIZE = 8 because your RTX 3050 has 4GB VRAM.
# RoBERTa is large — 16 will likely cause out-of-memory error.
# GRAD_ACCUM_STEPS = 4 means gradients accumulate over 4 steps
# before updating weights — effective batch size = 8 × 4 = 32.
# Same training quality as batch size 32, less VRAM used.
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
MAX_LEN = 512
NUM_LABELS = NUM_CLASSES

# ── LIME settings ─────────────────────────────────────────────
# NUM_SAMPLES = how many perturbed versions of the text LIME generates
# NUM_FEATURES = how many top tokens to highlight in the explanation
LIME_NUM_SAMPLES = 500
LIME_NUM_FEATURES = 10

# ── Evaluation ────────────────────────────────────────────────
# macro averaging treats all classes equally regardless of size
AVERAGING = "macro"

# ── HuggingFace Hub ───────────────────────────────────────────
# Change this to your HuggingFace username/reponame later
# after you create an account at huggingface.co
HF_REPO_NAME = "Adithya-257/mental-health-roberta"

# ── Paths ─────────────────────────────────────────────────────
# os.path.abspath(__file__) gives the full path to config.py
# os.path.dirname twice goes up two levels to the project root
# All other paths are built relative to BASE_DIR so the project
# works on any machine regardless of where it's cloned.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
WORDCLOUD_DIR = os.path.join(ASSETS_DIR, "wordclouds")
CHARTS_DIR = os.path.join(ASSETS_DIR, "charts")