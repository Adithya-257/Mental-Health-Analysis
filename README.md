# Mental Health Monitoring Through Social Media Analytics

An end-to-end mental health text analysis system using fine-tuned RoBERTa, 
VADER sentiment analysis, LIME explainability, and a Streamlit dashboard.

## System Overview
- **Classification**: Fine-tuned roberta-base for 5-class mental health prediction
  (depression, anxiety, bipolar, PTSD, normal)
- **Sentiment**: VADER parallel sentiment scoring
- **Explainability**: LIME token-level prediction explanations
- **Dashboard**: Streamlit interactive visualization

## Dataset
- Primary: Reddit Mental Health Corpus (5-class)
- Secondary: Dreaddit (binary stress classification)

## Project Structure
- `/data` — datasets (gitignored)
- `/models` — saved checkpoints (gitignored, shared via HuggingFace Hub)
- `/assets` — generated charts and word clouds
- `/src` — all source code
- `/logs` — experiment logs

## Team
- Adithya KL — Data pipeline, EDA, RoBERTa training, baseline model
- Aman Tulsiyan — Evaluation, LIME explainability, error analysis
- Apoorv Anand — VADER sentiment, word clouds, Streamlit dashboard

## Setup
```bash
pip install -r requirements.txt
```

## Results
*(To be updated after training)*