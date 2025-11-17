#!/usr/bin/env python3
"""
Execute the remaining notebook cells for ML pipeline
"""
import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Executing ML Pipeline Steps")
print("="*60)

# Load and prepare data (assuming previous cells were run)
print("\n[Step 1] Loading and preparing data...")
df = pd.read_csv('/Users/sami/Desktop/Projects/Dogo/raw_dataset.csv')
df = df.dropna()
df = df.drop(columns=['id'])
sent_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['sentiment'].map(sent_map)
# Drop rows where label mapping failed (NaN)
df = df.dropna(subset=['label'])
df = df.drop(columns=['sentiment'])
df['description'] = df['description'].str.lower()

# Cell 25: Text cleaning
print("\n[Step 2] Text cleaning...")
def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['cleaned_description'] = df['description'].apply(clean_text)
df = df[df['cleaned_description'].str.len() > 0]

print(f"Shape after cleaning: {df.shape}")
print(f"\nSample cleaned text:")
print(df['cleaned_description'].head(3))

# Cell 26: Load BERT
print("\n[Step 3] Loading BERT tokenizer and model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Using device: {device}")

# Cell 27: Get BERT embeddings
print("\n[Step 4] Extracting BERT embeddings...")
def get_bert_embeddings(texts, batch_size=32, max_length=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch_texts)}/{len(texts)} texts...")
    return np.vstack(embeddings)

texts = df['cleaned_description'].tolist()
X_bert = get_bert_embeddings(texts)
y = df['label'].values

print(f"BERT embeddings shape: {X_bert.shape}")
print(f"Labels shape: {y.shape}")

# Cell 28: Train/test split
print("\n[Step 5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_bert, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"\nTrain label distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print(f"\nTest label distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# Cell 29: Baseline XGBoost
print("\n[Step 6] Training baseline XGBoost model...")
baseline_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"\nBaseline XGBoost Accuracy: {baseline_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=['positive', 'negative', 'neutral']))

# Cell 30: Optuna optimization
print("\n[Step 7] Starting Optuna hyperparameter optimization...")
print("This may take a while...\n")

def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n" + "="*50)
print("Optimization Complete!")
print("="*50)
print(f"\nBest trial:")
trial = study.best_trial
print(f"  Value (Accuracy): {trial.value:.4f}")
print(f"\n  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Cell 31: Final model
print("\n[Step 8] Training final XGBoost model with best parameters...")
best_params = study.best_params.copy()
best_params.update({
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
})

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\nFinal XGBoost Accuracy: {final_accuracy:.4f}")
print(f"Improvement over baseline: {final_accuracy - baseline_accuracy:.4f} ({((final_accuracy - baseline_accuracy) / baseline_accuracy * 100):.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['positive', 'negative', 'neutral']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# Save model and metadata
print("\n[Step 9] Saving model and metadata...")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Save XGBoost model
model_path = os.path.join(model_dir, "best_xgboost_model.json")
final_model.save_model(model_path)
print(f"Model saved to: {model_path}")

# Save metadata
metadata = {
    "model_type": "XGBoost",
    "bert_model": model_name,
    "max_length": 128,
    "baseline_accuracy": float(baseline_accuracy),
    "final_accuracy": float(final_accuracy),
    "improvement": float(final_accuracy - baseline_accuracy),
    "improvement_percent": float((final_accuracy - baseline_accuracy) / baseline_accuracy * 100),
    "best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()},
    "label_mapping": {v: k for k, v in sent_map.items()},  # Reverse mapping for inference
    "training_date": datetime.now().isoformat(),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "feature_dim": int(X_bert.shape[1])
}

metadata_path = os.path.join(model_dir, "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

# Save text cleaning function as a separate module for inference
text_cleaning_code = '''import re
import pandas as pd

def clean_text(text):
    """Clean text by removing URLs, special characters, extra spaces, etc."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\S+@\\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\\s.,!?]', '', text)
    text = re.sub(r'\\s+', ' ', text)
    text = text.strip()
    return text
'''

text_cleaning_path = os.path.join(model_dir, "text_cleaning.py")
with open(text_cleaning_path, 'w') as f:
    f.write(text_cleaning_code)
print(f"Text cleaning module saved to: {text_cleaning_path}")

print("\n" + "="*60)
print("Pipeline Complete!")
print("="*60)
print(f"\nModel files saved in '{model_dir}/' directory:")
print(f"  - best_xgboost_model.json (XGBoost model)")
print(f"  - model_metadata.json (Model metadata and parameters)")
print(f"  - text_cleaning.py (Text cleaning function)")

