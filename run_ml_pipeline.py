#!/usr/bin/env python3
"""
Run the ML pipeline: text cleaning, BERT tokenization, XGBoost training
"""
import numpy as np 
import pandas as pd 
import re
import string
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Starting ML Pipeline")
print("="*60)

# Load data
print("\n[Step 1] Loading data...")
df = pd.read_csv("/Users/sami/Desktop/Projects/Dogo/raw_dataset.csv")
df = df.dropna()
df = df.drop(columns=["id"])
sent_map = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['sentiment'].map(sent_map)
df = df.drop(columns=["sentiment"])
df['description'] = df['description'].str.lower()
print(f"Data shape: {df.shape}")

# Text cleaning
print("\n[Step 2] Cleaning text...")
def clean_text(text):
    if pd.isna(text):
        return ""
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
print(df['cleaned_description'].head(3).tolist())

# Load BERT
print("\n[Step 3] Loading BERT model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Using device: {device}")

# Get BERT embeddings
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

# Split data
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

# Baseline XGBoost
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

# Optuna optimization
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

# Final model
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

print("\n" + "="*60)
print("Pipeline Complete!")
print("="*60)

