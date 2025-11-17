#!/usr/bin/env python3
"""
Fine-tune FinBERT model for sentiment analysis using raw_dataset.csv
Uses HuggingFace Transformers with robust hyperparameter optimization
"""
import os
import json
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
try:
    from transformers import EarlyStoppingCallback
except ImportError:
    from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import optuna
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Force CPU to avoid MPS/GPU OOM on macOS and ensure portability
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def clean_text(text):
    """Clean text by removing URLs, special characters, extra spaces, etc."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }


def load_and_prepare_data(csv_path):
    """Load and prepare data from CSV"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop null values
    df = df.dropna(subset=['description', 'sentiment'])
    
    # Map sentiment labels
    sent_map = {'positive': 1, 'negative': 2, 'neutral': 0}
    df['label'] = df['sentiment'].map(sent_map)
    df = df.dropna(subset=['label'])
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_description'] = df['description'].apply(clean_text)
    df = df[df['cleaned_description'].str.len() > 0]
    
    # Convert to lowercase
    df['cleaned_description'] = df['cleaned_description'].str.lower()
    
    texts = df['cleaned_description'].tolist()
    labels = df['label'].astype(int).tolist()
    
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution:")
    print(pd.Series(labels).value_counts().sort_index())
    
    return texts, labels


def objective(trial, texts, labels, tokenizer, model_name, output_dir_base):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.3)
    warmup_steps = trial.suggest_int('warmup_steps', 0, 300)
    max_length = trial.suggest_categorical('max_length', [128, 256])
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    
    # Training arguments
    output_dir = os.path.join(output_dir_base, f"trial_{trial.number}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Shorter for hyperparameter search
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        no_cuda=True,
        use_mps_device=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Clean up
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    return eval_results['eval_accuracy']


def main():
    """Main training function"""
    print("="*60)
    print("FinBERT Fine-tuning for Sentiment Analysis")
    print("="*60)
    
    # Configuration
    csv_path = "/Users/sami/Desktop/Projects/Dogo/raw_dataset.csv"
    model_name = "yiyanghkust/finbert-tone"  # FinBERT specifically for sentiment analysis
    output_dir_base = "finbert_models"
    os.makedirs(output_dir_base, exist_ok=True)
    
    # Load data
    texts, labels = load_and_prepare_data(csv_path)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Hyperparameter optimization
    print("\n" + "="*60)
    print("Starting Hyperparameter Optimization with Optuna")
    print("="*60)
    print("This will test different hyperparameter combinations...\n")
    
    study = optuna.create_study(
        direction='maximize',
        study_name='finbert_optimization'
    )
    
    study.optimize(
        lambda trial: objective(trial, texts, labels, tokenizer, model_name, output_dir_base),
        n_trials=10,  # Number of hyperparameter search trials
        show_progress_bar=True
    )
    
    print("\n" + "="*60)
    print("Hyperparameter Optimization Complete!")
    print("="*60)
    print(f"\nBest trial:")
    best_trial = study.best_trial
    print(f"  Accuracy: {best_trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Use best hyperparameters for final training
    best_params = best_trial.params
    
    # Split data for final training
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, best_params['max_length'])
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, best_params['max_length'])
    test_dataset = SentimentDataset(X_test, y_test, tokenizer, best_params['max_length'])
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )
    
    # Final training arguments with best hyperparameters
    final_output_dir = os.path.join(output_dir_base, "final_model")
    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=50,  # Full 50 epochs as requested
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        warmup_steps=best_params['warmup_steps'],
        logging_dir=os.path.join(final_output_dir, 'logs'),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        fp16=False,
        report_to="none",
        seed=42,
        no_cuda=True,
        use_mps_device=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting Final Training (50 epochs)")
    print("="*60)
    print(f"Using best hyperparameters from optimization")
    print(f"Training will stop early if validation accuracy doesn't improve for 5 epochs\n")
    
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = y_test
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['positive', 'negative', 'neutral']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save model
    print(f"\nSaving model to {final_output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(final_output_dir)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "model_type": "FinBERT",
        "task": "sentiment_classification",
        "num_labels": 3,
        "label_mapping": {"0": "positive", "1": "negative", "2": "neutral"},
        "best_hyperparameters": {k: float(v) if isinstance(v, (int, float)) else v 
                                 for k, v in best_params.items()},
        "test_accuracy": float(test_results['eval_accuracy']),
        "test_f1": float(test_results['eval_f1']),
        "training_date": datetime.now().isoformat(),
        "num_epochs": 50,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "max_length": best_params['max_length']
    }
    
    metadata_path = os.path.join(final_output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModel saved to: {final_output_dir}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()

