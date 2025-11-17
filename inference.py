#!/usr/bin/env python3
"""
Inference script for sentiment analysis using trained XGBoost model with BERT embeddings
"""
import os
import sys
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import text cleaning function
import re
import pandas as pd

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


class SentimentPredictor:
    """Sentiment analysis predictor using BERT + XGBoost"""
    
    def __init__(self, model_dir="models"):
        """
        Initialize the predictor by loading model and metadata
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Please train the model first.")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loading model from {model_dir}...")
        print(f"Model accuracy: {self.metadata['final_accuracy']:.4f}")
        print(f"Training date: {self.metadata['training_date']}")
        
        # Load BERT tokenizer and model
        print("Loading BERT tokenizer and model...")
        self.bert_model_name = self.metadata['bert_model']
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        print(f"BERT model loaded. Using device: {self.device}")
        
        # Load XGBoost model
        print("Loading XGBoost model...")
        model_path = os.path.join(model_dir, "best_xgboost_model.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
        
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(model_path)
        print("XGBoost model loaded.")
        
        # Load label mapping
        self.label_mapping = self.metadata['label_mapping']
        self.max_length = self.metadata.get('max_length', 128)
        
        print("Model ready for inference!\n")
    
    def get_bert_embedding(self, text, max_length=128):
        """
        Get BERT embedding for a single text
        
        Args:
            text: Input text string
            max_length: Maximum sequence length
            
        Returns:
            BERT embedding vector
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get embedding (mean pooling)
        with torch.no_grad():
            outputs = self.bert_model(**encoded)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embedding
    
    def predict(self, text, return_proba=False):
        """
        Predict sentiment for a given text
        
        Args:
            text: Input text string
            return_proba: If True, return probability scores for all classes
            
        Returns:
            If return_proba=False: sentiment label (positive/negative/neutral)
            If return_proba=True: dict with label and probabilities
        """
        # Clean text
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) == 0:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'message': 'Empty text after cleaning'
            }
        
        # Convert to lowercase (as done in training)
        cleaned_text = cleaned_text.lower()
        
        # Get BERT embedding
        embedding = self.get_bert_embedding(cleaned_text, max_length=self.max_length)
        
        # Predict with XGBoost
        prediction = self.xgb_model.predict(embedding)[0]
        probabilities = self.xgb_model.predict_proba(embedding)[0]
        
        # Map prediction to label
        sentiment = self.label_mapping[str(int(prediction))]
        confidence = float(probabilities[int(prediction)])
        
        # Create probability dictionary
        prob_dict = {
            self.label_mapping[str(i)]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        if return_proba:
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': prob_dict,
                'text': text,
                'cleaned_text': cleaned_text
            }
        else:
            return sentiment


def main():
    """Main function for interactive inference"""
    # Initialize predictor
    try:
        predictor = SentimentPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python3 run_notebook_cells.py' first to train and save the model.")
        sys.exit(1)
    
    print("="*60)
    print("Sentiment Analysis Inference")
    print("="*60)
    print("\nEnter text to analyze sentiment (or 'quit' to exit):\n")
    
    while True:
        try:
            # Get user input
            user_input = input("Input text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break
            
            if not user_input:
                print("Please enter some text.\n")
                continue
            
            # Make prediction
            result = predictor.predict(user_input, return_proba=True)
            
            # Display results
            print("\n" + "-"*60)
            print("PREDICTION RESULTS")
            print("-"*60)
            print(f"Sentiment: {result['sentiment'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            for sentiment, prob in result['probabilities'].items():
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                print(f"  {sentiment:10s}: {prob:6.2%} {bar}")
            print("-"*60)
            print()
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError during prediction: {e}\n")
            continue


if __name__ == "__main__":
    main()

