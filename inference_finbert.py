#!/usr/bin/env python3
"""
Inference script for fine-tuned FinBERT sentiment analysis model
"""
import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
import warnings
warnings.filterwarnings('ignore')


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


class FinBERTPredictor:
    """FinBERT sentiment analysis predictor"""
    
    def __init__(self, model_dir="finbert_models/final_model", weight_file="model.safetensors"):
        """
        Initialize the predictor by loading model and metadata
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        self.weight_file = os.path.join(model_dir, weight_file) if weight_file else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "model_name": "yiyanghkust/finbert-tone",
                "label_mapping": {"0": "neutral", "1": "positive", "2": "negative"},
                "max_length": 256,
                "test_accuracy": None,
                "training_date": "unknown"
            }
        
        print(f"Loading FinBERT model from {model_dir}...")
        model_name = self.metadata.get('model_name', 'yiyanghkust/finbert-tone')
        print(f"Model: {model_name}")
        if self.metadata.get("test_accuracy") is not None:
            print(f"Test Accuracy: {self.metadata['test_accuracy']:.4f}")
        print(f"Training date: {self.metadata.get('training_date', 'unknown')}")
        
        # Load tokenizer and model
        print("Loading tokenizer and model...")
        config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.weight_file and os.path.exists(self.weight_file):
            state_dict = load_file(self.weight_file)
            self.model = AutoModelForSequenceClassification.from_config(config)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Using device: {self.device}")
        
        # Load label mapping
        if getattr(config, "id2label", None):
            self.label_mapping = {str(idx): label.lower() for idx, label in config.id2label.items()}
        else:
            self.label_mapping = self.metadata.get('label_mapping', {"0": "neutral", "1": "positive", "2": "negative"})
        self.max_length = self.metadata.get('max_length', 256)
        
        print("Model ready for inference!\n")
    
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
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        
        # Map prediction to label
        sentiment = self.label_mapping[str(prediction)]
        confidence = float(probabilities[prediction])
        
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
    
    def predict_batch(self, texts, batch_size=8):
        """
        Predict sentiment for a batch of texts
        
        Args:
            texts: List of input text strings
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            # Clean texts
            cleaned_texts = [clean_text(text).lower() for text in batch_texts]
            
            # Tokenize batch
            encodings = self.tokenizer(
                cleaned_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Process results
            for j, (text, pred, probs) in enumerate(zip(batch_texts, predictions, probabilities)):
                sentiment = self.label_mapping[str(int(pred))]
                confidence = float(probs[int(pred)])
                prob_dict = {
                    self.label_mapping[str(k)]: float(prob) 
                    for k, prob in enumerate(probs)
                }
                
                batch_results.append({
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': prob_dict,
                    'text': text
                })
            
            results.extend(batch_results)
        
        return results


def main():
    """Main function for interactive inference"""
    import sys
    
    # Initialize predictor
    try:
        predictor = FinBERTPredictor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run 'python3 finbert_finetuned.py' first to train and save the model.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print("="*60)
    print("FinBERT Sentiment Analysis Inference")
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

