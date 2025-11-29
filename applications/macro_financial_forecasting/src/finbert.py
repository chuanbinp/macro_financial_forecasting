from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import Config
from typing import List
from tqdm import tqdm

class FinBertSentiment:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.finbert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.finbert_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.optimal_batch_size = 256 if torch.cuda.is_available() else 32

    def get_sentiment_scores_batch(self, texts: List[str]) -> List[float]:
        """
        Process a batch of texts and return sentiment scores.
        Score = P(positive) - P(negative), range: [-1, 1]
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Increased from 128 for better context
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Vectorized calculation on GPU
        # probs shape: (batch_size, 3) where columns are [negative, neutral, positive]
        sentiment_scores = (probs[:, 2] - probs[:, 0]).cpu().tolist()
        
        return sentiment_scores
    
    
    def get_sentiment_scores(self, texts: List[str], batch_size: int = None) -> List[float]:
        """
        Process texts in batches with progress bar.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Batch size for processing (default: auto-optimized)
            
        Returns:
            List of sentiment scores in range [-1, 1]
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        all_scores = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT Sentiment", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            scores = self.get_sentiment_scores_batch(batch_texts)
            all_scores.extend(scores)
        
        return all_scores