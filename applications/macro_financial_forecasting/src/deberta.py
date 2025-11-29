from transformers import pipeline
import torch
from typing import List
from src.config import Config
from tqdm import tqdm

class DebertaIndustryClassifier:
    def __init__(self, config: Config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config.deberta_model,
            device=0 if torch.cuda.is_available() else -1,
            batch_size=32
        )
        self.industry_labels = config.industries
        self.max_chars = 1800 # Max chars to avoid token limit issues 
        self.optimal_batch_size = 32 if torch.cuda.is_available() else 8
    
    
    def _truncate_texts(self, texts: List[str]) -> List[str]:
        """Truncate texts to avoid token limit issues."""
        return [
            text[:self.max_chars] if len(text) > self.max_chars else text 
            for text in texts
        ]
    
    
    def classify_batch(self, texts: List[str]) -> List[str]:
        """
        Classify a batch of texts into industry sectors.
        
        Args:
            texts: List of text strings (will be truncated automatically)
            
        Returns:
            List of predicted industry labels
        """
        if not texts:
            return []
        
        # Truncate texts
        truncated_texts = self._truncate_texts(texts)
        
        # Process all texts at once (pipeline handles internal batching)
        results = self.classifier(
            truncated_texts,
            self.industry_labels,
            multi_label=False
        )
        
        # Handle both single and multiple text results
        if isinstance(results, dict):  # Single text
            return [results['labels'][0]]
        else:  # Multiple texts
            return [result['labels'][0] for result in results]
        
    def classify_industry(
        self, 
        texts: List[str], 
        batch_size: int = None
    ) -> List:
        """
        Classify all texts with progress bar.
        
        Args:
            texts: List of text strings to classify
            batch_size: Batch size for processing (default: auto-optimized)
            return_scores: If True, returns dicts with labels and scores
            
        Returns:
            List of industry labels or dicts with labels and scores
        """
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        all_results = []
        
        # Process in batches with progress bar
        for i in tqdm(
            range(0, len(texts), batch_size), 
            desc="Industry Classification",
            unit="batch"
        ):
            batch_texts = texts[i : i + batch_size]
            results = self.classify_batch(batch_texts)
            all_results.extend(results)
        
        return all_results