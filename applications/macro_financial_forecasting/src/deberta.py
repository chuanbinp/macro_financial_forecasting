from transformers import pipeline
import torch
import asyncio
import concurrent
from typing import List
from tqdm.asyncio import tqdm_asyncio
from config import Config


class DebertaIndustryClassifier:
    def __init__(self, config: Config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config.deberta_model,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # GICS sector labels
        self.industry_labels = config.industries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    
    def classify_industries_batch(self, texts: List[str]) -> List[str]:
        """
        Classify texts into industry sectors using zero-shot classification.
        Returns the top predicted industry label for each text.
        """
        results = []
        for text in texts:
            # Truncate text to avoid token limits (512 tokens)
            truncated_text = text[:512] if len(text) > 512 else text
            
            result = self.classifier(
                truncated_text,
                self.industry_labels,
                multi_label=False
            )
            
            # Get top label
            results.append(result['labels'][0])
        
        return results
    
    
    # def classify_industries_batch_with_scores(self, texts: List[str]) -> List[dict]:
    #     """
    #     Classify texts into industry sectors and return labels with confidence scores.
    #     Returns list of dicts with 'label' and 'score' keys.
    #     """
    #     results = []
    #     for text in texts:
    #         truncated_text = text[:512] if len(text) > 512 else text
            
    #         result = self.classifier(
    #             truncated_text,
    #             self.industry_labels,
    #             multi_label=False
    #         )
            
    #         # Return top label with confidence score
    #         results.append({
    #             'label': result['labels'][0],
    #             'score': result['scores'][0]
    #         })
        
    #     return results
    
    
    async def async_classify_industries(
        self, 
        texts: List[str], 
        batch_size: int = 128
    ) -> List[str]:
        """
        Asynchronously classify texts into industry sectors.
        Returns list of predicted industry labels.
        """
        all_labels = []
        
        for i in tqdm_asyncio(
            range(0, len(texts), batch_size), 
            desc="Industry Classification"
        ):
            batch_texts = texts[i : i + batch_size]
            labels = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                self.classify_industries_batch, 
                batch_texts
            )
            all_labels.extend(labels)
        
        return all_labels
    
    
    # async def async_classify_industries_with_scores(
    #     self, 
    #     texts: List[str], 
    #     batch_size: int = 32
    # ) -> List[dict]:
    #     """
    #     Asynchronously classify texts with confidence scores.
    #     Returns list of dicts with 'label' and 'score' keys.
    #     """
    #     all_results = []
        
    #     for i in tqdm_asyncio(
    #         range(0, len(texts), batch_size), 
    #         desc="Industry Classification"
    #     ):
    #         batch_texts = texts[i : i + batch_size]
    #         results = await asyncio.get_event_loop().run_in_executor(
    #             self.executor, 
    #             self.classify_industries_batch_with_scores, 
    #             batch_texts
    #         )
    #         all_results.extend(results)
        
    #     return all_results
