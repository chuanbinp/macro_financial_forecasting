from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import Config
import asyncio
import concurrent
from typing import List
from tqdm.asyncio import tqdm_asyncio 

class FinBertSentiment:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.finbert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.finbert_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def get_sentiment_scores_batch(self, texts: List[str]) -> List[float]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        scores = []
        for prob in probs:
            probs_dict = {self.id2label[i]: float(prob[i]) for i in range(len(prob))}
            score = probs_dict["positive"] - probs_dict["negative"]
            scores.append(score)
        return scores
    
    async def async_get_sentiment_scores(self, texts: List[str], batch_size: int = 128) -> List[float]:
        all_scores = []
        # process texts in batches
        for i in tqdm_asyncio(range(0, len(texts), batch_size), desc="FinBERT Sentiment"):
            batch_texts = texts[i : i + batch_size]
            scores = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.get_sentiment_scores_batch, batch_texts
            )
            all_scores.extend(scores)
        return all_scores