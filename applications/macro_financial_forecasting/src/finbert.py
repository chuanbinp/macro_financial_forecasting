from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import Config

class FinBertSentiment:
    def __init__(self, config: Config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.finbert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.finbert_model)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def get_sentiment_score(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        probs_dict = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        # Score from -1 (neg) to +1 (pos)
        score = probs_dict["positive"] - probs_dict["negative"]
        return score