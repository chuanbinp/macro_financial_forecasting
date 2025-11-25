from pydantic import BaseModel, Field

class SentimentResult(BaseModel):
    # score: float = Field(description="Financial sentiment score ranging from -1 (negative) to +1 (positive).")
    explanation: str = Field(description="An explanation of the sentiment score assigned to the news article based on financial context.")