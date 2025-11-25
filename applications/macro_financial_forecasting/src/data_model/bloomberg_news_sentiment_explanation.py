from pydantic import BaseModel, Field

class SentimentResult(BaseModel):
    explanation: str = Field(description="An explanation of the sentiment score assigned to the news article based on financial context.")