from typing import List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl

# Read industry list dynamically in your app from env (see main.py)
# Define IndustryType dynamically (in main.py or loader)

class BloombergNewsEntry(BaseModel):
    Headline: str = Field(description="Title or headline of the news article.")
    Journalists: List[str] = Field(default_factory=list, description="List of journalists credited for the article.")
    Date: str = Field(description="Publication timestamp of the article in ISO 8601 format.")
    Link: HttpUrl = Field(description="URL link to the full article.")
    Article: str = Field(description="Full article text content.")
    Industry: Optional[str] = Field(default=None, description="Primary industry sector from allowed list.")
    KeyPoints: Optional[str] = Field(default=None, description="Summary bullet points of key insights.")
