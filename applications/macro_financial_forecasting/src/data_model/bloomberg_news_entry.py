from typing import List, Optional, Literal
from pydantic import BaseModel, Field, HttpUrl
from config import Config

config = Config("../config.env")

class BloombergNewsEntry(BaseModel):
    Headline: str = Field(description="Title or headline of the news article.")
    # Journalists: List[str] = Field(default_factory=list, description="List of journalists credited for the article.")
    Date: str = Field(description="Publication timestamp of the article (in UTC).")
    Link: str = Field(description="URL link to the full article.")
    Article: str = Field( description="Full article text content.")
    # New additions
    Industry: Optional[str] = Field(None, description=f"The primary industry sector this news is relevant to. Must be one of: {config.industries}.")
    KeyPoints: Optional[str] = Field(None, description="A bullet list summarizing the 5 most important points of the news article.")

    class Config:
        json_schema_extra = {
            "example": {
                "Headline": "Haunted Greeks Sell Real Estate EBay-Style to Evict Debt Specter",
                "Journalists": ["Maria Petrakis"],
                "Date": "2013-09-05T21:01:00Z",
                "Link": "http://www.bloomberg.com/news/2013-09-05/haunted-greeks-sell-real-estate-ebay-style-to-evict-debt-specter.html",
                "Article": "A legend that has swirled around the dilapidated mansion on Smolenski Street in Athens..."
            }
        }
