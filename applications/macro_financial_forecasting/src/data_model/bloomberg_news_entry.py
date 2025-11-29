from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from src.config import Config

config = Config("config.env")
IndustryType = Literal[tuple(config.industries)]

class BloombergNewsEntry(BaseModel):
    Headline: str = Field(description="Title or headline of the news article.")
    Date: str = Field(description="Publication timestamp of the article (in UTC).")
    Link: str = Field(description="URL link to the full article.")
    Article: str = Field( description="Full article text content.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Headline": "Haunted Greeks Sell Real Estate EBay-Style to Evict Debt Specter",
                "Date": "2013-09-05T21:01:00Z",
                "Link": "http://www.bloomberg.com/news/2013-09-05/haunted-greeks-sell-real-estate-ebay-style-to-evict-debt-specter.html",
                "Article": "A legend that has swirled around the dilapidated mansion on Smolenski Street in Athens..."
            }
        }