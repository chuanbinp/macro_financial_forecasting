from typing import List, Dict, Any
from pydantic import BaseModel
from langchain.tools import tool

from config import Config
from processor import NewsProcessor
from data_model.bloomberg_news_entry import BloombergNewsEntry

_processor_instance = None

def get_processor(config: Config) -> NewsProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = NewsProcessor(config)
    return _processor_instance

class ProcessNewsInput(BaseModel):
    data: List[BloombergNewsEntry]

@tool(args_schema=ProcessNewsInput)
async def process_bloomberg_news(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run the full NewsProcessor pipeline on raw Bloomberg RSS feed entries.
    Input: List of dicts with Headline, Link, Article, Date
    Output: Processed dataframe converted to list[dict]
    """
    from config import Config
    from processor import NewsProcessor

    config = Config()
    processor = get_processor(config)

    # Pipeline
    df = processor.enrich_news_entries_with_classifications(data)
    df = processor.group_by_date_and_industry(df)
    df = processor.filter_and_analyze_news(df)
    df = processor.extract_impactful_news(df, top_n=3)
    df = processor.get_consolidated_sentiment(df)
    df = await processor.get_explanation(df)

    return df.to_dict(orient="records")