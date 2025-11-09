import os
from dotenv import load_dotenv
from typing import List, Optional

class Config:
    def __init__(self, env_path: Optional[str] = ".env"):
        load_dotenv(dotenv_path=env_path)

        # LLM
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-5-nano-2025-08-07")

        # Industry List
        industries_str = os.getenv("INDUSTRIES", "")
        self.industries: List[str] = [i.strip() for i in industries_str.split(",") if i.strip()]

        # Training Dataset
        self.dataset_name: str = os.getenv("DATASET_NAME", "danidanou/BloombergFinancialNews")
        self.split_name: str = os.getenv("SPLIT_NAME", "train")

        # RSS Feeds - split by comma
        feeds_str = os.getenv("RSS_FEEDS", "")
        self.rss_feeds: List[str] = [f.strip() for f in feeds_str.split(",") if f.strip()]