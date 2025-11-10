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
        self.dataset_dir: str = os.getenv("DATASET_DIR", "../data")

        # RSS Feeds - split by comma
        feeds_str = os.getenv("RSS_FEEDS", "")
        self.rss_feeds: List[str] = [f.strip() for f in feeds_str.split(",") if f.strip()]

        # Prompt Instructions
        self.prompt_instructions: str = f'''You are a financial news analyst.
1. Read the article carefully and classify its **primary industry sector** as "Industry".

Industries must be one of:
{self.industries}

Guidelines:
- Choose **"General Market"** if the article covers overall economic conditions,
  government or central bank policies, currency movements, inflation, GDP,
  interest rates, IMF or World Bank decisions, or broad market sentiment that
  affects multiple sectors rather than one specific industry.
- Choose **"None"** if the have no financial effects to any sector or general market.
- If the article focuses on one company, classify it based on that company’s core sector.

2. Then, summarize the **5 most important points** of the article as "KeyPoints",
each starting with a bullet ("-").
'''