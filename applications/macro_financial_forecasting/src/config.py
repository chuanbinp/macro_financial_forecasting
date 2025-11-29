import os
from dotenv import load_dotenv
from typing import Dict, List, Optional

class Config:
    def __init__(self, env_path: Optional[str] = ".env"):
        load_dotenv(dotenv_path=env_path)

        # LLM
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-5-nano-2025-08-07")
        self.langgraph_model: str = os.getenv("LANGGRAPH_MODEL", "gpt-5-nano")

        # Industry List
        industries_str = os.getenv("INDUSTRIES", "")
        self.industries: List[str] = [i.strip() for i in industries_str.split(",") if i.strip()]

        # Training Dataset
        self.dataset_name: str = os.getenv("DATASET_NAME", "danidanou/BloombergFinancialNews")
        self.dataset_dir: str = os.getenv("DATASET_DIR", "../data")

        # RSS Feeds - split by comma
        feeds_str = os.getenv("RSS_FEEDS", "")
        self.rss_feeds: List[str] = [f.strip() for f in feeds_str.split(",") if f.strip()]

        self.finbert_model: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
        self.deberta_model: str = os.getenv("DEBERTA_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

        # Prompt Instructions
        self.prompt_instructions: Dict[str] = {
"sentiment_explanation": f'''You are a financial news analyst.
Think about how the news article impacts financial markets, especially in the Industry specified.
Use your knowledge of economics, market trends, and investor behavior.

Given the following news article, produce a brief explanation of your score as "Explanation".
'''}
    def __str__(self):
        return (
            f"Config(\n"
            f"  gemini_api_key: !secret!\n"
            f"  openai_api_key: !secret!\n"
            f"  llm_model: {self.llm_model}\n"
            f"  industries: {self.industries}\n"
            f"  dataset_name: {self.dataset_name}\n"
            f"  dataset_dir: {self.dataset_dir}\n"
            f"  rss_feeds: {self.rss_feeds}\n"
            f"  prompt_instructions: {self.prompt_instructions['sentiment_explanation'][:100]}... (truncated)\n"
            f")"
        )