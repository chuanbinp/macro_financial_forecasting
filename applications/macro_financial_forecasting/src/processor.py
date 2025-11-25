import asyncio
from typing import List, Tuple
from tqdm.asyncio import tqdm_asyncio
import instructor
import pandas as pd
import os

from config import Config
from data_model.bloomberg_news_entry import BloombergNewsEntry
from data_model.bloomberg_news_industry_and_keypoints import IndustryAndKeyPoints
from data_model.bloomberg_news_summary import NewsSummary
from data_model.bloomberg_news_sentiment_explanation import SentimentResult
from utils.pydantic_parquet_util import ParquetUtil

class NewsProcessor:
    def __init__(self, config: Config, concurrency_limit=32, batch_size=10_000):
        self.client = instructor.from_provider(
            config.llm_model,
            api_key=config.openai_api_key,
            async_client=True
        )
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.batch_size = batch_size
        self.data_dir = config.dataset_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.prompt_instructions = config.prompt_instructions

    async def extract_entry(self, entry: BloombergNewsEntry, prompt: str) -> BloombergNewsEntry:
        message_content = (
            f"{prompt}\n\n"
            f'Headline: "{entry.Headline}"\n'
            f'Date: "{entry.Date}"\n'
            f'Link: "{entry.Link}"\n'
            f'Article: """\n{entry.Article}\n"""\n'
        )
        async with self.semaphore:
            extracted = await self.client.chat.completions.create(
                response_model=IndustryAndKeyPoints,
                messages=[{"role": "user", "content": message_content}],
                max_retries=3
            )
        entry.Industry = extracted.Industry
        entry.KeyPoints = extracted.KeyPoints
        return entry

    async def transduce_news_entries_async(self, entries: List[BloombergNewsEntry], prompt: str = None, save_path_prefix: str = None):
        if prompt is None:
            prompt = self.prompt_instructions["classify_and_keypoints"]
        tasks = [self.extract_entry(entry, prompt) for entry in entries]
        results = []
        batch_number = 1

        for coro in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="Processing news", unit="entry"):
            result = await coro
            results.append(result)

            # Save in batches
            if save_path_prefix and len(results) % self.batch_size == 0:
                batch_filename = os.path.join(self.data_dir, f"{save_path_prefix}_batch_{batch_number}.parquet")
                ParquetUtil.save_pydantic_to_parquet(results, batch_filename)
                results = []
                batch_number += 1

        # Save any remaining entries after loop
        if save_path_prefix and results:
            batch_filename = os.path.join(self.data_dir, f"{save_path_prefix}_batch_{batch_number}.parquet")
            ParquetUtil.save_pydantic_to_parquet(results, batch_filename)

        return results
    
    def group_by_date_and_industry(self, entries: List[BloombergNewsEntry], save_path: str = None):
        df = pd.DataFrame([entry.dict() for entry in entries])
        df = (
            df.groupby(['Industry', 'Date'])
            .apply(lambda x: x.to_dict(orient='records'))
            .reset_index()
            .rename(columns={0: 'News'})
        )
        if save_path and df is not None:
            ParquetUtil.save_df_to_parquet(df, os.path.join(self.data_dir, f"{save_path}.parquet"))
        return df
    
    async def summarize_news(self, news_text: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = self.prompt_instructions["summarize_daily"]
        prompt += f"\nArticle:\n{news_text}"
        async with self.semaphore:
            result = await self.client.chat.completions.create(
                response_model=NewsSummary,
                messages=[{"role": "user", "content": prompt}],
                max_retries=3
            )
        return result.summary
    
    async def sentiment_and_explanation(self, news: str, gm_news: str = None, prompt: str = None) -> Tuple[float, str]:
        if prompt is None:
            prompt = self.prompt_instructions["sentiment_and_explanation"]
        prompt += f"\nArticle:\n{news}"
        if gm_news:
            prompt += f"\nGeneral Market News (same date):\n{gm_news}\n"

        async with self.semaphore:
            result = await self.client.chat.completions.create(
                response_model=SentimentResult,
                messages=[{"role": "user", "content": prompt}],
                max_retries=3
            )
        return result.score, result.explanation

    async def process_dataframe(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        # 1. Sort by date
        df = df.sort_values("Date").reset_index(drop=True)

        # 2. Summarize
        summarize_tasks = [
            self.summarize_news(news_text)
            for news_text in df["News"]
        ]
        summaries = []
        for coro in tqdm_asyncio(asyncio.as_completed(summarize_tasks), total=len(summarize_tasks), desc="Summarizing"):
            summaries.append(await coro)
        df["Summary"] = summaries

        # 3. Sentiment and explanation
        # Precompute General Market dict by date
        gm_by_date = df[df["Industry"] == "General Market"].groupby("Date")["News"].first().to_dict()
        sentiment_tasks = []
        for _, row in df.iterrows():
            gm_news = gm_by_date.get(row["Date"])
            sentiment_tasks.append(self.sentiment_and_explanation(row["News"], gm_news))
        
        sentiments, explanations = [], []
        for coro in tqdm_asyncio(asyncio.as_completed(sentiment_tasks), total=len(sentiment_tasks), desc="Sentiment & Explanation"):
            score, explanation = await coro
            sentiments.append(score)
            explanations.append(explanation)
        
        df["SentimentScore"] = sentiments
        df["SentimentExplanation"] = explanations

        if save_path and df is not None:
            ParquetUtil.save_df_to_parquet(df, os.path.join(self.data_dir, f"{save_path}.parquet"))

        return df