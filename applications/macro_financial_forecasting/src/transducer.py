import asyncio
from typing import List
from tqdm.asyncio import tqdm_asyncio
import instructor

from config import Config
from data_model.bloomberg_news_entry import BloombergNewsEntry
from data_model.bloomberg_news_industry_and_keypoints import IndustryAndKeyPoints
from utils.pydantic_parquet_util import PydanticParquetUtil

class NewsTransducer:
    def __init__(self, config: Config, concurrency_limit=32, batch_size=10_000):
        self.client = instructor.from_provider(
            config.llm_model,
            api_key=config.openai_api_key,
            async_client=True
        )
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.batch_size = batch_size

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

    async def process_news_entries_async(self, entries: List[BloombergNewsEntry], prompt: str):
        tasks = [self.extract_entry(entry, prompt) for entry in entries]
        results = []
        for coro in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="Processing news", unit="entry"):
            result = await coro
            results.append(result)
        return results

    async def process_news_entries_async(self, entries: List[BloombergNewsEntry], prompt: str, save_path_prefix: str = None):
        tasks = [self.extract_entry(entry, prompt) for entry in entries]
        results = []
        batch_number = 1

        for coro in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="Processing news", unit="entry"):
            result = await coro
            results.append(result)

            # Save in batches
            if save_path_prefix and len(results) % self.batch_size == 0:
                batch_filename = f"{save_path_prefix}_batch_{batch_number}.parquet"
                PydanticParquetUtil.save_to_parquet(results, batch_filename)
                results = []
                batch_number += 1

        # Save any remaining entries after loop
        if save_path_prefix and results:
            batch_filename = f"{save_path_prefix}_batch_{batch_number}.parquet"
            PydanticParquetUtil.save_to_parquet(results, batch_filename)

        return results