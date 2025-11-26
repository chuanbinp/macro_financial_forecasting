import asyncio
from typing import List
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
from finbert import FinBertSentiment
from deberta import DebertaIndustryClassifier
from hf_wrapper import HFInstructorClient

class NewsProcessor:
    def __init__(self, config: Config, concurrency_limit=64, batch_size=10_000):
        self.client = instructor.from_provider(
            config.llm_model,
            api_key=config.openai_api_key,
            async_client=True
        )
        # self.client = HFInstructorClient(model="google/gemma-2-2b")
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.batch_size = batch_size
        self.data_dir = config.dataset_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.prompt_instructions = config.prompt_instructions
        self.finbert = FinBertSentiment(config)
        self.deberta = DebertaIndustryClassifier(config)
    
    def group_by_date_and_industry(self, df: pd.DataFrame, save_path: str = None):
        df = (
            df.groupby(['Industry', 'Date'])
            .apply(lambda x: x.to_dict(orient='records'))
            .reset_index()
            .rename(columns={0: 'News'})
        )
        if save_path and df is not None:
            ParquetUtil.save_df_to_parquet(df, os.path.join(self.data_dir, f"{save_path}"))
        return df
    
    def prepare_impactful_news(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Process grouped news DataFrame to filter and identify impactful news.
        Expects DataFrame from group_by_date_and_industry() with columns: Industry, Date, News
        
        Args:
            df: Grouped DataFrame with 'News' column containing list of article dicts
            save_path: Optional path to save the resulting DataFrame
            
        Returns:
            DataFrame with ImpactfulNews column containing top 2 articles by sentiment
        """
        # 1. Drop rows where Industry == "None" and show count
        none_count = (df['Industry'] == 'None').sum()
        df_filtered = df[df['Industry'] != 'None'].copy()
        
        print(f"\n{'='*60}")
        print(f"Dropped {none_count} (Industry, Date) pairs with Industry='None'")
        print(f"Remaining pairs: {len(df_filtered)}")
        print(f"{'='*60}\n")
        
        # 2. Calculate article counts for each (Industry, Date) pair
        df_filtered['ArticleCount'] = df_filtered['News'].apply(len)
        
        print("Frequency Count by (Industry, Date):")
        freq_display = df_filtered[['Industry', 'Date', 'ArticleCount']].copy()
        print(freq_display.to_string(index=False))
        print(f"\n{'='*60}\n")
        
        # Display summary statistics
        print("Summary Statistics:")
        print(f"Total unique (Industry, Date) pairs: {len(df_filtered)}")
        print(f"Average articles per pair: {df_filtered['ArticleCount'].mean():.2f}")
        print(f"Max articles in a pair: {df_filtered['ArticleCount'].max()}")
        print(f"Min articles in a pair: {df_filtered['ArticleCount'].min()}")
        print(f"Total articles: {df_filtered['ArticleCount'].sum()}")
        print(f"\n{'='*60}\n")
        
        # 3. Create ImpactfulNews column with top 2 articles by absolute sentiment score
        def get_top_impactful(news_list):
            """Extract top 2 articles by absolute sentiment score from news list."""
            if not news_list:
                return []
            
            # Sort by absolute sentiment score
            sorted_news = sorted(
                news_list,
                key=lambda x: abs(x.get('SentimentScore', 0)),
                reverse=True
            )
            
            # Take top 2
            top_2 = sorted_news[:2]
            
            # Extract only needed fields
            impactful_news = [
                {
                    'Headline': article['Headline'],
                    # 'SentimentScore': article['SentimentScore'],
                    'Article': article['Article']
                }
                for article in top_2
            ]
            
            return impactful_news
        
        # Apply to each row
        df_filtered['ImpactfulNews'] = df_filtered['News'].apply(get_top_impactful)
        
        # Calculate average sentiment for each group
        # df_filtered['AvgSentiment'] = df_filtered['News'].apply(
        #     lambda news_list: sum(article.get('SentimentScore', 0) for article in news_list) / len(news_list) if news_list else 0
        # )
        
        # Sort by date and industry
        result = df_filtered.sort_values(['Date', 'Industry']).reset_index(drop=True)
        
        # Keep only necessary columns for final output
        # result = result[['Industry', 'Date', 'ArticleCount', 'AvgSentiment', 'ImpactfulNews']]
        
        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                result, 
                os.path.join(self.data_dir, save_path)
            )

        return result
    
    def enrich_news_entries_with_classifications(
        self, 
        entries: List[BloombergNewsEntry],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Enrich Bloomberg news entries with sentiment scores and industry classifications.
        
        Args:
            entries: List of BloombergNewsEntry pydantic objects
            save_path: Optional path to save the resulting DataFrame as parquet
            
        Returns:
            DataFrame with all entry fields plus SentimentScore and Industry columns
        """
        if not entries:
            return pd.DataFrame()

        # Extract article texts for batch processing
        news_texts = [entry.Headline + "\n\n" + entry.Article for entry in entries]

        print(f"Processing {len(news_texts)} news entries...")

        # Process sequentially (GPU-bound operations)
        sentiment_scores = self.finbert.get_sentiment_scores(news_texts)
        industry_results = self.deberta.classify_industry(news_texts)

        # Convert to DataFrame with all fields
        df = pd.DataFrame([entry.dict() for entry in entries])
        df['SentimentScore'] = sentiment_scores
        df['Industry'] = industry_results

        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df, 
                os.path.join(self.data_dir, save_path)
            )
        
        print(f"Completed processing {len(df)} entries")

        return df
    
    def prepare_impactful_news(
        self, 
        df: pd.DataFrame,
        save_path: str = None
        ) -> pd.DataFrame:
        """
        Process news DataFrame to filter, analyze, and identify impactful news.

        Args:
            df: DataFrame with columns: Industry, Date, SentimentScore, and news fields
            save_path: Optional path to save the resulting DataFrame
            
        Returns:
            DataFrame grouped by (Industry, Date) with ImpactfulNews column
        """
        # 1. Drop rows where Industry == "None" and show count
        none_count = (df['Industry'] == 'None').sum()
        df_filtered = df[df['Industry'] != 'None'].copy()

        print(f"\n{'='*60}")
        print(f"Dropped {none_count} articles with Industry='None'")
        print(f"Remaining articles: {len(df_filtered)}")
        print(f"{'='*60}\n")

        # 2. Get frequency count for each (Industry, Date) pair
        frequency = df_filtered.groupby(['Industry', 'Date']).size().reset_index(name='ArticleCount')

        print("Frequency Count by (Industry, Date):")
        print(frequency.to_string(index=False))
        print(f"\n{'='*60}\n")

        # Display summary statistics
        print("Summary Statistics:")
        print(f"Total unique (Industry, Date) pairs: {len(frequency)}")
        print(f"Average articles per pair: {frequency['ArticleCount'].mean():.2f}")
        print(f"Max articles in a pair: {frequency['ArticleCount'].max()}")
        print(f"Min articles in a pair: {frequency['ArticleCount'].min()}")
        print(f"\n{'='*60}\n")

        # 3. Create ImpactfulNews column with top 2 articles by absolute sentiment score
        def get_top_impactful(group):
            # Sort by absolute sentiment score (most extreme sentiment = most impactful)
            group_sorted = group.sort_values(
                by='SentimentScore', 
                key=lambda x: abs(x),  # Sort by absolute value
                ascending=False
            )
            
            # Take top 2
            top_2 = group_sorted.head(2)
            
            # Create list of impactful news with key fields
            impactful_news = top_2[[
                'Headline', 
                'Article'
            ]].to_dict('records')
            
            return pd.Series({
                'ImpactfulNews': impactful_news,
            })

        # Group and create final DataFrame
        result = df_filtered.groupby(['Industry', 'Date']).apply(get_top_impactful).reset_index()

        # Sort by date and industry
        result = result.sort_values(['Date', 'Industry']).reset_index(drop=True)

        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                result, 
                os.path.join(self.data_dir, save_path)
            )
            print(f"✓ Saved to {save_path}\n")

        return result
    
    # async def summarize_news(self, news_text: str, prompt: str = None) -> str:
    #     if prompt is None:
    #         prompt = self.prompt_instructions["summarize_daily"]
    #     prompt += f"\nArticle:\n{news_text}"
    #     async with self.semaphore:
    #         result = await self.client.chat.completions.create(
    #             response_model=NewsSummary,
    #             messages=[{"role": "user", "content": prompt}],
    #             max_retries=3
    #         )
    #     return result.summary

    # async def extract_entry(self, entry: BloombergNewsEntry, prompt: str) -> BloombergNewsEntry:
    #     message_content = (
    #         f"{prompt}\n\n"
    #         f'Headline: "{entry.Headline}"\n'
    #         f'Date: "{entry.Date}"\n'
    #         f'Link: "{entry.Link}"\n'
    #         f'Article: """\n{entry.Article}\n"""\n'
    #     )
    #     async with self.semaphore:
    #         extracted = await self.client.chat.completions.create(
    #             response_model=IndustryAndKeyPoints,
    #             messages=[{"role": "user", "content": message_content}],
    #             max_retries=3
    #         )
    #     entry.Industry = extracted.Industry
    #     entry.KeyPoints = extracted.KeyPoints
    #     return entry

    # async def transduce_news_entries_async(self, entries: List[BloombergNewsEntry], prompt: str = None, save_path_prefix: str = None):
    #     if prompt is None:
    #         prompt = self.prompt_instructions["classify_and_keypoints"]
    #     tasks = [self.extract_entry(entry, prompt) for entry in entries]
    #     results = []
    #     batch_number = 1

    #     for coro in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks), desc="Processing news", unit="entry"):
    #         result = await coro
    #         results.append(result)

    #         # Save in batches
    #         if save_path_prefix and len(results) % self.batch_size == 0:
    #             batch_filename = os.path.join(self.data_dir, f"{save_path_prefix}_batch_{batch_number}")
    #             ParquetUtil.save_pydantic_to_parquet(results, batch_filename)
    #             results = []
    #             batch_number += 1

    #     # Save any remaining entries after loop
    #     if save_path_prefix and results:
    #         batch_filename = os.path.join(self.data_dir, f"{save_path_prefix}_batch_{batch_number}")
    #         ParquetUtil.save_pydantic_to_parquet(results, batch_filename)

    #     return results

    # async def sentiment_explanation(
    #     self,
    #     industry: str,
    #     summary: str,
    #     finbert_score: float = None,
    #     gm_news: str = None,
    #     prompt: str = None
    # ) -> str:
    #     # Compose prompt for explanation with precomputed finbert_score
    #     if prompt is None:
    #         prompt = self.prompt_instructions["sentiment_explanation"]

    #     prompt += "\n"
    #     prompt += f"\nIndustry:\n{industry}"
    #     prompt += f"\nIndustry Summary:\n{summary}"
    #     if finbert_score is not None:
    #         prompt += f"\nFinBERT Score:\n{finbert_score:.3f}"
    #     if gm_news:
    #         prompt += "\n"
    #         prompt += f"\nTake into account the general market news for the same date to further inform your sentiment analysis.\n"
    #         prompt += f"\nGeneral Market News (same date):\n{gm_news}\n"

    #     async with self.semaphore:
    #         result = await self.client.chat.completions.create(
    #             response_model=SentimentResult,
    #             messages=[{"role": "user", "content": prompt}],
    #             max_retries=3
    #         )
    #     return result.explanation


    # async def process_dataframe(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    #     # 1. Sort by date
    #     df = df.sort_values("Date").reset_index(drop=True)

    #     # 2. Summarize asynchronously with order preserved
    #     summarize_tasks = [self.summarize_news(news_text) for news_text in df["News"]]
    #     summaries = await tqdm_asyncio.gather(*summarize_tasks, desc="Summarizing")
    #     df["Summary"] = summaries

    #     # 3. Batch FinBERT sentiment scoring
    #     news_list = [str(text) for text in df["News"].tolist()]
    #     finbert_scores = await self.batch_finbert_sentiment_scores(news_list)
    #     df["SentimentScore"] = finbert_scores

    #     # 4. Prepare general market news dict
    #     gm_by_date = df[df["Industry"] == "General Market"].groupby("Date")["Summary"].first().to_dict()

    #     # 5. Generate sentiment explanations preserving order
    #     sentiment_tasks = []
    #     for _, row in df.iterrows():
    #         gm_news = gm_by_date.get(row["Date"])
    #         sentiment_tasks.append(self.sentiment_explanation(
    #             row["Industry"],
    #             row["Summary"],
    #             finbert_score=row["SentimentScore"],
    #             gm_news=gm_news
    #         ))

    #     explanations = await tqdm_asyncio.gather(*sentiment_tasks, desc="Explanation")
    #     df["SentimentExplanation"] = explanations

    #     if save_path and df is not None:
    #         ParquetUtil.save_df_to_parquet(df, os.path.join(self.data_dir, f"{save_path}"))

    #     return df