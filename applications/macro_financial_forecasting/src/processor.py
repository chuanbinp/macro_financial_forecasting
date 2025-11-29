import asyncio
from typing import List
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import instructor
import pandas as pd
import os
import re

from src.config import Config
from src.data_model.bloomberg_news_entry import BloombergNewsEntry
from src.data_model.bloomberg_news_sentiment_explanation import SentimentResult
from src.utils.pydantic_parquet_util import ParquetUtil
from src.finbert import FinBertSentiment
from src.deberta import DebertaIndustryClassifier

class NewsProcessor:
    def __init__(self, config: Config, concurrency_limit=64, batch_size=10_000):
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
        self.finbert = FinBertSentiment(config)
        self.deberta = DebertaIndustryClassifier(config)

    def remove_redundant_info(self, entries: List[BloombergNewsEntry]) -> List[BloombergNewsEntry]:
        # Matches: To contact the editor(s)/reporter(s)...
        pattern = re.compile(r"To contact the (editor|editors|reporter|reporters).*", re.DOTALL)

        for entry in entries:
            # Remove everything from the boilerplate onward
            entry.Article = re.sub(pattern, "", entry.Article).strip()

        return entries
    
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
    
    def filter_and_analyze_news(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Filter out 'None' industries and display statistics about the grouped news data.
        
        Args:
            df: Grouped DataFrame with columns: Industry, Date, News
            save_path: Optional path to save the filtered DataFrame
            
        Returns:
            Filtered DataFrame with ArticleCount column added
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
        
        # print("Frequency Count by (Industry, Date):")
        # freq_display = df_filtered[['Industry', 'Date', 'ArticleCount']].copy()
        # print(freq_display.to_string(index=False))
        # print(f"\n{'='*60}\n")
        
        # Display summary statistics
        print("Summary Statistics:")
        print(f"Total unique (Industry, Date) pairs: {len(df_filtered)}")
        print(f"Average articles per pair: {df_filtered['ArticleCount'].mean():.2f}")
        print(f"Max articles in a pair: {df_filtered['ArticleCount'].max()}")
        print(f"Min articles in a pair: {df_filtered['ArticleCount'].min()}")
        print(f"25th percentile: {df_filtered['ArticleCount'].quantile(0.25)}")
        print(f"50th percentile: {df_filtered['ArticleCount'].quantile(0.5)}")
        print(f"75th percentile: {df_filtered['ArticleCount'].quantile(0.75)}")
        print(f"Number of pairs with at least 3 articles: {(df_filtered['ArticleCount'] >= 3).sum()}")
        print(f"Total articles: {df_filtered['ArticleCount'].sum()}")
        print(f"\n{'='*60}\n")
        
        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df_filtered, 
                os.path.join(self.data_dir, save_path)
            )
        
        return df_filtered

    def extract_impactful_news(
        self,
        df: pd.DataFrame,
        top_n: int = 3,
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Extract top N most impactful news articles for each (Industry, Date) pair.
        Articles are ranked by absolute sentiment score.
        
        Args:
            df: DataFrame with 'News' column containing list of article dicts
            top_n: Number of top articles to extract (default: 3)
            save_path: Optional path to save the resulting DataFrame
            
        Returns:
            DataFrame with ImpactfulNews column containing top N articles
        """
        df_result = df.copy()
        
        def get_top_impactful(news_list):
            """Extract top N articles by absolute sentiment score from news list."""
            if not news_list:
                return []
            
            # Sort by absolute sentiment score
            sorted_news = sorted(
                news_list,
                key=lambda x: abs(x.get('SentimentScore', 0)),
                reverse=True
            )
            
            # Take top N
            top_articles = sorted_news[:top_n]
            
            # Extract only needed fields
            impactful_news = [
                {
                    'Headline': article['Headline'],
                    'Article': article['Article'],
                    'SentimentScore': article['SentimentScore']
                }
                for article in top_articles
            ]

            # Calculate average sentiment of impactful news
            avg_sentiment = sum(article['SentimentScore'] for article in top_articles) / len(top_articles)
            
            return {
                'impactful_news': impactful_news,
                'avg_sentiment': avg_sentiment
            }
        
        # Apply to each row with progress bar
        print(f"Extracting top {top_n} impactful news per (Industry, Date) pair...")
        tqdm.pandas(desc="Processing groups")
        results = df_result['News'].progress_apply(get_top_impactful)
        df_result['ImpactfulNews'] = results.apply(lambda x: x['impactful_news'])
        df_result['AvgSentimentScore'] = results.apply(lambda x: x['avg_sentiment'])
        
        # Sort by date and industry
        df_result = df_result.sort_values(['Date', 'Industry']).reset_index(drop=True)
        
        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df_result, 
                os.path.join(self.data_dir, save_path)
            )
        
        return df_result
    
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
        df = pd.DataFrame([entry.dict() for entry in entries])

        sentiment_scores = self.finbert.get_sentiment_scores(news_texts)
        df['SentimentScore'] = sentiment_scores
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df, 
                os.path.join(self.data_dir, save_path)
            )

        industry_results = self.deberta.classify_industry(news_texts)
        df['Industry'] = industry_results
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df, 
                os.path.join(self.data_dir, save_path)
            )
        
        print(f"Completed processing {len(df)} entries")

        return df
    
    def get_consolidated_sentiment(
        self, 
        df: pd.DataFrame,
        save_path: str = None
    ) -> pd.DataFrame:
    
        if df.empty:
            return pd.DataFrame()

        # Extract article texts for batch processing
        news_texts = [
            "\n".join(
                entry["Headline"] + ". " + entry["Article"][:2048 // len(row)]
                for entry in row
            )
            for row in df["ImpactfulNews"]
            if len(row) > 0
        ]

        print(f"Processing {len(news_texts)} news entries...")

        # Process sequentially (GPU-bound operations)
        sentiment_scores = self.finbert.get_sentiment_scores(news_texts)
        df['SentimentScore'] = sentiment_scores

        # Save if path provided
        if save_path:
            ParquetUtil.save_df_to_parquet(
                df, 
                os.path.join(self.data_dir, save_path)
            )
        
        print(f"Completed processing {len(df)} entries")

        return df
    
    async def sentiment_explanation(
        self,
        industry: str,
        impact_news: str,
        combined_finbert_score: float = None,
        avg_finbert_score: float = None,
        gm_news: str = None,
        prompt: str = None
    ) -> str:
        # Compose prompt for explanation with precomputed finbert_score
        if prompt is None:
            prompt = self.prompt_instructions["sentiment_explanation"]

        prompt += "\n"
        prompt += f"\nIndustry:\n{industry}"
        prompt += f"\nNews:\n{impact_news}"
        if combined_finbert_score is not None and avg_finbert_score is not None:
            prompt += f"\nCombined/Average FinBERT (-1 to +1):\n{combined_finbert_score:.3f}/{avg_finbert_score:.3f}"
        if gm_news:
            prompt += "\n"
            prompt += f"\nTake into account the general market news to further inform your sentiment analysis.\n"
            prompt += f"\nGeneral Market News (same date):\n{gm_news}\n"

        async with self.semaphore:
            result = await self.client.chat.completions.create(
                response_model=SentimentResult,
                messages=[{"role": "user", "content": prompt}],
                max_retries=3
            )
        return result.explanation

    async def get_explanation(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        # 1. Prepare general market news dict
        gm_by_date = df[df["Industry"] == "General Market"].groupby("Date")["ImpactfulNews"].first().to_dict()

        # 2. Generate sentiment explanations preserving order
        sentiment_tasks = []
        for _, row in df.iterrows():
            gm_news = gm_by_date.get(row["Date"])
            sentiment_tasks.append(self.sentiment_explanation(
                row["Industry"],
                row["ImpactfulNews"],
                combined_finbert_score=row["SentimentScore"],
                avg_finbert_score=row["AvgSentimentScore"],
                gm_news=gm_news
            ))

        explanations = await tqdm_asyncio.gather(*sentiment_tasks, desc="Explanation")
        df["SentimentExplanation"] = explanations

        if save_path and df is not None:
            ParquetUtil.save_df_to_parquet(df, os.path.join(self.data_dir, f"{save_path}"))

        return df

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