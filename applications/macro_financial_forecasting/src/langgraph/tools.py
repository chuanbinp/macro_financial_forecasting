from datetime import datetime, timedelta, timezone
import feedparser
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

from src.data_model.bloomberg_news_entry import BloombergNewsEntry
from src.config import Config
from src.processor import NewsProcessor

## Singleton processor instance
def get_processor(config: Config) -> NewsProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = NewsProcessor(config)
    return _processor_instance

config = Config()
_processor_instance = None
get_processor(config)

## Functions to be used in the pipeline
def get_bloomberg_rss_feeds(days: int = 1) -> List[Dict[str, str]]:
    """Fetch all Bloomberg RSS news for the last N days. One call is enough."""

    feeds = config.rss_feeds[:3]

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    news = []

    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if hasattr(entry, "published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                if published > cutoff and hasattr(entry, 'summary'):
                    news.append({
                        "Headline": entry.title,
                        "Link": entry.link,
                        "Article": entry.summary,
                        "Date": datetime.strptime(published.isoformat(), "%Y-%m-%dT%H:%M:%S%z").date().strftime("%Y-%m-%d"),
                    })
    
    return [BloombergNewsEntry.model_validate(record) for record in tqdm(news, desc="Validating entries")]

def get_mock_bloomberg_rss_feeds() -> List[Dict[str, str]]:
    """Fetch mock Bloomberg RSS news"""
    data = [{'Headline': 'New Fortress Gets Tentative OK for $3 Billion Puerto Rico Deal',
        'Date': '2025-11-29',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/new-fortress-gets-tentative-ok-for-3-billion-puerto-rico-deal',
        'Article': 'Regulators in Puerto Rico tentatively approved a contentious deal with billionaire Wes Edens’ New Fortress Energy Inc. to supply liquefied natural gas to the US territory.'},
        {'Headline': 'Brazil Judge Frees Banco Master’s Vorcaro With Ankle Monitor',
        'Date': '2025-11-29',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/brazil-judge-frees-banco-master-s-vorcaro-with-ankle-monitor',
        'Article': 'A Brazilian judge has ordered the release of Daniel Vorcaro, the controlling shareholder of the failed Banco Master SA, who was detained last week.'},
        {'Headline': 'Trump to Pardon Ex-Honduras Leader Convicted of Drug Trafficking',
        'Date': '2025-11-29',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/trump-to-pardon-ex-honduras-leader-convicted-of-drug-trafficking',
        'Article': 'President Donald Trump said he plans to pardon a former president of Honduras who’s serving a decades-long US sentence for cocaine trafficking, two days before the nation’s election.'},
        {'Headline': 'Human Made Seeks Growth Beyond China as Tensions With Japan Rise',
        'Date': '2025-11-29',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/human-made-seeks-growth-beyond-china-as-tensions-with-japan-rise',
        'Article': 'Human Made Inc., the Japanese fashion label that counts China as central to its plans, is broadening its growth strategy to reduce reliance on any single market against the backdrop of rising tensions between Tokyo and Beijing.'},
        {'Headline': 'China’s Robotics Stocks Face Investor Scrutiny Over Bubble Fears',
        'Date': '2025-11-29',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/china-s-robotics-stocks-face-investor-scrutiny-over-bubble-fears',
        'Article': 'Investor hype over Chinese robotics stocks is giving way to deepening unease, with the latest government warning against a potential bubble bringing fresh scrutiny to the sector’s lofty valuations.'},
        {'Headline': 'Worldwide Markets Roiled by Data-Center Snafu in Chicago Suburb',
        'Date': '2025-11-28',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-28/worldwide-markets-roiled-by-data-center-snafu-in-chicago-suburb',
        'Article': 'One of the first signs of trouble arrived at 9:41 p.m. Eastern time on Thursday, when most of Wall Street was shut and traders were still enjoying the Thanksgiving holiday in the US.'},
        {'Headline': 'Nigeria’s Cardoso Opens Door to Resuming Rate Cuts in 2026',
        'Date': '2025-11-28',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-28/nigeria-s-cardoso-opens-door-to-resuming-rate-cuts-in-2026',
        'Article': 'Nigerian central bank Governor Olayemi Cardoso hinted that policymakers could resume interest-rate cuts next year, provided inflation continues to cool as expected.'},
        {'Headline': 'CME Trading Is Restored to Wrap Up Week After Hours-Long Outage',
        'Date': '2025-11-28',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-28/cme-partially-restores-operations-with-restart-of-forex-platform',
        'Article': 'The Chicago Mercantile Exchange restored trading operations in time to wrap up a holiday-shortened week in the US, recovering from an hours-long technical outage that had disrupted multiple financial markets across Asia and Europe.'},
        {'Headline': 'Traders Around the World Left Hanging After Glitch Took Out CME',
        'Date': '2025-11-28',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-28/frustration-and-confusion-ripple-across-markets-on-cme-outage',
        'Article': 'The Chicago Mercantile Exchange Group proudly describes itself as the place “where the world comes to manage risk.” Except on Friday, the world was shut out.'},
        {'Headline': 'S&amp;P 500 Extends Rally After CME Glitch, Reversing Monthly Slide',
        'Date': '2025-11-28',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-28/us-stocks-up-as-cme-trading-resumes-fed-rate-cut-bets-steady',
        'Article': 'US stocks notched a fifth-straight day of gains, as the Chicago Mercantile Exchange restarted operations following an earlier outage and expectations for a Federal Reserve interest-rate cut next month remained intact.'}]

    return [BloombergNewsEntry.model_validate(record) for record in tqdm(data[:3], desc="Validating entries")]

async def process_bloomberg_news(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Run the full NewsProcessor pipeline on raw Bloomberg RSS feed entries.
    Input: List of dicts with Headline, Link, Article, Date
    Output: Processed dataframe converted to list[dict]
    """
    processor = get_processor(config)

    # Pipeline
    df = processor.enrich_news_entries_with_classifications(data)
    df = processor.group_by_date_and_industry(df)
    df = processor.filter_and_analyze_news(df)
    df = processor.extract_impactful_news(df, top_n=3)
    df = processor.get_consolidated_sentiment(df)
    df = await processor.get_explanation(df)

    return df

def mock_process_bloomberg_news() -> pd.DataFrame:
    """Mock process Bloomberg RSS news"""
    data = [{'Industry': 'Consumer Discretionary',
        'Date': '2025-11-29',
        'News': [{'Headline': 'Trump to Pardon Ex-Honduras Leader Convicted of Drug Trafficking',
            'Date': '2025-11-29',
            'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/trump-to-pardon-ex-honduras-leader-convicted-of-drug-trafficking',
            'Article': 'President Donald Trump said he plans to pardon a former president of Honduras who’s serving a decades-long US sentence for cocaine trafficking, two days before the nation’s election.',
            'SentimentScore': -0.5947091579437256,
            'Industry': 'Consumer Discretionary'}],
        'ArticleCount': 1,
        'ImpactfulNews': [{'Headline': 'Trump to Pardon Ex-Honduras Leader Convicted of Drug Trafficking',
            'Article': 'President Donald Trump said he plans to pardon a former president of Honduras who’s serving a decades-long US sentence for cocaine trafficking, two days before the nation’s election.',
            'SentimentScore': -0.5947091579437256}],
        'AvgSentimentScore': -0.5947091579437256,
        'SentimentScore': -0.2907472252845764,
        'SentimentExplanation': 'The news signals elevated political risk and policy uncertainty (a US pardon related to a foreign leader), which tends to dampen risk appetite and weigh on global equities. For Consumer Discretionary, this translates into caution on names with international exposure and consumer confidence sensitivity, potentially increasing near-term volatility. The FinBERT score of about -0.59 reflects a clear negative sentiment and risk-off bias in the sector.'},
        {'Industry': 'Energy',
        'Date': '2025-11-29',
        'News': [{'Headline': 'New Fortress Gets Tentative OK for $3 Billion Puerto Rico Deal',
            'Date': '2025-11-29',
            'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/new-fortress-gets-tentative-ok-for-3-billion-puerto-rico-deal',
            'Article': 'Regulators in Puerto Rico tentatively approved a contentious deal with billionaire Wes Edens’ New Fortress Energy Inc. to supply liquefied natural gas to the US territory.',
            'SentimentScore': -0.8536778688430786,
            'Industry': 'Energy'}],
        'ArticleCount': 1,
        'ImpactfulNews': [{'Headline': 'New Fortress Gets Tentative OK for $3 Billion Puerto Rico Deal',
            'Article': 'Regulators in Puerto Rico tentatively approved a contentious deal with billionaire Wes Edens’ New Fortress Energy Inc. to supply liquefied natural gas to the US territory.',
            'SentimentScore': -0.8536778688430786}],
        'AvgSentimentScore': -0.8536778688430786,
        'SentimentScore': -0.7660529613494873,
        'SentimentExplanation': "The negative reading stems from regulatory contention surrounding New Fortress Energy's $3B LNG deal with Puerto Rico. Although tentative approval reduces some risk, the deal remains controversial and subject to execution and financing uncertainties, plus potential cost implications for ratepayers. This aligns with the article's strongly negative sentiment (about -0.854) and FinBERT scores in the same negative range, signaling near-term headwinds for energy equities tied to LNG projects."},
        {'Industry': 'Financials',
        'Date': '2025-11-29',
        'News': [{'Headline': 'Brazil Judge Frees Banco Master’s Vorcaro With Ankle Monitor',
            'Date': '2025-11-29',
            'Link': 'https://www.bloomberg.com/news/articles/2025-11-29/brazil-judge-frees-banco-master-s-vorcaro-with-ankle-monitor',
            'Article': 'A Brazilian judge has ordered the release of Daniel Vorcaro, the controlling shareholder of the failed Banco Master SA, who was detained last week.',
            'SentimentScore': 0.12870462238788605,
            'Industry': 'Financials'}],
        'ArticleCount': 1,
        'ImpactfulNews': [{'Headline': 'Brazil Judge Frees Banco Master’s Vorcaro With Ankle Monitor',
            'Article': 'A Brazilian judge has ordered the release of Daniel Vorcaro, the controlling shareholder of the failed Banco Master SA, who was detained last week.',
            'SentimentScore': 0.12870462238788605}],
        'AvgSentimentScore': 0.12870462238788605,
        'SentimentScore': 0.1460907757282257,
        'SentimentExplanation': 'Explanation: The modest positive sentiment score (~0.13) signals a potential stabilization effect from the release of Banco Master’s controlling shareholder, reducing governance-related uncertainty in Brazilian financials, though ongoing distress at Banco Master and broader regulatory risks keep upside limited.'}]

    return pd.DataFrame(data)

def predict_returns_next_day(raw_df: pd.DataFrame, lookback_days: int = 90) -> str:
    """
    Predict next-day returns based on processed financial news sentiment data.
    """
    # raw_list = normalize_to_list(json_data)

    # # convert each dict → Pydantic model
    # parsed_objects = [
    #     FinancialNewsSummary.model_validate(item)
    #     for item in raw_list
    # ]

    # # Convert into DataFrame properly
    # df = pd.DataFrame([obj.model_dump() for obj in parsed_objects])
    df = raw_df.copy()

    # Load artifacts once
    SCALER = joblib.load("artefacts/scaler_numeric.pkl")
    NUMERIC_FEATURE_NAMES = joblib.load("artefacts/numeric_feature_names.pkl")  # ["MKT","SentimentScore_std","ret","ret_vol_20d"]
    GB_FULL_MODEL = joblib.load("artefacts/gb_full_model.pkl")

    # Sentiment standardization constants (set from training)
    SENT_MEAN = 0.0  # replace with training df["SentimentScore"].mean()
    SENT_STD  = 1.0  # replace with training df["SentimentScore"].std()

    INDUSTRY_TO_INDEX = {
        "Information Technology": "XLK",
        "Health Care": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Communication Services": "VOX",
        "Industrials": "XLI",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "IYR",
        "Materials": "XLB",
        "General Market": "SPY",
        "None": None,
    }

    # df = pd.DataFrame(raw_data)

    # 1. Clean Date and Industry
    if df.index.name == "Date" and "Date" not in df.columns:
        df = df.reset_index()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    # Make Date timezone-naive for consistency
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert(None)

    df["Industry"] = df["Industry"].astype(str)

    print("Raw sentiment rows:", len(df))
    # print("Sample sentiment rows after cleaning:")
    # print(df[["Date", "Industry", "SentimentScore"]].head())

    # 2. Industry → ETF index mapping
    df["index_ticker"] = df["Industry"].map(INDUSTRY_TO_INDEX)
    df = df.dropna(subset=["index_ticker"])

    if df.empty:
        print("No rows after Industry → index_ticker mapping.")
        return df.assign(pred_ret_next=np.nan)

    print("Unique mapped tickers in sentiment:", sorted(df["index_ticker"].unique()))

    # 3. Download OHLCV for needed tickers over a rolling window
    tickers = sorted(df["index_ticker"].unique().tolist())
    if "SPY" not in tickers:
        tickers.append("SPY")

    max_date = df["Date"].max()
    start_date = max_date - pd.Timedelta(days=lookback_days)
    end_date = max_date
    print("Tickers:", tickers)
    print("OHLCV date range:", start_date, "to", end_date)

    ohlcv = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=False,
    )
    # Ensure OHLCV index is timezone-naive DatetimeIndex
    ohlcv.index = pd.to_datetime(ohlcv.index)
    if ohlcv.index.tz is not None:
        ohlcv.index = ohlcv.index.tz_convert(None)
    ohlcv = ohlcv.rename_axis("Date")

    print("OHLCV shape:", ohlcv.shape)
    print("OHLCV columns (first few):", list(ohlcv.columns)[:8])
    # print("Sample OHLCV head:")
    # print(ohlcv.head())

    if not isinstance(ohlcv.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (field, ticker) from yfinance.")

    # 4. Build long OHLCV with index_ticker
    rows = []
    for t in tickers:
        if ("Close", t) not in ohlcv.columns or ("Volume", t) not in ohlcv.columns:
            print(f"Missing Close/Volume for ticker {t} in OHLCV; skipping.")
            continue
        close = ohlcv[("Close", t)]
        vol   = ohlcv[("Volume", t)]
        tmp = pd.DataFrame({
            "Date": close.index,
            "index_ticker": t,
            "price": close.values,
            "Volume": vol.values,
        })
        rows.append(tmp)

    if not rows:
        print("No OHLCV rows built; check tickers/date range.")
        return df.assign(pred_ret_next=np.nan)

    ohlcv_long = pd.concat(rows, ignore_index=True)
    print("ohlcv_long shape:", ohlcv_long.shape)
    # print("Sample ohlcv_long rows:")
    # print(ohlcv_long.head())

    # 5. Price-based features on the OHLCV window
    ohlcv_long = ohlcv_long.sort_values(["index_ticker", "Date"])
    ohlcv_long["ret"] = ohlcv_long.groupby("index_ticker")["price"].pct_change()

    ohlcv_long["dollar_vol"] = ohlcv_long["price"] * ohlcv_long["Volume"]
    ohlcv_long["log_dollar_vol"] = np.log1p(ohlcv_long["dollar_vol"])

    window = 20
    ohlcv_long["ret_vol_20d"] = (
        ohlcv_long.groupby("index_ticker")["ret"]
                  .rolling(window).std()
                  .reset_index(level=0, drop=True)
    )
    ohlcv_long["ret_mom_20d"] = (
        ohlcv_long.groupby("index_ticker")["ret"]
                  .rolling(window).apply(lambda x: (1 + x).prod() - 1, raw=False)
                  .reset_index(level=0, drop=True)
    )
    ohlcv_long["dollar_vol_rel_20d"] = (
        ohlcv_long["dollar_vol"] /
        ohlcv_long.groupby("index_ticker")["dollar_vol"]
                  .rolling(window).mean()
                  .reset_index(level=0, drop=True)
    )

    # Next-day return (for evaluation only; not needed for inference)
    ohlcv_long["price_next"] = ohlcv_long.groupby("index_ticker")["price"].shift(-1)
    ohlcv_long["ret_next"] = ohlcv_long["price_next"] / ohlcv_long["price"] - 1

    # print("Sample ohlcv_long with features:")
    # print(ohlcv_long.head())

    # 6. Map each article Date to last available trading day ("feature_date")
    trading_dates = pd.Index(sorted(ohlcv.index.unique()))
    df["feature_date"] = df["Date"].map(
        lambda d: trading_dates[trading_dates.searchsorted(d, side="right") - 1]
        if d >= trading_dates[0] else pd.NaT
    )
    df = df.dropna(subset=["feature_date"])

    # print("Sample article -> feature_date mapping:")
    # print(df[["Date", "index_ticker", "feature_date"]].head())

    # 7. Merge OHLCV features using feature_date
    model_df = df.merge(
        ohlcv_long[
            [
                "Date", "index_ticker", "price", "ret",
                "Volume", "dollar_vol", "log_dollar_vol",
                "ret_vol_20d", "ret_mom_20d", "dollar_vol_rel_20d", "ret_next",
            ]
        ],
        left_on=["feature_date", "index_ticker"],
        right_on=["Date", "index_ticker"],
        how="inner",
        suffixes=("", "_feat"),
    )

    print("model_df shape after OHLCV merge:", model_df.shape)
    # print("Sample model_df after OHLCV merge:")
    # print(model_df[["Date", "feature_date", "index_ticker", "price", "ret"]].head())

    if model_df.empty:
        print("No rows after mapping to feature_date and merging with OHLCV.")
        return df.assign(pred_ret_next=np.nan)

    # 8. Build market factor MKT from SPY on feature_date using same-day return
    mkt = (
        ohlcv_long.loc[ohlcv_long["index_ticker"] == "SPY", ["Date", "ret"]]
                  .rename(columns={"ret": "MKT"})
                  .drop_duplicates(subset=["Date"])
    )
    mkt["Date"] = pd.to_datetime(mkt["Date"])
    if mkt["Date"].dt.tz is not None:
        mkt["Date"] = mkt["Date"].dt.tz_convert(None)
    model_df["feature_date"] = pd.to_datetime(model_df["feature_date"])
    if model_df["feature_date"].dt.tz is not None:
        model_df["feature_date"] = model_df["feature_date"].dt.tz_convert(None)

    model_df = model_df.merge(mkt, left_on="feature_date", right_on="Date", how="left", suffixes=("", "_mkt"))

    before_mkt = len(model_df)
    model_df = model_df.dropna(subset=["MKT", "SentimentScore"])
    after_mkt = len(model_df)
    print(f"Dropped {before_mkt - after_mkt} rows due to missing MKT or SentimentScore.")
    # print("Sample rows after adding MKT:")
    # print(model_df[["Date", "feature_date", "index_ticker", "MKT"]].head())

    if model_df.empty:
        print("No rows left after requiring MKT and SentimentScore; cannot score.")
        return df.assign(pred_ret_next=np.nan)

    # 9. Recompute standardized SentimentScore like training
    model_df["SentimentScore_std"] = (model_df["SentimentScore"] - SENT_MEAN) / SENT_STD

    # 10. Build numeric feature matrix in training order
    for col in ["MKT", "SentimentScore_std", "ret", "ret_vol_20d"]:
        if col not in model_df.columns:
            raise ValueError(f"Required feature column {col} is missing after merge.")

    X_num = model_df[NUMERIC_FEATURE_NAMES].fillna(0.0).values

    print("Numeric feature matrix shape:", X_num.shape)
    # print("Sample numeric features (first 5 rows):")
    # print(model_df[NUMERIC_FEATURE_NAMES].head())

    if X_num.shape[0] == 0:
        print("No samples to score after all preprocessing; returning empty predictions.")
        return df.assign(pred_ret_next=np.nan)

    # 11. Scale and predict
    X_scaled = SCALER.transform(X_num)
    pred = GB_FULL_MODEL.predict(X_scaled)

    model_df["pred_ret_next"] = pred

    return model_df