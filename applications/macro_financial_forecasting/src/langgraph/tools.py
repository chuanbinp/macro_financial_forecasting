from datetime import datetime, timedelta, timezone
import feedparser
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import yfinance as yf

from data_model.bloomberg_news_entry import BloombergNewsEntry
from config import Config
from processor import NewsProcessor

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

    from config import Config
    config = Config()
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
                        "Date": published.isoformat(),
                    })

    return [BloombergNewsEntry.model_validate(record) for record in tqdm(news, desc="Validating entries")]

def get_mock_bloomberg_rss_feeds() -> List[Dict[str, str]]:
    """Fetch mock Bloomberg RSS news"""
    data = [{'Headline': 'Asian Stocks Ebb as Global Rally Loses Momentum: Markets Wrap',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/asian-stocks-to-ebb-as-global-equity-rally-stalls-markets-wrap',
    'Article': 'Asian stocks wobbled at the open Friday as the sharp rebound in global equities over the past week showed signs of losing momentum.',
    'Date': '2025-11-27T22:29:53+00:00'},
    {'Headline': 'Gold Poised for Fourth Monthly Gain on Fed Rate-Cut Optimism',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/gold-poised-for-fourth-monthly-gain-on-fed-rate-cut-optimism',
    'Article': 'Gold was steady &mdash; and on track for a fourth monthly gain &mdash; as signs the Federal Reserve will cut rates next month supported the precious metal.',
    'Date': '2025-11-27T23:44:21+00:00'},
    {'Headline': 'Oil in Worst Monthly Run Since 2023 With OPEC+, Ukraine in Focus',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/latest-oil-market-news-and-analysis-for-nov-28',
    'Article': 'Oil headed for the longest run of monthly losses in more than two years, as traders looked ahead to an OPEC+ meeting this weekend and gauged US-led efforts to end the conflict in Ukraine.',
    'Date': '2025-11-27T23:33:36+00:00'},
    {'Headline': 'Vale Boosts Investor Payouts on Strong Output, Iron Ore Prices',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/vale-boosts-investor-payouts-on-strong-output-iron-ore-prices',
    'Article': 'Brazilian iron ore heavyweight Vale SA will pay a special dividend to shareholders on the back of its strong operational performance and high prices of the steelmaking ingredient this year.',
    'Date': '2025-11-27T23:53:06+00:00'},
    {'Headline': 'China EV Profit Woes Fuel Market Anxiety Over Challenging 2026',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/china-ev-profit-woes-fuel-market-anxiety-over-challenging-2026',
    'Article': 'Investors in Chinese electric vehicle stocks had been hoping for a strong earnings season to provide a fresh tailwind. Instead, disappointing results have stoked anxiety about what lies ahead.',
    'Date': '2025-11-27T23:30:00+00:00'},
    {'Headline': 'Emerging Assets Halt Rally in Quiet Session Amid US Holiday',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/china-sets-tone-as-rebound-in-emerging-market-assets-pauses',
    'Article': 'Emerging-market assets halted their recent advance on Thursday, with a gauge of currencies ending the day little changed and stocks falling slightly amid thin liquidity due to the Thanksgiving holiday.',
    'Date': '2025-11-27T11:52:17+00:00'},
    {'Headline': 'How China Came to Dominate Global Shipping Ports',
    'Link': 'https://www.bloomberg.com/news/videos/2025-11-27/china-s-ports-empire-video',
    'Article': 'China has invested billions of dollars in building ports around the globe, giving Beijing a strategic advantage as trade tensions rise, and sparking security concerns for the rest of the world.  (Source: Bloomberg)',
    'Date': '2025-11-27T22:00:03+00:00'},
    {'Headline': 'OBR Says Reeves Had Favorable UK Forecast Before Tax Speech',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/uk-s-obr-says-it-had-no-hand-in-tax-u-turn-that-shocked-markets',
    'Article': 'Rachel Reeves started preparing the public for a manifesto-breaking income-tax rise at the start of November in an unusual pre-budget speech. More surprising was the fact that she did so after privately learning that her fiscal room for maneuver was far better than expected.',
    'Date': '2025-11-27T15:23:41+00:00'},
    {'Headline': 'China Halts Some Brazil Soybean Imports Over Contamination',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/china-halts-some-brazil-soybean-imports-over-contamination',
    'Article': 'China halted soybean imports from five Brazilian plants owned by major global agricultural firms over sanitation concerns, according to people familiar with the matter.',
    'Date': '2025-11-27T14:44:51+00:00'},
    {'Headline': 'Mercuria, Vitol Are Among Bidders for Raizen Argentina Refinery',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/mercuria-vitol-are-among-bidders-for-raizen-argentina-refinery',
    'Article': 'Energy trading giants Mercuria Energy Group and Vitol Group are among finalists for a refinery and hundreds of gas stations in Argentina being sold by Raizen SA, according to people familiar with the matter.',
    'Date': '2025-11-27T19:41:50+00:00'},
    {'Headline': 'Still Constructive on Chinese Equities, Huynh Says',
    'Link': 'https://www.bloomberg.com/news/videos/2025-11-27/still-constructive-on-chinese-equities-huynh-says-video',
    'Article': 'Sophie Huynh, portfolio manager and strategist at BNP Paribas Asset Management, explains why China remains a pocket of diversification in the firm\'s portfolios. She speaks to Lizzy Burden on "Bloomberg Markets." (Source: Bloomberg)',
    'Date': '2025-11-27T19:06:21+00:00'},
    {'Headline': 'Hedge Fund Bond Market Bets Risk Yield Spikes, BIS Chief Warns',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/hedge-fund-bond-market-bets-risk-yield-spikes-bis-chief-warns',
    'Article': 'Bank for International Settlements head Pablo Hernandez de Cos has added his voice to escalating warnings about the role of non-bank firms including hedge funds in sovereign bond markets, at a time of historic government debt levels and a fraught geopolitical backdrop.',
    'Date': '2025-11-27T18:30:00+00:00'},
    {'Headline': 'Anglo-Teck Deal Clears Canada’s National Security Test, Globe Reports',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/anglo-teck-deal-clears-canada-s-national-security-test-globe-reports',
    'Article': 'The Canadian government has cleared Anglo American Plc’s proposed takeover of Teck Resources Ltd. on national security grounds, the Globe and Mail reported, removing one hurdle for the two companies to combine.',
    'Date': '2025-11-27T17:21:22+00:00'},
    {'Headline': 'Credit Market Can Handle Tech’s Debt Deluge, BI Panelists Say',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/credit-market-can-handle-tech-s-debt-deluge-bi-panellists-say',
    'Article': 'Concerns over massive debt issuance from tech giants such as Meta Platforms Inc. and Alphabet Inc. creating oversupply in the credit market are premature, said panelists at the Bloomberg Intelligence European credit market outlook conference in London on Thursday.',
    'Date': '2025-11-27T17:16:31+00:00'},
    {'Headline': 'Lenders Line Up for Chile Copper Smelter in Bet Glut Won’t Last',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/lenders-line-up-for-chile-copper-smelter-in-bet-glut-won-t-last',
    'Article': 'Chile’s Enami said it has received various financing offers for a $1.7 billion copper smelter and will name the banks to structure the deal as early as next week, even as the market remains depressed.',
    'Date': '2025-11-27T16:51:29+00:00'},
    {'Headline': 'Foreign Investment in Canada Falls to Lowest Since Start of ‘24',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/foreign-investment-in-canada-falls-to-lowest-since-start-of-24',
    'Article': 'Foreign direct investment into Canada fell in the third quarter, reaching the lowest level in a year and a half.',
    'Date': '2025-11-27T16:49:38+00:00'},
    {'Headline': 'European Stocks Head for Muted Finish to November; Puma Rallies',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/european-stocks-head-for-muted-finish-to-november-puma-rallies',
    'Article': 'European stocks edged higher on Thursday and were set to close November with a small gain, while Puma SE surged on chatter around a takeover.',
    'Date': '2025-11-27T08:07:35+00:00'},
    {'Headline': 'UK Bonds Fall as Budget Relief Gives Way to Longer-Term Concerns',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/uk-market-declines-offer-reality-check-on-post-budget-relief',
    'Article': 'UK’s bonds fell, ending a five-day rally, as investors analyzed the economic impact of Chancellor Rachel Reeves’s budget that delayed tax-raising measures until later this decade.',
    'Date': '2025-11-27T10:56:28+00:00'},
    {'Headline': 'Moody’s Cuts Brazil’s Raízen to Junk on Rising Debt Levels',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/brazil-s-raizen-cut-to-junk-by-moody-s-on-worsening-debt-metrics',
    'Article': 'Moody’s Ratings downgraded sugar powerhouse Raizen SA, cutting to junk one of Brazil’s last remaining investment-grade corporate credits.',
    'Date': '2025-11-27T13:08:22+00:00'},
    {'Headline': "Bonds Slip\xa0as Investors Assess the UK's Growth Prospects",
    'Link': 'https://www.bloomberg.com/news/live-blog/2025-11-27/uk-budget-2025-reeves-tax-hikes-stocks-pound-steady-markets-today',
    'Article': 'Rachel Reeves presents her budget in Parliament on Nov. 26. Source: UK Parliament',
    'Date': '2025-11-27T06:48:09+00:00'},
    {'Headline': 'Allfunds Said to Draw Takeover Interest From Deutsche Boerse',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/allfunds-said-to-attract-takeover-interest-from-deutsche-boerse',
    'Article': 'European fund distribution platform Allfunds Group Plc is attracting fresh takeover interest from suitors including Deutsche Boerse AG, according to people familiar with the matter.',
    'Date': '2025-11-27T15:18:30+00:00'},
    {'Headline': 'OPEC+ Still on Track to Pause Hikes in Early 2026, Delegates Say',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/opec-still-on-track-to-pause-hikes-in-early-2026-delegates-say',
    'Article': 'OPEC+ nations will probably stick with a decision to pause oil production increases in early 2026 when they meet this weekend, delegates said.',
    'Date': '2025-11-27T15:15:03+00:00'},
    {'Headline': "Westbourne's Whelan on 2026 Market Outlook and Global Stimulus",
    'Link': 'https://www.bloomberg.com/news/videos/2025-11-27/westbourne-s-whelan-on-2026-outlook-and-global-stimulus-video',
    'Article': "Westbourne Research Services Global Macro Strategist, Sharmila Whelan, discusses investment strategies in the context of recent geopolitical and economic developments. She speaks with Bloomberg's Tom Mackenzie on Daybreak Europe. (Source: Bloomberg)",
    'Date': '2025-11-27T14:21:30+00:00'},
    {'Headline': 'ECB Officials Saw Rate Level as Robust Enough to Manage Shocks',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/ecb-officials-saw-rate-level-as-robust-enough-to-manage-shocks',
    'Article': 'European Central Bank officials saw the existing level of borrowing costs as sufficient to deal with possible jolts to the economic outlook, according to an account of their October meeting.',
    'Date': '2025-11-27T12:57:18+00:00'},
    {'Headline': 'South Africa Corn Crop Rises to Second Largest on Record on Rain',
    'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/south-africa-corn-crop-rises-to-second-largest-on-record-on-rain',
    'Article': 'South African farmers probably produced the second-biggest corn crop on record in 2025 as the country experienced good rains earlier in the year.',
    'Date': '2025-11-27T12:56:38+00:00'}]

    return data[:3]

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
    data = [{'Industry': 'General Market',
    'Date': '2025-11-27T22:29:53+00:00',
    'News': [{'Headline': 'Asian Stocks Ebb as Global Rally Loses Momentum: Markets Wrap',
        'Date': '2025-11-27T22:29:53+00:00',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/asian-stocks-to-ebb-as-global-equity-rally-stalls-markets-wrap',
        'Article': 'Asian stocks wobbled at the open Friday as the sharp rebound in global equities over the past week showed signs of losing momentum.',
        'SentimentScore': 0.0039989622,
        'Industry': 'General Market'}],
    'ArticleCount': 1,
    'ImpactfulNews': [{'Headline': 'Asian Stocks Ebb as Global Rally Loses Momentum: Markets Wrap',
        'Article': 'Asian stocks wobbled at the open Friday as the sharp rebound in global equities over the past week showed signs of losing momentum.',
        'SentimentScore': 0.0039989622}],
    'AvgSentimentScore': 0.0039989622,
    'SentimentScore': 0.001944202,
    'SentimentExplanation': 'Explanation: The FinBERT score around 0.002–0.004 indicates a near-neutral sentiment for the general market. The article notes Asian equities wobbling as the global rally loses momentum, suggesting cautious investor posture and limited near-term directional moves as markets await clearer catalysts.'},
    {'Industry': 'Financials',
    'Date': '2025-11-27T23:33:36+00:00',
    'News': [{'Headline': 'Oil in Worst Monthly Run Since 2023 With OPEC+, Ukraine in Focus',
        'Date': '2025-11-27T23:33:36+00:00',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/latest-oil-market-news-and-analysis-for-nov-28',
        'Article': 'Oil headed for the longest run of monthly losses in more than two years, as traders looked ahead to an OPEC+ meeting this weekend and gauged US-led efforts to end the conflict in Ukraine.',
        'SentimentScore': -0.008944381,
        'Industry': 'Financials'}],
    'ArticleCount': 1,
    'ImpactfulNews': [{'Headline': 'Oil in Worst Monthly Run Since 2023 With OPEC+, Ukraine in Focus',
        'Article': 'Oil headed for the longest run of monthly losses in more than two years, as traders looked ahead to an OPEC+ meeting this weekend and gauged US-led efforts to end the conflict in Ukraine.',
        'SentimentScore': -0.008944381}],
    'AvgSentimentScore': -0.008944381,
    'SentimentScore': 0.0028017424,
    'SentimentExplanation': "The FinBERT sentiment score is approximately -0.009, i.e., near neutral with a slight negative tilt. This reflects the article's emphasis on oil's continued monthly losses and the uncertainty around OPEC+ decisions and the Ukraine conflict. While the energy sector could face near-term volatility, the near-zero score suggests a modest, not overpowering, impact on financial markets in the Financials space; investors may stay cautious ahead of policy meetings, watching for any signs of supply-tightening or conflict de-escalation that could shift oil demand expectations."},
    {'Industry': 'Financials',
    'Date': '2025-11-27T23:44:21+00:00',
    'News': [{'Headline': 'Gold Poised for Fourth Monthly Gain on Fed Rate-Cut Optimism',
        'Date': '2025-11-27T23:44:21+00:00',
        'Link': 'https://www.bloomberg.com/news/articles/2025-11-27/gold-poised-for-fourth-monthly-gain-on-fed-rate-cut-optimism',
        'Article': 'Gold was steady &mdash; and on track for a fourth monthly gain &mdash; as signs the Federal Reserve will cut rates next month supported the precious metal.',
        'SentimentScore': -0.8583589792,
        'Industry': 'Financials'}],
    'ArticleCount': 1,
    'ImpactfulNews': [{'Headline': 'Gold Poised for Fourth Monthly Gain on Fed Rate-Cut Optimism',
        'Article': 'Gold was steady &mdash; and on track for a fourth monthly gain &mdash; as signs the Federal Reserve will cut rates next month supported the precious metal.',
        'SentimentScore': -0.8583589792}],
    'AvgSentimentScore': -0.8583589792,
    'SentimentScore': -0.8677232862,
    'SentimentExplanation': 'The article carries a strong negative FinBERT sentiment (-0.858), indicating traders view it as unfavorable to financials in the near term. While gold benefits from Fed rate-cut expectations, the piece frames that environment as potentially dampening bank net interest margins and financial sector profitability, which drags sentiment in the Financials industry even as the commodity side improves.'}]

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