import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from pydantic import BaseModel, Field
from typing import Optional, List
import json
from datetime import datetime
from langchain.tools import tool

class NewsArticle(BaseModel):
    """
    Pydantic model for a single news item within the 'News' and 'ImpactfulNews' lists.
    """
    Headline: str = Field(..., description="The main title of the news article.")
    Article: str = Field(..., description="A short snippet or the full text of the article.")
    SentimentScore: float = Field(..., description="A numerical score representing the sentiment of the article (e.g., FinBERT score).")
    Date: Optional[datetime] = Field(None, description="The specific publication date and time of the article.")
    Link: Optional[str] = Field(None, description="The URL to the full news article.")
    Industry: Optional[str] = Field(None, description="The industry category of the news.")

class FinancialNewsSummary(BaseModel):
    """
    Pydantic model for a single financial news summary object (e.g., for one industry/date block).
    """
    Industry: str = Field(..., description="The main industry category for the summary (e.g., 'Financials', 'General Market').")
    Date: datetime = Field(..., description="The date and time the summary was generated/fetched.")
    News: List[NewsArticle] = Field(..., description="A list of all news articles in the summary.")
    ArticleCount: int = Field(..., description="The total number of articles included in the 'News' list.")
    ImpactfulNews: List[NewsArticle] = Field(..., description="A filtered list of articles deemed most impactful.")
    AvgSentimentScore: float = Field(..., description="The average sentiment score of all articles.")
    SentimentScore: float = Field(..., description="The main calculated/weighted sentiment score for the overall summary.")
    SentimentExplanation: str = Field(..., description="A detailed explanation of the sentiment score and its implications.")

class PredictReturnsNextDayArgs(BaseModel):
    """
    Pydantic model for the arguments of the predict_returns_next_day tool.
    """
    json_data: str = Field(..., description="A JSON string representing a list of FinancialNewsSummary objects, output from process_bloomberg_news.")

def normalize_to_list(raw):
    """
    Accepts: JSON string, Python list, dict, repr-ed JSON from LangChain.
    Returns: Python list of dicts.
    """

    # Case 1 — Already a Python list
    if isinstance(raw, list):
        return raw

    # Case 2 — Already a Python dict (single summary)
    if isinstance(raw, dict):
        return [raw]

    # Case 3 — LangChain sometimes gives a string with extra whitespace or quotes
    if isinstance(raw, str):
        cleaned = raw.strip()

        # remove surrounding quotes if agent added them
        # Check for '" at start and "' at end (nested quotes case)
        if cleaned.startswith("'\"") and cleaned.endswith("\"'"):
            cleaned = cleaned[2:-2] 
        elif cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]

        # Try parsing as JSON
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Case 4 — Sometimes LangChain double-escapes JSON
        try:
            # Attempt to decode unicode escapes first
            if '\\u' in cleaned or '\\\\' in cleaned: # Heuristic for double-escaped or unicode-escaped
                cleaned = cleaned.encode("utf-8").decode("unicode_escape")
            return json.loads(cleaned)
        except Exception:
            pass

    # If all else fails
    raise ValueError(f"Could not normalize input: {type(raw)}, value = {repr(raw)}")

@tool(args_schema=PredictReturnsNextDayArgs)
def predict_returns_next_day(json_data: str,
                               lookback_days: int = 90) -> str:

    raw_list = normalize_to_list(json_data)

    # convert each dict → Pydantic model
    parsed_objects = [
        FinancialNewsSummary.model_validate(item)
        for item in raw_list
    ]

    # Convert into DataFrame properly
    df = pd.DataFrame([obj.model_dump() for obj in parsed_objects])

    # Load artifacts once
    SCALER = joblib.load("../artefacts/scaler_numeric.pkl")
    NUMERIC_FEATURE_NAMES = joblib.load("../artefacts/numeric_feature_names.pkl")  # ["MKT","SentimentScore_std","ret","ret_vol_20d"]
    GB_FULL_MODEL = joblib.load("../artefacts/gb_full_model.pkl")

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

    return model_df.to_json(orient="records")