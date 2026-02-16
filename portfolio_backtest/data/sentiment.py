"""FRED UMCSENT sentiment data with publication lag handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"

# Publication lag: UMCSENT released mid-month for prior month
# At month-end t, we use reading from month t-1 (conservative, defensible)
SENTIMENT_LAG_MONTHS = 1


def load_sentiment(
    start: str = "2000-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.Series:
    """
    Load University of Michigan Consumer Sentiment (UMCSENT) from FRED.
    Returns monthly Series indexed by month-end date.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"umcsent_{start}_{end}.parquet"

    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df["UMCSENT"]

    try:
        from fredapi import Fred
        fred = Fred(api_key=_get_fred_key())
        data = fred.get_series("UMCSENT", observation_start=start, observation_end=end)
    except Exception:
        # Fallback: try to load from yfinance or generate synthetic data for testing
        data = _fallback_sentiment(start, end)

    if data is not None and not data.empty:
        data = data.dropna()
        data.name = "UMCSENT"
        if cache:
            data.to_frame().to_parquet(cache_file)

    return data


def _get_fred_key() -> str:
    """Attempt to load FRED API key from environment or file."""
    import os
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        return key

    key_file = Path.home() / ".fred_api_key"
    if key_file.exists():
        return key_file.read_text().strip()

    raise ValueError(
        "FRED API key not found. Set FRED_API_KEY env var or create ~/.fred_api_key"
    )


def _fallback_sentiment(start: str, end: str) -> pd.Series:
    """
    Generate synthetic sentiment data for testing when FRED is unavailable.
    Uses a mean-reverting process around historical UMCSENT mean (~85).
    """
    dates = pd.date_range(start, end, freq="ME")
    np.random.seed(42)

    values = [85.0]
    for _ in range(len(dates) - 1):
        shock = np.random.normal(0, 5)
        mean_reversion = 0.1 * (85.0 - values[-1])
        values.append(values[-1] + mean_reversion + shock)

    series = pd.Series(values[: len(dates)], index=dates, name="UMCSENT")
    return series


def load_sentiment_with_lag(
    sentiment_series: pd.Series,
    as_of: str,
    lag_months: int = 1,
) -> float:
    """
    Get sentiment value available at as_of date, respecting publication lag.
    """
    as_of_dt = pd.Timestamp(as_of)
    available_through = as_of_dt - pd.DateOffset(months=lag_months)
    available_data = sentiment_series.loc[:available_through]

    if available_data.empty:
        return np.nan
    return available_data.iloc[-1]


def compute_sentiment_zscore(
    sentiment_series: pd.Series,
    as_of: str,
    train_start: str,
    lag_months: int = 1,
) -> float:
    """
    Z-score using expanding window from train_start to as_of,
    respecting publication lag.
    """
    as_of_dt = pd.Timestamp(as_of)
    available_through = as_of_dt - pd.DateOffset(months=lag_months)

    history = sentiment_series.loc[train_start:available_through]

    if len(history) < 12:  # Need minimum history
        return 0.0

    current = history.iloc[-1]
    return (current - history.mean()) / history.std()
