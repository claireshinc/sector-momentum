"""Price data loader with yfinance + parquet caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"


def _cache_key(symbols: list[str], start: str, end: str) -> str:
    """Generate a stable cache key."""
    key = f"prices_{'_'.join(sorted(symbols))}_{start}_{end}"
    return hashlib.md5(key.encode()).hexdigest()


def load_prices(
    symbols: list[str],
    start: str = "2004-01-01",
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Load adjusted close prices for symbols via yfinance.
    Results are cached as parquet files.

    Returns:
        DataFrame with DatetimeIndex (US/Eastern trading days) and symbol columns.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(symbols, start, end)}.parquet"

    if cache and cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Ensure all requested symbols are present
        missing = set(symbols) - set(df.columns)
        if not missing:
            return df

    # Download from yfinance
    data = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Single symbol case
        prices = data[["Close"]].rename(columns={"Close": symbols[0]})

    # Drop rows where all prices are NaN
    prices = prices.dropna(how="all")

    # Forward-fill small gaps (weekends already excluded by yfinance)
    prices = prices.ffill(limit=5)

    if cache:
        prices.to_parquet(cache_file)

    return prices


def load_returns(
    symbol: str,
    start: str = "2004-01-01",
    end: str | None = None,
) -> pd.Series:
    """Load daily returns for a single symbol."""
    prices = load_prices([symbol], start=start, end=end)
    return prices[symbol].pct_change().dropna()
