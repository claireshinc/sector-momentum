"""Trading calendar utilities — NYSE holidays, month-ends, trading days."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=4)
def _get_nyse_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """Get NYSE trading days between start and end (cached)."""
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start, end_date=end)
    return schedule.index


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return NYSE trading days between start and end."""
    return _get_nyse_calendar(start, end)


def get_month_end_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Return last trading day of each month within range."""
    trading_days = get_trading_days(start, end)
    series = trading_days.to_series()
    month_ends = series.groupby(pd.Grouper(freq='ME')).last().dropna()
    return pd.DatetimeIndex(month_ends.values)


def get_next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    """Return the next NYSE trading day on or after the given date."""
    # Search a window of 10 days to find the next trading day
    start = date
    end = date + pd.Timedelta(days=10)
    trading_days = get_trading_days(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    if len(trading_days) > 0:
        return trading_days[0]
    return date


def is_trading_day(date: pd.Timestamp) -> bool:
    """Check if a date is a NYSE trading day."""
    d_str = date.strftime('%Y-%m-%d')
    trading_days = get_trading_days(d_str, d_str)
    return len(trading_days) > 0
