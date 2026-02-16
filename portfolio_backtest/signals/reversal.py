"""Reversal signals: 1-month, 1-week."""

from __future__ import annotations

import pandas as pd


def compute_reversal_1m(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    1-month reversal: negative of recent 1-month return.
    """
    p = prices.loc[:as_of]
    if len(p) < 21:
        return pd.Series(0.0, index=prices.columns)
    ret_1m = (p.iloc[-1] / p.iloc[-21]) - 1
    return -ret_1m  # Negative = reversal


def compute_reversal_1w(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    1-week reversal: negative of recent 1-week return.
    """
    p = prices.loc[:as_of]
    if len(p) < 5:
        return pd.Series(0.0, index=prices.columns)
    ret_1w = (p.iloc[-1] / p.iloc[-5]) - 1
    return -ret_1w
