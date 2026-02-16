"""Momentum signals: 12-1, 6-1, and blended."""

from __future__ import annotations

import pandas as pd


def compute_momentum_12_1(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    12-month return, skipping most recent month.
    Uses only data through as_of date.
    Return from t-252 to t-21 (skip last ~1 month).
    """
    p = prices.loc[:as_of]
    if len(p) < 252:
        # Not enough history — return zeros
        return pd.Series(0.0, index=prices.columns)
    mom = (p.iloc[-21] / p.iloc[-252]) - 1
    return mom


def compute_momentum_6_1(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    6-month return, skipping most recent month.
    Uses only data through as_of date.
    """
    p = prices.loc[:as_of]
    if len(p) < 126:
        return pd.Series(0.0, index=prices.columns)
    mom = (p.iloc[-21] / p.iloc[-126]) - 1
    return mom


def compute_momentum_blend(
    prices: pd.DataFrame,
    as_of: str,
    w_12: float = 0.5,
    w_6: float = 0.5,
) -> pd.Series:
    """Blended momentum signal."""
    mom_12 = compute_momentum_12_1(prices, as_of)
    mom_6 = compute_momentum_6_1(prices, as_of)
    return w_12 * mom_12 + w_6 * mom_6
