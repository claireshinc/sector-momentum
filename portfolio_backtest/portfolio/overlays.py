"""Overlays: sentiment gating, turnover constraints."""

from __future__ import annotations

import pandas as pd


def apply_sentiment_gate(
    weights: pd.Series,
    sentiment_z: float,
    high_threshold: float = 0.5,
    low_threshold: float = -0.5,
) -> pd.Series:
    """
    Gate short leg based on sentiment z-score.
    High sentiment -> full short (shorts profitable when optimism fades)
    Low sentiment -> no short (avoid falling knives)
    """
    gated = weights.copy()

    if sentiment_z > high_threshold:
        short_multiplier = 1.0  # Full short
    elif sentiment_z > low_threshold:
        short_multiplier = 0.5  # Half short
    else:
        short_multiplier = 0.0  # No short

    gated[gated < 0] *= short_multiplier

    # Leave as reduced gross (more conservative)
    return gated


def apply_turnover_cap(
    old_weights: pd.Series,
    new_weights: pd.Series,
    max_turnover: float = 1.0,
    min_trade_threshold: float = 0.005,
) -> pd.Series:
    """
    Cap turnover and ignore tiny trades.
    """
    combined_index = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(combined_index, fill_value=0)
    new = new_weights.reindex(combined_index, fill_value=0)

    trades = new - old

    # Ignore tiny trades
    trades[abs(trades) < min_trade_threshold] = 0

    # Cap total turnover
    turnover = abs(trades).sum() / 2
    if turnover > max_turnover:
        scale = max_turnover / turnover
        trades = trades * scale

    return old + trades
