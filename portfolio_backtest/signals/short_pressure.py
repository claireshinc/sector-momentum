"""Short pressure signal from FINRA daily short volume."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest.signals.momentum import compute_momentum_12_1
from portfolio_backtest.portfolio.construction import build_long_short_portfolio


def compute_short_pressure_score(
    short_data: pd.DataFrame,
    symbol: str,
    as_of: str,
    lookback: int = 5,
    train_start: str = None,
) -> float:
    """
    Compute short pressure z-score for a symbol.
    Uses smoothed short ratio vs expanding historical distribution.
    """
    sym_data = short_data[
        (short_data["symbol"] == symbol) & (short_data["date"] <= as_of)
    ].sort_values("date")

    if len(sym_data) < lookback + 20:
        return np.nan

    # Smoothed recent ratio
    recent_ratio = sym_data["short_ratio"].tail(lookback).mean()

    # Historical distribution (expanding window)
    if train_start:
        hist_data = sym_data[sym_data["date"] >= train_start]
    else:
        hist_data = sym_data

    hist_mean = hist_data["short_ratio"].mean()
    hist_std = hist_data["short_ratio"].std()

    if hist_std > 0:
        return (recent_ratio - hist_mean) / hist_std
    return 0.0


def strategy_c_signal(
    prices: pd.DataFrame,
    short_data: pd.DataFrame,
    as_of: str,
    train_start: str,
    sentiment_z: float,
) -> pd.Series:
    """
    Combine momentum + short pressure into trade signal.

    Avoid squeeze risk: high short pressure + improving momentum
    Target: weak momentum + high short pressure + high sentiment
    """
    symbols = prices.columns.tolist()

    # Compute momentum
    mom = compute_momentum_12_1(prices, as_of)
    mom_ranks = mom.rank()
    n = len(mom_ranks.dropna())

    if n < 3:
        return pd.Series(0.0, index=symbols)

    mom_tercile = pd.qcut(mom_ranks, 3, labels=[1, 2, 3], duplicates="drop")

    # Compute short pressure
    sp_scores = pd.Series(
        {
            sym: compute_short_pressure_score(
                short_data, sym, as_of, train_start=train_start
            )
            for sym in symbols
        }
    )
    sp_valid = sp_scores.dropna()
    if len(sp_valid) < 3:
        return pd.Series(0.0, index=symbols)

    sp_tercile = pd.qcut(sp_valid.rank(), 3, labels=[1, 2, 3], duplicates="drop")

    # Recent reversal (for squeeze risk detection)
    p = prices.loc[:as_of]
    if len(p) >= 21:
        ret_1m = (p.iloc[-1] / p.iloc[-21]) - 1
    else:
        ret_1m = pd.Series(0.0, index=symbols)

    # Build signals
    signal = pd.Series(0.0, index=symbols)

    for sym in symbols:
        mt = mom_tercile.get(sym)
        st = sp_tercile.get(sym)
        r1m = ret_1m.get(sym, 0)

        if pd.isna(mt) or pd.isna(st):
            continue

        # Long: strong momentum + low crowding
        if int(mt) == 3 and int(st) == 1:
            signal[sym] = 1.0

        # Short: weak momentum + high crowding + high sentiment
        # BUT avoid if recent return positive (squeeze risk)
        elif int(mt) == 1 and int(st) == 3 and sentiment_z > 0 and r1m < 0:
            signal[sym] = -1.0

    # Normalize to unit gross
    long_sum = signal[signal > 0].sum()
    short_sum = abs(signal[signal < 0].sum())

    if long_sum > 0:
        signal[signal > 0] /= long_sum
    if short_sum > 0:
        signal[signal < 0] /= short_sum

    return signal
