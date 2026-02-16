"""Long/Short portfolio construction with rank hysteresis."""

from __future__ import annotations

import pandas as pd


def build_long_short_portfolio(
    signal: pd.Series,
    k: int,
    prev_weights: pd.Series = None,
    rank_buffer: int = 1,
) -> pd.Series:
    """
    Long top-k, short bottom-k, equal weight.
    With rank hysteresis to reduce turnover.
    """
    ranks = signal.rank(ascending=False)
    n = len(signal)

    if n < 2 * k:
        # Not enough assets — equal weight all
        return pd.Series(0.0, index=signal.index)

    weights = pd.Series(0.0, index=signal.index)

    # Determine long/short cutoffs with hysteresis
    long_cutoff = k
    short_cutoff = n - k

    if prev_weights is not None:
        # Don't drop holdings unless rank crosses k + buffer
        for sym in signal.index:
            if prev_weights.get(sym, 0) > 0:  # Was long
                if ranks[sym] <= k + rank_buffer:
                    weights[sym] = 1.0 / k
            elif prev_weights.get(sym, 0) < 0:  # Was short
                if ranks[sym] >= n - k - rank_buffer + 1:
                    weights[sym] = -1.0 / k

    # Fill remaining slots
    for sym in signal.index:
        if weights[sym] == 0:
            if ranks[sym] <= k:
                weights[sym] = 1.0 / k
            elif ranks[sym] > n - k:
                weights[sym] = -1.0 / k

    # Normalize legs
    long_sum = weights[weights > 0].sum()
    short_sum = abs(weights[weights < 0].sum())

    if long_sum > 0:
        weights[weights > 0] /= long_sum
    if short_sum > 0:
        # Divide negative weights by short_sum to normalize; they stay negative
        weights[weights < 0] /= short_sum

    return weights
