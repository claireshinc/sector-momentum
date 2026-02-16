"""No-lookahead tests — verify signals and pipeline don't use future data."""

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.signals.momentum import compute_momentum_12_1
from portfolio_backtest.data.sentiment import compute_sentiment_zscore
from portfolio_backtest.portfolio.crash_control import estimate_thresholds
from portfolio_backtest.backtest.engine import BacktestEngine
from portfolio_backtest.config.schema import BacktestConfig, SECTOR_ETFS


def _make_prices(symbols, start="2008-01-02", end="2021-12-31", seed=42):
    """Generate synthetic price data for testing."""
    np.random.seed(seed)
    dates = pd.bdate_range(start, end)
    data = {}
    for sym in symbols:
        returns = np.random.normal(0.0003, 0.015, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        data[sym] = prices
    return pd.DataFrame(data, index=dates)


def _make_sentiment(start="2000-01-01", end="2022-12-31", seed=42):
    """Generate synthetic sentiment data for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq="ME")
    values = 85 + np.cumsum(np.random.normal(0, 3, len(dates)))
    return pd.Series(values, index=dates, name="UMCSENT")


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Signal invariance — changing future data must not affect past signals
# ──────────────────────────────────────────────────────────────────────────────


def test_signal_invariance():
    """Changing future data must not affect past signals."""
    symbols = SECTOR_ETFS
    # Generate long dataset and truncate for short — ensures identical prices up to cutoff
    prices_long = _make_prices(symbols, end="2021-12-31")
    prices_short = prices_long.loc[:"2020-06-30"].copy()

    signal_short = compute_momentum_12_1(prices_short, "2020-06-30")
    signal_long = compute_momentum_12_1(prices_long, "2020-06-30")

    pd.testing.assert_series_equal(signal_short, signal_long, check_names=False)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Sentiment lag — must not use data beyond t - lag
# ──────────────────────────────────────────────────────────────────────────────


def test_sentiment_lag():
    """Sentiment at time t must not use data beyond t - lag."""
    sentiment = _make_sentiment()

    # At 2020-06-30 with 1-month lag, should use May 2020 value or earlier
    z = compute_sentiment_zscore(sentiment, "2020-06-30", "2010-01-01", lag_months=1)

    # Verify the latest data point used is May 2020
    available = sentiment.loc[:"2020-05-31"]
    assert len(available) > 0, "Should have sentiment data through May 2020"

    # The z-score should be a finite number
    assert not np.isnan(z), "Z-score should not be NaN"

    # Verify that changing June 2020+ data doesn't change the z-score
    sentiment_modified = sentiment.copy()
    sentiment_modified.loc["2020-06-01":] = 999.0

    z_modified = compute_sentiment_zscore(
        sentiment_modified, "2020-06-30", "2010-01-01", lag_months=1
    )
    assert z == z_modified, "Z-score should not depend on future sentiment data"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Pipeline invariance — corrupting future data must not change results
# ──────────────────────────────────────────────────────────────────────────────


def test_pipeline_invariance():
    """Full pipeline returns must not change when future data is corrupted."""
    symbols = SECTOR_ETFS + ["SPY"]
    prices = _make_prices(symbols, start="2008-01-02", end="2021-12-31")

    config = BacktestConfig()
    engine = BacktestEngine(prices, config)

    result_1 = engine.run(
        strategy="B",
        crash_controller=None,
        train_start="2010-01-01",
        train_end="2018-12-31",
        test_start="2019-01-01",
        test_end="2020-12-31",
        k=3,
    )

    # Corrupt 2021 data
    prices_corrupt = prices.copy()
    prices_corrupt.loc["2021-01-01":] *= 0.5

    engine_corrupt = BacktestEngine(prices_corrupt, config)
    result_2 = engine_corrupt.run(
        strategy="B",
        crash_controller=None,
        train_start="2010-01-01",
        train_end="2018-12-31",
        test_start="2019-01-01",
        test_end="2020-12-31",
        k=3,
    )

    # Returns through 2020 must match
    assert len(result_1.returns_net) > 0, "Should have returns"
    assert len(result_2.returns_net) > 0, "Should have returns"

    pd.testing.assert_series_equal(
        result_1.returns_net,
        result_2.returns_net,
        check_names=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Thresholds frozen — must be computed only from training data
# ──────────────────────────────────────────────────────────────────────────────


def test_threshold_frozen():
    """Thresholds must be computed only from training data."""
    symbols = ["SPY"]
    prices_short = _make_prices(symbols, start="2008-01-02", end="2022-12-31")
    prices_long = _make_prices(symbols, start="2008-01-02", end="2025-12-31")

    spy_ret_short = prices_short["SPY"].pct_change().dropna()
    spy_ret_long = prices_long["SPY"].pct_change().dropna()

    thresholds_short = estimate_thresholds(spy_ret_short, "2018-12-31")
    thresholds_long = estimate_thresholds(spy_ret_long, "2018-12-31")

    for key in thresholds_short:
        assert np.isclose(
            thresholds_short[key], thresholds_long[key], rtol=1e-10
        ), f"Threshold {key} should not change with extended data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
