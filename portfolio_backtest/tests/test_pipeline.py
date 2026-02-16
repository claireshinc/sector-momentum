"""Integration tests for the full backtest pipeline."""

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.backtest.engine import BacktestEngine
from portfolio_backtest.config.schema import BacktestConfig, SECTOR_ETFS
from portfolio_backtest.portfolio.construction import build_long_short_portfolio
from portfolio_backtest.signals.momentum import (
    compute_momentum_12_1,
    compute_momentum_6_1,
    compute_momentum_blend,
)
from portfolio_backtest.portfolio.overlays import apply_sentiment_gate, apply_turnover_cap
from portfolio_backtest.portfolio.crash_control import RegimeTracker


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


def test_momentum_12_1_basic():
    """Test 12-1 momentum signal with synthetic data."""
    symbols = ["A", "B", "C", "D", "E"]
    prices = _make_prices(symbols, end="2020-12-31")

    signal = compute_momentum_12_1(prices, "2020-06-30")

    assert len(signal) == len(symbols)
    assert not signal.isna().any(), "Signal should have no NaN for valid data"


def test_momentum_6_1_basic():
    """Test 6-1 momentum signal."""
    symbols = ["A", "B", "C", "D"]
    prices = _make_prices(symbols, end="2020-12-31")

    signal = compute_momentum_6_1(prices, "2020-06-30")
    assert len(signal) == len(symbols)


def test_momentum_blend():
    """Test blended momentum signal."""
    symbols = ["A", "B", "C", "D"]
    prices = _make_prices(symbols, end="2020-12-31")

    blend = compute_momentum_blend(prices, "2020-06-30")
    mom_12 = compute_momentum_12_1(prices, "2020-06-30")
    mom_6 = compute_momentum_6_1(prices, "2020-06-30")

    expected = 0.5 * mom_12 + 0.5 * mom_6
    pd.testing.assert_series_equal(blend, expected)


def test_portfolio_construction():
    """Test L/S portfolio construction."""
    signal = pd.Series(
        {"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.1, "E": -0.3, "F": -0.5}
    )

    weights = build_long_short_portfolio(signal, k=2)

    # Top 2 should be long, bottom 2 should be short
    assert weights["A"] > 0
    assert weights["B"] > 0
    assert weights["E"] < 0
    assert weights["F"] < 0

    # Long leg should sum to 1, short leg to -1
    assert np.isclose(weights[weights > 0].sum(), 1.0, atol=0.01)
    assert np.isclose(weights[weights < 0].sum(), -1.0, atol=0.01)


def test_sentiment_gate():
    """Test sentiment gating of short leg."""
    weights = pd.Series({"A": 0.5, "B": 0.5, "C": -0.5, "D": -0.5})

    # High sentiment: full shorts
    gated_high = apply_sentiment_gate(weights, sentiment_z=1.0)
    assert gated_high["C"] < 0
    assert gated_high["D"] < 0

    # Low sentiment: no shorts
    gated_low = apply_sentiment_gate(weights, sentiment_z=-1.0)
    assert gated_low["C"] == 0
    assert gated_low["D"] == 0


def test_turnover_cap():
    """Test turnover capping."""
    old = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    new = pd.Series({"A": 0.0, "B": 0.0, "C": 1.0})

    # Without cap
    capped = apply_turnover_cap(old, new, max_turnover=10.0)
    pd.testing.assert_series_equal(capped, new, atol=0.01)

    # With tight cap
    capped_tight = apply_turnover_cap(old, new, max_turnover=0.25)
    # Should be partially moved toward new
    assert capped_tight["A"] > 0  # Still has some A
    assert capped_tight["C"] < 1.0  # Hasn't fully moved to C


def test_regime_tracker():
    """Test regime persistence filter."""
    tracker = RegimeTracker(min_persist_periods=2)

    # First detection: pending, no switch yet (count=1 < 2)
    assert tracker.update("crash") == "normal"
    # Second detection: count=2 >= min_persist=2, now switches
    assert tracker.update("crash") == "crash"
    assert tracker.current_regime == "crash"

    # Back to normal also requires persistence
    assert tracker.update("normal") == "crash"  # count=1, still crash
    assert tracker.update("normal") == "normal"  # count=2, now switches


def test_full_pipeline():
    """Test full backtest pipeline runs without error."""
    symbols = SECTOR_ETFS + ["SPY"]
    prices = _make_prices(symbols, start="2008-01-02", end="2021-12-31")

    config = BacktestConfig()
    engine = BacktestEngine(prices, config)

    result = engine.run(
        strategy="B",
        crash_controller=None,
        train_start="2010-01-01",
        train_end="2018-12-31",
        test_start="2019-01-01",
        test_end="2020-12-31",
        k=3,
    )

    assert len(result.returns_net) > 0, "Should have net returns"
    assert "sharpe" in result.metrics, "Should compute Sharpe ratio"
    assert "max_drawdown" in result.metrics, "Should compute max drawdown"
    assert result.metrics["max_drawdown"] <= 0, "Max drawdown should be <= 0"


def test_pipeline_with_crash_controller():
    """Test pipeline with crash controller."""
    symbols = SECTOR_ETFS + ["SPY"]
    prices = _make_prices(symbols, start="2008-01-02", end="2021-12-31")

    config = BacktestConfig()
    engine = BacktestEngine(prices, config)

    for ctrl in ["panic_throttle", "signal_switch", "vol_target"]:
        result = engine.run(
            strategy="B",
            crash_controller=ctrl,
            train_start="2010-01-01",
            train_end="2018-12-31",
            test_start="2019-01-01",
            test_end="2020-12-31",
            k=3,
        )
        assert len(result.returns_net) > 0, f"Should have returns for {ctrl}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
