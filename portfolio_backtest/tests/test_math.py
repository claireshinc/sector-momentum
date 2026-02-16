"""Test correctness of drawdown and return calculations."""

import numpy as np
import pandas as pd
import pytest

from portfolio_backtest.validation.metrics import (
    compute_drawdown_series,
    compute_metrics,
    compute_recovery_days,
)


def test_drawdown_wealth_based():
    """Drawdown must be computed on wealth (cumulative product), not sum."""
    # Create returns that go up 10% then down 10%
    # Wealth: 1.0 -> 1.1 -> 0.99 (not 1.0!)
    returns = pd.Series([0.10, -0.10], index=pd.bdate_range("2020-01-01", periods=2))

    dd = compute_drawdown_series(returns)

    # After going up 10% and down 10%, wealth = 1.1 * 0.9 = 0.99
    # Peak = 1.1, drawdown = (0.99 - 1.1) / 1.1 = -0.1/1.1 ≈ -0.0909
    expected_dd = (0.99 - 1.1) / 1.1
    assert np.isclose(dd.iloc[-1], expected_dd, rtol=1e-6), (
        f"Drawdown should be wealth-based: {expected_dd:.4f}, got {dd.iloc[-1]:.4f}"
    )


def test_drawdown_not_sum():
    """Ensure drawdown is NOT computed as simple sum of returns."""
    returns = pd.Series(
        [0.10, -0.05, -0.05, 0.02],
        index=pd.bdate_range("2020-01-01", periods=4),
    )

    dd = compute_drawdown_series(returns)
    max_dd = dd.min()

    # If someone used sum: 0.10 - 0.05 - 0.05 + 0.02 = 0.02 (no drawdown!)
    # Correct (wealth):
    # Day 1: 1.10, peak=1.10, dd=0
    # Day 2: 1.10*0.95=1.045, peak=1.10, dd=(1.045-1.10)/1.10 = -0.05
    # Day 3: 1.045*0.95=0.99275, peak=1.10, dd=(0.99275-1.10)/1.10 ≈ -0.0975
    # Day 4: 0.99275*1.02=1.01261, peak=1.10, dd=(1.01261-1.10)/1.10 ≈ -0.0794

    assert max_dd < -0.05, f"Max drawdown should be < -5%, got {max_dd:.4f}"


def test_cumulative_return_compound():
    """Cumulative return must use compound math, not sum."""
    returns = pd.Series(
        [0.10, 0.10, 0.10],
        index=pd.bdate_range("2020-01-01", periods=3),
    )

    metrics = compute_metrics(returns)

    # Compound: (1.1)(1.1)(1.1) - 1 = 0.331
    # Sum would give: 0.30
    expected = 1.1**3 - 1
    assert np.isclose(metrics["total_return"], expected, rtol=1e-6), (
        f"Total return should be compound: {expected:.4f}, got {metrics['total_return']:.4f}"
    )


def test_recovery_days():
    """Test recovery day calculation."""
    # Create a drawdown and recovery pattern
    returns = pd.Series(
        [0.02, 0.02, -0.10, -0.10, 0.05, 0.05, 0.05, 0.05, 0.05],
        index=pd.bdate_range("2020-01-01", periods=9),
    )

    # Recovery from the drawdown window
    recovery = compute_recovery_days(returns, "2020-01-03", "2020-01-06")
    # Should eventually recover (or return -1)
    assert isinstance(recovery, int)


def test_sharpe_uses_cagr():
    """Sharpe ratio should use CAGR, not mean return."""
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, 504),
        index=pd.bdate_range("2020-01-01", periods=504),
    )

    metrics = compute_metrics(returns)

    # Verify Sharpe = CAGR / vol
    expected_sharpe = metrics["cagr"] / metrics["annual_vol"]
    assert np.isclose(metrics["sharpe"], expected_sharpe, rtol=1e-6)


def test_metrics_with_benchmark():
    """Test that benchmark metrics (beta, alpha) are computed."""
    np.random.seed(42)
    n = 504

    bench_ret = pd.Series(
        np.random.normal(0.0004, 0.01, n),
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    # Strategy correlated with benchmark
    strat_ret = bench_ret * 0.8 + pd.Series(
        np.random.normal(0.0002, 0.005, n),
        index=pd.bdate_range("2020-01-01", periods=n),
    )

    metrics = compute_metrics(strat_ret, bench_ret)

    assert "beta" in metrics, "Should compute beta"
    assert "alpha" in metrics, "Should compute alpha"
    assert 0 < metrics["beta"] < 2, f"Beta should be reasonable: {metrics['beta']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
