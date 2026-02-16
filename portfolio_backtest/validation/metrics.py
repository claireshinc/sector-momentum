"""Performance metrics with correct math (wealth-based drawdown)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    """
    Compute performance metrics with correct math.
    """
    ann_factor = 252
    n_days = len(returns)
    n_years = n_days / ann_factor

    if n_days == 0:
        return {}

    # Cumulative return (compound, not sum!)
    cum_ret = (1 + returns).prod() - 1

    # CAGR
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility
    vol = returns.std() * np.sqrt(ann_factor)

    # Sharpe
    sharpe = cagr / vol if vol > 0 else 0

    # Sortino (downside deviation)
    downside_ret = returns[returns < 0]
    downside_vol = (
        downside_ret.std() * np.sqrt(ann_factor) if len(downside_ret) > 0 else 0
    )
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Drawdown (correct calculation on wealth)
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    metrics = {
        "total_return": cum_ret,
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "n_days": n_days,
    }

    # Benchmark comparison
    if benchmark is not None:
        aligned_bench = benchmark.reindex(returns.index).dropna()
        aligned_ret = returns.reindex(aligned_bench.index)

        if len(aligned_ret) > 20:
            cov_matrix = np.cov(aligned_ret, aligned_bench)
            beta = (
                cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            )

            bench_cagr = (
                (1 + aligned_bench).prod() ** (ann_factor / len(aligned_bench)) - 1
            )
            alpha = cagr - beta * bench_cagr

            metrics["beta"] = beta
            metrics["alpha"] = alpha

    return metrics


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series on wealth basis."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown


def compute_recovery_days(
    returns: pd.Series,
    start: str,
    end: str,
    allow_beyond_window: bool = True,
) -> int:
    """
    Days to recover to prior peak after a drawdown window.
    """
    wealth = (1 + returns).cumprod()

    # Peak before window
    pre_window = wealth.loc[:start]
    if pre_window.empty:
        return -1
    pre_window_peak = pre_window.max()

    # Find first day after window that exceeds pre-window peak
    post_window = wealth.loc[end:]

    if allow_beyond_window:
        recovery_dates = post_window[post_window >= pre_window_peak].index
    else:
        recovery_dates = wealth.loc[start:end][
            wealth.loc[start:end] >= pre_window_peak
        ].index

    if len(recovery_dates) > 0:
        window_end = pd.Timestamp(end)
        recovery_date = recovery_dates[0]
        return (recovery_date - window_end).days

    return -1  # Did not recover


def compute_leg_attribution(
    weights: pd.DataFrame,
    returns_gross: pd.Series,
) -> pd.DataFrame:
    """Compute return attribution for long and short legs."""
    if weights.empty or returns_gross.empty:
        return pd.DataFrame()

    results = []
    for col in weights.columns:
        w = weights[col]
        long_w = w[w > 0].sum() if (w > 0).any() else 0
        short_w = abs(w[w < 0].sum()) if (w < 0).any() else 0
        results.append(
            {"asset": col, "long_weight": long_w, "short_weight": short_w}
        )

    return pd.DataFrame(results)
