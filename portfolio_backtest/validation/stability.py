"""Multi-split out-of-sample stability analysis."""

from __future__ import annotations

import pandas as pd


def oos_stability_analysis(
    engine,
    strategy: str,
    crash_controller: str,
    train_start: str,
    split_dates: list[str],
    test_end: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Run backtest with multiple train_end dates to assess stability.
    """
    results = []

    for split_date in split_dates:
        # Test period is split_date+1 to test_end
        test_start = (
            pd.Timestamp(split_date) + pd.DateOffset(days=1)
        ).strftime("%Y-%m-%d")

        result = engine.run(
            strategy=strategy,
            crash_controller=crash_controller,
            train_start=train_start,
            train_end=split_date,
            test_start=test_start,
            test_end=test_end,
            **kwargs,
        )

        if result.metrics:
            results.append(
                {
                    "train_end": split_date,
                    "test_start": test_start,
                    "sharpe": result.metrics.get("sharpe", 0),
                    "max_dd": result.metrics.get("max_drawdown", 0),
                    "cagr": result.metrics.get("cagr", 0),
                }
            )

    df = pd.DataFrame(results)

    if not df.empty:
        print(f"OOS Stability Summary (n={len(df)} splits)")
        print(
            f"  Sharpe: median={df['sharpe'].median():.2f}, "
            f"IQR=[{df['sharpe'].quantile(0.25):.2f}, {df['sharpe'].quantile(0.75):.2f}]"
        )
        print(
            f"  MaxDD:  median={df['max_dd'].median():.1%}, "
            f"worst={df['max_dd'].min():.1%}"
        )

    return df
