"""Crisis window / stress test analysis."""

from __future__ import annotations

import pandas as pd

from portfolio_backtest.validation.metrics import (
    compute_drawdown_series,
    compute_recovery_days,
)

CRISIS_WINDOWS = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "Flash Crash": ("2010-05-01", "2010-05-31"),
    "Euro Crisis": ("2011-07-01", "2011-12-31"),
    "Taper Tantrum": ("2013-05-01", "2013-08-31"),
    "China Deval": ("2015-08-01", "2015-09-30"),
    "Volmageddon": ("2018-01-29", "2018-02-09"),
    "Q4 2018": ("2018-10-01", "2018-12-31"),
    "COVID Crash": ("2020-02-19", "2020-03-23"),
    "Inflation 2022": ("2022-01-01", "2022-10-31"),
    "Regional Banks": ("2023-03-01", "2023-05-15"),
}


def stress_test_table(
    returns: pd.Series,
    benchmark: pd.Series,
    windows: dict = None,
) -> pd.DataFrame:
    """
    Performance during crisis windows.
    """
    if windows is None:
        windows = CRISIS_WINDOWS

    results = []

    for name, (start, end) in windows.items():
        try:
            crisis_ret = returns.loc[start:end]
            bench_ret = benchmark.loc[start:end]

            if len(crisis_ret) == 0:
                continue

            strat_total = (1 + crisis_ret).prod() - 1
            bench_total = (1 + bench_ret).prod() - 1

            dd = compute_drawdown_series(crisis_ret)
            recovery = compute_recovery_days(returns, start, end)

            results.append(
                {
                    "Crisis": name,
                    "Period": f"{start} -> {end}",
                    "Strategy": f"{strat_total:.1%}",
                    "Benchmark": f"{bench_total:.1%}",
                    "Excess": f"{strat_total - bench_total:.1%}",
                    "Max DD": f"{dd.min():.1%}",
                    "Recovery (days)": recovery if recovery > 0 else "N/R",
                }
            )
        except Exception:
            continue

    return pd.DataFrame(results)
