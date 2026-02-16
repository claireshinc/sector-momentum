"""Transaction and holding cost model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CostModel:
    """Transaction and holding costs."""

    # Transaction costs (one-way, bps)
    etf_cost_bps: float = 5.0
    equity_cost_bps: float = 15.0

    # Short borrow costs (annualized, bps)
    etf_borrow_bps: float = 25.0  # ETFs generally easy to borrow
    equity_borrow_bps: float = 100.0  # Equities vary widely

    def transaction_cost(self, turnover: float, is_etf: bool = True) -> float:
        """Cost for given one-way turnover."""
        bps = self.etf_cost_bps if is_etf else self.equity_cost_bps
        return turnover * bps / 10000

    def daily_borrow_cost(self, short_weight: float, is_etf: bool = True) -> float:
        """Daily cost for holding short positions."""
        bps = self.etf_borrow_bps if is_etf else self.equity_borrow_bps
        annual_cost = abs(short_weight) * bps / 10000
        return annual_cost / 252


def compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """One-way turnover."""
    combined_index = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(combined_index, fill_value=0)
    new = new_weights.reindex(combined_index, fill_value=0)
    return abs(new - old).sum() / 2
