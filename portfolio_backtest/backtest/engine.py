"""Core backtest engine with strict no-lookahead guarantees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd

from portfolio_backtest.backtest.costs import CostModel, compute_turnover
from portfolio_backtest.config.schema import BacktestConfig
from portfolio_backtest.data.sentiment import compute_sentiment_zscore
from portfolio_backtest.portfolio.construction import build_long_short_portfolio
from portfolio_backtest.portfolio.crash_control import (
    RegimeTracker,
    apply_panic_throttle,
    apply_signal_switch,
    apply_vol_target,
    compute_regime_state,
    estimate_thresholds,
)
from portfolio_backtest.portfolio.overlays import apply_sentiment_gate, apply_turnover_cap
from portfolio_backtest.signals.momentum import compute_momentum_12_1
from portfolio_backtest.signals.short_pressure import strategy_c_signal
from portfolio_backtest.validation.metrics import compute_metrics


@dataclass
class BacktestResult:
    returns_gross: pd.Series
    returns_net: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    regime_states: pd.DataFrame
    actions: pd.DataFrame
    metrics: dict
    config: BacktestConfig


class BacktestEngine:
    def __init__(self, prices: pd.DataFrame, config: BacktestConfig):
        self.prices = prices
        self.returns = prices.pct_change()
        self.config = config
        self.cost_model = config.cost_model

    def run(
        self,
        strategy: Literal["A", "B", "C"],
        crash_controller: Optional[
            Literal["panic_throttle", "signal_switch", "vol_target"]
        ],
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        sentiment: pd.Series = None,
        short_data: pd.DataFrame = None,
        k: int = 3,
        blend_mode: bool = True,
        **kwargs,
    ) -> BacktestResult:
        """
        Main backtest loop with strict no-lookahead.
        """
        # 1. Estimate thresholds from training data (frozen)
        spy_col = "SPY" if "SPY" in self.returns.columns else self.returns.columns[0]
        spy_ret = self.returns[spy_col]

        if self.config.threshold_mode == "frozen_at_train_end":
            thresholds = estimate_thresholds(spy_ret, train_end)
        else:
            # Expanding: will re-estimate at each rebalance date
            thresholds = estimate_thresholds(spy_ret, train_end)

        # 2. Initialize trackers
        regime_tracker = RegimeTracker(self.config.regime_persist_periods)

        # 3. Get rebalance dates
        rebal_dates = self._get_rebalance_dates(test_start, test_end)

        # 4. Storage
        weights_history = {}
        regime_history = []
        actions_history = []

        prev_weights = None

        for t in rebal_dates:
            t_str = t.strftime("%Y-%m-%d") if hasattr(t, "strftime") else str(t)

            # --- NO LOOKAHEAD: all computations use data <= t ---

            # Expanding threshold mode: re-estimate using data through t
            if self.config.threshold_mode == "expanding":
                thresholds = estimate_thresholds(spy_ret, t_str)

            # Compute regime state
            regime_state = compute_regime_state(spy_ret, t_str, thresholds)

            # Compute sentiment z-score (with publication lag)
            if sentiment is not None:
                sent_z = compute_sentiment_zscore(
                    sentiment,
                    t_str,
                    train_start,
                    lag_months=self.config.sentiment_lag_months,
                )
            else:
                sent_z = 0.0

            # Compute raw signal based on strategy
            if strategy == "A":
                signal = compute_momentum_12_1(self.prices, t_str)
                weights = build_long_short_portfolio(signal, k, prev_weights)
                weights = apply_sentiment_gate(weights, sent_z)
                action = "momentum_sentiment_gated"

            elif strategy == "B":
                signal = compute_momentum_12_1(self.prices, t_str)
                weights = build_long_short_portfolio(signal, k, prev_weights)
                action = "momentum_baseline"

            elif strategy == "C":
                if short_data is None:
                    raise ValueError("Strategy C requires short_data")
                weights = strategy_c_signal(
                    self.prices, short_data, t_str, train_start, sent_z
                )
                action = "short_pressure_momentum"
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Apply crash controller
            if crash_controller == "panic_throttle":
                weights, ctrl_action = apply_panic_throttle(
                    weights, regime_state, regime_tracker
                )
                action = f"{action}|{ctrl_action}"

            elif crash_controller == "signal_switch":
                weights, ctrl_action = apply_signal_switch(
                    self.prices, t_str, regime_state, regime_tracker, k, blend_mode
                )
                action = f"{action}|{ctrl_action}"

            elif crash_controller == "vol_target":
                weights, ctrl_action = apply_vol_target(weights, regime_state)
                action = f"{action}|{ctrl_action}"

            # Apply turnover cap
            if prev_weights is not None:
                weights = apply_turnover_cap(
                    prev_weights,
                    weights,
                    self.config.max_turnover,
                    self.config.min_trade_threshold,
                )

            # Store
            weights_history[t] = weights
            regime_history.append({"date": t, **regime_state})
            actions_history.append({"date": t, "action": action})
            prev_weights = weights

        # 5. Compute returns
        return self._compute_returns(
            weights_history, regime_history, actions_history, spy_col
        )

    def _compute_returns(
        self, weights_history, regime_history, actions_history, spy_col
    ):
        """Compute gross and net returns from weights."""
        dates = sorted(weights_history.keys())

        if len(dates) < 2:
            empty_series = pd.Series(dtype=float)
            return BacktestResult(
                returns_gross=empty_series,
                returns_net=empty_series,
                weights=pd.DataFrame(),
                turnover=empty_series,
                costs=empty_series,
                regime_states=pd.DataFrame(regime_history),
                actions=pd.DataFrame(actions_history),
                metrics={},
                config=self.config,
            )

        daily_returns_gross = []
        daily_returns_net = []
        daily_turnover = []
        daily_costs = []

        prev_weights = None

        for i, t in enumerate(dates[:-1]):
            next_t = dates[i + 1]
            weights = weights_history[t]

            # Returns between rebalance dates
            period_prices = self.prices.loc[t:next_t]
            period_returns = period_prices.pct_change().iloc[1:]  # Skip first (t itself)

            for day, day_ret in period_returns.iterrows():
                # Portfolio return
                aligned_weights = weights.reindex(day_ret.index, fill_value=0)
                port_ret = (aligned_weights * day_ret).sum()
                daily_returns_gross.append({"date": day, "return": port_ret})

                # Costs (transaction on rebalance day, borrow daily)
                cost = 0.0
                if day == period_returns.index[0] and prev_weights is not None:
                    turnover = compute_turnover(prev_weights, weights)
                    cost += self.cost_model.transaction_cost(turnover)
                    daily_turnover.append({"date": day, "turnover": turnover})

                # Daily borrow cost
                short_weight = abs(aligned_weights[aligned_weights < 0].sum())
                cost += self.cost_model.daily_borrow_cost(short_weight)

                daily_costs.append({"date": day, "cost": cost})
                daily_returns_net.append({"date": day, "return": port_ret - cost})

            prev_weights = weights

        # Convert to Series
        if not daily_returns_gross:
            empty_series = pd.Series(dtype=float)
            return BacktestResult(
                returns_gross=empty_series,
                returns_net=empty_series,
                weights=pd.DataFrame(weights_history).T,
                turnover=empty_series,
                costs=empty_series,
                regime_states=pd.DataFrame(regime_history),
                actions=pd.DataFrame(actions_history),
                metrics={},
                config=self.config,
            )

        returns_gross = pd.DataFrame(daily_returns_gross).set_index("date")["return"]
        returns_net = pd.DataFrame(daily_returns_net).set_index("date")["return"]
        turnover = (
            pd.DataFrame(daily_turnover).set_index("date")["turnover"]
            if daily_turnover
            else pd.Series(dtype=float)
        )
        costs = pd.DataFrame(daily_costs).set_index("date")["cost"]

        # Compute metrics
        spy_ret = self.returns[spy_col].reindex(returns_net.index)
        metrics = compute_metrics(returns_net, spy_ret)

        return BacktestResult(
            returns_gross=returns_gross,
            returns_net=returns_net,
            weights=pd.DataFrame(weights_history).T,
            turnover=turnover,
            costs=costs,
            regime_states=pd.DataFrame(regime_history),
            actions=pd.DataFrame(actions_history),
            metrics=metrics,
            config=self.config,
        )

    def _get_rebalance_dates(self, start: str, end: str) -> list:
        """Get rebalance dates based on frequency."""
        dates = self.prices.loc[start:end].index

        if len(dates) == 0:
            return []

        if self.config.rebalance_freq == "M":
            # Last trading day of each month
            return dates.to_series().groupby(pd.Grouper(freq="ME")).last().dropna().tolist()
        elif self.config.rebalance_freq == "W":
            return (
                dates.to_series()
                .groupby(pd.Grouper(freq="W-FRI"))
                .last()
                .dropna()
                .tolist()
            )
        else:  # Daily
            return dates.tolist()
