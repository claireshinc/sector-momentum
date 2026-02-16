"""Crash control: regime detection, controllers, vol targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest.signals.momentum import compute_momentum_12_1
from portfolio_backtest.signals.reversal import compute_reversal_1m
from portfolio_backtest.portfolio.construction import build_long_short_portfolio


# ──────────────────────────────────────────────────────────────────────────────
# Regime Detection
# ──────────────────────────────────────────────────────────────────────────────


def compute_regime_state(
    spy_returns: pd.Series,
    as_of: str,
    thresholds: dict,
) -> dict:
    """
    Detect crash/panic regime using data through as_of only.
    Returns regime flags + computed values.
    """
    ret = spy_returns.loc[:as_of]

    if len(ret) < 126:
        return {
            "in_drawdown": False,
            "high_vol": False,
            "rebounding": False,
            "is_panic": False,
            "is_crash_risk": False,
            "crash_score": 0.0,
            "vol_1m": 0.0,
            "ret_1m": 0.0,
            "ret_6m": 0.0,
        }

    # Cumulative returns (NOT sum of pct_change!)
    ret_6m = (1 + ret.tail(126)).prod() - 1
    ret_1m = (1 + ret.tail(21)).prod() - 1

    # Realized volatility (annualized)
    vol_1m = ret.tail(21).std() * np.sqrt(252)

    # Regime flags
    in_drawdown = ret_6m < -0.05  # 5% drawdown threshold
    high_vol = vol_1m > thresholds["vol_80pct"]
    rebounding = ret_1m > thresholds["ret_1m_70pct"]

    # Crash risk score (0 to 1) for blending
    crash_score = 0.0
    if in_drawdown:
        crash_score += 0.4
    if high_vol:
        crash_score += 0.4
    if rebounding:
        crash_score += 0.2

    return {
        "in_drawdown": in_drawdown,
        "high_vol": high_vol,
        "rebounding": rebounding,
        "is_panic": in_drawdown and high_vol and rebounding,
        "is_crash_risk": in_drawdown and high_vol,
        "crash_score": crash_score,
        "vol_1m": vol_1m,
        "ret_1m": ret_1m,
        "ret_6m": ret_6m,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Threshold Estimation
# ──────────────────────────────────────────────────────────────────────────────


def estimate_thresholds(spy_returns: pd.Series, train_end: str) -> dict:
    """
    Estimate regime thresholds using training data only.
    These are FROZEN at train_end for pure OOS testing.
    """
    train_ret = spy_returns.loc[:train_end]

    # Rolling metrics over training period
    rolling_vol = train_ret.rolling(21).std() * np.sqrt(252)
    rolling_ret_1m = train_ret.rolling(21).apply(lambda x: (1 + x).prod() - 1, raw=False)

    return {
        "vol_80pct": rolling_vol.quantile(0.80),
        "vol_90pct": rolling_vol.quantile(0.90),
        "ret_1m_70pct": rolling_ret_1m.quantile(0.70),
        "ret_1m_80pct": rolling_ret_1m.quantile(0.80),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Regime State Tracker (With Persistence)
# ──────────────────────────────────────────────────────────────────────────────


class RegimeTracker:
    """
    Track regime state with persistence filter to avoid whipsaw.
    Regime must be ON for min_persist periods before switching.
    """

    def __init__(self, min_persist_periods: int = 1):
        self.min_persist = min_persist_periods
        self.current_regime = "normal"
        self.pending_regime = None
        self.pending_count = 0

    def update(self, detected_regime: str) -> str:
        """
        Update tracker and return effective regime.
        """
        if detected_regime == self.current_regime:
            self.pending_regime = None
            self.pending_count = 0
            return self.current_regime

        if detected_regime == self.pending_regime:
            self.pending_count += 1
            if self.pending_count >= self.min_persist:
                self.current_regime = detected_regime
                self.pending_regime = None
                self.pending_count = 0
        else:
            self.pending_regime = detected_regime
            self.pending_count = 1

        return self.current_regime

    def reset(self):
        self.current_regime = "normal"
        self.pending_regime = None
        self.pending_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# Controller 1: Panic-State Throttle
# ──────────────────────────────────────────────────────────────────────────────


def apply_panic_throttle(
    weights: pd.Series,
    regime_state: dict,
    regime_tracker: RegimeTracker,
) -> tuple[pd.Series, str]:
    """
    Remove short leg during panic state.
    Returns (adjusted_weights, action_taken).
    """
    detected = "panic" if regime_state["is_panic"] else "normal"
    effective_regime = regime_tracker.update(detected)

    if effective_regime == "panic":
        adjusted = weights.clip(lower=0)
        # Renormalize long leg
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()
        return adjusted, "shorts_removed"

    return weights, "no_action"


# ──────────────────────────────────────────────────────────────────────────────
# Controller 2: Signal Switching (With Blending)
# ──────────────────────────────────────────────────────────────────────────────


def apply_signal_switch(
    prices: pd.DataFrame,
    as_of: str,
    regime_state: dict,
    regime_tracker: RegimeTracker,
    k: int,
    blend_mode: bool = True,
) -> tuple[pd.Series, str]:
    """
    Switch or blend signals based on regime.

    blend_mode=True: Gradual blend using crash_score
    blend_mode=False: Hard switch at regime change
    """
    detected = "crash_risk" if regime_state["is_crash_risk"] else "normal"
    effective_regime = regime_tracker.update(detected)

    # Compute both signals
    mom_signal = compute_momentum_12_1(prices, as_of)
    rev_signal = compute_reversal_1m(prices, as_of)

    mom_weights = build_long_short_portfolio(mom_signal, k)

    if effective_regime == "normal":
        return mom_weights, "momentum_ls"

    if blend_mode:
        # Gradual blend: alpha increases with crash_score
        alpha = min(regime_state["crash_score"], 1.0)

        # In crash: blend toward long-only momentum + reversal
        mom_long_only = mom_weights.clip(lower=0)
        if mom_long_only.sum() > 0:
            mom_long_only = mom_long_only / mom_long_only.sum()

        rev_weights = build_long_short_portfolio(rev_signal, k)

        # Blend: (1-alpha)*mom_ls + alpha*(0.5*mom_long + 0.5*rev_ls)
        crash_portfolio = 0.5 * mom_long_only + 0.5 * rev_weights
        blended = (1 - alpha) * mom_weights + alpha * crash_portfolio

        return blended, f"blended_alpha_{alpha:.2f}"
    else:
        # Hard switch to long-only momentum
        long_only = mom_weights.clip(lower=0)
        if long_only.sum() > 0:
            long_only = long_only / long_only.sum()
        return long_only, "momentum_long_only"


# ──────────────────────────────────────────────────────────────────────────────
# Controller 3: Volatility Targeting
# ──────────────────────────────────────────────────────────────────────────────


def apply_vol_target(
    weights: pd.Series,
    regime_state: dict,
    target_vol: float = 0.10,
    max_leverage: float = 1.5,
    min_leverage: float = 0.25,
) -> tuple[pd.Series, str]:
    """
    Scale exposure to target portfolio volatility.
    """
    realized_vol = regime_state["vol_1m"]

    if realized_vol > 0:
        raw_leverage = target_vol / realized_vol
        leverage = np.clip(raw_leverage, min_leverage, max_leverage)
    else:
        leverage = 1.0

    scaled = weights * leverage
    action = f"vol_scaled_{leverage:.2f}x"

    return scaled, action
