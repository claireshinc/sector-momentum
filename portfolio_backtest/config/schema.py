"""Pydantic configuration schema and preset loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel


SECTOR_ETFS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE']
BENCHMARK = 'SPY'

DEFAULT_EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
    'NFLX', 'CRM', 'INTC', 'VZ', 'KO', 'PEP', 'MRK', 'ABT', 'TMO', 'CSCO',
    'AVGO', 'ACN',
]

ThresholdMode = Literal['frozen_at_train_end', 'expanding']


class UniverseConfig(BaseModel):
    type: str = "sector_etfs"
    symbols: list[str] = SECTOR_ETFS
    benchmark: str = BENCHMARK
    file: Optional[str] = None


class DatesConfig(BaseModel):
    train_start: str = "2005-01-01"
    train_end: str = "2018-12-31"
    test_start: str = "2019-01-01"
    test_end: str = "2024-12-31"


class ParamsConfig(BaseModel):
    k: int = 3
    momentum_lookback: int = 252
    skip_recent: int = 21
    short_pressure_lookback: int = 5
    short_pressure_smooth: int = 5


class CostsConfig(BaseModel):
    etf_cost_bps: float = 5.0
    etf_borrow_bps: float = 25.0
    equity_cost_bps: float = 15.0
    equity_borrow_bps: float = 100.0


class RegimeConfig(BaseModel):
    threshold_mode: ThresholdMode = "frozen_at_train_end"
    persist_periods: int = 1
    blend_mode: bool = False


class SentimentConfig(BaseModel):
    enabled: bool = False
    lag_months: int = 1
    high_threshold: float = 0.5
    low_threshold: float = -0.5


class StrategyPreset(BaseModel):
    name: str = "Unnamed Strategy"
    description: str = ""
    strategy: str = "A"
    crash_controller: Optional[str] = None
    universe: UniverseConfig = UniverseConfig()
    dates: DatesConfig = DatesConfig()
    params: ParamsConfig = ParamsConfig()
    costs: CostsConfig = CostsConfig()
    regime: RegimeConfig = RegimeConfig()
    sentiment: SentimentConfig = SentimentConfig()

    def to_backtest_config(self) -> "BacktestConfig":
        from portfolio_backtest.backtest.costs import CostModel
        return BacktestConfig(
            rebalance_freq='M',
            execution='next_close',
            cost_model=CostModel(
                etf_cost_bps=self.costs.etf_cost_bps,
                equity_cost_bps=self.costs.equity_cost_bps,
                etf_borrow_bps=self.costs.etf_borrow_bps,
                equity_borrow_bps=self.costs.equity_borrow_bps,
            ),
            max_turnover=1.0,
            min_trade_threshold=0.005,
            threshold_mode=self.regime.threshold_mode,
            regime_persist_periods=self.regime.persist_periods,
            sentiment_lag_months=self.sentiment.lag_months,
        )


@dataclass
class BacktestConfig:
    rebalance_freq: Literal['D', 'W', 'M'] = 'M'
    execution: Literal['next_close'] = 'next_close'
    cost_model: object = None
    max_turnover: float = 1.0
    min_trade_threshold: float = 0.005
    threshold_mode: ThresholdMode = 'frozen_at_train_end'
    regime_persist_periods: int = 1
    sentiment_lag_months: int = 1

    def __post_init__(self):
        if self.cost_model is None:
            from portfolio_backtest.backtest.costs import CostModel
            self.cost_model = CostModel()


PRESETS_DIR = Path(__file__).parent / "presets"


def load_preset(name: str) -> StrategyPreset:
    """Load a strategy preset from YAML file."""
    yaml_path = PRESETS_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Preset not found: {yaml_path}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return StrategyPreset(**data)


def list_presets() -> list[str]:
    """List available preset names."""
    if not PRESETS_DIR.exists():
        return []
    return [p.stem for p in PRESETS_DIR.glob("*.yaml")]


def load_universe(path: str = None) -> list[str]:
    """Load universe from CSV or return default."""
    if path and Path(path).exists():
        import pandas as pd
        df = pd.read_csv(path)
        return df['symbol'].tolist()
    return DEFAULT_EQUITY_UNIVERSE
