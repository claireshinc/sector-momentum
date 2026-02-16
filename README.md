# Sector Momentum Backtest Framework

Crash-aware sector momentum with rigorous backtesting and counterfactual comparison.

---

## What This Does

Builds and backtests long/short sector momentum strategies on US sector ETFs, with three crash control mechanisms and a sentiment gating overlay. The core research question: **does crash-aware momentum improve risk-adjusted returns, and by how much?**

The dashboard runs all crash controllers side-by-side as counterfactuals against the same baseline, so you can see exactly what each layer adds or costs.

---

## Architecture

```
                    SIGNAL LAYER                    OVERLAY LAYER                 EXECUTION
                    ────────────                    ─────────────                 ─────────
               ┌─ 12-1 Momentum ─┐            ┌─ Sentiment Gate ─┐
  Prices ─────►│  6-1 Momentum   │──► Ranks ──►│  Crash Control   │──► Weights ──► Backtest
  (yfinance)   │  Blend          │             │  Vol Targeting    │    Engine      Engine
               └─────────────────┘             │  Turnover Cap     │
                                               └──────────────────┘
```

### Universe
- **10 SPDR Sector ETFs**: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE
- **Benchmark**: SPY

### Signal: 12-1 Momentum
```
momentum = price[t-21] / price[t-252] - 1
```
12-month return, skipping the most recent month to avoid short-term reversal contamination.

### Portfolio Construction
- Long top-k sectors, short bottom-k (default k=3)
- Equal weight within each leg
- Rank hysteresis buffer of 1 to reduce turnover
- Monthly rebalance on last NYSE trading day

### Three Strategies

| Strategy | Description |
|----------|-------------|
| **A** | Momentum L/S + sentiment gating (UMCSENT z-score scales short leg) |
| **B** | Pure momentum L/S baseline |
| **C** | Short pressure + momentum on equity universe (FINRA short volume) |

### Three Crash Controllers

| Controller | Mechanism |
|------------|-----------|
| **Panic Throttle** | Remove shorts during panic regime (drawdown + high vol + rebound) |
| **Signal Switch** | Blend toward reversal signal as crash score increases: `(1-a)*mom + a*(0.5*mom_long + 0.5*reversal)` |
| **Vol Target** | Scale exposure: `leverage = clip(10% / realized_vol, 0.25x, 1.5x)` |

Regime detection uses SPY data only through current date. Thresholds (vol 80th pct, return 70th pct) are estimated from training data and frozen for out-of-sample testing.

### Cost Model
- ETF transaction: 5 bps one-way
- Short borrow: 25 bps annualized
- Turnover cap: 100% one-way per rebalance
- Trades < 0.5% position zeroed out

### No-Lookahead Guarantees
All signals use data through rebalance date only. Sentiment applies a 1-month publication lag. Regime thresholds are frozen at training end. 19 tests verify these guarantees:
- Signal invariance (future data changes don't affect past signals)
- Sentiment lag verification
- Full pipeline invariance (corrupted future data doesn't change results)
- Threshold freeze verification

### Metrics (Correct Math)
- Returns: compounded `(1+r).prod() - 1`, not summed
- Drawdown: wealth-based `(wealth - peak) / peak`
- Sharpe: `CAGR / annualized_vol`
- Alpha/Beta: computed against SPY

---

## Quick Start

### Prerequisites
- Python 3.11+

### Install

```bash
git clone https://github.com/claireshinc/sector-momentum.git
cd sector-momentum
pip install -r requirements.txt
pip install -e .
```

### Run Tests

```bash
pytest portfolio_backtest/tests/ -v
```

Expected: 19 passed.

### Launch Dashboard

```bash
streamlit run portfolio_backtest/app/dashboard.py
```

1. Configure strategy, dates, and parameters in the sidebar
2. Click **Run Backtest**
3. Go to **Crash Control** tab and click **Run Counterfactual Comparison** to see all controllers side-by-side

### Run From Command Line

```bash
python run_backtest.py
```

Runs both presets (baseline and sentiment-gated) and prints comparison metrics.

### Run From Preset

```python
from portfolio_backtest.config.schema import load_preset
from portfolio_backtest.backtest.engine import BacktestEngine
from portfolio_backtest.data.prices import load_prices

preset = load_preset("sector_momentum_baseline")
config = preset.to_backtest_config()
prices = load_prices(preset.universe.symbols + [preset.universe.benchmark])
engine = BacktestEngine(prices, config)

result = engine.run(
    strategy=preset.strategy,
    crash_controller=preset.crash_controller,
    train_start=preset.dates.train_start,
    train_end=preset.dates.train_end,
    test_start=preset.dates.test_start,
    test_end=preset.dates.test_end,
    k=preset.params.k,
)

print(result.metrics)
```

---

## Project Structure

```
portfolio_backtest/
├── config/
│   ├── presets/                    # YAML strategy configs
│   │   ├── sector_momentum_baseline.yaml
│   │   └── sentiment_gated_crash_aware.yaml
│   └── schema.py                  # Pydantic validation + preset loader
├── data/
│   ├── prices.py                  # yfinance + parquet caching
│   ├── sentiment.py               # FRED UMCSENT with publication lag
│   ├── short_volume.py            # FINRA daily short volume
│   └── calendar.py                # NYSE trading days
├── signals/
│   ├── momentum.py                # 12-1, 6-1, blend
│   ├── reversal.py                # 1-month, 1-week
│   └── short_pressure.py          # FINRA-based composite
├── portfolio/
│   ├── construction.py            # L/S builder with rank hysteresis
│   ├── crash_control.py           # Regime detection + 3 controllers
│   └── overlays.py                # Sentiment gate, turnover cap
├── backtest/
│   ├── engine.py                  # Core loop with no-lookahead
│   ├── costs.py                   # Transaction + borrow costs
│   └── integrity.py               # Backtest integrity checks
├── validation/
│   ├── metrics.py                 # Wealth-based drawdown, CAGR, Sharpe
│   ├── stress.py                  # 10 crisis windows
│   └── stability.py               # Multi-split OOS analysis
├── app/
│   └── dashboard.py               # Streamlit dashboard (7 tabs)
└── tests/
    ├── test_lookahead.py          # 4 no-lookahead tests
    ├── test_math.py               # 6 correctness tests
    └── test_pipeline.py           # 9 integration tests
```

---

## Dashboard Tabs

| Tab | What It Shows |
|-----|---------------|
| **Performance** | Equity curve, drawdown, Sharpe/CAGR/MaxDD/Alpha metrics |
| **Crash Control** | Counterfactual comparison of all controllers on same data |
| **Weights** | Portfolio weights over time, turnover per rebalance |
| **Stress Test** | Returns during 10 crisis windows (GFC, COVID, etc.) vs SPY |
| **Stability** | OOS robustness across multiple train/test splits |
| **Compare Runs** | Side-by-side config + metric diff of saved runs |
| **Research Memo** | Hypothesis, implementation choices, known limitations |

---

## Presets

**sector_momentum_baseline** — Pure momentum L/S, no overlays, no crash control. The counterfactual baseline.

**sentiment_gated_crash_aware** — Momentum with UMCSENT sentiment gating on the short leg and signal-switching crash control with gradual blending.

Both use frozen thresholds estimated from 2005-2018 training data and test on 2019-2024.

---

## OOS Results (2019-2024)

| Metric | Baseline (B) | Sentiment+Crash (A) |
|--------|-------------|---------------------|
| CAGR | 2.97% | 1.66% |
| Sharpe | 0.18 | 0.10 |
| Max DD | -28.87% | -32.69% |
| Vol | 16.18% | 16.86% |
| Beta | -0.13 | -0.05 |
| Alpha | +5.08% | +2.38% |

---

## Known Limitations

1. **Survivorship bias** — ETF universe is the current list, not point-in-time
2. **Execution** — Assumes next-day close with fixed costs; no market impact modeling
3. **Sentiment proxy** — UMCSENT is monthly and lagged; may not capture real-time shifts
4. **Borrow costs** — Simplified flat rate; real costs vary by name and time
5. **Small universe** — 10 sectors limits signal dispersion and diversification
6. **Dollar-neutral** — By design, will underperform in sustained bull markets; alpha is the right metric, not absolute return
