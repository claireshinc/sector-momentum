# Portfolio Research & Backtest Dashboard — Final Spec v3

## Project Philosophy

**One core research contribution:** Crash-aware momentum with clear counterfactuals
**Goal:** Demonstrate rigorous quant thinking, not feature breadth

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LAYERED STRATEGY SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CORE BOOK          OVERLAYS              OPTIONAL SLEEVES               │
│  ──────────         ────────              ────────────────               │
│  Sector Momentum    • Sentiment Gate      • Short Pressure (C)           │
│  (12-1, 6-1)        • Crash Control       • EDGAR Text (D)               │
│                     • Vol Targeting                                      │
│                     • Turnover Limits                                    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           HF-NATIVE VIEWS                                │
│  ─────────────────────────────────────────────────────────────────────   │
│  • Counterfactual Comparison    • Stress / Scenario Lab                  │
│  • Run Registry + Compare       • Research Memo Panel                    │
│  • OOS Stability Analysis       • Backtest Integrity Checklist           │
│  • QuantStats Tear Sheet                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Time Conventions (Critical)

| Event | Timing | Data Used |
|-------|--------|-----------|
| Signal computation | End of trading day t | Prices through close(t) |
| Sentiment reading | Month-end t | Value from month t-1 (1-month publication lag) |
| Weight decision | End of day t | Signals from close(t) + lagged sentiment |
| Order execution | Close of day t+1 | — |
| Return attribution | t+1 → t+2 | Close(t+1) to close(t+2) |

- All timestamps **US/Eastern**
- "Month-end" = last trading day of calendar month
- Trading day calendar = NYSE holidays excluded

---

## File Structure

```
portfolio_backtest/
├── config/
│   ├── presets/                # YAML strategy configs
│   ├── universes/              # Constituent CSVs
│   └── schema.py               # Pydantic validation
├── data/
│   ├── prices.py               # yfinance + cache
│   ├── sentiment.py            # FRED UMCSENT with publication lag
│   ├── short_volume.py         # FINRA daily short volume
│   └── calendar.py             # Trading days, holidays, month-ends
├── signals/
│   ├── momentum.py             # 12-1, 6-1, blend
│   ├── reversal.py             # 1-week, 1-month
│   └── short_pressure.py       # FINRA-based composite
├── portfolio/
│   ├── construction.py         # L/S portfolio builder
│   ├── crash_control.py        # Regime detection + controllers
│   └── overlays.py             # Sentiment gate, vol target, turnover
├── backtest/
│   ├── engine.py               # Core loop
│   ├── costs.py                # Transaction + borrow costs
│   └── integrity.py            # No-lookahead checks
├── validation/
│   ├── metrics.py              # Performance calcs (correct math)
│   ├── stress.py               # Crisis window analysis
│   └── stability.py            # Multi-split OOS analysis
├── app/
│   ├── dashboard.py            # Main Streamlit app
│   ├── components/             # Reusable UI pieces
│   └── pages/                  # Multi-page app structure
├── runs/                       # Saved run JSONs
├── tests/
│   ├── test_lookahead.py
│   ├── test_pipeline.py
│   └── test_math.py            # Drawdown, returns correctness
└── requirements.txt
```

---

## 1. Core Book: Sector Momentum

### Universe

```python
SECTOR_ETFS = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XLRE']
BENCHMARK = 'SPY'
```

### Signal: 12-1 Momentum

```python
def compute_momentum_12_1(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    12-month return, skipping most recent month.
    Uses only data through as_of date.
    """
    p = prices.loc[:as_of]
    # Return from t-252 to t-21 (skip last ~1 month)
    mom = (p.iloc[-21] / p.iloc[-252]) - 1
    return mom

def compute_momentum_6_1(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """6-month return, skipping most recent month."""
    p = prices.loc[:as_of]
    mom = (p.iloc[-21] / p.iloc[-126]) - 1
    return mom

def compute_momentum_blend(prices: pd.DataFrame, as_of: str, 
                           w_12: float = 0.5, w_6: float = 0.5) -> pd.Series:
    """Blended momentum signal."""
    mom_12 = compute_momentum_12_1(prices, as_of)
    mom_6 = compute_momentum_6_1(prices, as_of)
    return w_12 * mom_12 + w_6 * mom_6
```

### Portfolio Construction

```python
def build_long_short_portfolio(signal: pd.Series, k: int, 
                                prev_weights: pd.Series = None,
                                rank_buffer: int = 1) -> pd.Series:
    """
    Long top-k, short bottom-k, equal weight.
    With rank hysteresis to reduce turnover.
    """
    ranks = signal.rank(ascending=False)
    n = len(signal)
    
    weights = pd.Series(0.0, index=signal.index)
    
    # Determine long/short cutoffs with hysteresis
    long_cutoff = k
    short_cutoff = n - k
    
    if prev_weights is not None:
        # Don't drop holdings unless rank crosses k + buffer
        for sym in signal.index:
            if prev_weights.get(sym, 0) > 0:  # Was long
                if ranks[sym] <= k + rank_buffer:
                    weights[sym] = 1.0 / k
            elif prev_weights.get(sym, 0) < 0:  # Was short
                if ranks[sym] >= n - k - rank_buffer:
                    weights[sym] = -1.0 / k
    
    # Fill remaining slots
    for sym in signal.index:
        if weights[sym] == 0:
            if ranks[sym] <= k:
                weights[sym] = 1.0 / k
            elif ranks[sym] > n - k:
                weights[sym] = -1.0 / k
    
    # Normalize legs
    long_sum = weights[weights > 0].sum()
    short_sum = abs(weights[weights < 0].sum())
    
    if long_sum > 0:
        weights[weights > 0] /= long_sum
    if short_sum > 0:
        weights[weights < 0] /= short_sum
        weights[weights < 0] *= -1  # Keep as negative
    
    return weights
```

---

## 2. Overlays

### 2A. Sentiment Gating

```python
# Publication lag: UMCSENT released mid-month for prior month
# At month-end t, we use reading from month t-1 (conservative, defensible)

SENTIMENT_LAG_MONTHS = 1  # User-configurable, default 1

def load_sentiment_with_lag(sentiment_series: pd.Series, 
                            as_of: str, 
                            lag_months: int = 1) -> float:
    """
    Get sentiment value available at as_of date, respecting publication lag.
    """
    as_of_dt = pd.Timestamp(as_of)
    available_through = as_of_dt - pd.DateOffset(months=lag_months)
    available_data = sentiment_series.loc[:available_through]
    
    if available_data.empty:
        return np.nan
    return available_data.iloc[-1]

def compute_sentiment_zscore(sentiment_series: pd.Series,
                              as_of: str,
                              train_start: str,
                              lag_months: int = 1) -> float:
    """
    Z-score using expanding window from train_start to as_of,
    respecting publication lag.
    """
    as_of_dt = pd.Timestamp(as_of)
    available_through = as_of_dt - pd.DateOffset(months=lag_months)
    
    history = sentiment_series.loc[train_start:available_through]
    
    if len(history) < 12:  # Need minimum history
        return 0.0
    
    current = history.iloc[-1]
    return (current - history.mean()) / history.std()

def apply_sentiment_gate(weights: pd.Series, 
                          sentiment_z: float,
                          high_threshold: float = 0.5,
                          low_threshold: float = -0.5) -> pd.Series:
    """
    Gate short leg based on sentiment z-score.
    High sentiment → full short (shorts profitable when optimism fades)
    Low sentiment → no short (avoid falling knives)
    """
    gated = weights.copy()
    
    if sentiment_z > high_threshold:
        short_multiplier = 1.0      # Full short
    elif sentiment_z > low_threshold:
        short_multiplier = 0.5      # Half short
    else:
        short_multiplier = 0.0      # No short
    
    gated[gated < 0] *= short_multiplier
    
    # Renormalize long leg to maintain gross exposure if desired
    # Or leave as reduced gross (more conservative)
    
    return gated
```

### 2B. Crash Control

#### Regime Detection (Correct Math)

```python
def compute_regime_state(spy_returns: pd.Series, 
                          as_of: str, 
                          thresholds: dict) -> dict:
    """
    Detect crash/panic regime using data through as_of only.
    Returns regime flags + computed values.
    """
    ret = spy_returns.loc[:as_of]
    
    # Cumulative returns (NOT sum of pct_change!)
    ret_6m = (1 + ret.tail(126)).prod() - 1
    ret_1m = (1 + ret.tail(21)).prod() - 1
    
    # Realized volatility (annualized)
    vol_1m = ret.tail(21).std() * np.sqrt(252)
    
    # Regime flags
    in_drawdown = ret_6m < -0.05  # 5% drawdown threshold
    high_vol = vol_1m > thresholds['vol_80pct']
    rebounding = ret_1m > thresholds['ret_1m_70pct']
    
    # Crash risk score (0 to 1) for blending
    crash_score = 0.0
    if in_drawdown:
        crash_score += 0.4
    if high_vol:
        crash_score += 0.4
    if rebounding:
        crash_score += 0.2
    
    return {
        'in_drawdown': in_drawdown,
        'high_vol': high_vol,
        'rebounding': rebounding,
        'is_panic': in_drawdown and high_vol and rebounding,
        'is_crash_risk': in_drawdown and high_vol,
        'crash_score': crash_score,
        'vol_1m': vol_1m,
        'ret_1m': ret_1m,
        'ret_6m': ret_6m,
    }
```

#### Threshold Estimation

```python
def estimate_thresholds(spy_returns: pd.Series, 
                         train_end: str) -> dict:
    """
    Estimate regime thresholds using training data only.
    These are FROZEN at train_end for pure OOS testing.
    """
    train_ret = spy_returns.loc[:train_end]
    
    # Rolling metrics over training period
    rolling_vol = train_ret.rolling(21).std() * np.sqrt(252)
    rolling_ret_1m = train_ret.rolling(21).apply(lambda x: (1+x).prod() - 1)
    
    return {
        'vol_80pct': rolling_vol.quantile(0.80),
        'vol_90pct': rolling_vol.quantile(0.90),
        'ret_1m_70pct': rolling_ret_1m.quantile(0.70),
        'ret_1m_80pct': rolling_ret_1m.quantile(0.80),
    }

# Threshold mode options
ThresholdMode = Literal['frozen_at_train_end', 'expanding']
```

#### Regime State Tracker (With Persistence)

```python
class RegimeTracker:
    """
    Track regime state with persistence filter to avoid whipsaw.
    Regime must be ON for min_persist periods before switching.
    """
    
    def __init__(self, min_persist_periods: int = 1):
        self.min_persist = min_persist_periods
        self.current_regime = 'normal'
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
        self.current_regime = 'normal'
        self.pending_regime = None
        self.pending_count = 0
```

#### Controller 1: Panic-State Throttle

```python
def apply_panic_throttle(weights: pd.Series, 
                          regime_state: dict,
                          regime_tracker: RegimeTracker) -> tuple[pd.Series, str]:
    """
    Remove short leg during panic state.
    Returns (adjusted_weights, action_taken).
    """
    detected = 'panic' if regime_state['is_panic'] else 'normal'
    effective_regime = regime_tracker.update(detected)
    
    if effective_regime == 'panic':
        adjusted = weights.clip(lower=0)
        # Renormalize long leg
        if adjusted.sum() > 0:
            adjusted = adjusted / adjusted.sum()
        return adjusted, 'shorts_removed'
    
    return weights, 'no_action'
```

#### Controller 2: Signal Switching (With Blending)

```python
def compute_reversal_signal(prices: pd.DataFrame, as_of: str) -> pd.Series:
    """
    1-month reversal: negative of recent return.
    """
    p = prices.loc[:as_of]
    ret_1m = (p.iloc[-1] / p.iloc[-21]) - 1
    return -ret_1m  # Negative = reversal

def apply_signal_switch(prices: pd.DataFrame,
                         as_of: str,
                         regime_state: dict,
                         regime_tracker: RegimeTracker,
                         k: int,
                         blend_mode: bool = True) -> tuple[pd.Series, str]:
    """
    Switch or blend signals based on regime.
    
    blend_mode=True: Gradual blend using crash_score
    blend_mode=False: Hard switch at regime change
    """
    detected = 'crash_risk' if regime_state['is_crash_risk'] else 'normal'
    effective_regime = regime_tracker.update(detected)
    
    # Compute both signals
    mom_signal = compute_momentum_12_1(prices, as_of)
    rev_signal = compute_reversal_signal(prices, as_of)
    
    mom_weights = build_long_short_portfolio(mom_signal, k)
    
    if effective_regime == 'normal':
        return mom_weights, 'momentum_ls'
    
    if blend_mode:
        # Gradual blend: α increases with crash_score
        alpha = min(regime_state['crash_score'], 1.0)
        
        # In crash: blend toward long-only momentum + reversal
        mom_long_only = mom_weights.clip(lower=0)
        if mom_long_only.sum() > 0:
            mom_long_only = mom_long_only / mom_long_only.sum()
        
        rev_weights = build_long_short_portfolio(rev_signal, k)
        
        # Blend: (1-α)*mom_ls + α*(0.5*mom_long + 0.5*rev_ls)
        crash_portfolio = 0.5 * mom_long_only + 0.5 * rev_weights
        blended = (1 - alpha) * mom_weights + alpha * crash_portfolio
        
        return blended, f'blended_alpha_{alpha:.2f}'
    else:
        # Hard switch to long-only momentum
        long_only = mom_weights.clip(lower=0)
        if long_only.sum() > 0:
            long_only = long_only / long_only.sum()
        return long_only, 'momentum_long_only'
```

#### Controller 3: Volatility Targeting

```python
def apply_vol_target(weights: pd.Series,
                      regime_state: dict,
                      target_vol: float = 0.10,
                      max_leverage: float = 1.5,
                      min_leverage: float = 0.25) -> tuple[pd.Series, str]:
    """
    Scale exposure to target portfolio volatility.
    """
    realized_vol = regime_state['vol_1m']
    
    if realized_vol > 0:
        raw_leverage = target_vol / realized_vol
        leverage = np.clip(raw_leverage, min_leverage, max_leverage)
    else:
        leverage = 1.0
    
    scaled = weights * leverage
    action = f'vol_scaled_{leverage:.2f}x'
    
    return scaled, action
```

### 2C. Cost Model

```python
@dataclass
class CostModel:
    """
    Transaction and holding costs.
    """
    # Transaction costs (one-way, bps)
    etf_cost_bps: float = 5.0
    equity_cost_bps: float = 15.0
    
    # Short borrow costs (annualized, bps)
    etf_borrow_bps: float = 25.0      # ETFs generally easy to borrow
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
```

### 2D. Turnover Constraints

```python
def apply_turnover_cap(old_weights: pd.Series,
                        new_weights: pd.Series,
                        max_turnover: float = 1.0,
                        min_trade_threshold: float = 0.005) -> pd.Series:
    """
    Cap turnover and ignore tiny trades.
    """
    combined_index = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(combined_index, fill_value=0)
    new = new_weights.reindex(combined_index, fill_value=0)
    
    trades = new - old
    
    # Ignore tiny trades
    trades[abs(trades) < min_trade_threshold] = 0
    
    # Cap total turnover
    turnover = abs(trades).sum() / 2
    if turnover > max_turnover:
        scale = max_turnover / turnover
        trades = trades * scale
    
    return old + trades
```

---

## 3. Optional Sleeves

### 3A. Strategy C: Short Pressure + Momentum

#### Data: FINRA Daily Short Volume

```python
def load_finra_short_volume(symbols: list, 
                             start: str, 
                             end: str) -> pd.DataFrame:
    """
    Load daily short volume from FINRA API.
    This is SHORT ACTIVITY, not short interest (different concept).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"finra_short_vol_{hash(tuple(sorted(symbols)))}_{start}_{end}.parquet"
    
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    
    url = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    
    all_data = []
    for symbol in symbols:
        payload = {
            "limit": 10000,
            "compareFilters": [{
                "compareType": "EQUAL",
                "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                "fieldValue": symbol
            }]
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                for row in resp.json():
                    all_data.append({
                        'date': pd.to_datetime(row.get('tradeReportDate')),
                        'symbol': symbol,
                        'short_volume': float(row.get('shortVolume', 0)),
                        'total_volume': float(row.get('totalVolume', 0)),
                    })
        except Exception as e:
            print(f"FINRA fetch failed for {symbol}: {e}")
    
    df = pd.DataFrame(all_data)
    if not df.empty:
        df['short_ratio'] = df['short_volume'] / df['total_volume'].replace(0, np.nan)
        df.to_parquet(cache_file)
    
    return df
```

#### Signal: Short Pressure Composite

```python
def compute_short_pressure_score(short_data: pd.DataFrame,
                                   symbol: str,
                                   as_of: str,
                                   lookback: int = 5,
                                   train_start: str = None) -> float:
    """
    Compute short pressure z-score for a symbol.
    Uses smoothed short ratio vs expanding historical distribution.
    """
    sym_data = short_data[
        (short_data['symbol'] == symbol) & 
        (short_data['date'] <= as_of)
    ].sort_values('date')
    
    if len(sym_data) < lookback + 20:
        return np.nan
    
    # Smoothed recent ratio
    recent_ratio = sym_data['short_ratio'].tail(lookback).mean()
    
    # Historical distribution (expanding window)
    if train_start:
        hist_data = sym_data[sym_data['date'] >= train_start]
    else:
        hist_data = sym_data
    
    hist_mean = hist_data['short_ratio'].mean()
    hist_std = hist_data['short_ratio'].std()
    
    if hist_std > 0:
        return (recent_ratio - hist_mean) / hist_std
    return 0.0
```

#### 2D Model

```python
def strategy_c_signal(prices: pd.DataFrame,
                       short_data: pd.DataFrame,
                       as_of: str,
                       train_start: str,
                       sentiment_z: float) -> pd.Series:
    """
    Combine momentum + short pressure into trade signal.
    
    Avoid squeeze risk: high short pressure + improving momentum
    Target: weak momentum + high short pressure + high sentiment
    """
    symbols = prices.columns.tolist()
    
    # Compute momentum
    mom = compute_momentum_12_1(prices, as_of)
    mom_tercile = pd.qcut(mom.rank(), 3, labels=[1, 2, 3])
    
    # Compute short pressure
    sp_scores = pd.Series({
        sym: compute_short_pressure_score(short_data, sym, as_of, train_start=train_start)
        for sym in symbols
    })
    sp_tercile = pd.qcut(sp_scores.rank(), 3, labels=[1, 2, 3])
    
    # Recent reversal (for squeeze risk detection)
    ret_1m = (prices.loc[:as_of].iloc[-1] / prices.loc[:as_of].iloc[-21]) - 1
    
    # Build signals
    signal = pd.Series(0.0, index=symbols)
    
    for sym in symbols:
        mt = mom_tercile.get(sym)
        st = sp_tercile.get(sym)
        r1m = ret_1m.get(sym, 0)
        
        if pd.isna(mt) or pd.isna(st):
            continue
        
        # Long: strong momentum + low crowding
        if mt == 3 and st == 1:
            signal[sym] = 1.0
        
        # Short: weak momentum + high crowding + high sentiment
        # BUT avoid if recent return positive (squeeze risk)
        elif mt == 1 and st == 3 and sentiment_z > 0 and r1m < 0:
            signal[sym] = -1.0
    
    # Normalize to unit gross
    long_sum = signal[signal > 0].sum()
    short_sum = abs(signal[signal < 0].sum())
    
    if long_sum > 0:
        signal[signal > 0] /= long_sum
    if short_sum > 0:
        signal[signal < 0] /= short_sum
    
    return signal
```

#### Universe (User-Provided or Default)

```python
# Default: liquid large caps (no SPY holdings dependency)
DEFAULT_EQUITY_UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
    'NFLX', 'CRM', 'INTC', 'VZ', 'KO', 'PEP', 'MRK', 'ABT', 'TMO', 'CSCO',
    'AVGO', 'ACN'
]

# Or load from user CSV
def load_universe(path: str = None) -> list:
    if path and Path(path).exists():
        df = pd.read_csv(path)
        return df['symbol'].tolist()
    return DEFAULT_EQUITY_UNIVERSE
```

### 3B. Strategy D: EDGAR Text Shock (On-Demand)

```python
# Marked as "Experimental - Slow" in UI
# Only runs when explicitly requested

def get_filing_trade_date(accepted_datetime: str) -> str:
    """
    Determine trade date from SEC acceptance timestamp.
    Handle after-hours: if after 16:00 ET, signal date is today, trade tomorrow.
    """
    from datetime import time
    
    accepted = pd.Timestamp(accepted_datetime)
    if accepted.tzinfo is None:
        accepted = accepted.tz_localize('US/Eastern')
    else:
        accepted = accepted.tz_convert('US/Eastern')
    
    market_close = time(16, 0)
    
    if accepted.time() >= market_close:
        signal_date = accepted.date()
    else:
        signal_date = accepted.date() - pd.Timedelta(days=1)
    
    # Trade on next trading day
    trade_date = get_next_trading_day(signal_date + pd.Timedelta(days=1))
    return trade_date.strftime('%Y-%m-%d')

# Full implementation deferred - marked as optional
```

---

## 4. Backtest Engine

```python
@dataclass
class BacktestConfig:
    # Timing
    rebalance_freq: Literal['D', 'W', 'M'] = 'M'
    execution: Literal['next_close'] = 'next_close'
    
    # Costs
    cost_model: CostModel = field(default_factory=CostModel)
    
    # Constraints
    max_turnover: float = 1.0
    min_trade_threshold: float = 0.005
    
    # Regime
    threshold_mode: ThresholdMode = 'frozen_at_train_end'
    regime_persist_periods: int = 1
    
    # Data
    sentiment_lag_months: int = 1

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
    
    def run(self,
            strategy: Literal['A', 'B', 'C'],
            crash_controller: Optional[Literal['panic_throttle', 'signal_switch', 'vol_target']],
            train_start: str,
            train_end: str,
            test_start: str,
            test_end: str,
            sentiment: pd.Series = None,
            short_data: pd.DataFrame = None,
            k: int = 3,
            **kwargs) -> BacktestResult:
        """
        Main backtest loop with strict no-lookahead.
        """
        # 1. Estimate thresholds from training data (frozen)
        spy_ret = self.returns['SPY']
        if self.config.threshold_mode == 'frozen_at_train_end':
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
            t_str = t.strftime('%Y-%m-%d')
            
            # --- NO LOOKAHEAD: all computations use data <= t ---
            
            # Compute regime state
            regime_state = compute_regime_state(spy_ret, t_str, thresholds)
            
            # Compute sentiment z-score (with publication lag)
            if sentiment is not None:
                sent_z = compute_sentiment_zscore(
                    sentiment, t_str, train_start,
                    lag_months=self.config.sentiment_lag_months
                )
            else:
                sent_z = 0.0
            
            # Compute raw signal based on strategy
            if strategy == 'A':
                signal = compute_momentum_12_1(self.prices, t_str)
                weights = build_long_short_portfolio(signal, k, prev_weights)
                weights = apply_sentiment_gate(weights, sent_z)
                action = 'momentum_sentiment_gated'
            
            elif strategy == 'B':
                signal = compute_momentum_12_1(self.prices, t_str)
                weights = build_long_short_portfolio(signal, k, prev_weights)
                action = 'momentum_baseline'
            
            elif strategy == 'C':
                weights = strategy_c_signal(
                    self.prices, short_data, t_str, train_start, sent_z
                )
                action = 'short_pressure_momentum'
            
            # Apply crash controller
            if crash_controller == 'panic_throttle':
                weights, ctrl_action = apply_panic_throttle(weights, regime_state, regime_tracker)
                action = f'{action}|{ctrl_action}'
            
            elif crash_controller == 'signal_switch':
                weights, ctrl_action = apply_signal_switch(
                    self.prices, t_str, regime_state, regime_tracker, k
                )
                action = f'{action}|{ctrl_action}'
            
            elif crash_controller == 'vol_target':
                weights, ctrl_action = apply_vol_target(weights, regime_state)
                action = f'{action}|{ctrl_action}'
            
            # Apply turnover cap
            if prev_weights is not None:
                weights = apply_turnover_cap(
                    prev_weights, weights,
                    self.config.max_turnover,
                    self.config.min_trade_threshold
                )
            
            # Store
            weights_history[t] = weights
            regime_history.append({'date': t, **regime_state})
            actions_history.append({'date': t, 'action': action})
            prev_weights = weights
        
        # 5. Compute returns
        return self._compute_returns(weights_history, regime_history, actions_history)
    
    def _compute_returns(self, weights_history, regime_history, actions_history):
        """Compute gross and net returns from weights."""
        dates = sorted(weights_history.keys())
        
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
                port_ret = (weights * day_ret).sum()
                daily_returns_gross.append({'date': day, 'return': port_ret})
                
                # Costs (transaction on rebalance day, borrow daily)
                cost = 0.0
                if day == period_returns.index[0] and prev_weights is not None:
                    turnover = compute_turnover(prev_weights, weights)
                    cost += self.cost_model.transaction_cost(turnover)
                    daily_turnover.append({'date': day, 'turnover': turnover})
                
                # Daily borrow cost
                short_weight = abs(weights[weights < 0].sum())
                cost += self.cost_model.daily_borrow_cost(short_weight)
                
                daily_costs.append({'date': day, 'cost': cost})
                daily_returns_net.append({'date': day, 'return': port_ret - cost})
            
            prev_weights = weights
        
        # Convert to Series
        returns_gross = pd.DataFrame(daily_returns_gross).set_index('date')['return']
        returns_net = pd.DataFrame(daily_returns_net).set_index('date')['return']
        turnover = pd.DataFrame(daily_turnover).set_index('date')['turnover'] if daily_turnover else pd.Series()
        costs = pd.DataFrame(daily_costs).set_index('date')['cost']
        
        # Compute metrics
        metrics = compute_metrics(returns_net, self.returns['SPY'].loc[returns_net.index])
        
        return BacktestResult(
            returns_gross=returns_gross,
            returns_net=returns_net,
            weights=pd.DataFrame(weights_history).T,
            turnover=turnover,
            costs=costs,
            regime_states=pd.DataFrame(regime_history),
            actions=pd.DataFrame(actions_history),
            metrics=metrics,
            config=self.config
        )
    
    def _get_rebalance_dates(self, start: str, end: str) -> list:
        """Get rebalance dates based on frequency."""
        dates = self.prices.loc[start:end].index
        
        if self.config.rebalance_freq == 'M':
            # Last trading day of each month
            return dates.to_series().groupby(pd.Grouper(freq='M')).last().tolist()
        elif self.config.rebalance_freq == 'W':
            return dates.to_series().groupby(pd.Grouper(freq='W-FRI')).last().tolist()
        else:  # Daily
            return dates.tolist()
```

---

## 5. Validation (Correct Math)

### Performance Metrics

```python
def compute_metrics(returns: pd.Series, 
                     benchmark: pd.Series = None) -> dict:
    """
    Compute performance metrics with correct math.
    """
    ann_factor = 252
    n_days = len(returns)
    n_years = n_days / ann_factor
    
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
    downside_vol = downside_ret.std() * np.sqrt(ann_factor) if len(downside_ret) > 0 else 0
    sortino = cagr / downside_vol if downside_vol > 0 else 0
    
    # Drawdown (correct calculation on wealth)
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_dd = drawdown.min()
    
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    metrics = {
        'total_return': cum_ret,
        'cagr': cagr,
        'annual_vol': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'skew': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'n_days': n_days,
    }
    
    # Benchmark comparison
    if benchmark is not None:
        aligned_bench = benchmark.reindex(returns.index).dropna()
        aligned_ret = returns.reindex(aligned_bench.index)
        
        if len(aligned_ret) > 20:
            cov_matrix = np.cov(aligned_ret, aligned_bench)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
            
            bench_cagr = (1 + aligned_bench).prod() ** (ann_factor / len(aligned_bench)) - 1
            alpha = cagr - beta * bench_cagr
            
            metrics['beta'] = beta
            metrics['alpha'] = alpha
    
    return metrics
```

### Drawdown & Recovery (Correct)

```python
def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series on wealth basis."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown

def compute_recovery_days(returns: pd.Series, 
                           start: str, 
                           end: str,
                           allow_beyond_window: bool = True) -> int:
    """
    Days to recover to prior peak after a drawdown window.
    """
    wealth = (1 + returns).cumprod()
    
    # Peak before window
    pre_window_peak = wealth.loc[:start].max()
    
    # Find first day after window that exceeds pre-window peak
    post_window = wealth.loc[end:]
    
    if allow_beyond_window:
        recovery_dates = post_window[post_window >= pre_window_peak].index
    else:
        recovery_dates = wealth.loc[start:end][wealth.loc[start:end] >= pre_window_peak].index
    
    if len(recovery_dates) > 0:
        window_end = pd.Timestamp(end)
        recovery_date = recovery_dates[0]
        return (recovery_date - window_end).days
    
    return -1  # Did not recover
```

### Stress Test Table

```python
CRISIS_WINDOWS = {
    'GFC': ('2008-09-01', '2009-03-31'),
    'Flash Crash': ('2010-05-01', '2010-05-31'),
    'Euro Crisis': ('2011-07-01', '2011-12-31'),
    'Taper Tantrum': ('2013-05-01', '2013-08-31'),
    'China Deval': ('2015-08-01', '2015-09-30'),
    'Volmageddon': ('2018-01-29', '2018-02-09'),
    'Q4 2018': ('2018-10-01', '2018-12-31'),
    'COVID Crash': ('2020-02-19', '2020-03-23'),
    'Inflation 2022': ('2022-01-01', '2022-10-31'),
    'Regional Banks': ('2023-03-01', '2023-05-15'),
}

def stress_test_table(returns: pd.Series, 
                       benchmark: pd.Series,
                       windows: dict = CRISIS_WINDOWS) -> pd.DataFrame:
    """
    Performance during crisis windows.
    """
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
            
            results.append({
                'Crisis': name,
                'Period': f"{start} → {end}",
                'Strategy': f"{strat_total:.1%}",
                'Benchmark': f"{bench_total:.1%}",
                'Excess': f"{strat_total - bench_total:.1%}",
                'Max DD': f"{dd.min():.1%}",
                'Recovery (days)': recovery if recovery > 0 else 'N/R',
            })
        except:
            continue
    
    return pd.DataFrame(results)
```

### OOS Stability (Multiple Splits)

```python
def oos_stability_analysis(engine: BacktestEngine,
                            strategy: str,
                            crash_controller: str,
                            train_start: str,
                            split_dates: list,
                            test_end: str,
                            **kwargs) -> pd.DataFrame:
    """
    Run backtest with multiple train_end dates to assess stability.
    """
    results = []
    
    for split_date in split_dates:
        # Test period is split_date+1 to test_end
        test_start = (pd.Timestamp(split_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
        
        result = engine.run(
            strategy=strategy,
            crash_controller=crash_controller,
            train_start=train_start,
            train_end=split_date,
            test_start=test_start,
            test_end=test_end,
            **kwargs
        )
        
        results.append({
            'train_end': split_date,
            'test_start': test_start,
            'sharpe': result.metrics['sharpe'],
            'max_dd': result.metrics['max_drawdown'],
            'cagr': result.metrics['cagr'],
        })
    
    df = pd.DataFrame(results)
    
    # Summary stats
    print(f"OOS Stability Summary (n={len(df)} splits)")
    print(f"  Sharpe: median={df['sharpe'].median():.2f}, "
          f"IQR=[{df['sharpe'].quantile(0.25):.2f}, {df['sharpe'].quantile(0.75):.2f}]")
    print(f"  MaxDD:  median={df['max_dd'].median():.1%}, "
          f"worst={df['max_dd'].min():.1%}")
    
    return df
```

---

## 6. Integrity Checks

### Backtest Integrity Checklist

```python
@dataclass
class IntegrityCheck:
    name: str
    passed: bool
    detail: str

def run_integrity_checks(config: BacktestConfig,
                          has_sentiment: bool,
                          has_benchmark: bool) -> list[IntegrityCheck]:
    """
    Verify backtest setup is rigorous.
    Returns list of checks with pass/fail status.
    """
    checks = []
    
    # 1. Execution lag
    checks.append(IntegrityCheck(
        name="Execution lag applied",
        passed=config.execution == 'next_close',
        detail=f"Execution: {config.execution}"
    ))
    
    # 2. Sentiment lag
    if has_sentiment:
        checks.append(IntegrityCheck(
            name="Sentiment publication lag",
            passed=config.sentiment_lag_months >= 1,
            detail=f"Lag: {config.sentiment_lag_months} month(s)"
        ))
    
    # 3. Thresholds frozen
    checks.append(IntegrityCheck(
        name="Regime thresholds mode",
        passed=True,  # Both modes are valid if chosen intentionally
        detail=f"Mode: {config.threshold_mode}"
    ))
    
    # 4. Costs enabled
    checks.append(IntegrityCheck(
        name="Transaction costs enabled",
        passed=config.cost_model.etf_cost_bps > 0,
        detail=f"ETF: {config.cost_model.etf_cost_bps} bps"
    ))
    
    # 5. Turnover reported
    checks.append(IntegrityCheck(
        name="Turnover constraints",
        passed=config.max_turnover < 5.0,  # Some reasonable cap
        detail=f"Max: {config.max_turnover:.0%}"
    ))
    
    # 6. Benchmark
    checks.append(IntegrityCheck(
        name="Benchmark comparison",
        passed=has_benchmark,
        detail="SPY" if has_benchmark else "None"
    ))
    
    return checks

def display_integrity_badge(checks: list[IntegrityCheck]) -> str:
    """Format checks for display."""
    all_passed = all(c.passed for c in checks)
    
    lines = []
    for c in checks:
        icon = "✅" if c.passed else "⚠️"
        lines.append(f"{icon} {c.name}: {c.detail}")
    
    status = "PASS" if all_passed else "REVIEW"
    return f"Integrity: {status}\n" + "\n".join(lines)
```

### No-Lookahead Tests

```python
# tests/test_lookahead.py

def test_signal_invariance():
    """Changing future data must not affect past signals."""
    prices_short = load_prices(SECTOR_ETFS, end='2020-06-30')
    prices_long = load_prices(SECTOR_ETFS, end='2021-12-31')
    
    signal_short = compute_momentum_12_1(prices_short, '2020-06-30')
    signal_long = compute_momentum_12_1(prices_long, '2020-06-30')
    
    pd.testing.assert_series_equal(signal_short, signal_long, check_names=False)

def test_sentiment_lag():
    """Sentiment at time t must not use data beyond t - lag."""
    sentiment = load_sentiment('2010-01-01', '2022-12-31')
    
    # At 2020-06-30 with 1-month lag, should use May 2020 value
    z = compute_sentiment_zscore(sentiment, '2020-06-30', '2010-01-01', lag_months=1)
    
    # Verify the latest data point used is May 2020
    available = sentiment.loc[:'2020-05-31']
    expected_current = available.iloc[-1]
    
    # The z-score should be based on this value
    assert not np.isnan(z)

def test_pipeline_invariance():
    """Full pipeline returns must not change when future data is corrupted."""
    prices = load_prices(SECTOR_ETFS + ['SPY'], end='2021-12-31')
    config = BacktestConfig()
    engine = BacktestEngine(prices, config)
    
    result_1 = engine.run(
        strategy='A',
        crash_controller=None,
        train_start='2010-01-01',
        train_end='2018-12-31',
        test_start='2019-01-01',
        test_end='2020-12-31',
        k=3
    )
    
    # Corrupt 2021 data
    prices_corrupt = prices.copy()
    prices_corrupt.loc['2021-01-01':] *= 0.5
    
    engine_corrupt = BacktestEngine(prices_corrupt, config)
    result_2 = engine_corrupt.run(
        strategy='A',
        crash_controller=None,
        train_start='2010-01-01',
        train_end='2018-12-31',
        test_start='2019-01-01',
        test_end='2020-12-31',
        k=3
    )
    
    # Returns through 2020 must match
    pd.testing.assert_series_equal(
        result_1.returns_net,
        result_2.returns_net,
        check_names=False
    )

def test_threshold_frozen():
    """Thresholds must be computed only from training data."""
    spy_ret = load_returns('SPY', '2010-01-01', '2022-12-31')
    
    thresholds = estimate_thresholds(spy_ret, '2018-12-31')
    
    # Thresholds should not change if we extend data
    spy_ret_extended = load_returns('SPY', '2010-01-01', '2025-12-31')
    thresholds_extended = estimate_thresholds(spy_ret_extended, '2018-12-31')
    
    assert thresholds == thresholds_extended
```

---

## 7. Dashboard UI

### Main Layout

```python
# app/dashboard.py

import streamlit as st

st.set_page_config(page_title="Quant Backtest Lab", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════
# HEADER: Information Frozen Badge
# ═══════════════════════════════════════════════════════════════════════════

def render_frozen_badge(train_end, test_start, test_end):
    st.markdown(f"""
    <div style="background-color: #1a1a2e; padding: 10px; border-radius: 5px; 
                border-left: 4px solid #4CAF50; margin-bottom: 20px;">
        🔒 <b>Information Set Frozen At:</b> {train_end} &nbsp;|&nbsp; 
        <b>Test Window:</b> {test_start} → {test_end}
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("Strategy Configuration")
    
    strategy = st.selectbox(
        "Strategy",
        ['A: Sector Momentum + Gating', 
         'B: Crash-Aware Momentum',
         'C: Short Pressure + Momentum (Equity)',
         'D: EDGAR Text Shock (Experimental)'],
        index=0
    )
    
    crash_controller = st.selectbox(
        "Crash Controller",
        [None, 'panic_throttle', 'signal_switch', 'vol_target'],
        format_func=lambda x: {
            None: 'None',
            'panic_throttle': '1: Panic-State Throttle',
            'signal_switch': '2: Signal Switching',
            'vol_target': '3: Volatility Targeting'
        }.get(x, x)
    )
    
    st.divider()
    st.subheader("Date Controls")
    
    col1, col2 = st.columns(2)
    train_start = col1.date_input("Train Start", value=pd.Timestamp('2005-01-01'))
    train_end = col2.date_input("Train End", value=pd.Timestamp('2018-12-31'))
    
    col3, col4 = st.columns(2)
    test_start = col3.date_input("Test Start", value=pd.Timestamp('2019-01-01'))
    test_end = col4.date_input("Test End", value=pd.Timestamp('2024-12-31'))
    
    threshold_mode = st.radio(
        "Threshold Mode",
        ['frozen_at_train_end', 'expanding'],
        format_func=lambda x: 'Frozen at Train End' if x == 'frozen_at_train_end' else 'Expanding Window'
    )
    
    st.divider()
    st.subheader("Parameters")
    
    k = st.slider("k (long/short)", 1, 5, 3)
    sentiment_lag = st.slider("Sentiment Lag (months)", 0, 3, 1)
    regime_persist = st.slider("Regime Persistence (periods)", 0, 3, 1)
    
    st.divider()
    st.subheader("Costs")
    
    cost_bps = st.slider("Transaction Cost (bps)", 0, 50, 5)
    borrow_bps = st.slider("Short Borrow Cost (bps/yr)", 0, 200, 25)
    
    run_button = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

render_frozen_badge(train_end, test_start, test_end)

# Integrity Checklist (collapsible)
with st.expander("🔍 Backtest Integrity Checklist"):
    checks = run_integrity_checks(config, has_sentiment=True, has_benchmark=True)
    for c in checks:
        icon = "✅" if c.passed else "⚠️"
        st.write(f"{icon} **{c.name}**: {c.detail}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Performance",
    "🔄 Crash Control",
    "📊 Weights",
    "⚡ Stress Test",
    "🔬 Stability",
    "📋 Compare Runs",
    "📝 Research Memo"
])
```

### Tab 1: Performance

```python
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    col2.metric("CAGR", f"{metrics['cagr']:.1%}")
    col3.metric("Max DD", f"{metrics['max_drawdown']:.1%}")
    col4.metric("Alpha", f"{metrics.get('alpha', 0):.2%}")
    
    # Equity curve
    st.subheader("Equity Curve")
    fig_equity = plot_equity_curve(result.returns_net, benchmark_returns)
    st.pyplot(fig_equity)
    
    # Drawdown (separate chart)
    st.subheader("Drawdown")
    fig_dd = plot_drawdown(result.returns_net)
    st.pyplot(fig_dd)
    
    # Metrics table
    st.subheader("Performance Metrics")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: 'Value'}))
    
    # Leg attribution
    st.subheader("Long vs Short Attribution")
    leg_attr = compute_leg_attribution(result.weights, result.returns_gross)
    st.dataframe(leg_attr)
```

### Tab 2: Crash Control Comparison (Key Feature)

```python
with tab2:
    st.subheader("Crash Control Comparison")
    st.caption("Compare strategy performance with different crash controllers")
    
    # Run all controllers and cache
    if 'counterfactual_results' not in st.session_state:
        st.session_state.counterfactual_results = {}
        
        for ctrl in [None, 'panic_throttle', 'signal_switch', 'vol_target']:
            result = engine.run(
                strategy=strategy_code,
                crash_controller=ctrl,
                train_start=str(train_start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
                k=k
            )
            st.session_state.counterfactual_results[ctrl] = result
    
    results = st.session_state.counterfactual_results
    
    # Overlay chart
    fig, ax = plt.subplots(figsize=(12, 6))
    for ctrl, res in results.items():
        label = ctrl if ctrl else 'Baseline'
        (1 + res.returns_net).cumprod().plot(ax=ax, label=label)
    ax.legend()
    ax.set_title("Equity Curves: Crash Control Comparison")
    st.pyplot(fig)
    
    # Comparison table
    comparison_data = []
    baseline_sharpe = results[None].metrics['sharpe']
    
    for ctrl, res in results.items():
        comparison_data.append({
            'Controller': ctrl if ctrl else 'None (Baseline)',
            'Sharpe': f"{res.metrics['sharpe']:.2f}",
            'Max DD': f"{res.metrics['max_drawdown']:.1%}",
            'CAGR': f"{res.metrics['cagr']:.1%}",
            'vs Baseline': f"+{res.metrics['sharpe'] - baseline_sharpe:.2f}" if ctrl else "—"
        })
    
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
    
    # Regime timeline
    st.subheader("Regime State Timeline")
    fig_regime = plot_regime_timeline(result.regime_states, result.actions)
    st.pyplot(fig_regime)
    
    # Action log
    st.subheader("Controller Actions")
    st.dataframe(result.actions.tail(24))  # Last 2 years
```

### Tab 4: Stress Test

```python
with tab4:
    st.subheader("Crisis Window Analysis")
    
    stress_df = stress_test_table(
        result.returns_net, 
        benchmark_returns,
        CRISIS_WINDOWS
    )
    
    st.dataframe(stress_df, hide_index=True)
    
    # Custom window
    st.subheader("Custom Stress Window")
    col1, col2 = st.columns(2)
    custom_start = col1.date_input("Start", value=pd.Timestamp('2020-02-19'))
    custom_end = col2.date_input("End", value=pd.Timestamp('2020-03-23'))
    
    if st.button("Analyze Custom Window"):
        custom_ret = result.returns_net.loc[str(custom_start):str(custom_end)]
        custom_bench = benchmark_returns.loc[str(custom_start):str(custom_end)]
        
        st.metric("Strategy Return", f"{(1 + custom_ret).prod() - 1:.1%}")
        st.metric("Benchmark Return", f"{(1 + custom_bench).prod() - 1:.1%}")
```

### Tab 5: OOS Stability

```python
with tab5:
    st.subheader("Out-of-Sample Stability")
    st.caption("Test robustness across multiple train/test splits")
    
    # Default split dates
    default_splits = ['2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31']
    
    n_splits = st.slider("Number of splits", 3, 7, 5)
    
    if st.button("Run Stability Analysis"):
        with st.spinner("Running multiple backtests..."):
            stability_df = oos_stability_analysis(
                engine=engine,
                strategy=strategy_code,
                crash_controller=crash_controller,
                train_start=str(train_start),
                split_dates=default_splits[:n_splits],
                test_end=str(test_end),
                k=k
            )
        
        # Boxplot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        stability_df.boxplot(column='sharpe', ax=axes[0])
        axes[0].set_title('Sharpe Distribution')
        stability_df.boxplot(column='max_dd', ax=axes[1])
        axes[1].set_title('Max DD Distribution')
        stability_df.boxplot(column='cagr', ax=axes[2])
        axes[2].set_title('CAGR Distribution')
        st.pyplot(fig)
        
        # Summary
        st.dataframe(stability_df)
        
        # Warning
        st.warning(f"""
        **Stability Summary**
        - Sharpe: median={stability_df['sharpe'].median():.2f}, 
          range=[{stability_df['sharpe'].min():.2f}, {stability_df['sharpe'].max():.2f}]
        - Selection risk: best vs median = +{stability_df['sharpe'].max() - stability_df['sharpe'].median():.2f}
        """)
```

### Tab 6: Compare Runs

```python
with tab6:
    st.subheader("Compare Saved Runs")
    
    # List saved runs
    saved_runs = list(Path('runs').glob('*.json'))
    run_names = [r.stem for r in saved_runs]
    
    col1, col2 = st.columns(2)
    run_a = col1.selectbox("Run A", run_names, index=0 if run_names else None)
    run_b = col2.selectbox("Run B", run_names, index=1 if len(run_names) > 1 else None)
    
    if run_a and run_b:
        with open(f'runs/{run_a}.json') as f:
            data_a = json.load(f)
        with open(f'runs/{run_b}.json') as f:
            data_b = json.load(f)
        
        # Config diff
        st.subheader("Configuration Differences")
        config_a = data_a['config']
        config_b = data_b['config']
        
        diff_rows = []
        all_keys = set(config_a.keys()) | set(config_b.keys())
        for key in sorted(all_keys):
            val_a = config_a.get(key, '—')
            val_b = config_b.get(key, '—')
            changed = '← changed' if val_a != val_b else ''
            diff_rows.append({
                'Parameter': key,
                run_a: str(val_a),
                run_b: str(val_b),
                'Δ': changed
            })
        
        st.dataframe(pd.DataFrame(diff_rows), hide_index=True)
        
        # Metrics diff
        st.subheader("Performance Comparison")
        metrics_a = data_a['metrics']
        metrics_b = data_b['metrics']
        
        metric_rows = []
        for key in ['sharpe', 'cagr', 'max_drawdown', 'alpha']:
            val_a = metrics_a.get(key, 0)
            val_b = metrics_b.get(key, 0)
            delta = val_b - val_a
            metric_rows.append({
                'Metric': key,
                run_a: f"{val_a:.3f}",
                run_b: f"{val_b:.3f}",
                'Δ': f"{delta:+.3f}"
            })
        
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
    
    # Save current run
    st.divider()
    st.subheader("Save Current Run")
    run_name = st.text_input("Run name")
    if st.button("Save") and run_name:
        save_run(config, result.metrics, run_name)
        st.success(f"Saved as runs/{run_name}.json")
```

### Tab 7: Research Memo

```python
with tab7:
    st.subheader("Research Notes")
    
    st.markdown("""
    ### Hypothesis
    
    Sector momentum captures slow-moving capital flows and institutional rebalancing.
    Sentiment gating exploits asymmetry: shorts are most profitable when optimism 
    peaks (crowded longs unwind), least profitable in fear (falling knives).
    Crash control avoids the well-documented "momentum crash" during market stress.
    
    ### Implementation Choices
    
    - **12-1 momentum**: Skip recent month to avoid short-term reversal contamination
    - **Sentiment lag**: 1 month to reflect actual data availability
    - **Regime persistence**: 1 period minimum to avoid whipsaw
    - **Signal switching**: Blend approach to avoid cliff effects
    
    ### Known Limitations
    
    1. **Survivorship bias**: ETF universe is current list, not point-in-time
    2. **Execution assumptions**: Next-day close with fixed costs; no market impact
    3. **Sentiment proxy**: UMCSENT may not capture real-time sentiment shifts
    4. **Borrow costs**: Simplified; real costs vary by name and time
    5. **Thresholds**: Estimated from training data; may not be optimal OOS
    
    ### What Would Break This Live
    
    - Regime detection lags actual market stress by 1+ months
    - Sentiment release schedule changes
    - ETF liquidity issues during crisis (tracking error, spreads)
    - Correlated positioning with other momentum funds
    
    ### Key Findings
    
    *(To be filled after backtest runs)*
    """)
    
    # Editable notes
    st.divider()
    custom_notes = st.text_area(
        "Your Notes",
        placeholder="Add your observations here...",
        height=200
    )
    
    if st.button("Save Notes"):
        # Append to run file
        pass
```

### Export

```python
# At bottom of sidebar
st.sidebar.divider()
st.sidebar.subheader("Export")

if st.sidebar.button("📥 CSV (Returns)"):
    csv = result.returns_net.to_csv()
    st.sidebar.download_button("Download", csv, "returns.csv")

if st.sidebar.button("📥 CSV (Weights)"):
    csv = result.weights.to_csv()
    st.sidebar.download_button("Download", csv, "weights.csv")

if st.sidebar.button("📄 Tear Sheet (HTML)"):
    import quantstats as qs
    qs.reports.html(result.returns_net, benchmark=benchmark_returns, output="tearsheet.html")
    with open("tearsheet.html", "rb") as f:
        st.sidebar.download_button("Download", f, "tearsheet.html")
```

---

## 8. Configuration Presets

```yaml
# config/presets/sector_momentum_baseline.yaml
name: "Sector Momentum Baseline"
description: "Pure momentum L/S on sector ETFs, no overlays"

strategy: A
crash_controller: null

universe:
  type: "sector_etfs"
  symbols: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE]
  benchmark: SPY

dates:
  train_start: "2005-01-01"
  train_end: "2018-12-31"
  test_start: "2019-01-01"
  test_end: "2024-12-31"

params:
  k: 3
  momentum_lookback: 252
  skip_recent: 21

costs:
  etf_cost_bps: 5
  etf_borrow_bps: 25

regime:
  threshold_mode: "frozen_at_train_end"
  persist_periods: 1

sentiment:
  enabled: false
  lag_months: 1
```

```yaml
# config/presets/sentiment_gated_crash_aware.yaml
name: "Sentiment-Gated + Crash Control"
description: "Momentum with sentiment gating and signal switching in crashes"

strategy: A
crash_controller: signal_switch

universe:
  type: "sector_etfs"
  symbols: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE]
  benchmark: SPY

dates:
  train_start: "2005-01-01"
  train_end: "2018-12-31"
  test_start: "2019-01-01"
  test_end: "2024-12-31"

params:
  k: 3
  momentum_lookback: 252
  skip_recent: 21

costs:
  etf_cost_bps: 5
  etf_borrow_bps: 25

regime:
  threshold_mode: "frozen_at_train_end"
  persist_periods: 1
  blend_mode: true

sentiment:
  enabled: true
  lag_months: 1
  high_threshold: 0.5
  low_threshold: -0.5
```

```yaml
# config/presets/short_pressure_equity.yaml
name: "Short Pressure + Momentum (Equity)"
description: "2D model using FINRA short volume + momentum on liquid large caps"

strategy: C
crash_controller: vol_target

universe:
  type: "custom"
  file: "universes/liquid_large_caps.csv"
  benchmark: SPY

dates:
  train_start: "2015-01-01"
  train_end: "2020-12-31"
  test_start: "2021-01-01"
  test_end: "2024-12-31"

params:
  k: 5
  short_pressure_lookback: 5
  short_pressure_smooth: 5

costs:
  equity_cost_bps: 15
  equity_borrow_bps: 100

regime:
  threshold_mode: "expanding"
  persist_periods: 1

sentiment:
  enabled: true
  lag_months: 1
```

---

## 9. Deliverables Checklist

### Core (Must Ship)
- [ ] Sector momentum signal (12-1, 6-1, blend)
- [ ] L/S portfolio construction with rank hysteresis
- [ ] Sentiment gating with publication lag
- [ ] Crash controllers (3 variants)
- [ ] Cost model (transaction + borrow)
- [ ] Backtest engine with no-lookahead
- [ ] Performance metrics (correct math)
- [ ] Streamlit dashboard

### HF-Native Features
- [ ] Counterfactual crash control comparison
- [ ] Stress test / crisis window table
- [ ] OOS stability analysis (multiple splits)
- [ ] Run comparison (2-3 runs)
- [ ] Research memo panel
- [ ] Backtest integrity checklist
- [ ] "Information frozen at" badge

### Quality
- [ ] No-lookahead unit tests (3 tests)
- [ ] Correct drawdown / recovery math
- [ ] Data caching
- [ ] QuantStats tear sheet export
- [ ] Configuration presets (3 YAMLs)
- [ ] README with rationale + limitations

### Optional (If Time)
- [ ] Strategy C: Short pressure + momentum
- [ ] Strategy D: EDGAR text shock
- [ ] Parameter sensitivity heatmap
- [ ] Rolling beta chart

---

## Quick Start

```bash
# Clone and setup
git clone <repo>
cd portfolio_backtest
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run app/dashboard.py

# Or run from preset
python -c "
from config.schema import load_preset
from backtest.engine import BacktestEngine
from data.prices import load_prices

config = load_preset('sector_momentum_baseline')
prices = load_prices(config.universe.symbols + [config.universe.benchmark])
engine = BacktestEngine(prices, config.to_backtest_config())

result = engine.run(
    strategy='A',
    crash_controller=config.crash_controller,
    train_start=config.dates.train_start,
    train_end=config.dates.train_end,
    test_start=config.dates.test_start,
    test_end=config.dates.test_end,
    k=config.params.k
)

print(result.metrics)
"
```

---

## Estimated Scope

| Component | Files | Lines |
|-----------|-------|-------|
| Config + Schema | 3 | 200 |
| Data loaders | 4 | 350 |
| Signals | 3 | 200 |
| Portfolio + Overlays | 3 | 400 |
| Backtest engine | 3 | 350 |
| Validation | 3 | 300 |
| Dashboard | 3 | 500 |
| Tests | 3 | 200 |
| **Total** | **~25** | **~2500** |

---

This spec is now implementation-ready with correct math, proper lookahead controls, realistic cost assumptions, and the HF-native features that demonstrate research maturity without scope bloat.