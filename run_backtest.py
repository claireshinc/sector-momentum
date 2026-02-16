"""Run full backtest with default (baseline) config and print metrics."""

from portfolio_backtest.config.schema import load_preset, BacktestConfig, SECTOR_ETFS
from portfolio_backtest.backtest.engine import BacktestEngine
from portfolio_backtest.data.prices import load_prices
from portfolio_backtest.data.sentiment import load_sentiment
from portfolio_backtest.backtest.integrity import run_integrity_checks, display_integrity_badge


def main():
    # Load baseline preset
    preset = load_preset("sector_momentum_baseline")
    config = preset.to_backtest_config()
    symbols = preset.universe.symbols + [preset.universe.benchmark]

    print(f"=== {preset.name} ===")
    print(f"Strategy: {preset.strategy} | Controller: {preset.crash_controller}")
    print(f"Train: {preset.dates.train_start} -> {preset.dates.train_end}")
    print(f"Test:  {preset.dates.test_start} -> {preset.dates.test_end}")
    print(f"Universe: {len(preset.universe.symbols)} symbols + {preset.universe.benchmark}")
    print()

    # Load data
    print("Loading price data...")
    prices = load_prices(symbols, start=preset.dates.train_start, end=preset.dates.test_end)
    print(f"  Loaded {len(prices)} trading days, {len(prices.columns)} symbols")
    print()

    # Run integrity checks
    checks = run_integrity_checks(config, has_sentiment=False, has_benchmark=True)
    print(display_integrity_badge(checks))
    print()

    # Run backtest
    print("Running backtest...")
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

    # Print metrics
    print("\n=== PERFORMANCE METRICS ===")
    for key, val in result.metrics.items():
        if isinstance(val, float):
            if "return" in key or "cagr" in key or "drawdown" in key or "alpha" in key or "vol" in key:
                print(f"  {key:20s}: {val:>10.2%}")
            else:
                print(f"  {key:20s}: {val:>10.4f}")
        else:
            print(f"  {key:20s}: {val}")

    print(f"\n  Net return days: {len(result.returns_net)}")
    print(f"  Rebalance dates: {len(result.weights)}")
    if len(result.turnover) > 0:
        print(f"  Avg turnover:    {result.turnover.mean():.2%}")
    print(f"  Total costs:     {result.costs.sum():.4f}")

    # Also run the sentiment-gated version for comparison
    print("\n\n" + "=" * 60)
    preset2 = load_preset("sentiment_gated_crash_aware")
    config2 = preset2.to_backtest_config()

    print(f"\n=== {preset2.name} ===")
    print(f"Strategy: {preset2.strategy} | Controller: {preset2.crash_controller}")

    sentiment = load_sentiment(start=preset2.dates.train_start, end=preset2.dates.test_end)
    print(f"  Loaded {len(sentiment)} sentiment observations")

    engine2 = BacktestEngine(prices, config2)
    result2 = engine2.run(
        strategy=preset2.strategy,
        crash_controller=preset2.crash_controller,
        train_start=preset2.dates.train_start,
        train_end=preset2.dates.train_end,
        test_start=preset2.dates.test_start,
        test_end=preset2.dates.test_end,
        sentiment=sentiment,
        k=preset2.params.k,
    )

    print("\n=== PERFORMANCE METRICS ===")
    for key, val in result2.metrics.items():
        if isinstance(val, float):
            if "return" in key or "cagr" in key or "drawdown" in key or "alpha" in key or "vol" in key:
                print(f"  {key:20s}: {val:>10.2%}")
            else:
                print(f"  {key:20s}: {val:>10.4f}")
        else:
            print(f"  {key:20s}: {val}")

    # Summary comparison
    print("\n\n=== COMPARISON ===")
    print(f"{'Metric':20s} | {'Baseline':>12s} | {'Sent+Crash':>12s} | {'Delta':>10s}")
    print("-" * 62)
    for key in ["sharpe", "cagr", "max_drawdown", "annual_vol"]:
        v1 = result.metrics.get(key, 0)
        v2 = result2.metrics.get(key, 0)
        delta = v2 - v1
        if "cagr" in key or "drawdown" in key or "vol" in key:
            print(f"  {key:18s} | {v1:>11.2%} | {v2:>11.2%} | {delta:>+9.2%}")
        else:
            print(f"  {key:18s} | {v1:>11.4f} | {v2:>11.4f} | {delta:>+9.4f}")


if __name__ == "__main__":
    main()
