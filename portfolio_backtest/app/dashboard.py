"""Main Streamlit dashboard — Quant Backtest Lab."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path (needed for Streamlit Cloud)
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from portfolio_backtest.backtest.costs import CostModel
from portfolio_backtest.backtest.engine import BacktestEngine
from portfolio_backtest.backtest.integrity import run_integrity_checks
from portfolio_backtest.config.schema import (
    SECTOR_ETFS,
    BacktestConfig,
    list_presets,
    load_preset,
)
from portfolio_backtest.data.prices import load_prices
from portfolio_backtest.data.sentiment import load_sentiment
from portfolio_backtest.validation.metrics import (
    compute_drawdown_series,
    compute_leg_attribution,
    compute_metrics,
)
from portfolio_backtest.validation.stability import oos_stability_analysis
from portfolio_backtest.validation.stress import CRISIS_WINDOWS, stress_test_table

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Quant Backtest Lab", layout="wide")

RUNS_DIR = Path(__file__).parent.parent.parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: frozen badge
# ──────────────────────────────────────────────────────────────────────────────


def render_frozen_badge(train_end, test_start, test_end):
    st.markdown(
        f"""
    <div style="background-color: #1a1a2e; padding: 10px; border-radius: 5px;
                border-left: 4px solid #4CAF50; margin-bottom: 20px; color: #e0e0e0;">
        <b>Information Set Frozen At:</b> {train_end} &nbsp;|&nbsp;
        <b>Test Window:</b> {test_start} &rarr; {test_end}
    </div>
    """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_equity_curve(returns_net, benchmark_returns):
    fig, ax = plt.subplots(figsize=(12, 5))
    (1 + returns_net).cumprod().plot(ax=ax, label="Strategy", linewidth=1.5)
    if benchmark_returns is not None:
        aligned = benchmark_returns.reindex(returns_net.index).dropna()
        (1 + aligned).cumprod().plot(ax=ax, label="SPY", linewidth=1, alpha=0.7)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_drawdown(returns_net):
    fig, ax = plt.subplots(figsize=(12, 3))
    dd = compute_drawdown_series(returns_net)
    dd.plot(ax=ax, color="red", alpha=0.7)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.3, color="red")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_regime_timeline(regime_states, actions):
    fig, ax = plt.subplots(figsize=(12, 3))
    if regime_states.empty:
        return fig

    dates = regime_states["date"]
    crash_risk = regime_states.get("is_crash_risk", pd.Series(False, index=regime_states.index))
    panic = regime_states.get("is_panic", pd.Series(False, index=regime_states.index))

    ax.fill_between(dates, 0, 1, where=crash_risk, alpha=0.3, color="orange", label="Crash Risk")
    ax.fill_between(dates, 0, 1, where=panic, alpha=0.5, color="red", label="Panic")
    ax.set_title("Regime State Timeline")
    ax.legend(loc="upper right")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def save_run(config_dict, metrics, run_name):
    """Save a run to JSON."""
    data = {
        "config": config_dict,
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metrics.items()},
    }
    with open(RUNS_DIR / f"{run_name}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Strategy Configuration")

    # Preset loader
    available_presets = list_presets()
    if available_presets:
        selected_preset = st.selectbox(
            "Load Preset",
            ["(Custom)"] + available_presets,
        )
        if selected_preset != "(Custom)":
            preset = load_preset(selected_preset)
            st.success(f"Loaded: {preset.name}")
    else:
        selected_preset = "(Custom)"

    strategy = st.selectbox(
        "Strategy",
        [
            "A: Sector Momentum + Gating",
            "B: Crash-Aware Momentum",
            "C: Short Pressure + Momentum (Equity)",
        ],
        index=0,
    )
    strategy_code = strategy.split(":")[0].strip()

    crash_controller = st.selectbox(
        "Crash Controller",
        [None, "panic_throttle", "signal_switch", "vol_target"],
        format_func=lambda x: {
            None: "None",
            "panic_throttle": "1: Panic-State Throttle",
            "signal_switch": "2: Signal Switching",
            "vol_target": "3: Volatility Targeting",
        }.get(x, x),
    )

    st.divider()
    st.subheader("Date Controls")

    col1, col2 = st.columns(2)
    train_start = col1.date_input("Train Start", value=pd.Timestamp("2005-01-01"))
    train_end = col2.date_input("Train End", value=pd.Timestamp("2018-12-31"))

    col3, col4 = st.columns(2)
    test_start = col3.date_input("Test Start", value=pd.Timestamp("2019-01-01"))
    test_end = col4.date_input("Test End", value=pd.Timestamp("2024-12-31"))

    threshold_mode = st.radio(
        "Threshold Mode",
        ["frozen_at_train_end", "expanding"],
        format_func=lambda x: (
            "Frozen at Train End" if x == "frozen_at_train_end" else "Expanding Window"
        ),
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

    run_button = st.button("Run Backtest", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────

render_frozen_badge(train_end, test_start, test_end)

# Build config
config = BacktestConfig(
    rebalance_freq="M",
    execution="next_close",
    cost_model=CostModel(etf_cost_bps=cost_bps, etf_borrow_bps=borrow_bps),
    max_turnover=1.0,
    min_trade_threshold=0.005,
    threshold_mode=threshold_mode,
    regime_persist_periods=regime_persist,
    sentiment_lag_months=sentiment_lag,
)

# Integrity Checklist (collapsible)
with st.expander("Backtest Integrity Checklist"):
    has_sentiment = strategy_code == "A"
    checks = run_integrity_checks(config, has_sentiment=has_sentiment, has_benchmark=True)
    for c in checks:
        icon = "PASS" if c.passed else "WARN"
        st.write(f"**[{icon}] {c.name}**: {c.detail}")

# Run backtest
if run_button:
    with st.spinner("Loading data..."):
        symbols = SECTOR_ETFS + ["SPY"]
        try:
            prices = load_prices(symbols, start=str(train_start), end=str(test_end))
        except Exception as e:
            st.error(f"Failed to load prices: {e}")
            st.stop()

        st.caption(
            f"Loaded {prices.shape[0]} days x {prices.shape[1]} symbols "
            f"({prices.index.min().date()} to {prices.index.max().date()})"
        )

        sentiment = None
        if strategy_code in ("A", "C"):
            sentiment = load_sentiment(start=str(train_start), end=str(test_end))

        short_data = None
        if strategy_code == "C":
            from portfolio_backtest.data.short_volume import load_finra_short_volume
            short_data = load_finra_short_volume(
                SECTOR_ETFS, start=str(train_start), end=str(test_end)
            )

    with st.spinner("Running backtest..."):
        engine = BacktestEngine(prices, config)
        result = engine.run(
            strategy=strategy_code,
            crash_controller=crash_controller,
            train_start=str(train_start),
            train_end=str(train_end),
            test_start=str(test_start),
            test_end=str(test_end),
            sentiment=sentiment,
            short_data=short_data,
            k=k,
        )

    if not result.metrics:
        st.error(
            f"Backtest returned no results. "
            f"Rebalance dates: {len(result.weights)}, "
            f"Net return days: {len(result.returns_net)}"
        )

    st.session_state["result"] = result
    st.session_state["engine"] = engine
    st.session_state["prices"] = prices
    st.session_state["sentiment"] = sentiment
    st.session_state["short_data"] = short_data
    st.session_state["config_dict"] = {
        "strategy": strategy_code,
        "crash_controller": crash_controller,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "k": k,
        "threshold_mode": threshold_mode,
        "cost_bps": cost_bps,
        "borrow_bps": borrow_bps,
        "sentiment_lag": sentiment_lag,
        "regime_persist": regime_persist,
    }

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

if "result" in st.session_state:
    result = st.session_state["result"]
    engine = st.session_state["engine"]
    prices = st.session_state["prices"]
    metrics = result.metrics

    if not metrics:
        st.warning("No metrics computed. Check date ranges and data availability.")
    else:
        benchmark_returns = prices["SPY"].pct_change().dropna()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "Performance",
                "Crash Control",
                "Weights",
                "Stress Test",
                "Stability",
                "Compare Runs",
                "Research Memo",
            ]
        )

        # ── Tab 1: Performance ───────────────────────────────────────────
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
            col2.metric("CAGR", f"{metrics.get('cagr', 0):.1%}")
            col3.metric("Max DD", f"{metrics.get('max_drawdown', 0):.1%}")
            col4.metric("Alpha", f"{metrics.get('alpha', 0):.2%}")

            st.subheader("Equity Curve")
            fig_equity = plot_equity_curve(result.returns_net, benchmark_returns)
            st.pyplot(fig_equity)
            plt.close(fig_equity)

            st.subheader("Drawdown")
            fig_dd = plot_drawdown(result.returns_net)
            st.pyplot(fig_dd)
            plt.close(fig_dd)

            st.subheader("Performance Metrics")
            metrics_display = {
                k: f"{v:.4f}" if isinstance(v, float) else str(v)
                for k, v in metrics.items()
            }
            st.dataframe(
                pd.DataFrame([metrics_display]).T.rename(columns={0: "Value"})
            )

        # ── Tab 2: Crash Control Comparison (Key Feature) ────────────────
        with tab2:
            st.subheader("Crash Control Comparison")
            st.caption("Compare strategy performance with different crash controllers")

            if st.button("Run Counterfactual Comparison"):
                counterfactual_results = {}
                sentiment = st.session_state.get("sentiment")
                short_data = st.session_state.get("short_data")

                with st.spinner("Running counterfactual backtests..."):
                    for ctrl in [None, "panic_throttle", "signal_switch", "vol_target"]:
                        res = engine.run(
                            strategy=strategy_code,
                            crash_controller=ctrl,
                            train_start=str(train_start),
                            train_end=str(train_end),
                            test_start=str(test_start),
                            test_end=str(test_end),
                            sentiment=sentiment,
                            short_data=short_data,
                            k=k,
                        )
                        counterfactual_results[ctrl] = res

                st.session_state["counterfactual_results"] = counterfactual_results

            if "counterfactual_results" in st.session_state:
                results = st.session_state["counterfactual_results"]

                # Overlay chart
                fig, ax = plt.subplots(figsize=(12, 6))
                for ctrl, res in results.items():
                    label = ctrl if ctrl else "Baseline"
                    if len(res.returns_net) > 0:
                        (1 + res.returns_net).cumprod().plot(ax=ax, label=label)
                ax.legend()
                ax.set_title("Equity Curves: Crash Control Comparison")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

                # Comparison table
                comparison_data = []
                baseline_sharpe = results[None].metrics.get("sharpe", 0) if results[None].metrics else 0

                for ctrl, res in results.items():
                    if not res.metrics:
                        continue
                    m = res.metrics
                    comparison_data.append(
                        {
                            "Controller": ctrl if ctrl else "None (Baseline)",
                            "Sharpe": f"{m.get('sharpe', 0):.2f}",
                            "Max DD": f"{m.get('max_drawdown', 0):.1%}",
                            "CAGR": f"{m.get('cagr', 0):.1%}",
                            "vs Baseline": (
                                f"+{m.get('sharpe', 0) - baseline_sharpe:.2f}"
                                if ctrl
                                else "--"
                            ),
                        }
                    )

                st.dataframe(pd.DataFrame(comparison_data), hide_index=True)

                # Regime timeline
                st.subheader("Regime State Timeline")
                fig_regime = plot_regime_timeline(result.regime_states, result.actions)
                st.pyplot(fig_regime)
                plt.close(fig_regime)

                # Action log
                st.subheader("Controller Actions")
                if not result.actions.empty:
                    st.dataframe(result.actions.tail(24))

        # ── Tab 3: Weights ───────────────────────────────────────────────
        with tab3:
            st.subheader("Portfolio Weights Over Time")
            if not result.weights.empty:
                fig, ax = plt.subplots(figsize=(12, 6))
                result.weights.plot.bar(ax=ax, stacked=True)
                ax.set_title("Portfolio Weights at Each Rebalance")
                ax.set_ylabel("Weight")
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Latest Weights")
                latest = result.weights.iloc[-1].sort_values(ascending=False)
                st.dataframe(latest.to_frame("Weight"))

            st.subheader("Turnover")
            if len(result.turnover) > 0:
                fig, ax = plt.subplots(figsize=(12, 3))
                result.turnover.plot(ax=ax, kind="bar")
                ax.set_title("Turnover per Rebalance")
                ax.set_ylabel("One-Way Turnover")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ── Tab 4: Stress Test ───────────────────────────────────────────
        with tab4:
            st.subheader("Crisis Window Analysis")

            stress_df = stress_test_table(
                result.returns_net,
                benchmark_returns,
                CRISIS_WINDOWS,
            )

            if not stress_df.empty:
                st.dataframe(stress_df, hide_index=True)
            else:
                st.info("No crisis windows overlap with the test period.")

            # Custom window
            st.subheader("Custom Stress Window")
            col1, col2 = st.columns(2)
            custom_start = col1.date_input(
                "Start", value=pd.Timestamp("2020-02-19"), key="stress_start"
            )
            custom_end = col2.date_input(
                "End", value=pd.Timestamp("2020-03-23"), key="stress_end"
            )

            if st.button("Analyze Custom Window"):
                custom_ret = result.returns_net.loc[str(custom_start) : str(custom_end)]
                custom_bench = benchmark_returns.loc[str(custom_start) : str(custom_end)]

                if len(custom_ret) > 0:
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Strategy Return",
                        f"{(1 + custom_ret).prod() - 1:.1%}",
                    )
                    c2.metric(
                        "Benchmark Return",
                        f"{(1 + custom_bench).prod() - 1:.1%}",
                    )
                else:
                    st.warning("No data in selected window.")

        # ── Tab 5: OOS Stability ─────────────────────────────────────────
        with tab5:
            st.subheader("Out-of-Sample Stability")
            st.caption("Test robustness across multiple train/test splits")

            default_splits = [
                "2016-12-31",
                "2017-12-31",
                "2018-12-31",
                "2019-12-31",
                "2020-12-31",
            ]

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
                        k=k,
                    )

                if not stability_df.empty:
                    # Boxplots
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    stability_df.boxplot(column="sharpe", ax=axes[0])
                    axes[0].set_title("Sharpe Distribution")
                    stability_df.boxplot(column="max_dd", ax=axes[1])
                    axes[1].set_title("Max DD Distribution")
                    stability_df.boxplot(column="cagr", ax=axes[2])
                    axes[2].set_title("CAGR Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(stability_df)

                    st.warning(
                        f"""
                    **Stability Summary**
                    - Sharpe: median={stability_df['sharpe'].median():.2f},
                      range=[{stability_df['sharpe'].min():.2f}, {stability_df['sharpe'].max():.2f}]
                    - Selection risk: best vs median = +{stability_df['sharpe'].max() - stability_df['sharpe'].median():.2f}
                    """
                    )
                else:
                    st.info("No results from stability analysis.")

        # ── Tab 6: Compare Runs ──────────────────────────────────────────
        with tab6:
            st.subheader("Compare Saved Runs")

            saved_runs = list(RUNS_DIR.glob("*.json"))
            run_names = [r.stem for r in saved_runs]

            if len(run_names) >= 2:
                col1, col2 = st.columns(2)
                run_a = col1.selectbox("Run A", run_names, index=0)
                run_b = col2.selectbox(
                    "Run B", run_names, index=1 if len(run_names) > 1 else 0
                )

                if run_a and run_b:
                    with open(RUNS_DIR / f"{run_a}.json") as f:
                        data_a = json.load(f)
                    with open(RUNS_DIR / f"{run_b}.json") as f:
                        data_b = json.load(f)

                    # Config diff
                    st.subheader("Configuration Differences")
                    config_a = data_a["config"]
                    config_b = data_b["config"]

                    diff_rows = []
                    all_keys = set(config_a.keys()) | set(config_b.keys())
                    for key in sorted(all_keys):
                        val_a = config_a.get(key, "--")
                        val_b = config_b.get(key, "--")
                        changed = "<- changed" if val_a != val_b else ""
                        diff_rows.append(
                            {
                                "Parameter": key,
                                run_a: str(val_a),
                                run_b: str(val_b),
                                "D": changed,
                            }
                        )

                    st.dataframe(pd.DataFrame(diff_rows), hide_index=True)

                    # Metrics diff
                    st.subheader("Performance Comparison")
                    metrics_a = data_a["metrics"]
                    metrics_b = data_b["metrics"]

                    metric_rows = []
                    for key in ["sharpe", "cagr", "max_drawdown", "alpha"]:
                        val_a = metrics_a.get(key, 0)
                        val_b = metrics_b.get(key, 0)
                        delta = val_b - val_a
                        metric_rows.append(
                            {
                                "Metric": key,
                                run_a: f"{val_a:.3f}",
                                run_b: f"{val_b:.3f}",
                                "D": f"{delta:+.3f}",
                            }
                        )

                    st.dataframe(pd.DataFrame(metric_rows), hide_index=True)
            elif len(run_names) == 1:
                st.info("Save at least 2 runs to compare.")
            else:
                st.info("No saved runs yet. Run a backtest and save it below.")

            # Save current run
            st.divider()
            st.subheader("Save Current Run")
            run_name = st.text_input("Run name")
            if st.button("Save") and run_name and "result" in st.session_state:
                save_run(
                    st.session_state["config_dict"],
                    st.session_state["result"].metrics,
                    run_name,
                )
                st.success(f"Saved as runs/{run_name}.json")

        # ── Tab 7: Research Memo ─────────────────────────────────────────
        with tab7:
            st.subheader("Research Notes")

            st.markdown(
                """
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
            """
            )

            st.divider()
            custom_notes = st.text_area(
                "Your Notes",
                placeholder="Add your observations here...",
                height=200,
            )

    # ── Sidebar Export ────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Export")

    if st.sidebar.button("CSV (Returns)"):
        csv = result.returns_net.to_csv()
        st.sidebar.download_button("Download Returns CSV", csv, "returns.csv")

    if st.sidebar.button("CSV (Weights)"):
        csv = result.weights.to_csv()
        st.sidebar.download_button("Download Weights CSV", csv, "weights.csv")

    if st.sidebar.button("Tear Sheet (HTML)"):
        try:
            import quantstats as qs

            qs.reports.html(
                result.returns_net,
                benchmark=benchmark_returns,
                output="tearsheet.html",
            )
            with open("tearsheet.html", "rb") as f:
                st.sidebar.download_button(
                    "Download Tear Sheet", f, "tearsheet.html"
                )
        except Exception as e:
            st.sidebar.error(f"Tear sheet generation failed: {e}")

else:
    st.info("Configure strategy in the sidebar and click **Run Backtest** to begin.")
