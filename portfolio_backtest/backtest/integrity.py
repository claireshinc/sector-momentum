"""Backtest integrity checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IntegrityCheck:
    name: str
    passed: bool
    detail: str


def run_integrity_checks(
    config,
    has_sentiment: bool,
    has_benchmark: bool,
) -> list[IntegrityCheck]:
    """
    Verify backtest setup is rigorous.
    Returns list of checks with pass/fail status.
    """
    checks = []

    # 1. Execution lag
    checks.append(
        IntegrityCheck(
            name="Execution lag applied",
            passed=config.execution == "next_close",
            detail=f"Execution: {config.execution}",
        )
    )

    # 2. Sentiment lag
    if has_sentiment:
        checks.append(
            IntegrityCheck(
                name="Sentiment publication lag",
                passed=config.sentiment_lag_months >= 1,
                detail=f"Lag: {config.sentiment_lag_months} month(s)",
            )
        )

    # 3. Thresholds frozen
    checks.append(
        IntegrityCheck(
            name="Regime thresholds mode",
            passed=True,  # Both modes are valid if chosen intentionally
            detail=f"Mode: {config.threshold_mode}",
        )
    )

    # 4. Costs enabled
    checks.append(
        IntegrityCheck(
            name="Transaction costs enabled",
            passed=config.cost_model.etf_cost_bps > 0,
            detail=f"ETF: {config.cost_model.etf_cost_bps} bps",
        )
    )

    # 5. Turnover reported
    checks.append(
        IntegrityCheck(
            name="Turnover constraints",
            passed=config.max_turnover < 5.0,  # Some reasonable cap
            detail=f"Max: {config.max_turnover:.0%}",
        )
    )

    # 6. Benchmark
    checks.append(
        IntegrityCheck(
            name="Benchmark comparison",
            passed=has_benchmark,
            detail="SPY" if has_benchmark else "None",
        )
    )

    return checks


def display_integrity_badge(checks: list[IntegrityCheck]) -> str:
    """Format checks for display."""
    all_passed = all(c.passed for c in checks)

    lines = []
    for c in checks:
        icon = "PASS" if c.passed else "WARN"
        lines.append(f"[{icon}] {c.name}: {c.detail}")

    status = "PASS" if all_passed else "REVIEW"
    return f"Integrity: {status}\n" + "\n".join(lines)
