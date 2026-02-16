"""Streamlit Cloud entry point."""
import runpy
import sys
from pathlib import Path

# Ensure project root is importable
root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)

runpy.run_module("portfolio_backtest.app.dashboard", run_name="__main__", alter_sys=True)
