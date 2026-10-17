"""Unified third-party integration layer.

Combines the ecosystem around this platform into one namespace:

- ``integrations.registry``     : catalog of the 100 curated repos + status
- ``integrations.market_data``  : akshare, tushare, FinanceDatabase, FinanceToolkit,
                                  SEC EDGAR, OpenBB, Quandl, Google Finance, insider feed
- ``integrations.analytics``    : TA-Lib, Riskfolio, PyPortfolioOpt, cvxportfolio,
                                  FinQuant, FinancePy, py_vollib, precise, alphalens, pyfolio
- ``integrations.backends``     : backtrader, backtesting.py, vectorbt, qlib, freqtrade
- ``integrations.storage``      : DuckDB, Polars, ArcticDB, Parquet, JSON snapshots
- ``integrations.agents``       : multi-agent LLM research team (Ollama) w/ rule fallback

Every adapter degrades to the native ``quant``/``models`` implementations when
the optional dependency is missing, so nothing in the platform hard-fails.
"""

from __future__ import annotations

__all__ = [
    "REPOS", "CATEGORIES", "get_repo", "integration_summary",
    "DATA_SOURCE_STATUS", "ANALYTICS_STATUS", "BACKEND_STATUS", "STORAGE_STATUS",
    "LLM_STATUS", "full_status", "analyze_symbol",
]

from integrations.registry import (
    REPOS, CATEGORIES, get_repo, list_by_category, list_by_status,
    integration_summary, to_records,
)
from integrations.market_data import DATA_SOURCE_STATUS
from integrations.analytics import ANALYTICS_STATUS
from integrations.backends import BACKEND_STATUS
from integrations.storage import STORAGE_STATUS
from integrations.agents import LLM_STATUS, analyze_symbol


def full_status() -> dict:
    """Snapshot of every integration's availability."""
    return {
        "registry": integration_summary(),
        "data_sources": DATA_SOURCE_STATUS,
        "analytics": ANALYTICS_STATUS,
        "backends": BACKEND_STATUS,
        "storage": STORAGE_STATUS,
        "llm": LLM_STATUS,
    }
