"""Catalog of the 100 curated trading repositories and their integration status.

Status legend
-------------
core      : directly wired into this system (primary dependency or native module)
adapter   : optional pip integration shipped in ``integrations/`` with graceful fallback
wrapped   : key techniques reimplemented natively in ``quant/`` or ``models/``
reference : cataloged for discovery; not mergeable code (desktop/Java/Rust apps,
            awesome-lists, deprecated projects) - see docs/INTEGRATIONS.md
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

__all__ = [
    "Repo",
    "REPOS",
    "CATEGORIES",
    "get_repo",
    "list_by_category",
    "list_by_status",
    "integration_summary",
    "to_records",
]


@dataclass(frozen=True)
class Repo:
    id: int
    name: str
    url: str
    category: str
    status: str
    where: str = ""


def _r(id_, name, category, status, where=""):
    return Repo(id_, name, f"https://github.com/{name}", category, status, where)


CATEGORIES = {
    "research": "Research Platforms & Terminals",
    "ai": "LLMs & Agentic AI Frameworks",
    "quant": "Quantitative & Algorithmic Trading Frameworks",
    "strategy": "Market Strategy Libraries & Catalogs",
    "analytics": "Mathematical Analytics & Indicators",
    "data": "APIs, Market Crawlers & Historical Data",
    "infra": "Infrastructure, Charts & Low-Level Architecture",
    "awesome": "Curated Resource Lists & Aggregators",
}

REPOS: list[Repo] = [
    # 1. Research platforms
    _r(1, "OpenBB-finance/OpenBBTerminal", "research", "adapter", "integrations/market_data.py"),
    _r(2, "Fincept-Corporation/FinceptTerminal", "research", "reference"),
    _r(3, "shubhamsborkar/greeksoup", "research", "reference"),
    _r(4, "portfolio-performance/portfolio", "research", "reference"),
    _r(5, "ghostfolio/ghostfolio", "research", "reference"),
    _r(6, "firefly-iii/firefly-iii", "research", "reference"),
    _r(7, "actualbudget/actual", "research", "reference"),
    _r(8, "maybe-finance/maybe", "research", "reference"),
    _r(9, "midday-ai/midday", "research", "reference"),
    # 2. LLM / agents
    _r(10, "TauricResearch/TradingAgents", "ai", "adapter", "integrations/agents.py"),
    _r(11, "AI4Finance-Foundation/FinGPT", "ai", "reference"),
    _r(12, "HKUDS/Vibe-Trading", "ai", "reference"),
    _r(13, "The-Swarm-Corporation/AutoHedge", "ai", "adapter", "integrations/agents.py"),
    _r(14, "burnertube-bit/WallStreetAgents", "ai", "reference"),
    _r(15, "Richorrific-Rea/insider-agent", "ai", "adapter", "integrations/market_data.py"),
    _r(16, "OpenByteInc/QuantDinger", "ai", "reference"),
    _r(17, "The-Swarm-Corporation/Volara", "ai", "reference"),
    _r(18, "ellenzishanli/SignalForge", "ai", "reference"),
    _r(19, "valuecell-labs/valuecell", "ai", "reference"),
    _r(20, "LangAlpha/LangAlpha", "ai", "reference"),
    _r(21, "ai-berkshire/ai-berkshire", "ai", "reference"),
    # 3. Quant frameworks
    _r(22, "QuantConnect/Lean", "quant", "reference"),
    _r(23, "mementum/backtrader", "quant", "adapter", "integrations/backends.py"),
    _r(24, "microsoft/qlib", "quant", "adapter", "integrations/backends.py"),
    _r(25, "vnpy/vnpy", "quant", "reference"),
    _r(26, "freqtrade/freqtrade", "quant", "adapter", "integrations/backends.py"),
    _r(27, "nautilus-trader/nautilus_trader", "quant", "adapter", "integrations/backends.py"),
    _r(28, "pmartin/backtesting.py", "quant", "adapter", "integrations/backends.py"),
    _r(29, "polars-algo/pybroker", "quant", "reference"),
    _r(30, "hummingbot/hummingbot", "quant", "reference"),
    _r(31, "AI4Finance-Foundation/FinRL", "quant", "reference"),
    _r(32, "StockSharp/StockSharp", "quant", "reference"),
    _r(33, "jesse-ai/jesse", "quant", "reference"),
    _r(34, "tensortrade-org/tensortrade", "quant", "reference"),
    _r(35, "openalgo/openalgo", "quant", "reference"),
    _r(36, "blankly-finance/blankly", "quant", "reference"),
    _r(37, "VisualHFT/VisualHFT", "quant", "reference"),
    _r(38, "tradytics/eiten", "quant", "wrapped", "quant/strategies.py"),
    _r(39, "aat-core/aat", "quant", "reference"),
    _r(40, "zvt-project/zvt", "quant", "reference"),
    # 4. Strategy libraries
    _r(41, "freqtrade/freqtrade-strategies", "strategy", "wrapped", "quant/strategies.py"),
    _r(42, "je-suis-tm/quant-trading", "strategy", "wrapped", "quant/strategies.py"),
    _r(43, "yli188/WorldQuant_alpha101_code", "strategy", "wrapped", "quant/alpha101.py"),
    _r(44, "fmzquant/strategies", "strategy", "reference"),
    _r(45, "Screencast/Stock-Prediction-Models", "strategy", "wrapped", "models/ensemble.py"),
    _r(46, "hudson-and-thames/financial-machine-learning", "strategy", "wrapped", "quant/backtest.py"),
    _r(47, "stefan-jansen/machine-learning-for-trading", "strategy", "wrapped", "models/, predictor/"),
    _r(48, "czsc/czsc", "strategy", "adapter", "integrations/analytics.py"),
    _r(49, "xFFFFF/Gekko-Strategies", "strategy", "reference"),
    _r(50, "asavinov/intelligent-trading-bot", "strategy", "reference"),
    # 5. Analytics & indicators
    _r(51, "TA-Lib/ta-lib", "analytics", "adapter", "integrations/analytics.py"),
    _r(52, "twopirllc/pandas-ta", "analytics", "adapter", "quant/pandasta_wrapper.py"),
    _r(53, "Riskfolio-Lib/Riskfolio-Lib", "analytics", "adapter", "integrations/analytics.py"),
    _r(54, "robertmartin8/PyPortfolioOpt", "analytics", "adapter", "integrations/analytics.py"),
    _r(55, "cvxgrp/cvxportfolio", "analytics", "adapter", "integrations/analytics.py"),
    _r(56, "fmilthaler/FinQuant", "analytics", "adapter", "integrations/analytics.py"),
    _r(57, "ranaroussi/quantstats", "analytics", "adapter", "quant/quantstats_reports.py"),
    _r(58, "yhilpisch/finance", "analytics", "wrapped", "quant/indicators.py"),
    _r(59, "FinancePy/FinancePy", "analytics", "adapter", "integrations/analytics.py"),
    _r(60, "quantmind/vollib", "analytics", "adapter", "integrations/analytics.py"),
    _r(61, "microprediction/precise", "analytics", "adapter", "integrations/analytics.py"),
    _r(62, "yuy0e/alphalens", "analytics", "adapter", "integrations/analytics.py"),
    _r(63, "quantopian/pyfolio", "analytics", "adapter", "integrations/analytics.py"),
    _r(64, "volest/volest", "analytics", "wrapped", "quant/risk.py"),
    # 6. Data
    _r(65, "ranaroussi/yfinance", "data", "core", "data/fetcher.py"),
    _r(66, "pyalgotrade/pyalgotrade", "data", "reference"),
    _r(67, "waditu/tushare", "data", "adapter", "integrations/market_data.py"),
    _r(68, "shutil/akshare", "data", "adapter", "integrations/market_data.py"),
    _r(69, "jerry-git/FinanceDatabase", "data", "adapter", "integrations/market_data.py"),
    _r(70, "JerBouma/FinanceToolkit", "data", "adapter", "integrations/market_data.py"),
    _r(71, "manahl/arcticdb", "data", "adapter", "integrations/storage.py"),
    _r(72, "ccxt/ccxt", "data", "adapter", "quant/ccxt_fetcher.py"),
    _r(73, "quandl/quandl-python", "data", "adapter", "integrations/market_data.py"),
    _r(74, "erdogant/stocksight", "data", "reference"),
    _r(75, "sec-edgar/sec-edgar", "data", "adapter", "integrations/market_data.py"),
    _r(76, "man-ahl/phis", "data", "reference"),
    _r(77, "twosigma/beakerx", "data", "reference"),
    _r(78, "goldmansachs/gs-quant", "data", "adapter", "integrations/analytics.py"),
    # 7. Infra
    _r(79, "tradingview/lightweight-charts", "infra", "core", "web/index.html"),
    _r(80, "LMAX-Exchange/Disruptor", "infra", "reference"),
    _r(81, "exchange-core/exchange-core", "infra", "reference"),
    _r(82, "duckdb/duckdb", "infra", "adapter", "integrations/storage.py"),
    _r(83, "bmoscon/cryptofeed", "infra", "adapter", "integrations/market_data.py"),
    _r(84, "pola-rs/polars", "infra", "adapter", "integrations/storage.py"),
    _r(85, "coala/coala", "infra", "reference"),
    _r(86, "enricoschumann/PMwR", "infra", "reference"),
    _r(87, "tradingticket/TradeItIosTicketSDK2", "infra", "reference"),
    _r(88, "ticker-terminal/ticker", "infra", "reference"),
    # 8. Awesome lists
    _r(89, "wilsonfreitas/awesome-quant", "awesome", "reference"),
    _r(90, "georgezouq/awesome-ai-in-finance", "awesome", "reference"),
    _r(91, "paperswithbacktest/awesome-systematic-trading", "awesome", "reference"),
    _r(92, "romainsimon/awesome-investing", "awesome", "reference"),
    _r(93, "joelowj/awesome-algorithmic-trading", "awesome", "reference"),
    _r(94, "cbailes/awesome-deep-trading", "awesome", "reference"),
    _r(95, "LLMQuant/awesome-trading-agents", "awesome", "reference"),
    _r(96, "artemiusgreat/Quantitative-Resources", "awesome", "reference"),
    _r(97, "EliteQuant/EliteQuant", "awesome", "reference"),
    _r(98, "Jera-Value/awesome-investing-tools-and-software-directory", "awesome", "reference"),
    _r(99, "melg8/awesome-investment", "awesome", "reference"),
    _r(100, "alpha-miner/quant-developer-resources", "awesome", "reference"),
]

_BY_ID = {r.id: r for r in REPOS}


def get_repo(repo_id: int) -> Repo:
    return _BY_ID[repo_id]


def list_by_category(category: str) -> list[Repo]:
    return [r for r in REPOS if r.category == category]


def list_by_status(status: str) -> list[Repo]:
    return [r for r in REPOS if r.status == status]


def integration_summary() -> dict[str, int]:
    out: dict[str, int] = {}
    for r in REPOS:
        out[r.status] = out.get(r.status, 0) + 1
    out["total"] = len(REPOS)
    return out


def to_records() -> list[dict]:
    return [asdict(r) for r in REPOS]
