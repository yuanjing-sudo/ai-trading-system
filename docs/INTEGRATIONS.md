# Third-Party Integration Catalog

This repository consolidates the 100-repo trading/quant ecosystem watchlist.
Every entry below is tracked in `integrations/registry.py` and exposed via
`GET /api/integrations` and `GET /api/integrations/repos`.

## Status summary

| Status | Count | Meaning |
| --- | --- | --- |
| Core (wired in) | 2 | Primary dependency or first-class module |
| Adapter (optional pip) | 34 | Optional package with graceful fallback |
| Wrapped (native reimplementation) | 9 | Techniques reimplemented in quant/ or models/ |
| Reference (cataloged) | 55 | Cataloged for discovery (apps, awesome-lists, non-Python) |
| **Total** | **100** | |

## Research Platforms & Terminals

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 1 | [OpenBB-finance/OpenBBTerminal](https://github.com/OpenBB-finance/OpenBBTerminal) | Adapter (optional pip) | `integrations/market_data.py` |
| 2 | [Fincept-Corporation/FinceptTerminal](https://github.com/Fincept-Corporation/FinceptTerminal) | Reference (cataloged) | - |
| 3 | [shubhamsborkar/greeksoup](https://github.com/shubhamsborkar/greeksoup) | Reference (cataloged) | - |
| 4 | [portfolio-performance/portfolio](https://github.com/portfolio-performance/portfolio) | Reference (cataloged) | - |
| 5 | [ghostfolio/ghostfolio](https://github.com/ghostfolio/ghostfolio) | Reference (cataloged) | - |
| 6 | [firefly-iii/firefly-iii](https://github.com/firefly-iii/firefly-iii) | Reference (cataloged) | - |
| 7 | [actualbudget/actual](https://github.com/actualbudget/actual) | Reference (cataloged) | - |
| 8 | [maybe-finance/maybe](https://github.com/maybe-finance/maybe) | Reference (cataloged) | - |
| 9 | [midday-ai/midday](https://github.com/midday-ai/midday) | Reference (cataloged) | - |

## LLMs & Agentic AI Frameworks

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 10 | [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | Adapter (optional pip) | `integrations/agents.py` |
| 11 | [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | Reference (cataloged) | - |
| 12 | [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading) | Reference (cataloged) | - |
| 13 | [The-Swarm-Corporation/AutoHedge](https://github.com/The-Swarm-Corporation/AutoHedge) | Adapter (optional pip) | `integrations/agents.py` |
| 14 | [burnertube-bit/WallStreetAgents](https://github.com/burnertube-bit/WallStreetAgents) | Reference (cataloged) | - |
| 15 | [Richorrific-Rea/insider-agent](https://github.com/Richorrific-Rea/insider-agent) | Adapter (optional pip) | `integrations/market_data.py` |
| 16 | [OpenByteInc/QuantDinger](https://github.com/OpenByteInc/QuantDinger) | Reference (cataloged) | - |
| 17 | [The-Swarm-Corporation/Volara](https://github.com/The-Swarm-Corporation/Volara) | Reference (cataloged) | - |
| 18 | [ellenzishanli/SignalForge](https://github.com/ellenzishanli/SignalForge) | Reference (cataloged) | - |
| 19 | [valuecell-labs/valuecell](https://github.com/valuecell-labs/valuecell) | Reference (cataloged) | - |
| 20 | [LangAlpha/LangAlpha](https://github.com/LangAlpha/LangAlpha) | Reference (cataloged) | - |
| 21 | [ai-berkshire/ai-berkshire](https://github.com/ai-berkshire/ai-berkshire) | Reference (cataloged) | - |

## Quantitative & Algorithmic Trading Frameworks

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 22 | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | Reference (cataloged) | - |
| 23 | [mementum/backtrader](https://github.com/mementum/backtrader) | Adapter (optional pip) | `integrations/backends.py` |
| 24 | [microsoft/qlib](https://github.com/microsoft/qlib) | Adapter (optional pip) | `integrations/backends.py` |
| 25 | [vnpy/vnpy](https://github.com/vnpy/vnpy) | Reference (cataloged) | - |
| 26 | [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | Adapter (optional pip) | `integrations/backends.py` |
| 27 | [nautilus-trader/nautilus_trader](https://github.com/nautilus-trader/nautilus_trader) | Adapter (optional pip) | `integrations/backends.py` |
| 28 | [pmartin/backtesting.py](https://github.com/pmartin/backtesting.py) | Adapter (optional pip) | `integrations/backends.py` |
| 29 | [polars-algo/pybroker](https://github.com/polars-algo/pybroker) | Reference (cataloged) | - |
| 30 | [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | Reference (cataloged) | - |
| 31 | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Reference (cataloged) | - |
| 32 | [StockSharp/StockSharp](https://github.com/StockSharp/StockSharp) | Reference (cataloged) | - |
| 33 | [jesse-ai/jesse](https://github.com/jesse-ai/jesse) | Reference (cataloged) | - |
| 34 | [tensortrade-org/tensortrade](https://github.com/tensortrade-org/tensortrade) | Reference (cataloged) | - |
| 35 | [openalgo/openalgo](https://github.com/openalgo/openalgo) | Reference (cataloged) | - |
| 36 | [blankly-finance/blankly](https://github.com/blankly-finance/blankly) | Reference (cataloged) | - |
| 37 | [VisualHFT/VisualHFT](https://github.com/VisualHFT/VisualHFT) | Reference (cataloged) | - |
| 38 | [tradytics/eiten](https://github.com/tradytics/eiten) | Wrapped (native reimplementation) | `quant/strategies.py` |
| 39 | [aat-core/aat](https://github.com/aat-core/aat) | Reference (cataloged) | - |
| 40 | [zvt-project/zvt](https://github.com/zvt-project/zvt) | Reference (cataloged) | - |

## Market Strategy Libraries & Catalogs

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 41 | [freqtrade/freqtrade-strategies](https://github.com/freqtrade/freqtrade-strategies) | Wrapped (native reimplementation) | `quant/strategies.py` |
| 42 | [je-suis-tm/quant-trading](https://github.com/je-suis-tm/quant-trading) | Wrapped (native reimplementation) | `quant/strategies.py` |
| 43 | [yli188/WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code) | Wrapped (native reimplementation) | `quant/alpha101.py` |
| 44 | [fmzquant/strategies](https://github.com/fmzquant/strategies) | Reference (cataloged) | - |
| 45 | [Screencast/Stock-Prediction-Models](https://github.com/Screencast/Stock-Prediction-Models) | Wrapped (native reimplementation) | `models/ensemble.py` |
| 46 | [hudson-and-thames/financial-machine-learning](https://github.com/hudson-and-thames/financial-machine-learning) | Wrapped (native reimplementation) | `quant/backtest.py` |
| 47 | [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | Wrapped (native reimplementation) | `models/, predictor/` |
| 48 | [czsc/czsc](https://github.com/czsc/czsc) | Adapter (optional pip) | `integrations/analytics.py` |
| 49 | [xFFFFF/Gekko-Strategies](https://github.com/xFFFFF/Gekko-Strategies) | Reference (cataloged) | - |
| 50 | [asavinov/intelligent-trading-bot](https://github.com/asavinov/intelligent-trading-bot) | Reference (cataloged) | - |

## Mathematical Analytics & Indicators

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 51 | [TA-Lib/ta-lib](https://github.com/TA-Lib/ta-lib) | Adapter (optional pip) | `integrations/analytics.py` |
| 52 | [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta) | Adapter (optional pip) | `quant/pandasta_wrapper.py` |
| 53 | [Riskfolio-Lib/Riskfolio-Lib](https://github.com/Riskfolio-Lib/Riskfolio-Lib) | Adapter (optional pip) | `integrations/analytics.py` |
| 54 | [robertmartin8/PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | Adapter (optional pip) | `integrations/analytics.py` |
| 55 | [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio) | Adapter (optional pip) | `integrations/analytics.py` |
| 56 | [fmilthaler/FinQuant](https://github.com/fmilthaler/FinQuant) | Adapter (optional pip) | `integrations/analytics.py` |
| 57 | [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) | Adapter (optional pip) | `quant/quantstats_reports.py` |
| 58 | [yhilpisch/finance](https://github.com/yhilpisch/finance) | Wrapped (native reimplementation) | `quant/indicators.py` |
| 59 | [FinancePy/FinancePy](https://github.com/FinancePy/FinancePy) | Adapter (optional pip) | `integrations/analytics.py` |
| 60 | [quantmind/vollib](https://github.com/quantmind/vollib) | Adapter (optional pip) | `integrations/analytics.py` |
| 61 | [microprediction/precise](https://github.com/microprediction/precise) | Adapter (optional pip) | `integrations/analytics.py` |
| 62 | [yuy0e/alphalens](https://github.com/yuy0e/alphalens) | Adapter (optional pip) | `integrations/analytics.py` |
| 63 | [quantopian/pyfolio](https://github.com/quantopian/pyfolio) | Adapter (optional pip) | `integrations/analytics.py` |
| 64 | [volest/volest](https://github.com/volest/volest) | Wrapped (native reimplementation) | `quant/risk.py` |

## APIs, Market Crawlers & Historical Data

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 65 | [ranaroussi/yfinance](https://github.com/ranaroussi/yfinance) | Core (wired in) | `data/fetcher.py` |
| 66 | [pyalgotrade/pyalgotrade](https://github.com/pyalgotrade/pyalgotrade) | Reference (cataloged) | - |
| 67 | [waditu/tushare](https://github.com/waditu/tushare) | Adapter (optional pip) | `integrations/market_data.py` |
| 68 | [shutil/akshare](https://github.com/shutil/akshare) | Adapter (optional pip) | `integrations/market_data.py` |
| 69 | [jerry-git/FinanceDatabase](https://github.com/jerry-git/FinanceDatabase) | Adapter (optional pip) | `integrations/market_data.py` |
| 70 | [JerBouma/FinanceToolkit](https://github.com/JerBouma/FinanceToolkit) | Adapter (optional pip) | `integrations/market_data.py` |
| 71 | [manahl/arcticdb](https://github.com/manahl/arcticdb) | Adapter (optional pip) | `integrations/storage.py` |
| 72 | [ccxt/ccxt](https://github.com/ccxt/ccxt) | Adapter (optional pip) | `quant/ccxt_fetcher.py` |
| 73 | [quandl/quandl-python](https://github.com/quandl/quandl-python) | Adapter (optional pip) | `integrations/market_data.py` |
| 74 | [erdogant/stocksight](https://github.com/erdogant/stocksight) | Reference (cataloged) | - |
| 75 | [sec-edgar/sec-edgar](https://github.com/sec-edgar/sec-edgar) | Adapter (optional pip) | `integrations/market_data.py` |
| 76 | [man-ahl/phis](https://github.com/man-ahl/phis) | Reference (cataloged) | - |
| 77 | [twosigma/beakerx](https://github.com/twosigma/beakerx) | Reference (cataloged) | - |
| 78 | [goldmansachs/gs-quant](https://github.com/goldmansachs/gs-quant) | Adapter (optional pip) | `integrations/analytics.py` |

## Infrastructure, Charts & Low-Level Architecture

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 79 | [tradingview/lightweight-charts](https://github.com/tradingview/lightweight-charts) | Core (wired in) | `web/index.html` |
| 80 | [LMAX-Exchange/Disruptor](https://github.com/LMAX-Exchange/Disruptor) | Reference (cataloged) | - |
| 81 | [exchange-core/exchange-core](https://github.com/exchange-core/exchange-core) | Reference (cataloged) | - |
| 82 | [duckdb/duckdb](https://github.com/duckdb/duckdb) | Adapter (optional pip) | `integrations/storage.py` |
| 83 | [bmoscon/cryptofeed](https://github.com/bmoscon/cryptofeed) | Adapter (optional pip) | `integrations/market_data.py` |
| 84 | [pola-rs/polars](https://github.com/pola-rs/polars) | Adapter (optional pip) | `integrations/storage.py` |
| 85 | [coala/coala](https://github.com/coala/coala) | Reference (cataloged) | - |
| 86 | [enricoschumann/PMwR](https://github.com/enricoschumann/PMwR) | Reference (cataloged) | - |
| 87 | [tradingticket/TradeItIosTicketSDK2](https://github.com/tradingticket/TradeItIosTicketSDK2) | Reference (cataloged) | - |
| 88 | [ticker-terminal/ticker](https://github.com/ticker-terminal/ticker) | Reference (cataloged) | - |

## Curated Resource Lists & Aggregators

| # | Repository | Status | Where |
| --- | --- | --- | --- |
| 89 | [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | Reference (cataloged) | - |
| 90 | [georgezouq/awesome-ai-in-finance](https://github.com/georgezouq/awesome-ai-in-finance) | Reference (cataloged) | - |
| 91 | [paperswithbacktest/awesome-systematic-trading](https://github.com/paperswithbacktest/awesome-systematic-trading) | Reference (cataloged) | - |
| 92 | [romainsimon/awesome-investing](https://github.com/romainsimon/awesome-investing) | Reference (cataloged) | - |
| 93 | [joelowj/awesome-algorithmic-trading](https://github.com/joelowj/awesome-algorithmic-trading) | Reference (cataloged) | - |
| 94 | [cbailes/awesome-deep-trading](https://github.com/cbailes/awesome-deep-trading) | Reference (cataloged) | - |
| 95 | [LLMQuant/awesome-trading-agents](https://github.com/LLMQuant/awesome-trading-agents) | Reference (cataloged) | - |
| 96 | [artemiusgreat/Quantitative-Resources](https://github.com/artemiusgreat/Quantitative-Resources) | Reference (cataloged) | - |
| 97 | [EliteQuant/EliteQuant](https://github.com/EliteQuant/EliteQuant) | Reference (cataloged) | - |
| 98 | [Jera-Value/awesome-investing-tools-and-software-directory](https://github.com/Jera-Value/awesome-investing-tools-and-software-directory) | Reference (cataloged) | - |
| 99 | [melg8/awesome-investment](https://github.com/melg8/awesome-investment) | Reference (cataloged) | - |
| 100 | [alpha-miner/quant-developer-resources](https://github.com/alpha-miner/quant-developer-resources) | Reference (cataloged) | - |

## Using the integrations

```python
from integrations import full_status
print(full_status())        # availability of every optional dependency

from integrations.agents import analyze_symbol
result = analyze_symbol('AAPL', ohlcv_df)   # multi-agent research note

from integrations.backends import compare_backends
compare_backends(prices, signals)           # native vs installed frameworks
```

Install optional dependencies: `pip install -r requirements-optional.txt`
or any subset - every adapter falls back to the native implementation.
