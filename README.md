# AI Trading System

Institutional-Grade Adaptive Market Prediction Engine with web interface.

## Features

- Multi-horizon prediction (1d, 5d, 10d, 20d) with per-horizon models
- Adaptive ensemble: XGBoost + ExtraTrees + HistGradientBoosting + Ridge meta
- Online incremental learning via passive-aggressive SGD overlay
- Regime detection (statistical rules + Gaussian mixture)
- Kalman filter trend extraction
- Bayesian uncertainty via bootstrap + quantile regression
- Purged, embargoed walk-forward cross-validation
- Fractional differentiation for stationarity with memory preservation
- Volatility forecasting (EWMA + GARCH-lite + realized-vol blend)
- Cross-sectional z-scoring / rank normalization
- Feature selection via mutual information + permutation importance
- Conformal prediction intervals for calibrated uncertainty
- Automatic feature health monitoring (drift detection via PSI)
- Multi-asset cross-sectional alpha + per-asset time-series alpha fusion
- Web UI with search bar for any stock, ETF, or crypto
- WorldQuant Alpha101 factor library (101 formulaic alphas)
- Quant toolkit: 100+ indicators, 50+ factors, 25 strategies, risk metrics, backtester, walk-forward optimizer
- pandas-ta / TA-Lib wrappers, CCXT crypto cross-exchange data, quantstats tear sheets
- Live annotatable Plotly charts (drawing tools, persistent notes per symbol)
- Multi-agent AI research team (technical / fundamental / sentiment / risk / PM): local Ollama at home, Groq free tier (`llama-3.3-70b-versatile`) in the cloud workflow when the `GROQ_API_KEY` secret is set, deterministic offline fallback everywhere
- 100-repo integration catalog with optional adapters (`integrations/`, see `docs/INTEGRATIONS.md`)
- GitHub cloud data pipeline: scheduled Actions snapshots stored in-repo and served via GitHub Pages

## Quick Start

### Web UI

```bash
pip install -r requirements.txt
python -m api.server
```

Open http://localhost:8000 in your browser.

### Cloud snapshot (GitHub as free cloud storage)

```bash
python scripts/cloud_snapshot.py --tickers AAPL MSFT NVDA --days 400
# writes docs/data/dashboard.json + docs/data/bars/*.json (served by GitHub Pages)
```

The `.github/workflows/cloud-data.yml` workflow refreshes the snapshot every weekday and commits it to the repo - the Pages site shows it live under "Live Cloud Snapshot".

### Optional integrations

```bash
pip install -r requirements-optional.txt   # any subset; all adapters degrade gracefully
python test_integrations.py                # verify available/fake deps and fallbacks
```

### CLI

```bash
python main.py --mode predictor --symbols AAPL MSFT NVDA --start 2020-01-01
python main.py --mode scan --symbols AAPL MSFT
python main.py --mode backtest --symbols AAPL MSFT NVDA TSLA
```

### API

```bash
curl http://localhost:8000/api/analyze/AAPL
curl http://localhost:8000/api/analyze_batch?symbols=AAPL,MSFT,NVDA
curl http://localhost:8000/api/market_state
curl http://localhost:8000/api/integrations
curl http://localhost:8000/api/agents/AAPL
```

## Architecture

```
ai_trading_system/
├── api/server.py          # FastAPI web server
├── web/index.html         # Frontend with search bar + annotatable charts
├── config.py              # Configuration management
├── data/                  # Data ingestion & feature engineering
├── models/                # XGBoost, AI analyst, ensemble
├── predictor/             # Advanced prediction engine subsystem
├── quant/                 # Standalone quant library (indicators/factors/strategies/risk/backtest/optimizer/alpha101)
├── integrations/          # 100-repo integration layer (data, analytics, backends, storage, agents)
├── scripts/               # Cloud snapshot + docs generators
├── docs/                  # GitHub Pages site + cloud data
├── scanner/               # Multi-factor scanner & patterns
├── strategy/              # Signals, risk, portfolio construction
├── backtest/              # Backtesting engine & metrics
├── execution/             # Broker abstraction (paper + IBKR)
└── main.py                # CLI entrypoint
```

## What Makes This Institutional-Grade

| Dimension | Implementation |
|---|---|
| No look-ahead bias | Walk-forward training with embargo |
| Model diversity | XGBoost + RF + GBM + logistic stacking |
| AI analysis | FinBERT sentiment + regime detection + LLM commentary |
| Risk management | Vol targeting, VaR caps, sector limits, correlation filter |
| Regime-adaptive | Volatility & trend regime modifies risk multiplier |
| Cross-sectional | Rank features across universe |
| Costs modeled | Commission + slippage bps |
| Realistic backtest | Weekly rebalance, position caps, mark-to-market |
| Production hooks | Paper broker + IBKR adapter, caching, structured logging |
| Analytics | Sharpe, Sortino, Calmar, VaR, CVaR, tail ratio, beta/alpha, IR |
