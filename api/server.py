from __future__ import annotations
import os
import sys
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from data.fetcher import MarketDataFetcher
from predictor.bridge import PredictorBridge
from predictor.engine import PredictorConfig, PredictionTarget, REGIME_NAMES
from utils.logger import setup_logger

logger = setup_logger("API", "INFO")

app = FastAPI(title="AI Trading System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    symbol: str
    horizon: int = 5


class AnalysisResponse(BaseModel):
    symbol: str
    predictions: List[Dict]
    signals: List[Dict]
    metrics: List[Dict]
    market_state: Dict
    feature_importance: List[Dict]


class SearchResult(BaseModel):
    symbol: str
    name: str
    type: str


_bridge_cache: Dict[str, PredictorBridge] = {}
_fetcher = MarketDataFetcher()


def _get_bridge(symbols: List[str], start: str, end: str) -> PredictorBridge:
    cache_key = f"{','.join(sorted(symbols))}_{start}_{end}"
    if cache_key not in _bridge_cache:
        price_data = _fetcher.fetch(symbols, start, end)
        benchmark = _fetcher.fetch_benchmark("SPY", start, end)
        pred_cfg = PredictorConfig(
            horizons=(1, 5, 10, 20),
            primary_horizon=5,
            target_type=PredictionTarget.DIRECTION,
            min_history=260,
            n_bootstrap=15,
            xgb_n_estimators=300,
            feature_selection_enabled=True,
            max_features=80,
            log_level="WARNING",
        )
        bridge = PredictorBridge(cfg=pred_cfg)
        bridge.fit(price_data, benchmark)
        _bridge_cache[cache_key] = bridge
    return _bridge_cache[cache_key]


@app.get("/")
async def root():
    html_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AI Trading System API</h1><p>Web UI not found. Use /docs for API.</p>")


@app.get("/api/search")
async def search_symbols(q: str = Query(..., min_length=1, max_length=20)):
    import yfinance as yf
    results = []
    try:
        tickers = yf.search(q) if hasattr(yf, 'search') else []
        if not tickers:
            try:
                t = yf.Ticker(q)
                info = t.info
                results.append(SearchResult(
                    symbol=q.upper(),
                    name=info.get("longName", info.get("shortName", q.upper())),
                    type=info.get("quoteType", "unknown")
                ))
            except Exception:
                results.append(SymbolResult(symbol=q.upper(), name=q.upper(), type="unknown"))
        else:
            for t in tickers[:10]:
                results.append(SearchResult(
                    symbol=t.get("symbol", ""),
                    name=t.get("shortname", t.get("longname", t.get("symbol", ""))),
                    type=t.get("quoteType", "unknown")
                ))
    except Exception as e:
        logger.warning(f"Search failed: {e}")
        results.append(SymbolResult(symbol=q.upper(), name=q.upper(), type="unknown"))
    return results


@app.get("/api/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    horizon: int = 5,
):
    try:
        bridge = _get_bridge([symbol], start, end)
        pred_cfg = bridge.predictor.cfg

        price_data = _fetcher.fetch([symbol], start, end)
        benchmark = _fetcher.fetch_benchmark("SPY", start, end)

        preds = bridge.predict(price_data, benchmark)
        if preds.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        signals = bridge.get_signals_df().to_dict("records")
        metrics = bridge.metrics_report().to_dict("records")
        mstate = bridge.get_market_state(benchmark)
        imp = bridge.feature_importance(top_k=20)
        feature_importance = [{"feature": k, "importance": float(v)} for k, v in imp.items()]

        if "date" in preds.columns:
            preds["date"] = preds["date"].astype(str)

        response = AnalysisResponse(
            symbol=symbol.upper(),
            predictions=preds.to_dict("records"),
            signals=signals,
            metrics=metrics,
            market_state=mstate,
            feature_importance=feature_importance,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze_batch")
async def analyze_batch(
    symbols: str = Query(..., description="Comma-separated symbols"),
    start: str = "2020-01-01",
    end: str = "2024-12-31",
):
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    if len(sym_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 symbols per batch")

    try:
        price_data = _fetcher.fetch(sym_list, start, end)
        benchmark = _fetcher.fetch_benchmark("SPY", start, end)

        pred_cfg = PredictorConfig(
            horizons=(1, 5, 10, 20),
            primary_horizon=5,
            target_type=PredictionTarget.DIRECTION,
            min_history=260,
            n_bootstrap=15,
            xgb_n_estimators=300,
            feature_selection_enabled=True,
            max_features=80,
            log_level="WARNING",
        )
        bridge = PredictorBridge(cfg=pred_cfg)
        bridge.fit(price_data, benchmark)

        preds = bridge.predict(price_data, benchmark)
        signals = bridge.get_signals_df()
        mstate = bridge.get_market_state(benchmark)

        if "date" in preds.columns:
            preds["date"] = preds["date"].astype(str)

        return {
            "symbols": sym_list,
            "predictions": preds.to_dict("records"),
            "signals": signals.to_dict("records"),
            "market_state": mstate,
            "metrics": bridge.metrics_report().to_dict("records"),
        }
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market_state")
async def market_state():
    benchmark = _fetcher.fetch_benchmark("SPY", "2023-01-01", "2024-12-31")
    from predictor.engine import RegimeDetector
    detector = RegimeDetector()
    regime = detector.classify_single(benchmark["close"])
    return {
        "regime": REGIME_NAMES.get(regime, "unknown"),
        "risk_multiplier": RegimeDetector.regime_risk_multiplier(regime),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


def start():
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
