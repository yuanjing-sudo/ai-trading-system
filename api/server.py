from __future__ import annotations
import os
import sys
import json
import asyncio
from typing import Dict, List, Optional, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from data.fetcher import MarketDataFetcher
from predictor.bridge import PredictorBridge
from predictor.engine import PredictorConfig, PredictionTarget, REGIME_NAMES
from crypto.bridge import CryptoPredictorBridge
from crypto.engine import CryptoPredictorConfig, CryptoPredictionTarget
from utils.logger import setup_logger

logger = setup_logger("API", "INFO")

app = FastAPI(title="AI Trading System", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)
_bridge_cache: Dict[str, PredictorBridge] = {}
_crypto_bridge_cache: Dict[str, CryptoPredictorBridge] = {}
_fetcher = MarketDataFetcher()
_httpx_client = httpx.AsyncClient(timeout=30.0)

STOCK_UNIVERSE: Set[str] = set()
CRYPTO_UNIVERSE: Set[str] = set()
ETF_UNIVERSE: Set[str] = set()
FOREX_UNIVERSE: Set[str] = set()
COMMODITY_UNIVERSE: Set[str] = set()

CRYPTO_ALIASES = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "BNB": "BNB-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "DOGE": "DOGE-USD", "DOT": "DOT-USD",
    "MATIC": "MATIC-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD", "UNI": "UNI-USD",
    "LTC": "LTC-USD", "BCH": "BCH-USD", "XLM": "XLM-USD", "ALGO": "ALGO-USD",
    "VET": "VET-USD", "FIL": "FIL-USD", "ICP": "ICP-USD", "ETC": "ETC-USD",
    "XMR": "XMR-USD", "XTZ": "XTZ-USD", "ATOM": "ATOM-USD", "AAVE": "AAVE-USD",
    "EGLD": "EGLD-USD", "THETA": "THETA-USD", "FTM": "FTM-USD", "HBAR": "HBAR-USD",
    "EOS": "EOS-USD", "NEO": "NEO-USD", "KSM": "KSM-USD", "WAVES": "WAVES-USD",
    "ZEC": "ZEC-USD", "DASH": "DASH-USD", "COMP": "COMP-USD", "SNX": "SNX-USD",
    "YFI": "YFI-USD", "SUSHI": "SUSHI-USD", "CRV": "CRV-USD", "1INCH": "1INCH-USD",
    "ENJ": "ENJ-USD", "CHZ": "CHZ-USD", "BAT": "BAT-USD", "MANA": "MANA-USD",
    "SAND": "SAND-USD", "AXS": "AXS-USD", "GALA": "GALA-USD", "ILV": "ILV-USD",
    "DYDX": "DYDX-USD", "GRT": "GRT-USD", "LRC": "LRC-USD", "IMX": "IMX-USD",
    "RNDR": "RNDR-USD", "FET": "FET-USD", "AGIX": "AGIX-USD", "OCEAN": "OCEAN-USD",
    "AR": "AR-USD", "STORJ": "STORJ-USD", "SKL": "SKL-USD",
    "GOLD": "GC=F", "SILVER": "SI=F", "OIL": "CL=F", "NATGAS": "NG=F",
    "COPPER": "HG=F", "PLATINUM": "PL=F", "PALLADIUM": "PA=F",
    "WHEAT": "ZW=F", "CORN": "ZC=F", "SOYBEANS": "ZS=F",
    "COFFEE": "KC=F", "SUGAR": "SB=F", "COTTON": "CT=F",
    "COCOA": "CC=F", "ORANGE": "OJ=F",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "USDCNY": "USDCNY=X", "USDHKD": "USDHKD=X",
    "USDSGD": "USDSGD=X", "USDKRW": "USDKRW=X", "USDINR": "USDINR=X",
}


def _resolve_symbol(symbol: str) -> str:
    upper = symbol.upper()
    if upper in CRYPTO_ALIASES:
        return CRYPTO_ALIASES[upper]
    return symbol

CURRENCY_RATES: Dict[str, float] = {"USD": 1.0}
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "KRW": "₩", "INR": "₹", "CAD": "C$", "AUD": "A$", "CHF": "Fr",
    "SGD": "S$", "HKD": "HK$", "BRL": "R$", "MXN": "MX$",
}


def _load_universes():
    global STOCK_UNIVERSE, CRYPTO_UNIVERSE, ETF_UNIVERSE, FOREX_UNIVERSE, COMMODITY_UNIVERSE
    try:
        import yfinance as yf
        tickers = yf.Tickers('')
    except Exception:
        pass
    CRYPTO_UNIVERSE = {
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "DOT-USD", "MATIC-USD", "AVAX-USD", "LINK-USD", "UNI-USD",
        "LTC-USD", "BCH-USD", "XLM-USD", "ALGO-USD", "VET-USD", "FIL-USD",
        "ICP-USD", "ETC-USD", "XMR-USD", "XTZ-USD", "ATOM-USD", "AAVE-USD",
        "EGLD-USD", "THETA-USD", "FTM-USD", "HBAR-USD", "EOS-USD", "NEO-USD",
        "KSM-USD", "WAVES-USD", "ZEC-USD", "DASH-USD", "COMP-USD", "SNX-USD",
        "YFI-USD", "SUSHI-USD", "CRV-USD", "1INCH-USD", "ENJ-USD", "CHZ-USD",
        "BAT-USD", "MANA-USD", "SAND-USD", "AXS-USD", "GALA-USD", "ILV-USD",
        "DYDX-USD", "GRT-USD", "LRC-USD", "IMX-USD", "RNDR-USD", "FET-USD",
        "AGIX-USD", "OCEAN-USD", "FET-USD", "AR-USD", "STORJ-USD", "SKL-USD",
    }
    ETF_UNIVERSE = {
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "BND",
        "AGG", "GLD", "SLV", "UNG", "USO", "TLT", "HYG", "LQD", "EMB",
        "EEM", "EFA", "IXUS", "VT", "RSP", "SCHD", "VUG", "VO", "VB",
        "VHT", "VGT", "VDC", "VDE", "VNQ", "VAW", "VIS", "VCR", "VPU",
        "XLF", "XLE", "XLV", "XLI", "XLP", "XLK", "XLU", "XLY", "XLB",
        "MDY", "IJR", "IJH", "IWF", "IWD", "SCHX", "ITOT", "USMV", "QUAL",
        "MTUM", "SMLF", "VLUE", "EFV", "EFAV", "EFG", "SPYG", "SUSL",
        "CRBN", "ESGV", "ESGD", "ESGE", "AOK", "AOM", "AOA", "MNA",
        "QEMM", "ACWI", "ACWX", "SCZ", "IEFA", "IEF", "SHY", "IEI",
        "GOVT", "TIP", "MBB", "JNK", "BKLN", "PFF", "SPLV", "SPHD",
        "NOBL", "HDV", "VYM", "SCHD", "JEPI", "JEPQ", "PPLT", "PALL",
        "DBC", "GSG", "RJI", "FTGC", "PDBC", "USCI", "FIVG", "FBCG",
        "FILL", "FXI", "ASHR", "FXN", "FXR", "FXZ", "FXU", "FXB", "FXH",
    }
    FOREX_UNIVERSE = {
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X",
        "AUDUSD=X", "NZDUSD=X", "USDCNY=X", "USDHKD=X", "USDSGD=X",
        "USDKRW=X", "USDINR=X", "USDMXN=X", "USDBRL=X", "USDZAR=X",
        "USDRUB=X", "USDTRY=X", "USDNOK=X", "USDSEK=X", "USDDKK=X",
        "USDXDR=X", "USDCNH=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
        "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X", "EURCHF=X",
        "EURAUD=X", "EURCAD=X", "GBPAUD=X", "GBPCAD=X", "GBPNZD=X",
        "AUDCAD=X", "AUDNZD=X", "USDCOP=X", "USDPHP=X", "USDTHB=X",
        "USDMYR=X", "USDIDR=X", "USDTWD=X", "USDPEN=X", "USDCLP=X",
    }
    COMMODITY_UNIVERSE = {
        "GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F", "NG=F", "RB=F",
        "HO=F", "BZ=F", "ZC=F", "ZW=F", "ZS=F", "ZL=F", "ZM=F", "ZR=F",
        "KC=F", "CT=F", "SB=F", "CC=F", "OJ=F", "LE=F", "HE=F", "GF=F",
        "LBS=F", "CC=F", "BTC=F", "ETH=F", "NG=F", "GB=F", "SF=F",
    }


class AnalysisRequest(BaseModel):
    symbol: str
    horizon: int = 5
    asset_class: str = "stocks"
    currency: str = "USD"


class AnalysisResponse(BaseModel):
    symbol: str
    asset_class: str
    currency: str
    currency_symbol: str
    predictions: List[Dict]
    signals: List[Dict]
    metrics: List[Dict]
    market_state: Dict
    feature_importance: List[Dict]
    asset_specific: Dict


class SearchResult(BaseModel):
    symbol: str
    name: str
    type: str
    asset_class: str
    price: Optional[float] = None
    currency: str = "USD"


def _get_bridge(symbols: List[str], start: str, end: str) -> PredictorBridge:
    cache_key = f"{','.join(sorted(symbols))}_{start}_{end}"
    if cache_key not in _bridge_cache:
        price_data = _fetcher.fetch(symbols, start, end)
        benchmark = _fetcher.fetch_benchmark("SPY", start, end)
        pred_cfg = PredictorConfig(
            horizons=(1, 5, 10, 20),
            primary_horizon=5,
            target_type=PredictionTarget.DIRECTION,
            min_history=200,
            n_bootstrap=10,
            xgb_n_estimators=200,
            xgb_max_depth=5,
            feature_selection_enabled=True,
            max_features=60,
            online_enabled=False,
            log_level="WARNING",
        )
        bridge = PredictorBridge(cfg=pred_cfg)
        bridge.fit(price_data, benchmark)
        _bridge_cache[cache_key] = bridge
    return _bridge_cache[cache_key]


def _get_crypto_bridge(symbols: List[str], start: str, end: str) -> CryptoPredictorBridge:
    cache_key = f"{','.join(sorted(symbols))}_{start}_{end}"
    if cache_key not in _crypto_bridge_cache:
        price_data = _fetcher.fetch(symbols, start, end)
        bench_data = _fetcher.fetch(["BTC-USD"], start, end)
        benchmark = bench_data.get("BTC-USD", pd.DataFrame())
        if benchmark.empty:
            benchmark = _fetcher.fetch_benchmark("SPY", start, end)
        pred_cfg = CryptoPredictorConfig(
            horizons_hours=(1, 4, 12, 24, 72, 168),
            primary_horizon_hours=24,
            target_type=CryptoPredictionTarget.DIRECTION,
            min_history=500,
            n_bootstrap=15,
            xgb_n_estimators=300,
            xgb_max_depth=5,
            feature_selection_enabled=True,
            max_features=100,
            online_enabled=False,
            log_level="WARNING",
        )
        bridge = CryptoPredictorBridge(cfg=pred_cfg)
        bridge.fit(price_data, benchmark)
        _crypto_bridge_cache[cache_key] = bridge
    return _crypto_bridge_cache[cache_key]


async def _fetch_crypto_indicators() -> Dict:
    try:
        btc = _fetcher.fetch(["BTC-USD"], "2023-01-01", "2024-12-31")
        total = _fetcher.fetch(["BTC-USD", "ETH-USD"], "2023-01-01", "2024-12-31")
        btc_df = btc.get("BTC-USD", pd.DataFrame())
        eth_df = total.get("ETH-USD", pd.DataFrame())
        if btc_df.empty or eth_df.empty:
            return {"btc_dominance": 50.0, "altcoin_season": False, "total_market_cap_trend": "neutral"}
        btc_mcap_proxy = btc_df["close"] * 19000000
        eth_mcap_proxy = eth_df["close"] * 120000000
        total_mcap = btc_mcap_proxy + eth_mcap_proxy
        btc_dom = (btc_mcap_proxy / total_mcap * 100).iloc[-1]
        btc_30d = btc_df["close"].pct_change(30).iloc[-1] if len(btc_df) > 30 else 0
        eth_30d = eth_df["close"].pct_change(30).iloc[-1] if len(eth_df) > 30 else 0
        alt_season = eth_30d > btc_30d
        return {
            "btc_dominance": float(btc_dom),
            "altcoin_season": bool(alt_season),
            "btc_30d_return": float(btc_30d * 100),
            "eth_30d_return": float(eth_30d * 100),
            "total_market_cap_trend": "up" if btc_30d > 0 else "down",
        }
    except Exception:
        return {}


async def _fetch_tokenomics(symbol: str) -> Dict:
    return {
        "total_supply": "N/A",
        "circulating_supply": "N/A",
        "max_supply": "N/A",
        "staking_apy": "N/A",
        "tvl": "N/A",
        "note": "Tokenomics data requires external API (DeFiLlama/CoinGecko).",
    }


async def _fetch_forex_indicators(symbol: str) -> Dict:
    try:
        df = _fetcher.fetch([symbol.replace("=X", "-USD") if symbol.endswith("=X") else symbol], "2024-01-01", "2024-12-31")
        d = list(df.values())[0] if df else pd.DataFrame()
        if d.empty:
            return {}
        price = float(d["close"].iloc[-1])
        ma50 = float(d["close"].rolling(50).mean().iloc[-1])
        ma200 = float(d["close"].rolling(200).mean().iloc[-1])
        vol = float(d["close"].pct_change().rolling(20).std().iloc[-1] * 100)
        return {
            "current_rate": price,
            "trend": "up" if price > ma50 > ma200 else "down" if price < ma50 < ma200 else "mixed",
            "ma50": ma50,
            "ma200": ma200,
            "volatility_20d": vol,
        }
    except Exception:
        return {}


async def _fetch_commodity_indicators(symbol: str) -> Dict:
    try:
        df = _fetcher.fetch([symbol], "2024-01-01", "2024-12-31")
        d = list(df.values())[0] if df else pd.DataFrame()
        if d.empty:
            return {}
        price = float(d["close"].iloc[-1])
        ma50 = float(d["close"].rolling(50).mean().iloc[-1])
        ma200 = float(d["close"].rolling(200).mean().iloc[-1])
        vol = float(d["close"].pct_change().rolling(20).std().iloc[-1] * 100)
        return {
            "current_price": price,
            "trend": "up" if price > ma50 > ma200 else "down" if price < ma50 < ma200 else "mixed",
            "ma50": ma50,
            "ma200": ma200,
            "volatility_20d": vol,
            "contango_backwardation": "contango" if price < ma50 else "backwardation",
        }
    except Exception:
        return {}


async def _fetch_etf_indicators(symbol: str) -> Dict:
    try:
        df = _fetcher.fetch([symbol], "2024-01-01", "2024-12-31")
        d = list(df.values())[0] if df else pd.DataFrame()
        if d.empty:
            return {}
        price = float(d["close"].iloc[-1])
        ma50 = float(d["close"].rolling(50).mean().iloc[-1])
        ma200 = float(d["close"].rolling(200).mean().iloc[-1])
        vol = float(d["close"].pct_change().rolling(20).std().iloc[-1] * 100)
        adv = float((d["close"] * d["volume"]).rolling(20).mean().iloc[-1])
        return {
            "current_price": price,
            "trend": "up" if price > ma50 > ma200 else "down" if price < ma50 < ma200 else "mixed",
            "ma50": ma50,
            "ma200": ma200,
            "volatility_20d": vol,
            "adv_20d": adv,
        }
    except Exception:
        return {}


async def _fetch_currency_rates() -> Dict[str, float]:
    global CURRENCY_RATES
    try:
        pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "AUDUSD=X", "USDCNY=X", "USDINR=X", "USDKRW=X", "USDSGD=X", "USDMXN=X", "USDBRL=X"]
        data = _fetcher.fetch(pairs, "2024-12-01", "2024-12-31")
        for pair, df in data.items():
            if not df.empty:
                rate = float(df["close"].iloc[-1])
                if pair.startswith("USD"):
                    currency = pair.replace("USD", "").replace("=X", "")
                    if currency and currency != "USD":
                        CURRENCY_RATES[currency] = rate
                elif pair.endswith("=X"):
                    base = pair[:3]
                    CURRENCY_RATES[base] = 1.0 / rate
    except Exception:
        pass
    return CURRENCY_RATES


@app.on_event("startup")
async def startup():
    _load_universes()
    await _fetch_currency_rates()


@app.get("/")
async def root():
    html_path = Path(__file__).resolve().parent.parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AI Trading System API</h1>")


@app.get("/api/search")
async def search_symbols(q: str = Query(..., min_length=1, max_length=50)):
    import yfinance as yf
    results = []
    seen = set()
    q_upper = q.upper().strip()

    resolved = _resolve_symbol(q_upper)
    if resolved != q_upper and resolved not in seen:
        seen.add(resolved)
        results.append(SearchResult(
            symbol=resolved, name=resolved, type="crypto", asset_class="crypto",
            price=None, currency="USD"
        ))

    try:
        try:
            tickers = yf.search(q)
            for t in tickers[:20]:
                sym = t.get("symbol", "")
                if sym and sym not in seen:
                    seen.add(sym)
                    asset_cls = _classify_symbol(sym)
                    price = t.get("regularMarketPrice", None)
                    currency = t.get("currency", "USD")
                    results.append(SearchResult(
                        symbol=sym,
                        name=t.get("shortname", t.get("longname", sym)),
                        type=t.get("quoteType", "unknown"),
                        asset_class=asset_cls,
                        price=price,
                        currency=currency,
                    ))
        except Exception:
            pass

        for universe, cls in [(CRYPTO_UNIVERSE, "crypto"), (ETF_UNIVERSE, "etf"),
                               (FOREX_UNIVERSE, "forex"), (COMMODITY_UNIVERSE, "commodities")]:
            for sym in sorted(universe):
                if q_upper in sym.upper() and sym not in seen:
                    seen.add(sym)
                    results.append(SearchResult(
                        symbol=sym, name=sym, type=cls, asset_class=cls,
                        price=None, currency="USD"
                    ))

        for alias, resolved_sym in CRYPTO_ALIASES.items():
            if q_upper in alias.upper() and resolved_sym not in seen:
                seen.add(resolved_sym)
                asset_cls = _classify_symbol(resolved_sym)
                results.append(SearchResult(
                    symbol=resolved_sym, name=resolved_sym, type=asset_cls,
                    asset_class=asset_cls, price=None, currency="USD"
                ))

        if not results:
            try:
                t = yf.Ticker(q)
                info = t.info
                sym = info.get("symbol", q_upper)
                asset_cls = _classify_symbol(sym)
                results.append(SearchResult(
                    symbol=sym,
                    name=info.get("longName", info.get("shortName", sym)),
                    type=info.get("quoteType", "unknown"),
                    asset_class=asset_cls,
                    price=info.get("regularMarketPrice", None),
                    currency=info.get("currency", "USD"),
                ))
            except Exception:
                sym = q_upper
                results.append(SearchResult(
                    symbol=sym, name=sym, type="unknown",
                    asset_class=_classify_symbol(sym), price=None, currency="USD"
                ))
    except Exception as e:
        logger.warning(f"Search failed: {e}")
    return results[:50]


def _classify_symbol(symbol: str) -> str:
    if symbol in CRYPTO_UNIVERSE or symbol.endswith("-USD"):
        return "crypto"
    if symbol in ETF_UNIVERSE:
        return "etf"
    if symbol in FOREX_UNIVERSE:
        return "forex"
    if symbol in COMMODITY_UNIVERSE:
        return "commodities"
    return "stocks"


@app.get("/api/analyze/{symbol}")
async def analyze_symbol(
    symbol: str,
    asset_class: str = "stocks",
    currency: str = "USD",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
):
    symbol = _resolve_symbol(symbol)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, _analyze_sync, symbol, asset_class, currency, start, end
    )
    return result


def _analyze_sync(symbol: str, asset_class: str, currency: str, start: str, end: str) -> Dict:
    try:
        price_data = _fetcher.fetch([symbol], start, end)
        if not price_data:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        rate = CURRENCY_RATES.get(currency, 1.0)
        ccy_sym = CURRENCY_SYMBOLS.get(currency, "$")

        if asset_class == "crypto":
            bench_data = _fetcher.fetch(["BTC-USD"], start, end)
            benchmark = bench_data.get("BTC-USD", pd.DataFrame())
            if benchmark.empty:
                benchmark = _fetcher.fetch_benchmark("SPY", start, end)
            bridge = _get_crypto_bridge([symbol], start, end)
            preds = bridge.predict(price_data, benchmark)
            signals = bridge.get_signals_df()
            metrics = bridge.metrics_report()
            mstate = bridge.get_market_state(benchmark)
            imp = bridge.feature_importance(top_k=20)
            feature_importance = [{"feature": k, "importance": float(v)} for k, v in imp.items()]
            asset_specific = asyncio.get_event_loop().run_until_complete(
                _fetch_crypto_indicators()
            )
            asset_specific["tokenomics"] = asyncio.get_event_loop().run_until_complete(
                _fetch_tokenomics(symbol)
            )
        else:
            benchmark = _fetcher.fetch_benchmark("SPY", start, end)
            bridge = _get_bridge([symbol], start, end)
            preds = bridge.predict(price_data, benchmark)
            signals = bridge.get_signals_df()
            metrics = bridge.metrics_report()
            mstate = bridge.get_market_state(benchmark)
            imp = bridge.feature_importance(top_k=20)
            feature_importance = [{"feature": k, "importance": float(v)} for k, v in imp.items()]
            asset_specific = {}
            if asset_class == "forex":
                asset_specific = asyncio.get_event_loop().run_until_complete(
                    _fetch_forex_indicators(symbol)
                )
            elif asset_class == "commodities":
                asset_specific = asyncio.get_event_loop().run_until_complete(
                    _fetch_commodity_indicators(symbol)
                )
            elif asset_class == "etf":
                asset_specific = asyncio.get_event_loop().run_until_complete(
                    _fetch_etf_indicators(symbol)
                )

        if "date" in preds.columns:
            preds["date"] = preds["date"].astype(str)
        if "timestamp" in preds.columns:
            preds["timestamp"] = preds["timestamp"].astype(str)

        for col in ["expected_return", "expected_return_lower", "expected_return_upper"]:
            if col in preds.columns:
                preds[col] = preds[col].apply(lambda x: x * rate)

        return AnalysisResponse(
            symbol=symbol.upper(),
            asset_class=asset_class,
            currency=currency,
            currency_symbol=ccy_sym,
            predictions=preds.to_dict("records"),
            signals=signals.to_dict("records"),
            metrics=metrics.to_dict("records"),
            market_state=mstate,
            feature_importance=feature_importance,
            asset_specific=asset_specific,
        ).dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyze_batch")
async def analyze_batch(
    symbols: str = Query(...),
    asset_class: str = "stocks",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
):
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    if len(sym_list) > 15:
        raise HTTPException(status_code=400, detail="Maximum 15 symbols per batch")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, _analyze_batch_sync, sym_list, asset_class, start, end
    )
    return result


def _analyze_batch_sync(sym_list: List[str], asset_class: str, start: str, end: str) -> Dict:
    try:
        price_data = _fetcher.fetch(sym_list, start, end)
        benchmark = _fetcher.fetch_benchmark("SPY", start, end)
        bridge = _get_bridge(sym_list, start, end)
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
        "currencies": CURRENCY_RATES,
    }


@app.get("/api/currencies")
async def currencies():
    await _fetch_currency_rates()
    return {
        "symbols": CURRENCY_SYMBOLS,
        "rates": CURRENCY_RATES,
    }


@app.get("/api/universe/{asset_class}")
async def get_universe(asset_class: str):
    universes = {
        "stocks": sorted(STOCK_UNIVERSE),
        "crypto": sorted(CRYPTO_UNIVERSE),
        "etf": sorted(ETF_UNIVERSE),
        "forex": sorted(FOREX_UNIVERSE),
        "commodities": sorted(COMMODITY_UNIVERSE),
    }
    if asset_class not in universes:
        raise HTTPException(status_code=404, detail=f"Unknown asset class: {asset_class}")
    return {"asset_class": asset_class, "symbols": universes[asset_class]}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}


def start():
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
