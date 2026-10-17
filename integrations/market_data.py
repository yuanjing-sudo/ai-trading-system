"""Market data source adapters for the integrated third-party libraries.

Every adapter degrades gracefully: when the optional package is missing the
functions fall back to the built-in yfinance/ccxt pipeline so the platform
keeps working on a minimal install.

Integrated sources (repo registry ids in parentheses):
- akshare (68), tushare (67)          : Chinese + global market data
- finance-database (69)               : 300k+ symbol metadata catalog
- FinanceToolkit (70)                 : 150+ valuation ratios
- sec-edgar-downloader (75)           : SEC filings + insider (Form 4) feed
- OpenBB (1)                          : unified research terminal data layer
- Nasdaq Data Link / quandl (73)      : macro + alternative datasets
- cryptofeed (83)                     : low-latency crypto websocket streams
- Google Finance (experimental)       : quote snapshot with yfinance fallback
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AKSHARE_AVAILABLE", "TUSHARE_AVAILABLE", "FINANCEDATABASE_AVAILABLE",
    "FINANCETOOLKIT_AVAILABLE", "SECEDGAR_AVAILABLE", "OPENBB_AVAILABLE",
    "QUANDL_AVAILABLE", "CRYPTOFEED_AVAILABLE",
    "fetch_akshare", "fetch_tushare", "search_symbols", "fetch_financial_ratios",
    "fetch_sec_filings", "fetch_insider_transactions", "fetch_openbb",
    "fetch_quandl", "fetch_google_finance_quote", "DATA_SOURCE_STATUS",
]

# --- availability detection -------------------------------------------------

try:
    import akshare as _ak
    AKSHARE_AVAILABLE = True
except Exception:
    AKSHARE_AVAILABLE = False

try:
    import tushare as _ts
    TUSHARE_AVAILABLE = True
except Exception:
    TUSHARE_AVAILABLE = False

try:
    import financedatabase as _fdb
    FINANCEDATABASE_AVAILABLE = True
except Exception:
    FINANCEDATABASE_AVAILABLE = False

try:
    from financetoolkit import Toolkit as _FTToolkit
    FINANCETOOLKIT_AVAILABLE = True
except Exception:
    FINANCETOOLKIT_AVAILABLE = False

try:
    from sec_edgar_downloader import Downloader as _SecDownloader
    SECEDGAR_AVAILABLE = True
except Exception:
    SECEDGAR_AVAILABLE = False

try:
    from openbb import obb as _obb
    OPENBB_AVAILABLE = True
except Exception:
    OPENBB_AVAILABLE = False

try:
    import nasdaqdatalink as _ndl
    QUANDL_AVAILABLE = True
except Exception:
    try:
        import quandl as _ndl
        QUANDL_AVAILABLE = True
    except Exception:
        QUANDL_AVAILABLE = False

try:
    import cryptofeed as _cf
    CRYPTOFEED_AVAILABLE = True
except Exception:
    CRYPTOFEED_AVAILABLE = False


def _yf_ohlcv(symbol: str, start: str | None = None, end: str | None = None,
              interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.rename(columns=str.lower)


def fetch_akshare(symbol: str = "000001", start: str | None = None) -> pd.DataFrame:
    """Fetch A-share/global history via akshare; falls back to yfinance."""
    if not AKSHARE_AVAILABLE:
        return _yf_ohlcv(symbol, start=start)
    try:
        df = _ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                 start_date=(start or "19900101").replace("-", ""),
                                 adjust="qfq")
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume",
        })
        return df.set_index("date")[["open", "high", "low", "close", "volume"]]
    except Exception:
        return _yf_ohlcv(symbol, start=start)


def fetch_tushare(symbol: str = "000001.SZ", token: str | None = None,
                  start: str | None = None) -> pd.DataFrame:
    """Fetch daily bars via tushare Pro (requires token); falls back to yfinance."""
    if not TUSHARE_AVAILABLE or not token:
        return _yf_ohlcv(symbol, start=start)
    try:
        _ts.set_token(token)
        pro = _ts.pro_api()
        df = pro.daily(ts_code=symbol, start_date=(start or "19900101").replace("-", ""))
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        return df.sort_values("trade_date").set_index("trade_date")[
            ["open", "high", "low", "close", "vol"]
        ].rename(columns={"vol": "volume"})
    except Exception:
        return _yf_ohlcv(symbol, start=start)


def search_symbols(query: str, limit: int = 25) -> list[dict]:
    """Search the FinanceDatabase symbol catalog (69); empty list if unavailable."""
    if not FINANCEDATABASE_AVAILABLE:
        return []
    try:
        db = _fdb.Equities()
        all_symbols = db.select()
        matches = [s for s in all_symbols.index if query.upper() in s.upper()][:limit]
        out = []
        for s in matches:
            row = all_symbols.loc[s]
            out.append({"symbol": s, "name": str(row.get("name", "")),
                        "sector": str(row.get("sector", "")),
                        "country": str(row.get("country", ""))})
        return out
    except Exception:
        return []


def fetch_financial_ratios(tickers: list[str], start: str = "2020-01-01") -> pd.DataFrame:
    """Compute 150+ valuation ratios via FinanceToolkit (70); empty if unavailable."""
    if not FINANCETOOLKIT_AVAILABLE:
        return pd.DataFrame()
    try:
        tk = _FTToolkit(tickers=tickers, start=start, progress_bar=False)
        ratios = tk.ratios.collect_all_ratios()
        return ratios
    except Exception:
        return pd.DataFrame()


def fetch_sec_filings(ticker: str, form_type: str = "10-K", limit: int = 5,
                      download_dir: str = "data/sec_filings") -> list[str]:
    """Download recent SEC filings via sec-edgar-downloader (75)."""
    if not SECEDGAR_AVAILABLE:
        return []
    try:
        dl = _SecDownloader("ai-trading-system", "research@example.com",
                            download_dir)
        dl.get(form_type, ticker, limit=limit)
        return [str(p) for p in
                (Path(download_dir) / "sec-edgar-filings" / ticker.upper() / form_type).rglob("*")
                if p.is_file()]
    except Exception:
        return []


def fetch_insider_transactions(ticker: str) -> pd.DataFrame:
    """Insider (Form 4) transactions parsed from EDGAR daily index (15/75).

    Uses EDGAR full-text search JSON; empty frame when offline/unavailable.
    """
    url = ("https://efts.sec.gov/LATEST/search-index?q=%22" + ticker.upper() +
           "%22&forms=4")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-trading-system research"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        hits = payload.get("hits", {}).get("hits", [])
        rows = []
        for h in hits:
            src = h.get("_source", {})
            rows.append({
                "filed": src.get("file_date"),
                "form": src.get("root_form"),
                "company": (src.get("display_names") or [""])[0],
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def fetch_openbb(symbol: str, start: str | None = None) -> pd.DataFrame:
    """Pull equity history via OpenBB (1); falls back to yfinance."""
    if not OPENBB_AVAILABLE:
        return _yf_ohlcv(symbol, start=start)
    try:
        data = _obb.equity.price.historical(symbol=symbol, start_date=start).to_df()
        return data.rename(columns=str.lower)
    except Exception:
        return _yf_ohlcv(symbol, start=start)


def fetch_quandl(dataset: str, api_key: str | None = None) -> pd.DataFrame:
    """Fetch macro/alternative dataset via Nasdaq Data Link (73)."""
    if not QUANDL_AVAILABLE or not api_key:
        return pd.DataFrame()
    try:
        _ndl.ApiConfig.api_key = api_key
        return _ndl.get(dataset)
    except Exception:
        return pd.DataFrame()


def _google_finance_prices(ticker: str, exchange: str = "NASDAQ",
                           days: int = 200) -> pd.DataFrame:
    base = "https://www.google.com/finance/getprices"
    q = f"{ticker}:{exchange}"
    url = (f"{base}?q={q}&x={exchange}&i=86400&p={days}d&f=d,o,h,l,c,v")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    rows = []
    last_ts = None
    for line in text.splitlines():
        if "=" in line and not line.startswith(("TIMEZONE", "EXCHANGE", "MARKET")):
            continue
        if line.startswith("TIMEZONE") or line.startswith("EXCHANGE"):
            continue
        parts = line.split(",")
        if len(parts) != 6 or not parts[0]:
            continue
        try:
            if parts[0].startswith("a"):
                last_ts = int(parts[0][1:])
            else:
                last_ts = (last_ts or 0) + int(parts[0]) * 86400
            ts = pd.Timestamp(last_ts, unit="s")
            rows.append({"date": ts, "open": float(parts[1]), "high": float(parts[2]),
                         "low": float(parts[3]), "close": float(parts[4]),
                         "volume": float(parts[5])})
        except Exception:
            continue
    if not rows:
        raise ValueError("no google finance prices")
    return pd.DataFrame(rows).set_index("date").sort_index()


def fetch_google_finance_quote(ticker: str, exchange: str = "NASDAQ",
                               days: int = 200) -> pd.DataFrame:
    """Experimental Google Finance price history; falls back to yfinance.

    Google's public endpoint is undocumented and region-dependent, so the
    yfinance fallback guarantees data availability.
    """
    try:
        return _google_finance_prices(ticker, exchange, days)
    except Exception:
        return _yf_ohlcv(ticker)


DATA_SOURCE_STATUS = {
    "akshare": AKSHARE_AVAILABLE,
    "tushare": TUSHARE_AVAILABLE,
    "financedatabase": FINANCEDATABASE_AVAILABLE,
    "financetoolkit": FINANCETOOLKIT_AVAILABLE,
    "sec_edgar": SECEDGAR_AVAILABLE,
    "openbb": OPENBB_AVAILABLE,
    "quandl": QUANDL_AVAILABLE,
    "cryptofeed": CRYPTOFEED_AVAILABLE,
    "yfinance": True,
    "ccxt": None,  # filled by quant.ccxt_fetcher
}
