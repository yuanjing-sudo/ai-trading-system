"""CCXT Crypto Exchange Data Fetcher.

Provides real-time and historical cryptocurrency market data from 100+
crypto exchanges via the CCXT library. Falls back to yfinance if CCXT
is not installed or exchange data is unavailable.

Supported exchanges include: Binance, Coinbase, Kraken, KuCoin, OKX,
Bybit, Bitfinex, Huobi, Bitstamp, Gemini, and many more.

Reference: https://github.com/ccxt/ccxt
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Check if ccxt is available
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

from quant.pandasta_wrapper import PANDAS_TA_AVAILABLE

__all__ = [
    "CCXT_AVAILABLE",
    "get_exchanges",
    "get_markets",
    "fetch_ohlcv",
    "fetch_ticker",
    "fetch_order_book",
    "fetch_trades",
    "fetch_balance",
    "CryptoDataFetcher",
    "resolve_symbol",
]


# ---------------------------------------------------------------------------
# Exchange discovery
# ---------------------------------------------------------------------------

def get_exchanges() -> list[str]:
    """Return list of all CCXT-supported exchange IDs."""
    if CCXT_AVAILABLE:
        return ccxt.exchanges
    return []


def get_markets(exchange_id: str = "binance", api_key: str | None = None, secret: str | None = None) -> dict:
    """Return available market symbols for an exchange."""
    if not CCXT_AVAILABLE:
        return {}
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"apiKey": api_key, "secret": secret})
        markets = exchange.load_markets()
        return markets
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Data fetching functions
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    timeframe: str = "1d",
    since: str | int | None = None,
    limit: int = 500,
    api_key: str | None = None,
    secret: str | None = None,
    sandbox: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data from a crypto exchange.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTC/USDT", "ETH/USDT").
    exchange_id : str
        CCXT exchange ID (e.g., "binance", "coinbase", "kraken").
    timeframe : str
        Candle timeframe (e.g., "1m", "5m", "1h", "4h", "1d", "1w").
    since : str or int, optional
        Start timestamp (ISO string or milliseconds).
    limit : int
        Maximum number of candles to fetch.
    api_key, secret : str, optional
        Exchange API credentials for authenticated endpoints.
    sandbox : bool
        Use sandbox/testnet mode if supported.

    Returns
    -------
    DataFrame
        OHLCV data with columns: open, high, low, close, volume.
        Index is datetime.
    """
    if not CCXT_AVAILABLE:
        return _fetch_yfinance_fallback(symbol, timeframe, since, limit)

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "sandbox": sandbox,
        })

        # Convert since to milliseconds if string
        since_ms = None
        if since is not None:
            if isinstance(since, str):
                since_ms = exchange.parse8601(since)
            else:
                since_ms = int(since)

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)

        if not ohlcv:
            return _fetch_yfinance_fallback(symbol, timeframe, since, limit)

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype("float64")
        return df

    except Exception:
        return _fetch_yfinance_fallback(symbol, timeframe, since, limit)


def fetch_ticker(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    api_key: str | None = None,
    secret: str | None = None,
) -> dict:
    """Fetch current ticker (price, bid, ask, volume) for a symbol."""
    if not CCXT_AVAILABLE:
        return _fetch_yfinance_ticker(symbol)

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
        ticker = exchange.fetch_ticker(symbol)
        return {
            "symbol": ticker.get("symbol", symbol),
            "timestamp": ticker.get("timestamp"),
            "datetime": ticker.get("datetime"),
            "high": ticker.get("high"),
            "low": ticker.get("low"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "vwap": ticker.get("vwap"),
            "open": ticker.get("open"),
            "close": ticker.get("close"),
            "last": ticker.get("last"),
            "previousClose": ticker.get("previousClose"),
            "change": ticker.get("change"),
            "percentage": ticker.get("percentage"),
            "average": ticker.get("average"),
            "baseVolume": ticker.get("baseVolume"),
            "quoteVolume": ticker.get("quoteVolume"),
            "info": ticker.get("info"),
        }
    except Exception:
        return _fetch_yfinance_ticker(symbol)


def fetch_order_book(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    limit: int = 20,
    api_key: str | None = None,
    secret: str | None = None,
) -> dict:
    """Fetch order book (bids and asks) for a symbol."""
    if not CCXT_AVAILABLE:
        return {}

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        return {
            "symbol": order_book.get("symbol", symbol),
            "timestamp": order_book.get("timestamp"),
            "datetime": order_book.get("datetime"),
            "bids": order_book.get("bids", []),
            "asks": order_book.get("asks", []),
            "nonce": order_book.get("nonce"),
        }
    except Exception:
        return {}


def fetch_trades(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    since: str | int | None = None,
    limit: int = 100,
    api_key: str | None = None,
    secret: str | None = None,
) -> list[dict]:
    """Fetch recent trades for a symbol."""
    if not CCXT_AVAILABLE:
        return []

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"apiKey": api_key, "secret": secret, "enableRateLimit": True})

        since_ms = None
        if since is not None:
            if isinstance(since, str):
                since_ms = exchange.parse8601(since)
            else:
                since_ms = int(since)

        trades = exchange.fetch_trades(symbol, since=since_ms, limit=limit)
        return trades
    except Exception:
        return []


def fetch_balance(
    exchange_id: str = "binance",
    api_key: str | None = None,
    secret: str | None = None,
    password: str | None = None,
) -> dict:
    """Fetch account balance (requires API credentials)."""
    if not CCXT_AVAILABLE:
        return {}

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "password": password,
            "enableRateLimit": True,
        })
        balance = exchange.fetch_balance()
        return {
            "free": balance.get("free", {}),
            "used": balance.get("used", {}),
            "total": balance.get("total", {}),
            "timestamp": balance.get("timestamp"),
            "datetime": balance.get("datetime"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------

def resolve_symbol(symbol: str, exchange_id: str = "binance") -> str:
    """Resolve a symbol to exchange-specific format.

    Converts common formats like "BTC-USD" or "BTCUSDT" to "BTC/USDT".
    """
    symbol = symbol.upper().strip()

    # Already in CCXT format
    if "/" in symbol:
        return symbol

    # Common crypto quote currencies
    quote_currencies = ["USDT", "USD", "USDC", "BUSD", "BTC", "ETH", "EUR", "GBP", "JPY"]

    for quote in quote_currencies:
        if symbol.endswith(quote) and len(symbol) > len(quote):
            base = symbol[: -len(quote)].rstrip("-/_ ")
            if base:
                return f"{base}/{quote}"

    # Default: assume USDT quote
    return f"{symbol}/USDT"


# ---------------------------------------------------------------------------
# Fallback to yfinance
# ---------------------------------------------------------------------------

def _fetch_yfinance_fallback(
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    since: str | int | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """Fallback to yfinance when CCXT is unavailable."""
    import yfinance as yf

    # Convert symbol format
    yf_symbol = symbol.replace("/", "-").replace("--", "-")
    if yf_symbol.endswith("-USDT"):
        yf_symbol = yf_symbol[: -len("-USDT")] + "-USD"
    if not yf_symbol.endswith("-USD"):
        yf_symbol = yf_symbol + "-USD"

    # Map timeframe to yfinance interval
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo",
    }
    interval = interval_map.get(timeframe, "1d")

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="max", interval=interval)
        if df.empty:
            return pd.DataFrame()

        # Limit rows
        if limit and len(df) > limit:
            df = df.tail(limit)

        # Rename columns to match CCXT format
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


def _fetch_yfinance_ticker(symbol: str = "BTC/USDT") -> dict:
    """Fallback ticker from yfinance."""
    import yfinance as yf

    yf_symbol = symbol.replace("/", "-").replace("--", "-")
    if yf_symbol.endswith("-USDT"):
        yf_symbol = yf_symbol[: -len("-USDT")] + "-USD"
    if not yf_symbol.endswith("-USD"):
        yf_symbol = yf_symbol + "-USD"

    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        hist = ticker.history(period="1d")
        if hist.empty:
            return {}

        last = hist.iloc[-1]
        return {
            "symbol": symbol,
            "timestamp": int(hist.index[-1].timestamp() * 1000),
            "datetime": str(hist.index[-1]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "bid": float(info.get("bid", last["Close"])),
            "ask": float(info.get("ask", last["Close"])),
            "vwap": float(last["Close"]),
            "open": float(last["Open"]),
            "close": float(last["Close"]),
            "last": float(last["Close"]),
            "previousClose": float(info.get("previousClose", last["Close"])),
            "change": float(last["Close"] - info.get("previousClose", last["Close"])),
            "percentage": float(
                (last["Close"] - info.get("previousClose", last["Close"]))
                / info.get("previousClose", last["Close"]) * 100
                if info.get("previousClose") else 0
            ),
            "average": float((last["High"] + last["Low"]) / 2),
            "baseVolume": float(last["Volume"]),
            "quoteVolume": float(last["Volume"] * last["Close"]),
            "info": info,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# High-level fetcher class
# ---------------------------------------------------------------------------

class CryptoDataFetcher:
    """High-level crypto data fetcher with exchange failover.

    Automatically tries multiple exchanges if the primary fails.
    """

    def __init__(
        self,
        primary_exchange: str = "binance",
        fallback_exchanges: list[str] | None = None,
        api_key: str | None = None,
        secret: str | None = None,
        sandbox: bool = False,
    ):
        self.primary_exchange = primary_exchange
        self.fallback_exchanges = fallback_exchanges or ["coinbase", "kraken", "kucoin", "okx"]
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox

    def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        since: str | int | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV with automatic exchange failover."""
        symbol = resolve_symbol(symbol, self.primary_exchange)

        exchanges_to_try = [self.primary_exchange] + self.fallback_exchanges
        last_error = None

        for exchange_id in exchanges_to_try:
            try:
                df = fetch_ohlcv(
                    symbol=symbol,
                    exchange_id=exchange_id,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                    api_key=self.api_key,
                    secret=self.secret,
                    sandbox=self.sandbox,
                )
                if not df.empty:
                    return df
            except Exception as e:
                last_error = e
                continue

        # All exchanges failed, try yfinance
        return _fetch_yfinance_fallback(symbol, timeframe, since, limit)

    def fetch_ticker(
        self,
        symbol: str = "BTC/USDT",
    ) -> dict:
        """Fetch ticker with automatic exchange failover."""
        symbol = resolve_symbol(symbol, self.primary_exchange)

        exchanges_to_try = [self.primary_exchange] + self.fallback_exchanges
        for exchange_id in exchanges_to_try:
            try:
                ticker = fetch_ticker(
                    symbol=symbol,
                    exchange_id=exchange_id,
                    api_key=self.api_key,
                    secret=self.secret,
                )
                if ticker:
                    return ticker
            except Exception:
                continue

        return _fetch_yfinance_ticker(symbol)

    def get_available_symbols(self, exchange_id: str | None = None) -> list[str]:
        """Get list of available trading symbols for an exchange."""
        exchange_id = exchange_id or self.primary_exchange
        markets = get_markets(exchange_id, self.api_key, self.secret)
        return list(markets.keys()) if markets else []
