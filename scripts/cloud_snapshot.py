"""Cloud snapshot pipeline - GitHub Actions as free cloud storage.

Fetches daily bars for the configured universe, computes indicators,
agent verdicts and backtest statistics, then writes:

- ``docs/data/dashboard.json``   : summary consumed by the Pages dashboard
- ``docs/data/bars/{ticker}.json``: compact OHLCV history for charting
- ``data/snapshots/{date}.json`` : archival snapshot committed to the repo

Run locally:
    python scripts/cloud_snapshot.py --tickers AAPL MSFT NVDA --days 400
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from integrations.storage import export_json_snapshot

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "SPY", "QQQ",
    "BTC-USD", "ETH-USD", "GLD", "TLT", "XOM", "JPM",
]


def fetch_history(ticker: str, days: int) -> pd.DataFrame:
    try:
        import yfinance as yf
        period = f"{max(days, 30)}d"
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def compact_bars(df: pd.DataFrame, max_points: int = 400) -> dict:
    df = df.tail(max_points)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in df.index],
        "open": [round(float(v), 4) for v in df["open"]],
        "high": [round(float(v), 4) for v in df["high"]],
        "low": [round(float(v), 4) for v in df["low"]],
        "close": [round(float(v), 4) for v in df["close"]],
        "volume": [int(v) for v in df["volume"].fillna(0)],
    }


def technical_summary(df: pd.DataFrame) -> dict:
    close = df["close"]
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else 0.0
    ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 21 else 0.0
    ann_vol = float(close.pct_change().tail(60).std() * np.sqrt(252))
    high_52w = float(close.tail(252).max())
    low_52w = float(close.tail(252).min())
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).dropna().iloc[-1]) if len(close) > 20 else 50.0
    return {
        "price": round(float(close.iloc[-1]), 4),
        "ret_1d": round(ret_1d, 6),
        "ret_20d": round(ret_20d, 6),
        "rsi14": round(rsi, 2),
        "ma20": round(ma20, 4) if ma20 else None,
        "ma50": round(ma50, 4) if ma50 else None,
        "ann_vol": round(ann_vol, 4),
        "high_52w": round(high_52w, 4),
        "low_52w": round(low_52w, 4),
    }


def run_agents(ticker: str, df: pd.DataFrame) -> dict:
    try:
        from integrations.agents import analyze_symbol
        out = analyze_symbol(ticker, df)
        return {
            "decision": out["decision"]["decision"],
            "confidence": round(out["decision"]["confidence"], 3),
            "score": round(out["decision"]["score"], 3),
            "suggested_weight": round(out["decision"]["suggested_weight"], 3),
            "summary": out["decision"]["summary"],
        }
    except Exception as exc:
        return {"decision": "HOLD", "confidence": 0.0, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    ap.add_argument("--days", type=int, default=400)
    args = ap.parse_args()

    docs_data = ROOT / "docs" / "data"
    bars_dir = docs_data / "bars"
    bars_dir.mkdir(parents=True, exist_ok=True)

    today = dt.date.today().isoformat()
    dashboard = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "universe": [],
        "sources": {"yfinance": True},
    }

    for ticker in args.tickers:
        df = fetch_history(ticker, args.days)
        if df.empty:
            dashboard["universe"].append({"ticker": ticker, "error": "no data"})
            continue
        summary = technical_summary(df)
        summary["ticker"] = ticker
        summary["agents"] = run_agents(ticker, df)
        dashboard["universe"].append(summary)
        export_json_snapshot(compact_bars(df), bars_dir / f"{ticker.replace('/', '_')}.json")
        print(f"[ok] {ticker}: {summary['price']} ({summary['agents']['decision']})")

    export_json_snapshot(dashboard, docs_data / "dashboard.json")
    export_json_snapshot(dashboard, ROOT / "data" / "snapshots" / f"{today}.json")
    print(f"dashboard.json written with {len(dashboard['universe'])} tickers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
