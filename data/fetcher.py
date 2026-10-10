from __future__ import annotations
import os
import time
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logger import setup_logger
from utils.helpers import clean_ohlc

logger = setup_logger("DataFetcher")


class MarketDataFetcher:
    def __init__(self, cache_dir: str = "./cache", source: str = "yfinance"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.source = source

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}_{h}.pkl"

    def _cache_valid(self, path: Path, max_age_hours: int = 12) -> bool:
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < max_age_hours * 3600

    def _load_cache(self, path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_cache(self, path: Path, obj):
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        key = f"{'-'.join(sorted(symbols))}_{start}_{end}_{interval}"
        cpath = self._cache_path(key)
        if not force_refresh and self._cache_valid(cpath):
            logger.info(f"Loading cached data for {len(symbols)} symbols")
            return self._load_cache(cpath)

        logger.info(f"Fetching {len(symbols)} symbols from {self.source}")
        if self.source == "yfinance":
            data = self._fetch_yfinance(symbols, start, end, interval)
        elif self.source == "alpha_vantage":
            data = self._fetch_alpha_vantage(symbols, start, end)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

        self._save_cache(cpath, data)
        return data

    def _fetch_yfinance(self, symbols, start, end, interval) -> Dict[str, pd.DataFrame]:
        import yfinance as yf
        out: Dict[str, pd.DataFrame] = {}
        try:
            raw = yf.download(
                symbols, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, group_by="ticker",
                threads=True,
            )
        except Exception as e:
            logger.error(f"yfinance batch failed: {e}")
            raw = None

        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                for sym in symbols:
                    try:
                        df = raw[sym].dropna(how="all")
                        if not df.empty:
                            out[sym] = clean_ohlc(df)
                    except Exception:
                        continue
            else:
                out[symbols[0]] = clean_ohlc(raw)

        if not out:
            for sym in symbols:
                try:
                    df = yf.download(sym, start=start, end=end, interval=interval,
                                     auto_adjust=True, progress=False)
                    if not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        out[sym] = clean_ohlc(df)
                except Exception as e:
                    logger.warning(f"{sym} failed: {e}")
        logger.info(f"Fetched {len(out)} symbols successfully")
        return out

    def _fetch_alpha_vantage(self, symbols, start, end):
        raise NotImplementedError("Alpha Vantage adapter requires API key setup")

    def fetch_benchmark(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        d = self.fetch([ticker], start, end, force_refresh=False)
        return d.get(ticker, pd.DataFrame())
