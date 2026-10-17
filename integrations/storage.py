"""Storage adapters: DuckDB SQL analytics, Polars fast frames, ArcticDB snapshots.

Provides a uniform persistence layer used by the cloud snapshot pipeline:

- Parquet-based local cache (always available, stdlib + pandas)
- DuckDB (82) for SQL analytics over cached bars
- Polars (84) for fast dataframe operations
- ArcticDB (71) versioned tick/bar storage when installed
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DUCKDB_AVAILABLE", "POLARS_AVAILABLE", "ARCTICDB_AVAILABLE",
    "STORAGE_STATUS", "save_bars", "load_bars", "sql_query", "to_polars",
    "arctic_write", "arctic_read", "export_json_snapshot",
]

try:
    import duckdb as _duckdb
    DUCKDB_AVAILABLE = True
except Exception:
    DUCKDB_AVAILABLE = False

try:
    import polars as _pl
    POLARS_AVAILABLE = True
except Exception:
    POLARS_AVAILABLE = False

try:
    import arcticdb as _adbc
    ARCTICDB_AVAILABLE = True
except Exception:
    ARCTICDB_AVAILABLE = False


STORAGE_STATUS = {
    "parquet": True,
    "duckdb": DUCKDB_AVAILABLE,
    "polars": POLARS_AVAILABLE,
    "arcticdb": ARCTICDB_AVAILABLE,
}


def save_bars(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist an OHLCV frame; Parquet when an engine exists, else pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    try:
        out.to_parquet(path)
        return path
    except Exception:
        fallback = path.with_suffix(".pkl")
        out.to_pickle(fallback)
        return fallback


def load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet" and not path.exists():
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            return pd.read_pickle(pkl)
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def sql_query(sql: str, frames: dict[str, pd.DataFrame] | None = None,
              db_path: str = ":memory:", bars_dir: str | Path | None = None) -> pd.DataFrame:
    """Run SQL via DuckDB (82); falls back to pandas query error result."""
    if not DUCKDB_AVAILABLE:
        return pd.DataFrame({"error": ["duckdb not installed"]})
    con = _duckdb.connect(db_path)
    for name, frame in (frames or {}).items():
        con.register(name, frame)
    if bars_dir is not None:
        parquet_glob = str(Path(bars_dir) / "**" / "*.parquet")
        con.execute(
            f"CREATE OR REPLACE VIEW bars AS SELECT * FROM read_parquet('{parquet_glob}')"
        )
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def to_polars(df: pd.DataFrame):
    """Convert pandas -> polars (84); returns pandas if unavailable."""
    if not POLARS_AVAILABLE:
        return df
    return _pl.from_pandas(df)


def arctic_read(library: str, symbol: str, arctic_uri: str = "lmdb://data/arctic") -> pd.DataFrame:
    if not ARCTICDB_AVAILABLE:
        return pd.DataFrame()
    try:
        store = _adbc.Arctic(arctic_uri)
        lib = store.get_library(library, create_if_missing=True)
        return lib.read(symbol).data
    except Exception:
        return pd.DataFrame()


def arctic_write(df: pd.DataFrame, library: str, symbol: str,
                 arctic_uri: str = "lmdb://data/arctic") -> bool:
    if not ARCTICDB_AVAILABLE:
        return False
    try:
        store = _adbc.Arctic(arctic_uri)
        lib = store.get_library(library, create_if_missing=True)
        lib.write(symbol, df)
        return True
    except Exception:
        return False


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def export_json_snapshot(data: dict, path: str | Path) -> Path:
    """Write a dashboard-friendly JSON snapshot (used by GitHub Actions cloud pipeline)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_safe)
    return path
