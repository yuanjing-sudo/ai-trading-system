import numpy as np
import pandas as pd
from typing import Dict, List


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    rename = {"adj_close": "close", "adj close": "close"}
    df = df.rename(columns=rename)
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            df[col] = np.nan
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df = df[needed].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    for col in needed:
        if not isinstance(df[col], pd.Series):
            df[col] = pd.Series(np.asarray(df[col], dtype=float), index=df.index)
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def annualize_return(ret: float, periods: int, periods_per_year: int = 252) -> float:
    return (1 + ret) ** (periods_per_year / periods) - 1


def safe_div(a, b, default=np.nan):
    if b == 0 or b is None or np.isnan(b):
        return default
    return a / b
