from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd


def detect_breakout(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 1:
        return False
    hh = df["high"].rolling(lookback).max().shift(1).iloc[-1]
    return df["close"].iloc[-1] > hh


def detect_breakdown(df: pd.DataFrame, lookback: int = 20) -> bool:
    if len(df) < lookback + 1:
        return False
    ll = df["low"].rolling(lookback).min().shift(1).iloc[-1]
    return df["close"].iloc[-1] < ll


def detect_bull_engulf(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    p, c = df.iloc[-2], df.iloc[-1]
    return (p["close"] < p["open"] and c["close"] > c["open"]
            and c["close"] >= p["open"] and c["open"] <= p["close"])


def detect_bear_engulf(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    p, c = df.iloc[-2], df.iloc[-1]
    return (p["close"] > p["open"] and c["close"] < c["open"]
            and c["open"] >= p["close"] and c["close"] <= p["open"])


def detect_golden_cross(df: pd.DataFrame) -> bool:
    if len(df) < 201:
        return False
    ma50 = df["close"].rolling(50).mean()
    ma200 = df["close"].rolling(200).mean()
    return ma50.iloc[-2] < ma200.iloc[-2] and ma50.iloc[-1] > ma200.iloc[-1]


def detect_death_cross(df: pd.DataFrame) -> bool:
    if len(df) < 201:
        return False
    ma50 = df["close"].rolling(50).mean()
    ma200 = df["close"].rolling(200).mean()
    return ma50.iloc[-2] > ma200.iloc[-2] and ma50.iloc[-1] < ma200.iloc[-1]


def detect_consolidation(df: pd.DataFrame, window: int = 10,
                         max_range: float = 0.03) -> bool:
    if len(df) < window:
        return False
    seg = df.iloc[-window:]
    rng = (seg["high"].max() - seg["low"].min()) / seg["close"].mean()
    return rng < max_range


def pattern_score(df: pd.DataFrame) -> Dict[str, bool]:
    return {
        "breakout": detect_breakout(df),
        "breakdown": detect_breakdown(df),
        "bull_engulf": detect_bull_engulf(df),
        "bear_engulf": detect_bear_engulf(df),
        "golden_cross": detect_golden_cross(df),
        "death_cross": detect_death_cross(df),
        "consolidation": detect_consolidation(df),
    }
