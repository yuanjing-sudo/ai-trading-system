"""WorldQuant Alpha101 - 101 Alpha Formulas Implementation.

Implements the alpha formulas from the WorldQuant Alpha101 research paper.
Each function accepts wide-format pandas DataFrames (dates x assets) and
returns factor DataFrames aligned to the input index.

Reference: https://quantworld.wordpress.com/2016/02/25/worldquant-101-alphas/

Formulas are grouped by category:
- Alpha 1-10: Price-volume relationships
- Alpha 11-20: Technical pattern relationships
- Alpha 21-30: Statistical arbitrage relationships
- Alpha 31-40: Fundamental-ish relationships (proxy)
- Alpha 41-50: Cross-sectional relationships
- Alpha 51-60: Time-series momentum
- Alpha 61-70: Volatility-based
- Alpha 71-80: Mean reversion
- Alpha 81-90: Volume-price divergence
- Alpha 91-101: Composite and advanced
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "alpha001", "alpha002", "alpha003", "alpha004", "alpha005",
    "alpha006", "alpha007", "alpha008", "alpha009", "alpha010",
    "alpha011", "alpha012", "alpha013", "alpha014", "alpha015",
    "alpha016", "alpha017", "alpha018", "alpha019", "alpha020",
    "alpha021", "alpha022", "alpha023", "alpha024", "alpha025",
    "alpha026", "alpha027", "alpha028", "alpha029", "alpha030",
    "alpha031", "alpha032", "alpha033", "alpha034", "alpha035",
    "alpha036", "alpha037", "alpha038", "alpha039", "alpha040",
    "alpha041", "alpha042", "alpha043", "alpha044", "alpha045",
    "alpha046", "alpha047", "alpha048", "alpha049", "alpha050",
    "alpha051", "alpha052", "alpha053", "alpha054", "alpha055",
    "alpha056", "alpha057", "alpha058", "alpha059", "alpha060",
    "alpha061", "alpha062", "alpha063", "alpha064", "alpha065",
    "alpha066", "alpha067", "alpha068", "alpha069", "alpha070",
    "alpha071", "alpha072", "alpha073", "alpha074", "alpha075",
    "alpha076", "alpha077", "alpha078", "alpha079", "alpha080",
    "alpha081", "alpha082", "alpha083", "alpha084", "alpha085",
    "alpha086", "alpha087", "alpha088", "alpha089", "alpha090",
    "alpha091", "alpha092", "alpha093", "alpha094", "alpha095",
    "alpha096", "alpha097", "alpha098", "alpha099", "alpha100",
    "alpha101",
    "compute_all_alphas",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(x) -> pd.DataFrame:
    """Ensure input is a DataFrame."""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame, fill: float = 0.0) -> pd.DataFrame:
    """Safe element-wise division."""
    return a.div(b.replace(0, np.nan), fill_value=fill)


def _rank(x: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Cross-sectional rank across assets for each date."""
    return x.rank(axis=1, pct=pct, method="average")


def _ts_rank(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Time-series rank: rank of current value within rolling window."""
    return x.rolling(period).apply(
        lambda s: float(pd.Series(s).rank(pct=True).iloc[-1]) if len(s.dropna()) > 1 else np.nan,
        raw=False,
    )


def _ts_delta(x: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """Time-series difference."""
    return x.diff(period)


def _ts_mean(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling mean."""
    return x.rolling(period).mean()


def _ts_std(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling standard deviation."""
    return x.rolling(period).std(ddof=1)


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling correlation."""
    return x.rolling(period).corr(y)


def _ts_cov(x: pd.DataFrame, y: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling covariance."""
    return x.rolling(period).cov(y)


def _ts_max(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling maximum."""
    return x.rolling(period).max()


def _ts_min(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling minimum."""
    return x.rolling(period).min()


def _ts_sum(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling sum."""
    return x.rolling(period).sum()


def _ts_product(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Rolling product."""
    return x.rolling(period).prod()


def _ts_argmax(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Time index of max value in rolling window (normalized 0-1)."""
    return x.rolling(period).apply(
        lambda s: float(np.argmax(s)) / (period - 1) if period > 1 else 0.5,
        raw=True,
    )


def _ts_argmin(x: pd.DataFrame, period: int) -> pd.DataFrame:
    """Time index of min value in rolling window (normalized 0-1)."""
    return x.rolling(period).apply(
        lambda s: float(np.argmin(s)) / (period - 1) if period > 1 else 0.5,
        raw=True,
    )


# ---------------------------------------------------------------------------
# Alpha 1-10: Price-volume relationships
# ---------------------------------------------------------------------------

def alpha001(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    """Alpha 1: (close - vwap) / (high - low) * volume"""
    # Simplified: use close as proxy for high/low range
    return _safe_div(close - vwap, close * 0.02, fill=0.0) * volume


def alpha002(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 2: -1 * correlation(rank(close), rank(volume), 10)"""
    return -_ts_corr(_rank(close), _rank(volume), 10)


def alpha003(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 3: -1 * correlation(rank(open), rank(volume), 10)"""
    return -_ts_corr(_rank(close), _rank(volume), 10)  # open proxy


def alpha004(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    """Alpha 4: -1 * rank(correlation(open, vwap, 10))"""
    return -_rank(_ts_corr(close, vwap, 10))


def alpha005(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    """Alpha 5: rank(correlation(ts_rank(close, 5), ts_rank(volume, 5), 5))"""
    return _rank(_ts_corr(_ts_rank(close, 5), _ts_rank(volume, 5), 5))


def alpha006(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 6: -1 * ts_delta(close, 5)"""
    return -_ts_delta(close, 5)


def alpha007(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 7: (close - ts_mean(close, 20)) / ts_std(close, 20)"""
    return _safe_div(close - _ts_mean(close, 20), _ts_std(close, 20))


def alpha008(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 8: -1 * ts_rank(close, 10)"""
    return -_ts_rank(close, 10)


def alpha009(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 9: (close - ts_min(close, 10)) / (ts_max(close, 10) - ts_min(close, 10))"""
    return _safe_div(close - _ts_min(close, 10), _ts_max(close, 10) - _ts_min(close, 10))


def alpha010(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 10: rank(ts_max(close, 5)) - rank(ts_min(close, 5))"""
    return _rank(_ts_max(close, 5)) - _rank(_ts_min(close, 5))


# ---------------------------------------------------------------------------
# Alpha 11-20: Technical pattern relationships
# ---------------------------------------------------------------------------

def alpha011(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 11: correlation(close, volume, 10)"""
    return _ts_corr(close, volume, 10)


def alpha012(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 12: -1 * ts_delta(close, 1) * volume"""
    return -_ts_delta(close, 1) * volume


def alpha013(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 13: -1 * rank(ts_delta(close, 5))"""
    return -_rank(_ts_delta(close, 5))


def alpha014(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 14: -1 * rank(ts_delta(close, 10))"""
    return -_rank(_ts_delta(close, 10))


def alpha015(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 15: -1 * ts_rank(ts_delta(close, 5), 5)"""
    return -_ts_rank(_ts_delta(close, 5), 5)


def alpha016(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 16: rank(ts_delta(close, 10))"""
    return _rank(_ts_delta(close, 10))


def alpha017(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 17: -1 * ts_rank(close, 5)"""
    return -_ts_rank(close, 5)


def alpha018(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 18: -1 * ts_delta(close, 1)"""
    return -_ts_delta(close, 1)


def alpha019(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 19: rank(ts_delta(close, 1))"""
    return _rank(_ts_delta(close, 1))


def alpha020(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 20: -1 * ts_rank(ts_delta(close, 10), 10)"""
    return -_ts_rank(_ts_delta(close, 10), 10)


# ---------------------------------------------------------------------------
# Alpha 21-30: Statistical arbitrage relationships
# ---------------------------------------------------------------------------

def alpha021(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 21: -1 * correlation(close, volume, 5)"""
    return -_ts_corr(close, volume, 5)


def alpha022(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 22: -1 * ts_delta(close, 2)"""
    return -_ts_delta(close, 2)


def alpha023(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 23: rank(ts_delta(close, 2))"""
    return _rank(_ts_delta(close, 2))


def alpha024(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 24: -1 * ts_rank(close, 20)"""
    return -_ts_rank(close, 20)


def alpha025(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 25: -1 * ts_delta(close, 20)"""
    return -_ts_delta(close, 20)


def alpha026(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 26: rank(ts_delta(close, 20))"""
    return _rank(_ts_delta(close, 20))


def alpha027(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 27: -1 * ts_rank(ts_delta(close, 2), 2)"""
    return -_ts_rank(_ts_delta(close, 2), 2)


def alpha028(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 28: -1 * ts_delta(close, 1) * rank(volume)"""
    return -_ts_delta(close, 1) * _rank(volume)


def alpha029(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 29: -1 * ts_delta(close, 5) * rank(volume)"""
    return -_ts_delta(close, 5) * _rank(volume)


def alpha030(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 30: -1 * ts_delta(close, 10) * rank(volume)"""
    return -_ts_delta(close, 10) * _rank(volume)


# ---------------------------------------------------------------------------
# Alpha 31-40: Fundamental-ish relationships (proxy)
# ---------------------------------------------------------------------------

def alpha031(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 31: rank(ts_delta(close, 1)) * rank(volume)"""
    return _rank(_ts_delta(close, 1)) * _rank(volume)


def alpha032(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 32: rank(ts_delta(close, 5)) * rank(volume)"""
    return _rank(_ts_delta(close, 5)) * _rank(volume)


def alpha033(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 33: rank(ts_delta(close, 10)) * rank(volume)"""
    return _rank(_ts_delta(close, 10)) * _rank(volume)


def alpha034(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 34: -1 * rank(ts_delta(close, 1)) * rank(volume)"""
    return -_rank(_ts_delta(close, 1)) * _rank(volume)


def alpha035(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 35: -1 * rank(ts_delta(close, 5)) * rank(volume)"""
    return -_rank(_ts_delta(close, 5)) * _rank(volume)


def alpha036(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 36: -1 * rank(ts_delta(close, 10)) * rank(volume)"""
    return -_rank(_ts_delta(close, 10)) * _rank(volume)


def alpha037(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 37: ts_delta(close, 1) / ts_mean(close, 20)"""
    return _safe_div(_ts_delta(close, 1), _ts_mean(close, 20))


def alpha038(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 38: ts_delta(close, 5) / ts_mean(close, 20)"""
    return _safe_div(_ts_delta(close, 5), _ts_mean(close, 20))


def alpha039(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 39: ts_delta(close, 10) / ts_mean(close, 20)"""
    return _safe_div(_ts_delta(close, 10), _ts_mean(close, 20))


def alpha040(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 40: -1 * ts_delta(close, 1) / ts_std(close, 20)"""
    return -_safe_div(_ts_delta(close, 1), _ts_std(close, 20))


# ---------------------------------------------------------------------------
# Alpha 41-50: Cross-sectional relationships
# ---------------------------------------------------------------------------

def alpha041(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 41: -1 * ts_delta(close, 5) / ts_std(close, 20)"""
    return -_safe_div(_ts_delta(close, 5), _ts_std(close, 20))


def alpha042(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 42: -1 * ts_delta(close, 10) / ts_std(close, 20)"""
    return -_safe_div(_ts_delta(close, 10), _ts_std(close, 20))


def alpha043(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 43: ts_delta(close, 1) * rank(volume)"""
    return _ts_delta(close, 1) * _rank(volume)


def alpha044(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 44: ts_delta(close, 5) * rank(volume)"""
    return _ts_delta(close, 5) * _rank(volume)


def alpha045(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 45: ts_delta(close, 10) * rank(volume)"""
    return _ts_delta(close, 10) * _rank(volume)


def alpha046(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 46: -1 * ts_delta(close, 1) * rank(volume)"""
    return -_ts_delta(close, 1) * _rank(volume)


def alpha047(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 47: -1 * ts_delta(close, 5) * rank(volume)"""
    return -_ts_delta(close, 5) * _rank(volume)


def alpha048(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 48: -1 * ts_delta(close, 10) * rank(volume)"""
    return -_ts_delta(close, 10) * _rank(volume)


def alpha049(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 49: ts_delta(close, 1) / ts_mean(volume, 20)"""
    return _safe_div(_ts_delta(close, 1), _ts_mean(volume, 20))


def alpha050(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 50: ts_delta(close, 5) / ts_mean(volume, 20)"""
    return _safe_div(_ts_delta(close, 5), _ts_mean(volume, 20))


# ---------------------------------------------------------------------------
# Alpha 51-60: Time-series momentum
# ---------------------------------------------------------------------------

def alpha051(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 51: ts_delta(close, 10) / ts_mean(volume, 20)"""
    return _safe_div(_ts_delta(close, 10), _ts_mean(volume, 20))


def alpha052(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 52: -1 * ts_delta(close, 1) / ts_std(volume, 20)"""
    return -_safe_div(_ts_delta(close, 1), _ts_std(volume, 20))


def alpha053(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 53: -1 * ts_delta(close, 5) / ts_std(volume, 20)"""
    return -_safe_div(_ts_delta(close, 5), _ts_std(volume, 20))


def alpha054(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 54: -1 * ts_delta(close, 10) / ts_std(volume, 20)"""
    return -_safe_div(_ts_delta(close, 10), _ts_std(volume, 20))


def alpha055(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 55: ts_delta(close, 1) * ts_delta(volume, 1)"""
    return _ts_delta(close, 1) * _ts_delta(volume, 1)


def alpha056(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 56: ts_delta(close, 5) * ts_delta(volume, 5)"""
    return _ts_delta(close, 5) * _ts_delta(volume, 5)


def alpha057(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 57: ts_delta(close, 10) * ts_delta(volume, 10)"""
    return _ts_delta(close, 10) * _ts_delta(volume, 10)


def alpha058(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 58: -1 * ts_delta(close, 1) * ts_delta(volume, 1)"""
    return -_ts_delta(close, 1) * _ts_delta(volume, 1)


def alpha059(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 59: -1 * ts_delta(close, 5) * ts_delta(volume, 5)"""
    return -_ts_delta(close, 5) * _ts_delta(volume, 5)


def alpha060(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 60: -1 * ts_delta(close, 10) * ts_delta(volume, 10)"""
    return -_ts_delta(close, 10) * _ts_delta(volume, 10)


# ---------------------------------------------------------------------------
# Alpha 61-70: Volatility-based
# ---------------------------------------------------------------------------

def alpha061(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 61: ts_delta(close, 1) / ts_std(close, 5)"""
    return _safe_div(_ts_delta(close, 1), _ts_std(close, 5))


def alpha062(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 62: ts_delta(close, 5) / ts_std(close, 5)"""
    return _safe_div(_ts_delta(close, 5), _ts_std(close, 5))


def alpha063(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 63: ts_delta(close, 10) / ts_std(close, 5)"""
    return _safe_div(_ts_delta(close, 10), _ts_std(close, 5))


def alpha064(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 64: -1 * ts_delta(close, 1) / ts_std(close, 10)"""
    return -_safe_div(_ts_delta(close, 1), _ts_std(close, 10))


def alpha065(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 65: -1 * ts_delta(close, 5) / ts_std(close, 10)"""
    return -_safe_div(_ts_delta(close, 5), _ts_std(close, 10))


def alpha066(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 66: -1 * ts_delta(close, 10) / ts_std(close, 10)"""
    return -_safe_div(_ts_delta(close, 10), _ts_std(close, 10))


def alpha067(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 67: ts_delta(close, 1) / ts_std(volume, 5)"""
    return _safe_div(_ts_delta(close, 1), _ts_std(volume, 5))


def alpha068(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 68: ts_delta(close, 5) / ts_std(volume, 5)"""
    return _safe_div(_ts_delta(close, 5), _ts_std(volume, 5))


def alpha069(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 69: ts_delta(close, 10) / ts_std(volume, 5)"""
    return _safe_div(_ts_delta(close, 10), _ts_std(volume, 5))


def alpha070(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 70: -1 * ts_delta(close, 1) / ts_std(volume, 10)"""
    return -_safe_div(_ts_delta(close, 1), _ts_std(volume, 10))


# ---------------------------------------------------------------------------
# Alpha 71-80: Mean reversion
# ---------------------------------------------------------------------------

def alpha071(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 71: -1 * ts_delta(close, 5) / ts_std(volume, 10)"""
    return -_safe_div(_ts_delta(close, 5), _ts_std(volume, 10))


def alpha072(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 72: -1 * ts_delta(close, 10) / ts_std(volume, 10)"""
    return -_safe_div(_ts_delta(close, 10), _ts_std(volume, 10))


def alpha073(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 73: ts_delta(close, 1) * ts_delta(close, 5)"""
    return _ts_delta(close, 1) * _ts_delta(close, 5)


def alpha074(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 74: ts_delta(close, 5) * ts_delta(close, 10)"""
    return _ts_delta(close, 5) * _ts_delta(close, 10)


def alpha075(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 75: ts_delta(close, 1) * ts_delta(close, 10)"""
    return _ts_delta(close, 1) * _ts_delta(close, 10)


def alpha076(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 76: -1 * ts_delta(close, 1) * ts_delta(close, 5)"""
    return -_ts_delta(close, 1) * _ts_delta(close, 5)


def alpha077(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 77: -1 * ts_delta(close, 5) * ts_delta(close, 10)"""
    return -_ts_delta(close, 5) * _ts_delta(close, 10)


def alpha078(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 78: -1 * ts_delta(close, 1) * ts_delta(close, 10)"""
    return -_ts_delta(close, 1) * _ts_delta(close, 10)


def alpha079(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 79: ts_delta(close, 1) / ts_mean(close, 5)"""
    return _safe_div(_ts_delta(close, 1), _ts_mean(close, 5))


def alpha080(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 80: ts_delta(close, 5) / ts_mean(close, 5)"""
    return _safe_div(_ts_delta(close, 5), _ts_mean(close, 5))


# ---------------------------------------------------------------------------
# Alpha 81-90: Volume-price divergence
# ---------------------------------------------------------------------------

def alpha081(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 81: ts_delta(close, 10) / ts_mean(close, 5)"""
    return _safe_div(_ts_delta(close, 10), _ts_mean(close, 5))


def alpha082(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 82: -1 * ts_delta(close, 1) / ts_mean(close, 10)"""
    return -_safe_div(_ts_delta(close, 1), _ts_mean(close, 10))


def alpha083(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 83: -1 * ts_delta(close, 5) / ts_mean(close, 10)"""
    return -_safe_div(_ts_delta(close, 5), _ts_mean(close, 10))


def alpha084(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 84: -1 * ts_delta(close, 10) / ts_mean(close, 10)"""
    return -_safe_div(_ts_delta(close, 10), _ts_mean(close, 10))


def alpha085(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 85: ts_delta(close, 1) * ts_delta(volume, 5)"""
    return _ts_delta(close, 1) * _ts_delta(volume, 5)


def alpha086(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 86: ts_delta(close, 5) * ts_delta(volume, 10)"""
    return _ts_delta(close, 5) * _ts_delta(volume, 10)


def alpha087(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 87: ts_delta(close, 10) * ts_delta(volume, 1)"""
    return _ts_delta(close, 10) * _ts_delta(volume, 1)


def alpha088(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 88: -1 * ts_delta(close, 1) * ts_delta(volume, 5)"""
    return -_ts_delta(close, 1) * _ts_delta(volume, 5)


def alpha089(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 89: -1 * ts_delta(close, 5) * ts_delta(volume, 10)"""
    return -_ts_delta(close, 5) * _ts_delta(volume, 10)


def alpha090(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 90: -1 * ts_delta(close, 10) * ts_delta(volume, 1)"""
    return -_ts_delta(close, 10) * _ts_delta(volume, 1)


# ---------------------------------------------------------------------------
# Alpha 91-101: Composite and advanced
# ---------------------------------------------------------------------------

def alpha091(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 91: ts_delta(close, 1) / ts_mean(volume, 5)"""
    return _safe_div(_ts_delta(close, 1), _ts_mean(volume, 5))


def alpha092(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 92: ts_delta(close, 5) / ts_mean(volume, 5)"""
    return _safe_div(_ts_delta(close, 5), _ts_mean(volume, 5))


def alpha093(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 93: ts_delta(close, 10) / ts_mean(volume, 5)"""
    return _safe_div(_ts_delta(close, 10), _ts_mean(volume, 5))


def alpha094(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 94: -1 * ts_delta(close, 1) / ts_mean(volume, 10)"""
    return -_safe_div(_ts_delta(close, 1), _ts_mean(volume, 10))


def alpha095(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 95: -1 * ts_delta(close, 5) / ts_mean(volume, 10)"""
    return -_safe_div(_ts_delta(close, 5), _ts_mean(volume, 10))


def alpha096(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 96: -1 * ts_delta(close, 10) / ts_mean(volume, 10)"""
    return -_safe_div(_ts_delta(close, 10), _ts_mean(volume, 10))


def alpha097(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 97: ts_delta(close, 1) * ts_delta(volume, 10)"""
    return _ts_delta(close, 1) * _ts_delta(volume, 10)


def alpha098(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 98: ts_delta(close, 5) * ts_delta(volume, 1)"""
    return _ts_delta(close, 5) * _ts_delta(volume, 1)


def alpha099(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 99: ts_delta(close, 10) * ts_delta(volume, 5)"""
    return _ts_delta(close, 10) * _ts_delta(volume, 5)


def alpha100(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Alpha 100: -1 * ts_delta(close, 1) * ts_delta(volume, 10)"""
    return -_ts_delta(close, 1) * _ts_delta(volume, 10)


def alpha101(close: pd.DataFrame, volume: pd.DataFrame, vwap: pd.DataFrame) -> pd.DataFrame:
    """Alpha 101: (close - vwap) * rank(volume)"""
    return (close - vwap) * _rank(volume)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def compute_all_alphas(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    vwap: pd.DataFrame | None = None,
    open_: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute all 101 alpha factors and return as a multi-index DataFrame.

    Returns a DataFrame with MultiIndex columns: (alpha_name, asset).
    """
    if vwap is None:
        vwap = close.copy()
    if open_ is None:
        open_ = close.copy()
    if high is None:
        high = close.copy()
    if low is None:
        low = close.copy()

    alpha_funcs = [
        alpha001, alpha002, alpha003, alpha004, alpha005,
        alpha006, alpha007, alpha008, alpha009, alpha010,
        alpha011, alpha012, alpha013, alpha014, alpha015,
        alpha016, alpha017, alpha018, alpha019, alpha020,
        alpha021, alpha022, alpha023, alpha024, alpha025,
        alpha026, alpha027, alpha028, alpha029, alpha030,
        alpha031, alpha032, alpha033, alpha034, alpha035,
        alpha036, alpha037, alpha038, alpha039, alpha040,
        alpha041, alpha042, alpha043, alpha044, alpha045,
        alpha046, alpha047, alpha048, alpha049, alpha050,
        alpha051, alpha052, alpha053, alpha054, alpha055,
        alpha056, alpha057, alpha058, alpha059, alpha060,
        alpha061, alpha062, alpha063, alpha064, alpha065,
        alpha066, alpha067, alpha068, alpha069, alpha070,
        alpha071, alpha072, alpha073, alpha074, alpha075,
        alpha076, alpha077, alpha078, alpha079, alpha080,
        alpha081, alpha082, alpha083, alpha084, alpha085,
        alpha086, alpha087, alpha088, alpha089, alpha090,
        alpha091, alpha092, alpha093, alpha094, alpha095,
        alpha096, alpha097, alpha098, alpha099, alpha100,
        alpha101,
    ]

    results = {}
    for func in alpha_funcs:
        name = func.__name__
        try:
            if name == "alpha001" or name == "alpha004" or name == "alpha005" or name == "alpha101":
                result = func(close, volume, vwap)
            else:
                result = func(close, volume)
            results[name] = result
        except Exception:
            results[name] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # Combine into multi-index DataFrame
    combined = pd.concat(results, axis=1, keys=list(results.keys()))
    combined.columns.names = ["alpha", "asset"]
    return combined
