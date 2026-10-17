from __future__ import annotations

import os
import sys
import json
import time
import math
import uuid
import pickle
import hashlib
import logging
import warnings
import functools
import itertools
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional,
    Sequence, Set, Tuple, Union,
)
from collections import defaultdict, deque, OrderedDict, Counter
from pathlib import Path
from enum import Enum, IntEnum, auto

import numpy as np
import pandas as pd
from scipy import stats, signal as sp_signal, optimize as sp_optimize
from scipy.stats import norm, t as student_t, skew, kurtosis, jarque_bera
from scipy.special import expit, logit, erf

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest, AdaBoostClassifier,
)
from sklearn.linear_model import (
    Ridge, LogisticRegression, SGDClassifier, HuberRegressor,
    BayesianRidge, ElasticNet, Lasso, QuantileRegressor,
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
    f1_score, precision_score, recall_score,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger("crypto_predictor")


def _configure_logging(level: str = "INFO") -> None:
    if LOG.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    LOG.addHandler(h)
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))


CRYPTO_HORIZONS_HOURS: Tuple[int, ...] = (1, 4, 12, 24, 72, 168, 720)
HOURS_PER_YEAR = 24 * 365
MINUTES_PER_YEAR = 60 * 24 * 365
SECONDS_PER_YEAR = 60 * 60 * 24 * 365
DAYS_PER_YEAR_CRYPTO = 365

STABLECOIN_QUOTES: Set[str] = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP",
                               "FDUSD", "PYUSD", "USDD", "GUSD"}

CRYPTO_SECTORS: Dict[str, str] = {
    "BTC": "store_of_value", "WBTC": "store_of_value", "BCH": "store_of_value",
    "LTC": "store_of_value", "XMR": "privacy", "ZEC": "privacy",
    "ETH": "smart_contract", "SOL": "smart_contract", "AVAX": "smart_contract",
    "MATIC": "smart_contract", "DOT": "smart_contract", "ATOM": "smart_contract",
    "NEAR": "smart_contract", "ALGO": "smart_contract", "ADA": "smart_contract",
    "BNB": "exchange", "OKB": "exchange", "CRO": "exchange", "FTT": "exchange",
    "UNI": "defi", "SUSHI": "defi", "CRV": "defi", "AAVE": "defi",
    "COMP": "defi", "MKR": "defi", "SNX": "defi", "YFI": "defi",
    "LDO": "defi", "GMX": "defi", "DYDX": "defi",
    "LINK": "oracle", "BAND": "oracle", "API3": "oracle", "PYTH": "oracle",
    "DOGE": "meme", "SHIB": "meme", "PEPE": "meme", "WIF": "meme", "BONK": "meme",
    "XRP": "payments", "XLM": "payments", "ALGO_PAY": "payments",
    "ARB": "l2", "OP": "l2", "MATIC_L2": "l2", "STRK": "l2", "ZK": "l2",
    "IMX": "gaming", "GALA": "gaming", "SAND": "gaming", "MANA": "gaming",
    "AXS": "gaming", "APE": "gaming", "ENJ": "gaming",
    "FIL": "storage", "AR": "storage", "STORJ": "storage",
    "GRT": "infra", "RNDR": "infra", "FIL_INFRA": "infra",
    "FET": "ai", "AGIX": "ai", "OCEAN": "ai", "TAO": "ai", "RNDR_AI": "ai",
}

EPS = 1e-12


class CryptoPredictionTarget(Enum):
    DIRECTION = "direction"
    RETURN = "return"
    EXCESS_RETURN = "excess"
    TRIPLE_BARRIER = "triple"
    RANK = "rank"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"


class CryptoRegime(IntEnum):
    EXTREME_BEAR = 0
    BEAR = 1
    CHOP = 2
    BULL = 3
    EXTREME_BULL = 4
    CRASH = 5
    MELT_UP = 6
    DECOUPLED = 7


REGIME_NAMES_CRYPTO: Dict[int, str] = {
    0: "extreme_bear", 1: "bear", 2: "chop", 3: "bull",
    4: "extreme_bull", 5: "crash", 6: "melt_up", 7: "decoupled",
}


@dataclass
class CryptoPredictorConfig:
    horizons_hours: Tuple[int, ...] = CRYPTO_HORIZONS_HOURS
    primary_horizon_hours: int = 24
    target_type: CryptoPredictionTarget = CryptoPredictionTarget.DIRECTION
    return_threshold: float = 0.0
    benchmark_symbol: str = "BTC"

    timestamp_column: str = "timestamp"
    resample_frequency: Optional[str] = None
    timezone: str = "UTC"

    ma_windows: Tuple[int, ...] = (6, 12, 24, 48, 120, 240, 720)
    vol_windows: Tuple[int, ...] = (6, 24, 72, 168, 720)
    rsi_windows: Tuple[int, ...] = (6, 14, 24)
    atr_windows: Tuple[int, ...] = (14, 24, 72)
    macd_params: Tuple[int, int, int] = (12, 26, 9)
    bb_window: int = 20
    bb_std: float = 2.0
    use_fractional_diff: bool = True
    fractional_d: float = 0.4
    use_kalman: bool = True
    kalman_process_var: float = 2e-4
    kalman_obs_var: float = 2e-3
    use_cross_sectional: bool = True
    use_regime_features: bool = True
    use_pattern_features: bool = True
    use_volume_features: bool = True
    use_entropy_features: bool = True
    use_hurst: bool = True
    use_microstructure: bool = True
    use_onchain_proxies: bool = True
    use_funding_proxy: bool = True
    use_jump_features: bool = True
    use_leadlag: bool = True
    use_altcoin_beta: bool = True
    use_vol_of_vol: bool = True
    use_intraday_seasonality: bool = True

    winsorize_quantile: float = 0.005
    feature_clip_sigma: float = 6.0
    min_history: int = 500

    n_splits: int = 6
    embargo_hours: int = 24
    purge_hours: int = 24
    train_window_hours: int = 4000
    random_state: int = 42
    n_jobs: int = -1

    xgb_n_estimators: int = 1500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.02
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.85
    xgb_min_child_weight: int = 4
    xgb_gamma: float = 0.1
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.5
    xgb_early_stop: int = 100

    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgb": 0.45, "extratrees": 0.20, "histgbm": 0.20, "ridge_meta": 0.15,
    })

    n_bootstrap: int = 40
    conformal_alpha: float = 0.10
    quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

    online_enabled: bool = True
    online_alpha: float = 5e-5
    online_learning_rate: str = "optimal"

    feature_selection_enabled: bool = True
    max_features: int = 200
    min_mutual_info: float = 1e-4

    drift_psi_threshold: float = 0.25
    log_level: str = "INFO"

    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"

    funding_rate_aware: bool = True
    jump_threshold_sigma: float = 5.0
    crash_detector_threshold: float = -0.15


def safe_div(a, b, fill: float = 0.0):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b_arr) > EPS, a_arr / b_arr, fill)
    if isinstance(a, pd.Series):
        return pd.Series(out, index=a.index)
    return out if out.shape else float(out)


def robust_zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    q1 = np.nanpercentile(x, 25, axis=axis, keepdims=True)
    q3 = np.nanpercentile(x, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    scale = np.where(iqr > EPS, iqr, 1.0)
    return (x - med) / scale


def cross_section_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=ascending) - 0.5


def ewma_volatility(returns: pd.Series, halflife: int = 20,
                    annualize: bool = True, hours: bool = True) -> pd.Series:
    vol = returns.ewm(halflife=halflife, adjust=False).std()
    if annualize:
        factor = np.sqrt(HOURS_PER_YEAR if hours else DAYS_PER_YEAR_CRYPTO)
        vol = vol * factor
    return vol


def rolling_entropy(series: pd.Series, window: int = 24, bins: int = 12) -> pd.Series:
    def ent(x):
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return np.nan
        h, _ = np.histogram(x, bins=bins)
        p = h / (h.sum() + EPS)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + EPS)))
    return series.rolling(window).apply(ent, raw=True)


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < max_lag + 2:
        return np.nan
    lags = np.arange(2, max_lag)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.array(tau, dtype=float)
    valid = tau > 0
    if valid.sum() < 3:
        return np.nan
    try:
        slope = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0]
        return float(slope)
    except Exception:
        return np.nan


def fractionally_differentiate(series: pd.Series, d: float = 0.4,
                               threshold: float = 1e-4) -> pd.Series:
    x = series.values.astype(float)
    n = len(x)
    weights = [1.0]
    k = 1
    while k < n:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    width = len(weights)
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        window = x[i - width + 1: i + 1][::-1]
        out[i] = float(np.dot(weights, window))
    return pd.Series(out, index=series.index)


def psi(actual: np.ndarray, expected: np.ndarray, buckets: int = 10) -> float:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    mask_a = ~np.isnan(actual)
    mask_e = ~np.isnan(expected)
    if mask_a.sum() < 10 or mask_e.sum() < 10:
        return 0.0
    a, e = actual[mask_a], expected[mask_e]
    edges = np.percentile(e, np.linspace(0, 100, buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    a_counts, _ = np.histogram(a, bins=edges)
    e_counts, _ = np.histogram(e, bins=edges)
    a_pct = a_counts / (a_counts.sum() + EPS)
    e_pct = e_counts / (e_counts.sum() + EPS)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def normalize_crypto_symbol(symbol: str) -> str:
    s = symbol.upper().replace("-", "").replace("/", "").replace("_", "")
    for quote in sorted(STABLECOIN_QUOTES, key=len, reverse=True):
        if s.endswith(quote) and len(s) > len(quote):
            return s[:-len(quote)]
    for quote in ("USD", "EUR", "GBP", "JPY", "BTC", "ETH", "BNB"):
        if s.endswith(quote) and len(s) > len(quote):
            return s[:-len(quote)]
    return s


def crypto_sector(symbol: str) -> str:
    base = normalize_crypto_symbol(symbol)
    return CRYPTO_SECTORS.get(base, "other")


def infer_bar_hours(index: pd.DatetimeIndex) -> float:
    if len(index) < 3:
        return 1.0
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    med = np.median(deltas) / 3600.0
    return float(max(med, 1e-3))


def jarque_bera_pvalue(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return np.nan
    try:
        _, p = jarque_bera(x)
        return float(p)
    except Exception:
        return np.nan


def rolling_skew(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=max(5, w // 2)).skew()


def rolling_kurt(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=max(5, w // 2)).kurt()


def timed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        r = func(*args, **kwargs)
        LOG.debug(f"{func.__qualname__} took {(time.perf_counter()-t0)*1000:.2f} ms")
        return r
    return wrapper


class TechnicalIndicators:
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(2, window // 2)).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()

    @staticmethod
    def wma(series: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        up_ema = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        dn_ema = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        rs = safe_div(up_ema, dn_ema, fill=np.nan)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ef = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        es = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        line = ef - es
        sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = line - sig
        return line, sig, hist

    @staticmethod
    def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
        mid = series.rolling(window, min_periods=window).mean()
        sd = series.rolling(window, min_periods=window).std()
        return mid, mid + n_std * sd, mid - n_std * sd

    @staticmethod
    def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    @staticmethod
    def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        up = high.diff()
        down = -low.diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_v = tr.ewm(alpha=1 / window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_v
        minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_v
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
        return dx.ewm(alpha=1 / window, adjust=False).mean()

    @staticmethod
    def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3, smooth: int = 3):
        low_k = df["low"].rolling(k, min_periods=k).min()
        high_k = df["high"].rolling(k, min_periods=k).max()
        k_pct = 100 * (df["close"] - low_k) / (high_k - low_k + EPS)
        k_sm = k_pct.rolling(smooth).mean()
        return k_sm, k_sm.rolling(d).mean()

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        sign = np.sign(df["close"].diff().fillna(0.0))
        return (sign * df["volume"]).cumsum()

    @staticmethod
    def mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]
        diff = tp.diff()
        pos = mf.where(diff > 0, 0)
        neg = mf.where(diff < 0, 0)
        pos_sum = pos.rolling(window, min_periods=window).sum()
        neg_sum = neg.rolling(window, min_periods=window).sum()
        mfr = safe_div(pos_sum, neg_sum, fill=np.nan)
        return 100 - 100 / (1 + mfr)

    @staticmethod
    def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        ma = tp.rolling(window, min_periods=window).mean()
        md = tp.rolling(window, min_periods=window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - ma) / (0.015 * md + EPS)

    @staticmethod
    def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        hh = df["high"].rolling(window, min_periods=window).max()
        ll = df["low"].rolling(window, min_periods=window).min()
        return -100 * (hh - df["close"]) / (hh - ll + EPS)

    @staticmethod
    def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
        hl = (df["high"] - df["low"]) + EPS
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
        mfv = mfm * df["volume"]
        v = df["volume"].rolling(window, min_periods=window).sum()
        return mfv.rolling(window, min_periods=window).sum() / (v + EPS)

    @staticmethod
    def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        pv = tp * df["volume"]
        return pv.rolling(window).sum() / (df["volume"].rolling(window).sum() + EPS)

    @staticmethod
    def ultimate_oscillator(df: pd.DataFrame, s: int = 7, m: int = 14,
                            l: int = 28) -> pd.Series:
        prior_c = df["close"].shift()
        bp = df["close"] - pd.concat([df["low"], prior_c], axis=1).min(axis=1)
        tr = pd.concat([df["high"], prior_c], axis=1).max(axis=1) - \
             pd.concat([df["low"], prior_c], axis=1).min(axis=1)
        avg_s = bp.rolling(s).sum() / (tr.rolling(s).sum() + EPS)
        avg_m = bp.rolling(m).sum() / (tr.rolling(m).sum() + EPS)
        avg_l = bp.rolling(l).sum() / (tr.rolling(l).sum() + EPS)
        return 100 * (4 * avg_s + 2 * avg_m + avg_l) / 7

    @staticmethod
    def aroon(df: pd.DataFrame, window: int = 25) -> Tuple[pd.Series, pd.Series]:
        up = df["high"].rolling(window + 1).apply(lambda x: np.argmax(x), raw=True)
        dn = df["low"].rolling(window + 1).apply(lambda x: np.argmin(x), raw=True)
        return 100 * up / window, 100 * dn / window

    @staticmethod
    def keltner(df: pd.DataFrame, window: int = 20, mult: float = 2.0):
        mid = TechnicalIndicators.ema(df["close"], window)
        atr = TechnicalIndicators.atr(df, window)
        return mid, mid + mult * atr, mid - mult * atr

    @staticmethod
    def donchian(df: pd.DataFrame, window: int = 20):
        return (df["high"].rolling(window).max(),
                df["low"].rolling(window).min())


class KalmanFilter1D:
    def __init__(self, process_var: float = 2e-4, obs_var: float = 2e-3):
        self.q = process_var
        self.r = obs_var
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self._initialized = False

    def update(self, z: float) -> Tuple[float, float]:
        if not self._initialized:
            self.x = np.array([z, 0.0])
            self.P = np.eye(2)
            self._initialized = True
            return z, 0.0
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + np.eye(2) * self.q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.r
        K = P_pred @ self.H.T / S
        self.x = x_pred + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        return float(self.x[0]), float(self.x[1])

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        levels, slopes = [], []
        for v in series.values:
            if not np.isfinite(v):
                levels.append(np.nan)
                slopes.append(np.nan)
                continue
            lv, sl = self.update(v)
            levels.append(lv)
            slopes.append(sl)
        return (pd.Series(levels, index=series.index),
                pd.Series(slopes, index=series.index))


class CryptoRegimeDetector:
    def __init__(self, vol_window: int = 24, trend_window: int = 168,
                 vol_low: float = 0.40, vol_mid: float = 0.65,
                 vol_high: float = 1.10, crash_window: int = 24,
                 crash_threshold: float = -0.15):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_low = vol_low
        self.vol_mid = vol_mid
        self.vol_high = vol_high
        self.crash_window = crash_window
        self.crash_threshold = crash_threshold

    def classify(self, prices: pd.Series) -> int:
        if len(prices) < self.trend_window + 1:
            return int(CryptoRegime.CHOP)
        rets = prices.pct_change().dropna()
        if len(rets) < self.vol_window:
            return int(CryptoRegime.CHOP)
        vol = float(rets.tail(self.vol_window).std() * np.sqrt(HOURS_PER_YEAR))
        trend = float(prices.iloc[-1] / prices.iloc[-self.trend_window] - 1)
        recent_move = float(prices.iloc[-1] / prices.iloc[-min(self.crash_window, len(prices))] - 1)
        if recent_move < self.crash_threshold:
            return int(CryptoRegime.CRASH)
        if recent_move > 0.35:
            return int(CryptoRegime.MELT_UP)
        if vol > self.vol_high:
            return int(CryptoRegime.CRASH)
        if vol > self.vol_mid:
            if trend < -0.20:
                return int(CryptoRegime.EXTREME_BEAR)
            if trend > 0.20:
                return int(CryptoRegime.EXTREME_BULL)
            return int(CryptoRegime.BEAR)
        if vol < self.vol_low:
            if trend > 0.05:
                return int(CryptoRegime.BULL)
            return int(CryptoRegime.CHOP)
        if trend > 0.15:
            return int(CryptoRegime.BULL)
        if trend < -0.15:
            return int(CryptoRegime.BEAR)
        return int(CryptoRegime.CHOP)

    def classify_series(self, prices: pd.Series) -> pd.Series:
        out = []
        for i in range(len(prices)):
            if i < self.trend_window:
                out.append(int(CryptoRegime.CHOP))
                continue
            out.append(self.classify(prices.iloc[:i + 1]))
        return pd.Series(out, index=prices.index)

    @staticmethod
    def risk_multiplier(regime: int) -> float:
        return {
            int(CryptoRegime.EXTREME_BEAR): 0.25,
            int(CryptoRegime.BEAR): 0.55,
            int(CryptoRegime.CHOP): 0.75,
            int(CryptoRegime.BULL): 1.10,
            int(CryptoRegime.EXTREME_BULL): 1.30,
            int(CryptoRegime.CRASH): 0.15,
            int(CryptoRegime.MELT_UP): 1.20,
            int(CryptoRegime.DECOUPLED): 0.85,
        }.get(regime, 0.75)


class MicrostructureFeatures:
    @staticmethod
    def amihud_illiquidity(df: pd.DataFrame, window: int = 24) -> pd.Series:
        ret = df["close"].pct_change().abs()
        dollar_vol = (df["close"] * df["volume"]).replace(0, np.nan)
        illiq = ret / dollar_vol
        return illiq.rolling(window, min_periods=window // 2).mean() * 1e9

    @staticmethod
    def kyle_lambda(df: pd.DataFrame, window: int = 24) -> pd.Series:
        signed_vol = np.sign(df["close"].diff()) * df["volume"]
        ret = df["close"].pct_change().abs()

        def lambda_fn(idx):
            if len(idx) < 5:
                return np.nan
            sv = signed_vol.iloc[idx].values
            rr = ret.iloc[idx].values
            mask = np.isfinite(sv) & np.isfinite(rr)
            if mask.sum() < 3:
                return np.nan
            sv, rr = sv[mask], rr[mask]
            if sv.std() < EPS:
                return np.nan
            try:
                slope = np.polyfit(sv, rr, 1)[0]
                return float(slope)
            except Exception:
                return np.nan
        n = len(df)
        out = np.full(n, np.nan)
        for i in range(window, n):
            out[i] = lambda_fn(range(i - window, i))
        return pd.Series(out, index=df.index)

    @staticmethod
    def roll_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
        ret = df["close"].pct_change()
        cov = ret.rolling(window).cov(ret.shift(1))
        spread = 2 * np.sqrt(np.maximum(-cov, 0))
        return spread

    @staticmethod
    def volume_clock_ratio(df: pd.DataFrame, window: int = 24) -> pd.Series:
        v = df["volume"]
        return v / (v.rolling(window, min_periods=window // 2).mean() + EPS)

    @staticmethod
    def vwap_deviation(df: pd.DataFrame, window: int = 24) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (tp * df["volume"]).rolling(window).sum() / \
               (df["volume"].rolling(window).sum() + EPS)
        return (df["close"] - vwap) / (vwap + EPS)

    @staticmethod
    def trade_size_distribution(df: pd.DataFrame, window: int = 24) -> pd.Series:
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        avg_trade = df["volume"] / (df["close"] * rng.clip(lower=1e-6))
        return avg_trade.rolling(window, min_periods=window // 2).mean()

    @staticmethod
    def high_low_range_z(df: pd.DataFrame, window: int = 24) -> pd.Series:
        rng = (df["high"] - df["low"]) / (df["close"] + EPS)
        mu = rng.rolling(window, min_periods=window // 2).mean()
        sd = rng.rolling(window, min_periods=window // 2).std()
        return (rng - mu) / (sd + EPS)

    @staticmethod
    def realized_skew_kurt(df: pd.DataFrame, window: int = 24) -> Tuple[pd.Series, pd.Series]:
        ret = df["close"].pct_change()
        return rolling_skew(ret, window), rolling_kurt(ret, window)


class OnChainProxies:
    @staticmethod
    def exchange_flow_imbalance(df: pd.DataFrame, window: int = 24) -> pd.Series:
        ret = df["close"].pct_change()
        vol_z = (df["volume"] - df["volume"].rolling(72).mean()) / \
                (df["volume"].rolling(72).std() + EPS)
        flow = -np.sign(ret) * vol_z
        return flow.rolling(window).mean()

    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
        hl = (df["high"] - df["low"]) + EPS
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
        return (clv * df["volume"]).cumsum()

    @staticmethod
    def chaikin_oscillator(df: pd.DataFrame, fast: int = 3,
                           slow: int = 10) -> pd.Series:
        ad = OnChainProxies.accumulation_distribution(df)
        return ad.ewm(span=fast, adjust=False).mean() - \
               ad.ewm(span=slow, adjust=False).mean()

    @staticmethod
    def network_activity_proxy(df: pd.DataFrame, window: int = 72) -> pd.Series:
        v = df["volume"]
        return v.rolling(window, min_periods=window // 2).std() / \
               (v.rolling(window, min_periods=window // 2).mean() + EPS)

    @staticmethod
    def miner_pressure_proxy(df: pd.DataFrame, window: int = 168) -> pd.Series:
        ret = df["close"].pct_change()
        pressure = np.minimum(ret, 0) * df["volume"]
        return pressure.rolling(window, min_periods=window // 2).sum()


class FundingRateProxy:
    @staticmethod
    def basis_proxy(df: pd.DataFrame, window: int = 8) -> pd.Series:
        ret = df["close"].pct_change()
        return ret.rolling(window).mean() * HOURS_PER_YEAR

    @staticmethod
    def funding_carry_signal(df: pd.DataFrame,
                             short_window: int = 8,
                             long_window: int = 72) -> pd.Series:
        r_short = df["close"].pct_change().rolling(short_window).mean()
        r_long = df["close"].pct_change().rolling(long_window).mean()
        carry = (r_short - r_long) * HOURS_PER_YEAR
        return carry

    @staticmethod
    def open_interest_proxy(df: pd.DataFrame, window: int = 24) -> pd.Series:
        oi = df["close"] * df["volume"]
        return oi.pct_change(window)

    @staticmethod
    def liquidation_cascade_proxy(df: pd.DataFrame, window: int = 4) -> pd.Series:
        ret = df["close"].pct_change()
        vol_z = (df["volume"] - df["volume"].rolling(72).mean()) / \
                (df["volume"].rolling(72).std() + EPS)
        cascade = (ret.abs().rolling(window).sum() *
                   (vol_z > 2.0).astype(float).rolling(window).sum())
        return cascade


class JumpDetector:
    @staticmethod
    def detect(df: pd.DataFrame, threshold_sigma: float = 5.0,
               vol_window: int = 24) -> pd.Series:
        ret = df["close"].pct_change()
        vol = ret.rolling(vol_window, min_periods=vol_window // 2).std()
        z = ret / (vol + EPS)
        return (z.abs() > threshold_sigma).astype(int)

    @staticmethod
    def jump_magnitude(df: pd.DataFrame, threshold_sigma: float = 5.0,
                       vol_window: int = 24) -> pd.Series:
        ret = df["close"].pct_change()
        vol = ret.rolling(vol_window, min_periods=vol_window // 2).std()
        z = ret / (vol + EPS)
        return ret.where(z.abs() > threshold_sigma, 0.0)

    @staticmethod
    def bipower_variation(df: pd.DataFrame, window: int = 24) -> pd.Series:
        ret = df["close"].pct_change().abs()
        bpv = (ret * ret.shift(1)).rolling(window).sum() * (np.pi / 2)
        return np.sqrt(bpv + EPS)

    @staticmethod
    def jump_ratio(df: pd.DataFrame, window: int = 24) -> pd.Series:
        ret = df["close"].pct_change()
        rv = (ret ** 2).rolling(window).sum()
        bpv = JumpDetector.bipower_variation(df, window) ** 2
        return rv / (bpv + EPS)

    @staticmethod
    def jump_decay(df: pd.DataFrame, window: int = 24) -> pd.Series:
        is_jump = JumpDetector.detect(df, 5.0, 24)
        ret = df["close"].pct_change()
        drift = ret.where(is_jump == 1, np.nan).rolling(window).mean()
        return drift


class LeadLagFeatures:
    @staticmethod
    def cross_correlation(a: pd.Series, b: pd.Series, lag: int,
                          window: int) -> pd.Series:
        return a.rolling(window).corr(b.shift(lag))

    @staticmethod
    def build_leadlag(asset_ret: pd.Series, bench_ret: pd.Series,
                      lags: Sequence[int] = (-3, -1, 0, 1, 3),
                      window: int = 72) -> pd.DataFrame:
        out = pd.DataFrame(index=asset_ret.index)
        for lag in lags:
            out[f"ll_corr_{lag}"] = LeadLagFeatures.cross_correlation(
                asset_ret, bench_ret, lag, window)
        out["ll_ret_lag1"] = bench_ret.shift(1)
        out["ll_ret_lag3"] = bench_ret.shift(3)
        return out

    @staticmethod
    def rolling_beta(asset_ret: pd.Series, bench_ret: pd.Series,
                     window: int = 72) -> pd.Series:
        cov = asset_ret.rolling(window).cov(bench_ret)
        var = bench_ret.rolling(window).var()
        return cov / (var + EPS)

    @staticmethod
    def beta_stability(asset_ret: pd.Series, bench_ret: pd.Series,
                       short: int = 24, long: int = 168) -> pd.Series:
        b_short = LeadLagFeatures.rolling_beta(asset_ret, bench_ret, short)
        b_long = LeadLagFeatures.rolling_beta(asset_ret, bench_ret, long)
        return b_short - b_long


class VolOfVol:
    @staticmethod
    def compute(df: pd.DataFrame, vol_windows: Sequence[int] = (6, 24, 72, 168),
                vov_window: int = 72) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change()
        vols = {}
        for w in vol_windows:
            v = ret.rolling(w, min_periods=max(3, w // 3)).std() * np.sqrt(HOURS_PER_YEAR)
            vols[w] = v
            out[f"vov_{w}"] = v.rolling(vov_window, min_periods=vov_window // 3).std()
            out[f"vol_ratio_{w}"] = v / (v.rolling(vov_window, min_periods=10).mean() + EPS)
        if 24 in vols and 168 in vols:
            out["vol_regime"] = np.sign(vols[24] - vols[168])
        return out


class IntradaySeasonality:
    @staticmethod
    def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour.values

    @staticmethod
    def day_of_week(index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.values

    @staticmethod
    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        out = pd.DataFrame(index=idx)
        out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        out["is_asia"] = ((idx.hour >= 0) & (idx.hour < 8)).astype(int)
        out["is_eu"] = ((idx.hour >= 8) & (idx.hour < 16)).astype(int)
        out["is_us"] = ((idx.hour >= 16) | (idx.hour < 24)).astype(int)
        out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        return out


class CryptoFeatureEngine:
    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.ti = TechnicalIndicators()
        self.kf = KalmanFilter1D(cfg.kalman_process_var, cfg.kalman_obs_var)
        self.regime_detector = CryptoRegimeDetector()
        self.micro = MicrostructureFeatures()
        self.onchain = OnChainProxies()
        self.funding = FundingRateProxy()
        self.jumps = JumpDetector()
        self.volvol = VolOfVol()
        self.season = IntradaySeasonality()
        self.feature_cols_: List[str] = []

    def build(self, price_data: Dict[str, pd.DataFrame],
              benchmark: Optional[pd.DataFrame] = None) -> "CryptoFeatureSet":
        LOG.info(f"Building crypto features for {len(price_data)} symbols")
        frames = []
        for sym, df in price_data.items():
            if df is None or df.empty or len(df) < self.cfg.min_history:
                continue
            try:
                f = self._build_per_symbol(df, sym)
                frames.append(f)
            except Exception as e:
                LOG.warning(f"Feature build failed {sym}: {e}")
                continue
        if not frames:
            raise RuntimeError("No valid symbols produced features")
        panel = pd.concat(frames, axis=0)
        if self.cfg.use_cross_sectional:
            panel = self._add_cross_sectional(panel)
        if benchmark is not None and not benchmark.empty:
            panel = self._add_benchmark_relative(panel, benchmark)
        if self.cfg.use_regime_features:
            panel = self._add_regime_features(panel, price_data)
        panel = self._add_targets(panel)
        panel = panel.sort_index()
        self.feature_cols_ = self._infer_feature_cols(panel)
        target_cols = {h: f"target_{h}" for h in self.cfg.horizons_hours}
        timestamps = panel.index.get_level_values("timestamp").unique().tolist()
        symbols = panel.index.get_level_values("symbol").unique().tolist()
        LOG.info(f"Crypto panel: {panel.shape[0]} rows, {len(self.feature_cols_)} features, "
                 f"{len(symbols)} symbols, {len(timestamps)} timestamps")
        return CryptoFeatureSet(panel, self.feature_cols_, target_cols, symbols, timestamps)

    def _build_per_symbol(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = df.copy().sort_index()
        if "close" not in d.columns:
            raise ValueError(f"{symbol}: missing close")
        close = d["close"].astype(float)
        feat = pd.DataFrame(index=d.index)
        feat["symbol"] = symbol
        feat["close"] = close
        feat["volume"] = d.get("volume", pd.Series(0.0, index=d.index))
        feat["returns_1"] = close.pct_change()
        feat["log_returns_1"] = np.log(close / close.shift(1))
        day_key = d.index.normalize()
        feat["intraday_return"] = (
            close / close.groupby(day_key).transform("first") - 1)

        if self.cfg.use_fractional_diff:
            try:
                feat["frac_diff"] = fractionally_differentiate(close, self.cfg.fractional_d)
            except Exception:
                feat["frac_diff"] = np.nan

        if self.cfg.use_kalman:
            lvl, slope = self.kf.filter_series(close)
            feat["kalman_level"] = lvl
            feat["kalman_slope"] = slope
            feat["kalman_gap"] = (close - lvl) / (lvl.abs() + EPS)

        for w in (2, 3, 6, 12, 24, 72, 168, 720):
            feat[f"ret_{w}"] = close.pct_change(w)
            feat[f"logret_{w}"] = np.log(close / close.shift(w))

        for w in self.cfg.vol_windows:
            feat[f"vol_{w}"] = (feat["log_returns_1"]
                                .rolling(w, min_periods=max(3, w // 3)).std()
                                * np.sqrt(HOURS_PER_YEAR))
            feat[f"realized_vol_{w}"] = (
                np.sqrt((feat["log_returns_1"] ** 2).rolling(w).sum())
                * np.sqrt(HOURS_PER_YEAR))
        for (a, b) in itertools.combinations(self.cfg.vol_windows, 2):
            feat[f"vol_ratio_{a}_{b}"] = safe_div(
                feat[f"vol_{a}"], feat[f"vol_{b}"], np.nan)
        for hl in (12, 24, 72, 168):
            feat[f"ewma_vol_{hl}"] = ewma_volatility(
                feat["returns_1"], hl, annualize=True, hours=True)

        for w in self.cfg.ma_windows:
            ma = self.ti.sma(close, w)
            ema = self.ti.ema(close, w)
            feat[f"sma_{w}"] = ma
            feat[f"ema_{w}"] = ema
            feat[f"dist_sma_{w}"] = (close - ma) / (ma.abs() + EPS)
            feat[f"dist_ema_{w}"] = (close - ema) / (ema.abs() + EPS)
            feat[f"slope_sma_{w}"] = ma.diff(3) / (ma.abs() + EPS)
        for (a, b) in [(6, 24), (12, 48), (24, 120), (48, 240)]:
            if a in self.cfg.ma_windows and b in self.cfg.ma_windows:
                feat[f"cross_{a}_{b}"] = (feat[f"sma_{a}"] - feat[f"sma_{b}"]) / \
                                         (feat[f"sma_{b}"].abs() + EPS)

        for w in self.cfg.rsi_windows:
            rsi_v = self.ti.rsi(close, w)
            feat[f"rsi_{w}"] = rsi_v
            feat[f"rsi_slope_{w}"] = rsi_v.diff(3)

        f_, s_, sig_ = self.cfg.macd_params
        macd_line, macd_sig, macd_hist = self.ti.macd(close, f_, s_, sig_)
        feat["macd"] = macd_line / (close + EPS)
        feat["macd_signal"] = macd_sig / (close + EPS)
        feat["macd_hist"] = macd_hist / (close + EPS)
        feat["macd_slope"] = macd_line.diff(3) / (close + EPS)

        mid, up, lo = self.ti.bollinger(close, self.cfg.bb_window, self.cfg.bb_std)
        feat["bb_position"] = (close - lo) / (up - lo + EPS)
        feat["bb_width"] = (up - lo) / (mid.abs() + EPS)
        feat["bb_squeeze"] = feat["bb_width"] / (feat["bb_width"].rolling(72).mean() + EPS)

        for w in self.cfg.atr_windows:
            atr_v = self.ti.atr(d, w)
            feat[f"atr_{w}"] = atr_v / (close + EPS)
        adx_v = self.ti.adx(d, 14)
        feat["adx_14"] = adx_v
        feat["adx_slope"] = adx_v.diff(5)

        sk, sd = self.ti.stochastic(d, 14, 3, 3)
        feat["stoch_k"] = sk
        feat["stoch_d"] = sd
        feat["stoch_cross"] = sk - sd
        feat["williams_r"] = self.ti.williams_r(d, 14)
        feat["cci_20"] = self.ti.cci(d, 20)
        feat["uo"] = self.ti.ultimate_oscillator(d)
        up_a, dn_a = self.ti.aroon(d, 25)
        feat["aroon_up"] = up_a
        feat["aroon_dn"] = dn_a
        feat["aroon_osc"] = up_a - dn_a

        km, ku, kl = self.ti.keltner(d, 20, 2.0)
        feat["keltner_pos"] = (close - kl) / (ku - kl + EPS)
        dh, dl = self.ti.donchian(d, 24)
        feat["donchian_pos"] = (close - dl) / (dh - dl + EPS)
        feat["donchian_break"] = ((close >= dh * 0.999).astype(int) -
                                  (close <= dl * 1.001).astype(int))

        if self.cfg.use_volume_features and "volume" in d.columns:
            vol = d["volume"].astype(float)
            for w in (6, 24, 72, 168):
                vma = vol.rolling(w, min_periods=w // 2).mean()
                feat[f"vol_ratio_{w}"] = vol / (vma + EPS)
                feat[f"dollar_vol_{w}"] = (vol * close).rolling(w).mean()
            obv = self.ti.obv(d)
            feat["obv_z"] = (obv - obv.rolling(72).mean()) / (obv.rolling(72).std() + EPS)
            feat["obv_slope"] = obv.diff(12) / (obv.abs().rolling(72).mean() + EPS)
            feat["mfi_14"] = self.ti.mfi(d, 14)
            feat["cmf_20"] = self.ti.cmf(d, 20)
            feat["vwap_24"] = self.ti.vwap(d, 24) / (close + EPS)

        rng = (d["high"] - d["low"]).replace(0, np.nan)
        feat["upper_wick"] = (d["high"] - np.maximum(d["open"], close)) / (rng + EPS)
        feat["lower_wick"] = (np.minimum(d["open"], close) - d["low"]) / (rng + EPS)
        feat["body_pct"] = (close - d["open"]) / (rng + EPS)
        feat["range_pct"] = rng / (close + EPS)
        feat["body_to_range"] = abs(close - d["open"]) / (rng + EPS)

        for w in (24, 72, 168):
            feat[f"skew_{w}"] = rolling_skew(feat["log_returns_1"], w)
            feat[f"kurt_{w}"] = rolling_kurt(feat["log_returns_1"], w)
            feat[f"up_hours_{w}"] = (feat["returns_1"] > 0).rolling(w).mean()
            feat[f"ret_std_{w}"] = feat["returns_1"].rolling(w).std()
            feat[f"ret_mean_{w}"] = feat["returns_1"].rolling(w).mean()

        if self.cfg.use_entropy_features:
            feat["entropy_24"] = rolling_entropy(feat["log_returns_1"], 24)
            feat["entropy_72"] = rolling_entropy(feat["log_returns_1"], 72)
        if self.cfg.use_hurst:
            feat["hurst_72"] = feat["log_returns_1"].rolling(72).apply(
                lambda x: hurst_exponent(x, max_lag=12), raw=True)
            feat["hurst_168"] = feat["log_returns_1"].rolling(168).apply(
                lambda x: hurst_exponent(x, max_lag=15), raw=True)

        rm = close.cummax()
        feat["dd_from_peak"] = close / rm - 1
        feat["time_since_peak"] = ((close == rm).astype(int)
                                   .groupby((close == rm).cumsum()).cumcount())
        feat["dd_recovery_speed"] = feat["dd_from_peak"].diff(24)

        if self.cfg.use_microstructure:
            feat["amihud"] = self.micro.amihud_illiquidity(d, 24)
            feat["kyle_lambda"] = self.micro.kyle_lambda(d, 24)
            feat["roll_spread"] = self.micro.roll_spread(d, 20)
            feat["volume_clock"] = self.micro.volume_clock_ratio(d, 24)
            feat["vwap_dev"] = self.micro.vwap_deviation(d, 24)
            feat["trade_size"] = self.micro.trade_size_distribution(d, 24)
            feat["hl_range_z"] = self.micro.high_low_range_z(d, 24)
            sk_m, ku_m = self.micro.realized_skew_kurt(d, 24)
            feat["micro_skew"] = sk_m
            feat["micro_kurt"] = ku_m

        if self.cfg.use_onchain_proxies:
            feat["flow_imbalance"] = self.onchain.exchange_flow_imbalance(d, 24)
            ad = self.onchain.accumulation_distribution(d)
            feat["ad_z"] = (ad - ad.rolling(168).mean()) / (ad.rolling(168).std() + EPS)
            feat["chaikin_osc"] = self.onchain.chaikin_oscillator(d)
            feat["activity_proxy"] = self.onchain.network_activity_proxy(d, 72)
            feat["miner_pressure"] = self.onchain.miner_pressure_proxy(d, 168)

        if self.cfg.use_funding_proxy:
            feat["basis_proxy"] = self.funding.basis_proxy(d, 8)
            feat["funding_carry"] = self.funding.funding_carry_signal(d, 8, 72)
            feat["oi_proxy"] = self.funding.open_interest_proxy(d, 24)
            feat["cascade_proxy"] = self.funding.liquidation_cascade_proxy(d, 4)

        if self.cfg.use_jump_features:
            feat["is_jump"] = self.jumps.detect(d, self.cfg.jump_threshold_sigma, 24)
            feat["jump_mag"] = self.jumps.jump_magnitude(d, self.cfg.jump_threshold_sigma, 24)
            feat["bipower_vol"] = self.jumps.bipower_variation(d, 24)
            feat["jump_ratio"] = self.jumps.jump_ratio(d, 24)
            feat["jump_decay"] = self.jumps.jump_decay(d, 24)

        if self.cfg.use_vol_of_vol:
            vv = self.volvol.compute(d, (6, 24, 72, 168), 72)
            for c in vv.columns:
                feat[c] = vv[c]

        if self.cfg.use_intraday_seasonality:
            season = self.season.compute_features(d)
            for c in season.columns:
                feat[c] = season[c]

        if self.cfg.use_pattern_features:
            feat["is_doji"] = (abs(close - d["open"]) / (rng + EPS) < 0.1).astype(int)
            feat["is_hammer"] = (
                (feat["lower_wick"] > 2 * feat["body_pct"].abs()) &
                (feat["upper_wick"] < 0.3 * feat["body_pct"].abs())
            ).astype(int)
            feat["is_shooting_star"] = (
                (feat["upper_wick"] > 2 * feat["body_pct"].abs()) &
                (feat["lower_wick"] < 0.3 * feat["body_pct"].abs())
            ).astype(int)
            feat["is_bull_engulf"] = (
                (feat["body_pct"] > 0) &
                (feat["body_pct"].shift(1) < 0) &
                (close > d["open"].shift(1)) &
                (d["open"] < close.shift(1))
            ).astype(int)
            feat["is_bear_engulf"] = (
                (feat["body_pct"] < 0) &
                (feat["body_pct"].shift(1) > 0) &
                (close < d["open"].shift(1)) &
                (d["open"] > close.shift(1))
            ).astype(int)

        feat["price_volume_corr_24"] = close.pct_change().rolling(24).corr(
            feat["volume"].pct_change())
        feat["price_volume_corr_72"] = close.pct_change().rolling(72).corr(
            feat["volume"].pct_change())

        return feat

    def _add_cross_sectional(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        if "timestamp" not in panel.index.names:
            panel = panel.reset_index()
            for c in panel.columns:
                if np.issubdtype(panel[c].dtype, np.datetime64):
                    panel = panel.rename(columns={c: "timestamp"})
                    break
            else:
                panel["timestamp"] = panel.index
            panel = panel.set_index(["timestamp", "symbol"])

        rank_candidates = [
            "ret_1", "ret_24", "ret_168", "vol_24", "rsi_14", "macd_hist",
            "atr_14", "adx_14", "vol_ratio_24", "mfi_14", "bb_position",
            "kalman_slope", "hurst_72", "entropy_24", "amihud", "flow_imbalance",
            "funding_carry", "basis_proxy",
        ]
        by_ts = panel.groupby(level="timestamp", group_keys=False)
        for col in rank_candidates:
            if col in panel.columns:
                try:
                    panel[f"cs_rank_{col}"] = by_ts[col].rank(pct=True) - 0.5
                    panel[f"cs_z_{col}"] = by_ts[col].transform(
                        lambda s: (s - s.mean()) / (s.std() + EPS))
                except Exception:
                    continue
        return panel

    def _add_benchmark_relative(self, panel: pd.DataFrame,
                                benchmark: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        bench_ret = benchmark["close"].pct_change().rename("bench_ret")
        if not isinstance(panel.index.get_level_values("timestamp"), pd.DatetimeIndex):
            try:
                ts = pd.to_datetime(panel.index.get_level_values("timestamp"))
                panel.index = pd.MultiIndex.from_arrays(
                    [ts, panel.index.get_level_values("symbol")],
                    names=["timestamp", "symbol"])
            except Exception:
                pass
        rets_df = panel["returns_1"].unstack("symbol")
        excess = rets_df.sub(bench_ret, axis=0)
        panel["excess_ret_1"] = excess.stack().reindex(panel.index)
        try:
            beta_72 = rets_df.rolling(72).cov(bench_ret).div(
                bench_ret.rolling(72).var() + EPS, axis=0)
            panel["beta_72"] = beta_72.stack().reindex(panel.index)
            beta_168 = rets_df.rolling(168).cov(bench_ret).div(
                bench_ret.rolling(168).var() + EPS, axis=0)
            panel["beta_168"] = beta_168.stack().reindex(panel.index)
            corr_72 = rets_df.rolling(72).corr(bench_ret)
            panel["corr_bench_72"] = corr_72.stack().reindex(panel.index)
        except Exception:
            panel["beta_72"] = np.nan
            panel["beta_168"] = np.nan
            panel["corr_bench_72"] = np.nan

        if self.cfg.use_leadlag:
            try:
                ll_parts = []
                for sym in rets_df.columns:
                    a_ret = rets_df[sym]
                    ll = LeadLagFeatures.build_leadlag(
                        a_ret, bench_ret, lags=(-3, -1, 0, 1, 3), window=72)
                    ll["symbol"] = sym
                    ll = ll.set_index("symbol", append=True)
                    ll = ll.reorder_levels(["timestamp", "symbol"])
                    ll_parts.append(ll)
                if ll_parts:
                    ll_df = pd.concat(ll_parts)
                    for c in ll_df.columns:
                        panel[f"ll_{c}"] = ll_df[c].reindex(panel.index)
                    for sym in rets_df.columns:
                        bstab = LeadLagFeatures.beta_stability(
                            rets_df[sym], bench_ret, 24, 168)
                        panel.loc[(slice(None), sym), "beta_stability"] = bstab.values
            except Exception as e:
                LOG.debug(f"Lead-lag failed: {e}")
        return panel

    def _add_regime_features(self, panel: pd.DataFrame,
                             price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        panel = panel.copy()
        try:
            bench_sym = self.cfg.benchmark_symbol
            if bench_sym in price_data:
                prices = price_data[bench_sym]["close"]
            else:
                px = pd.DataFrame({s: d["close"] for s, d in price_data.items()})
                prices = px.median(axis=1)
            regimes = self.regime_detector.classify_series(prices)
            panel["regime"] = regimes.reindex(
                panel.index.get_level_values("timestamp")).values
            panel["regime_risk_mult"] = panel["regime"].apply(
                CryptoRegimeDetector.risk_multiplier)
            for r in CryptoRegime:
                panel[f"regime_is_{r.value}"] = (panel["regime"] == r.value).astype(int)
        except Exception as e:
            LOG.warning(f"Regime computation failed: {e}")
            panel["regime"] = int(CryptoRegime.CHOP)
            panel["regime_risk_mult"] = 0.75
        return panel

    def _add_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        close_df = panel["close"].unstack("symbol")
        for h in self.cfg.horizons_hours:
            fwd = close_df.shift(-h) / close_df - 1.0
            fwd_stacked = fwd.stack().reindex(panel.index)
            panel[f"fwd_ret_{h}"] = fwd_stacked
            if self.cfg.target_type == CryptoPredictionTarget.DIRECTION:
                panel[f"target_{h}"] = (fwd_stacked > self.cfg.return_threshold).astype(int)
            elif self.cfg.target_type == CryptoPredictionTarget.RETURN:
                panel[f"target_{h}"] = fwd_stacked
            elif self.cfg.target_type == CryptoPredictionTarget.EXCESS_RETURN:
                mean_fwd = fwd.mean(axis=1)
                excess = fwd.sub(mean_fwd, axis=0)
                panel[f"target_{h}"] = excess.stack().reindex(panel.index)
            elif self.cfg.target_type == CryptoPredictionTarget.RANK:
                ranks = fwd.rank(axis=1, pct=True)
                panel[f"target_{h}"] = ranks.stack().reindex(panel.index)
            elif self.cfg.target_type == CryptoPredictionTarget.TRIPLE_BARRIER:
                vol_series = panel["atr_24"] if "atr_24" in panel.columns else 0.02
                threshold = vol_series.fillna(0.02) * np.sqrt(h)
                target = pd.Series(0, index=panel.index)
                for step in range(1, h + 1):
                    r_step = close_df.shift(-step) / close_df - 1.0
                    r_step = r_step.stack().reindex(panel.index)
                    up_hit = (r_step > threshold) & (target == 0)
                    dn_hit = (r_step < -threshold) & (target == 0)
                    target = target.mask(up_hit, 1)
                    target = target.mask(dn_hit, -1)
                panel[f"target_{h}"] = target
            elif self.cfg.target_type == CryptoPredictionTarget.VOLATILITY:
                fwd_ret_series = panel["log_returns_1"]
                realized = (fwd_ret_series ** 2).rolling(h).sum().shift(-h)
                panel[f"target_{h}"] = np.sqrt(realized) * np.sqrt(HOURS_PER_YEAR / h)
            elif self.cfg.target_type == CryptoPredictionTarget.DRAWDOWN:
                min_fwd = close_df.rolling(h).min().shift(-h)
                panel[f"target_{h}"] = (min_fwd / close_df - 1).stack().reindex(panel.index)
        return panel

    @staticmethod
    def _infer_feature_cols(panel: pd.DataFrame) -> List[str]:
        drop = {"symbol", "regime"}
        drop |= {c for c in panel.columns if c.startswith("fwd_ret_")}
        drop |= {c for c in panel.columns if c.startswith("target_")}
        return [c for c in panel.columns
                if c not in drop and panel[c].dtype.kind in "if"]


@dataclass
class CryptoFeatureSet:
    panel: pd.DataFrame
    feature_cols: List[str]
    target_cols: Dict[int, str]
    symbol_index: List[str]
    timestamp_index: List[pd.Timestamp]


class Preprocessor:
    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.feature_names_: List[str] = []
        self.medians_: Dict[str, float] = {}
        self.winsor_bounds_: Dict[str, Tuple[float, float]] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "Preprocessor":
        self.feature_names_ = list(feature_cols)
        for col in self.feature_names_:
            if col not in df.columns:
                self.medians_[col] = 0.0
                self.winsor_bounds_[col] = (-6.0, 6.0)
                self.means_[col] = 0.0
                self.stds_[col] = 1.0
                continue
            s = df[col].replace([np.inf, -np.inf], np.nan)
            self.medians_[col] = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(self.medians_[col])
            lo = float(filled.quantile(self.cfg.winsorize_quantile))
            hi = float(filled.quantile(1 - self.cfg.winsorize_quantile))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -6.0, 6.0
            self.winsor_bounds_[col] = (lo, hi)
            clipped = filled.clip(lo, hi)
            mu = float(clipped.mean())
            sd = float(clipped.std())
            if not np.isfinite(sd) or sd < EPS:
                sd = 1.0
            self.means_[col] = mu
            self.stds_[col] = sd
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_names_:
            if col not in df.columns:
                out[col] = 0.0
                continue
            s = df[col].replace([np.inf, -np.inf], np.nan)
            s = s.fillna(self.medians_.get(col, 0.0))
            lo, hi = self.winsor_bounds_.get(col, (-6.0, 6.0))
            s = s.clip(lo, hi)
            mu = self.means_.get(col, 0.0)
            sd = self.stds_.get(col, 1.0)
            out[col] = (s - mu) / (sd + EPS)
        return out.clip(-self.cfg.feature_clip_sigma, self.cfg.feature_clip_sigma)

    def fit_transform(self, df: pd.DataFrame,
                      feature_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)


class PurgedEmbargoedSplit:
    def __init__(self, n_splits: int = 6, embargo_hours: int = 24,
                 purge_hours: int = 24):
        self.n_splits = n_splits
        self.embargo = embargo_hours
        self.purge = purge_hours

    def split(self, timestamps: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        unique_ts = np.sort(timestamps.unique())
        n = len(unique_ts)
        if n < self.n_splits * 3:
            fold = max(1, n // self.n_splits)
            for i in range(self.n_splits):
                start = i * fold
                end = min(start + fold, n)
                train_ts = unique_ts[:start]
                val_ts = unique_ts[start:end]
                tr = timestamps.isin(train_ts).values
                va = timestamps.isin(val_ts).values
                if tr.sum() > 0 and va.sum() > 0:
                    yield np.where(tr)[0], np.where(va)[0]
            return
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else n
            val_ts = unique_ts[val_start:val_end]
            purge_start = max(0, val_start - self.purge)
            train_ts_all = unique_ts[:purge_start]
            tr = timestamps.isin(train_ts_all).values
            va = timestamps.isin(val_ts).values
            if tr.sum() < 30 or va.sum() < 5:
                continue
            yield np.where(tr)[0], np.where(va)[0]


class BaseLearner:
    name = "base"

    def fit(self, X, y, X_val=None, y_val=None) -> "BaseLearner":
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def feature_importances(self) -> Optional[np.ndarray]:
        return None


class XGBLearner(BaseLearner):
    name = "xgb"

    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.model = None
        self.use_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            return r.returncode == 0
        except Exception:
            return False

    def fit(self, X, y, X_val=None, y_val=None):
        if not HAS_XGB:
            raise RuntimeError("xgboost missing")
        params = dict(
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=self.cfg.xgb_learning_rate,
            subsample=self.cfg.xgb_subsample,
            colsample_bytree=self.cfg.xgb_colsample,
            min_child_weight=self.cfg.xgb_min_child_weight,
            gamma=self.cfg.xgb_gamma,
            reg_alpha=self.cfg.xgb_reg_alpha,
            reg_lambda=self.cfg.xgb_reg_lambda,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            early_stopping_rounds=self.cfg.xgb_early_stop,
            use_label_encoder=False,
        )
        if self.use_gpu:
            try:
                params["device"] = "cuda"
            except Exception:
                pass
        self.model = xgb.XGBClassifier(**params)
        if X_val is not None and y_val is not None and len(y_val) > 0 and \
                len(np.unique(y_val)) >= 2:
            self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X, y, verbose=False)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        return self.model.feature_importances_


class ExtraTreesLearner(BaseLearner):
    name = "extratrees"

    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.model = ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=15,
            min_samples_split=30,
            max_features="sqrt",
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
            class_weight="balanced_subsample",
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        return self.model.feature_importances_


class HistGBMLearner(BaseLearner):
    name = "histgbm"

    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.model = HistGradientBoostingClassifier(
            max_iter=600,
            max_depth=6,
            learning_rate=0.035,
            max_bins=255,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=cfg.random_state,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class RidgeMetaLearner(BaseLearner):
    name = "ridge_meta"

    def __init__(self, cfg: CryptoPredictorConfig):
        self.model = LogisticRegression(
            C=1.0, max_iter=3000, solver="lbfgs",
            random_state=cfg.random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class AdaBoostLearner(BaseLearner):
    name = "adaboost"

    def __init__(self, cfg: CryptoPredictorConfig):
        self.model = AdaBoostClassifier(
            n_estimators=300, learning_rate=0.05,
            random_state=cfg.random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class QuantileReturnPredictor:
    def __init__(self, quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)):
        self.quantiles = tuple(quantiles)
        self.models: Dict[float, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileReturnPredictor":
        for q in self.quantiles:
            try:
                if q == 0.5:
                    m = BayesianRidge()
                else:
                    m = QuantileRegressor(quantile=q, alpha=1e-3,
                                          solver="highs", fit_intercept=True)
                m.fit(X, y)
                self.models[q] = m
            except Exception:
                try:
                    m = Ridge(alpha=1.0)
                    m.fit(X, y)
                    self.models[q] = m
                except Exception:
                    continue
        return self

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        out = {}
        for q, m in self.models.items():
            try:
                out[q] = m.predict(X)
            except Exception:
                out[q] = np.zeros(X.shape[0])
        return out


class ConformalPredictor:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.residual_q_ = 0.0

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor":
        residual = np.abs(y_true - y_pred)
        n = len(residual)
        if n == 0:
            self.residual_q_ = 0.0
            return self
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.residual_q_ = float(np.quantile(residual, level))
        return self

    def interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return y_pred - self.residual_q_, y_pred + self.residual_q_


class OnlineLearner:
    def __init__(self, n_features: int, cfg: CryptoPredictorConfig):
        self.n_features = n_features
        self.cfg = cfg
        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate=cfg.online_learning_rate,
            eta0=cfg.online_alpha * 100,
            alpha=cfg.online_alpha,
            penalty="l2",
            random_state=cfg.random_state,
            warm_start=True,
            max_iter=1,
        )
        self._fitted = False
        self._classes = np.array([0, 1])

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(np.unique(y)) < 2 and not self._fitted:
            return
        if not self._fitted:
            self.model.partial_fit(X, y, classes=self._classes)
            self._fitted = True
        else:
            try:
                self.model.partial_fit(X, y)
            except Exception:
                pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full(X.shape[0], 0.5)
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return np.full(X.shape[0], 0.5)


class FeatureSelector:
    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.selected_: List[int] = []
        self.importances_: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> "FeatureSelector":
        if not self.cfg.feature_selection_enabled or X.shape[1] <= self.cfg.max_features:
            self.selected_ = list(range(X.shape[1]))
            return self
        n = X.shape[0]
        idx = np.random.RandomState(self.cfg.random_state).choice(
            n, size=min(n, 10000), replace=False)
        X_s = X[idx]
        y_s = y[idx]
        try:
            mi = mutual_info_classif(X_s, y_s, random_state=self.cfg.random_state)
        except Exception:
            mi = np.zeros(X.shape[1])
        try:
            clf = ExtraTreesClassifier(
                n_estimators=200, max_depth=6, n_jobs=self.cfg.n_jobs,
                random_state=self.cfg.random_state)
            clf.fit(X_s, y_s)
            fi = clf.feature_importances_
        except Exception:
            fi = np.zeros(X.shape[1])
        combined = 0.5 * robust_zscore(mi) + 0.5 * robust_zscore(fi)
        for name, v in zip(feature_names, combined):
            self.importances_[name] = float(v)
        order = np.argsort(-combined)
        keep = [i for i in order if mi[i] >= self.cfg.min_mutual_info]
        if not keep:
            keep = list(order[:self.cfg.max_features])
        self.selected_ = keep[:self.cfg.max_features]
        LOG.info(f"Feature selection: {len(self.selected_)}/{X.shape[1]} kept")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_:
            return X
        return X[:, self.selected_]


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if len(np.unique(y)) < 2 or len(probs) < 50:
            self.calibrator = None
            return self
        try:
            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(probs, y)
                self.calibrator = ("iso", iso)
            else:
                lr = LogisticRegression(C=1.0, max_iter=1000)
                lr.fit(probs.reshape(-1, 1), y)
                self.calibrator = ("lr", lr)
        except Exception:
            self.calibrator = None
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return probs
        kind, model = self.calibrator
        try:
            if kind == "iso":
                return model.predict(probs)
            return model.predict_proba(probs.reshape(-1, 1))[:, 1]
        except Exception:
            return probs


class BootstrapUncertainty:
    def __init__(self, n_bootstrap: int = 40, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.models: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            base_factory: Callable[[], Any]) -> "BootstrapUncertainty":
        rng = np.random.RandomState(self.random_state)
        self.models = []
        n = X.shape[0]
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Xb, yb = X[idx], y[idx]
            if len(np.unique(yb)) < 2:
                continue
            try:
                m = base_factory()
                m.fit(Xb, yb)
                self.models.append(m)
            except Exception:
                continue
        LOG.info(f"Bootstrap: {len(self.models)} models")
        return self

    def predict_distribution(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            return np.full(X.shape[0], 0.5), np.zeros(X.shape[0])
        preds = []
        for m in self.models:
            try:
                p = m.predict_proba(X)[:, 1]
                preds.append(p)
            except Exception:
                continue
        if not preds:
            return np.full(X.shape[0], 0.5), np.zeros(X.shape[0])
        arr = np.vstack(preds)
        return arr.mean(axis=0), arr.std(axis=0)


class DriftMonitor:
    def __init__(self, cfg: CryptoPredictorConfig):
        self.cfg = cfg
        self.reference_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.psi_history_: List[Dict[str, float]] = []

    def set_reference(self, X: np.ndarray, feature_names: List[str]) -> None:
        self.reference_ = X.copy()
        self.feature_names_ = list(feature_names)

    def check(self, X: np.ndarray) -> Dict[str, float]:
        if self.reference_ is None or X.size == 0:
            return {}
        psi_vals = {}
        for i, name in enumerate(self.feature_names_):
            if i >= X.shape[1] or i >= self.reference_.shape[1]:
                continue
            psi_vals[name] = psi(X[:, i], self.reference_[:, i])
        self.psi_history_.append(psi_vals)
        n_drift = sum(1 for v in psi_vals.values()
                      if v > self.cfg.drift_psi_threshold)
        if n_drift > 0:
            LOG.warning(f"Drift: {n_drift} features PSI>{self.cfg.drift_psi_threshold}")
        return psi_vals

    def top_drifting(self, k: int = 10) -> List[Tuple[str, float]]:
        if not self.psi_history_:
            return []
        return sorted(self.psi_history_[-1].items(), key=lambda x: -x[1])[:k]


class HorizonEnsemble:
    def __init__(self, cfg: CryptoPredictorConfig, horizon_hours: int):
        self.cfg = cfg
        self.horizon = horizon_hours
        self.xgb = XGBLearner(cfg) if HAS_XGB else None
        self.extratrees = ExtraTreesLearner(cfg)
        self.histgbm = HistGBMLearner(cfg)
        self.meta = RidgeMetaLearner(cfg)
        self.selector = FeatureSelector(cfg)
        self.preprocessor = Preprocessor(cfg)
        self.calibrator = ProbabilityCalibrator(cfg.calibration_method)
        self.uncertainty: Optional[BootstrapUncertainty] = None
        self.metrics_: Optional[Dict] = None
        self.feature_names_: List[str] = []
        self.is_fitted = False
        self.training_time_s: float = 0.0

    def fit(self, panel: pd.DataFrame, feature_cols: List[str],
            target_col: str) -> "HorizonEnsemble":
        t0 = time.perf_counter()
        df = panel.dropna(subset=[target_col]).copy()
        df = df[df[feature_cols].notna().sum(axis=1) >= int(0.7 * len(feature_cols))]
        if len(df) < 200:
            raise RuntimeError(f"H{self.horizon}: only {len(df)} samples")
        X_df = df[feature_cols]
        y = df[target_col].values.astype(int)
        self.preprocessor.fit(X_df, feature_cols)
        X_scaled = self.preprocessor.transform(X_df).values
        self.feature_names_ = list(feature_cols)

        self.selector.fit(X_scaled, y, self.feature_names_)
        X_sel = self.selector.transform(X_scaled)

        splitter = PurgedEmbargoedSplit(
            n_splits=min(self.cfg.n_splits, 5),
            embargo_hours=self.cfg.embargo_hours,
            purge_hours=self.cfg.purge_hours,
        )
        ts_series = pd.Series(df.index.get_level_values("timestamp"))
        folds = list(splitter.split(ts_series))
        if folds:
            tr_idx, va_idx = folds[-1]
            X_tr, y_tr = X_sel[tr_idx], y[tr_idx]
            X_va, y_va = X_sel[va_idx], y[va_idx]
        else:
            cut = int(len(X_sel) * 0.8)
            X_tr, y_tr = X_sel[:cut], y[:cut]
            X_va, y_va = X_sel[cut:], y[cut:]

        cv_aucs = []
        for tr_i, va_i in folds:
            try:
                m = ExtraTreesClassifier(
                    n_estimators=200, max_depth=6, n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state)
                m.fit(X_sel[tr_i], y[tr_i])
                p = m.predict_proba(X_sel[va_i])[:, 1]
                if len(np.unique(y[va_i])) >= 2:
                    cv_aucs.append(roc_auc_score(y[va_i], p))
            except Exception:
                continue
        LOG.info(f"[H={self.horizon}h] CV AUC={np.mean(cv_aucs) if cv_aucs else float('nan'):.4f}")

        base_val, base_tr = [], []
        if self.xgb is not None:
            try:
                self.xgb.fit(X_tr, y_tr, X_va, y_va)
                base_val.append(self.xgb.predict_proba(X_va))
                base_tr.append(self.xgb.predict_proba(X_tr))
            except Exception as e:
                LOG.warning(f"[H={self.horizon}] XGB failed: {e}")
                base_val.append(np.full(len(X_va), 0.5))
                base_tr.append(np.full(len(X_tr), 0.5))
        else:
            base_val.append(np.full(len(X_va), 0.5))
            base_tr.append(np.full(len(X_tr), 0.5))
        try:
            self.extratrees.fit(X_tr, y_tr)
            base_val.append(self.extratrees.predict_proba(X_va))
            base_tr.append(self.extratrees.predict_proba(X_tr))
        except Exception:
            base_val.append(np.full(len(X_va), 0.5))
            base_tr.append(np.full(len(X_tr), 0.5))
        try:
            self.histgbm.fit(X_tr, y_tr)
            base_val.append(self.histgbm.predict_proba(X_va))
            base_tr.append(self.histgbm.predict_proba(X_tr))
        except Exception:
            base_val.append(np.full(len(X_va), 0.5))
            base_tr.append(np.full(len(X_tr), 0.5))

        X_meta_val = np.column_stack(base_val)
        X_meta_tr = np.column_stack(base_tr)
        try:
            self.meta.fit(X_meta_tr, y_tr)
            blended_val = self.meta.predict_proba(X_meta_val)
            blended_tr = self.meta.predict_proba(X_meta_tr)
        except Exception:
            blended_val = base_val[0]
            blended_tr = base_tr[0]

        if self.cfg.calibrate_probabilities:
            try:
                self.calibrator.fit(blended_val, y_va)
                blended_val_cal = self.calibrator.transform(blended_val)
            except Exception:
                blended_val_cal = blended_val
        else:
            blended_val_cal = blended_val

        self.metrics_ = self._compute_metrics(y_va, blended_val_cal,
                                              len(X_tr), len(X_va))

        try:
            self.uncertainty = BootstrapUncertainty(
                n_bootstrap=min(self.cfg.n_bootstrap, 30),
                random_state=self.cfg.random_state)
            self.uncertainty.fit(
                X_tr, y_tr,
                base_factory=lambda: ExtraTreesClassifier(
                    n_estimators=150, max_depth=6, n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state))
        except Exception:
            self.uncertainty = None

        self.is_fitted = True
        self.training_time_s = time.perf_counter() - t0
        m = self.metrics_
        LOG.info(f"[H={self.horizon}h] trained in {self.training_time_s:.1f}s | "
                 f"AUC={m['auc']:.4f} | Acc={m['accuracy']:.4f} | "
                 f"Brier={m['brier']:.4f} | n_train={m['n_train']}")
        return self

    def _compute_metrics(self, y, p, n_tr, n_va) -> Dict:
        if len(y) == 0 or len(np.unique(y)) < 2:
            return {k: 0.0 for k in
                    ["auc", "accuracy", "precision", "recall", "f1",
                     "brier", "logloss", "matthews", "hit_rate"]} | \
                   {"n_train": n_tr, "n_valid": n_va}
        labels = (p > 0.5).astype(int)
        tp = ((labels == 1) & (y == 1)).sum()
        fp = ((labels == 1) & (y == 0)).sum()
        fn = ((labels == 0) & (y == 1)).sum()
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = 0.0
        try:
            brier = brier_score_loss(y, p)
        except Exception:
            brier = 0.0
        try:
            ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        except Exception:
            ll = 0.0
        try:
            mcc = matthews_corrcoef(y, labels)
        except Exception:
            mcc = 0.0
        acc = accuracy_score(y, labels)
        hit = float((np.sign(p - 0.5) == np.sign(y - 0.5)).mean())
        return {
            "auc": float(auc), "accuracy": float(acc),
            "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "brier": float(brier), "logloss": float(ll),
            "matthews": float(mcc), "n_train": n_tr, "n_valid": n_va,
            "hit_rate": float(hit),
        }

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        X_df = panel[self.feature_names_]
        X_scaled = self.preprocessor.transform(X_df).values
        X_sel = self.selector.transform(X_scaled)
        preds = []
        if self.xgb is not None:
            try:
                preds.append(self.xgb.predict_proba(X_sel))
            except Exception:
                preds.append(np.full(X_sel.shape[0], 0.5))
        else:
            preds.append(np.full(X_sel.shape[0], 0.5))
        try:
            preds.append(self.extratrees.predict_proba(X_sel))
        except Exception:
            preds.append(np.full(X_sel.shape[0], 0.5))
        try:
            preds.append(self.histgbm.predict_proba(X_sel))
        except Exception:
            preds.append(np.full(X_sel.shape[0], 0.5))
        X_meta = np.column_stack(preds)
        try:
            blended = self.meta.predict_proba(X_meta)
        except Exception:
            blended = np.mean(preds, axis=0)
        if self.cfg.calibrate_probabilities:
            blended = self.calibrator.transform(blended)
        return blended

    def predict_uncertainty(self, panel: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.uncertainty is None:
            p = self.predict_proba(panel)
            return p, np.zeros_like(p)
        X_df = panel[self.feature_names_]
        X_scaled = self.preprocessor.transform(X_df).values
        X_sel = self.selector.transform(X_scaled)
        return self.uncertainty.predict_distribution(X_sel)

    def feature_importance(self) -> pd.Series:
        names = [self.feature_names_[i] for i in self.selector.selected_]
        agg = np.zeros(len(self.selector.selected_))
        count = 0
        for learner in [self.xgb, self.extratrees]:
            if learner is None:
                continue
            imp = learner.feature_importances()
            if imp is None:
                continue
            if len(imp) == len(self.selector.selected_):
                agg += imp
                count += 1
        if count:
            agg /= count
        return pd.Series(agg, index=names).sort_values(ascending=False)


class CryptoPredictor:
    def __init__(self, cfg: Optional[CryptoPredictorConfig] = None):
        self.cfg = cfg or CryptoPredictorConfig()
        _configure_logging(self.cfg.log_level)
        self.engine = CryptoFeatureEngine(self.cfg)
        self.ensembles: Dict[int, HorizonEnsemble] = {}
        self.online: Optional[OnlineLearner] = None
        self.drift_monitor = DriftMonitor(self.cfg)
        self.feature_set: Optional[CryptoFeatureSet] = None
        self.return_predictor: Optional[QuantileReturnPredictor] = None
        self.conformal: Optional[ConformalPredictor] = None
        self._fitted = False
        self.fit_timestamp: Optional[datetime] = None
        self.version = f"1.0.0-{uuid.uuid4().hex[:8]}"

    def fit(self, price_data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None) -> "CryptoPredictor":
        LOG.info(f"=== CryptoPredictor.fit() [{self.version}] ===")
        t0 = time.perf_counter()
        self.feature_set = self.engine.build(price_data, benchmark)
        panel = self.feature_set.panel
        feature_cols = self.feature_set.feature_cols

        feature_array = panel[feature_cols].fillna(0.0).values
        self.drift_monitor.set_reference(feature_array, feature_cols)

        for h in self.cfg.horizons_hours:
            target_col = f"target_{h}"
            if target_col not in panel.columns:
                LOG.warning(f"Target {target_col} missing")
                continue
            try:
                ens = HorizonEnsemble(self.cfg, horizon_hours=h)
                ens.fit(panel, feature_cols, target_col)
                self.ensembles[h] = ens
            except Exception as e:
                LOG.error(f"[H={h}h] Training failed: {e}")

        primary = self.cfg.primary_horizon_hours
        fwd_col = f"fwd_ret_{primary}"
        if fwd_col in panel.columns:
            df_ret = panel.dropna(subset=[fwd_col])
            if len(df_ret) >= 200:
                pre = Preprocessor(self.cfg)
                X_scaled = pre.fit_transform(
                    df_ret[feature_cols], feature_cols).values
                y_ret = df_ret[fwd_col].values
                try:
                    self.return_predictor = QuantileReturnPredictor(
                        quantiles=self.cfg.quantiles)
                    self.return_predictor.fit(X_scaled, y_ret)
                except Exception as e:
                    LOG.warning(f"Return predictor failed: {e}")
                    self.return_predictor = None

        if self.return_predictor is not None and fwd_col in panel.columns:
            try:
                df_ret = panel.dropna(subset=[fwd_col]).tail(400)
                if len(df_ret) > 50:
                    pre = Preprocessor(self.cfg)
                    Xs = pre.fit_transform(
                        df_ret[feature_cols], feature_cols).values
                    q_preds = self.return_predictor.predict(Xs)
                    y_true = df_ret[fwd_col].values
                    median_pred = q_preds.get(0.5, np.zeros(len(y_true)))
                    self.conformal = ConformalPredictor(
                        alpha=self.cfg.conformal_alpha).fit(y_true, median_pred)
            except Exception as e:
                LOG.debug(f"Conformal failed: {e}")

        if self.cfg.online_enabled:
            self.online = OnlineLearner(len(feature_cols), self.cfg)
            primary_ens = self.ensembles.get(primary)
            if primary_ens is not None:
                tail = panel.dropna(subset=[f"target_{primary}"]).tail(800)
                if len(tail) > 100:
                    X_tail = primary_ens.preprocessor.transform(
                        tail[primary_ens.feature_names_]).values
                    X_tail = primary_ens.selector.transform(X_tail)
                    y_tail = tail[f"target_{primary}"].values.astype(int)
                    for i in range(0, len(X_tail), 64):
                        self.online.partial_fit(
                            X_tail[i:i + 64], y_tail[i:i + 64])

        self._fitted = True
        self.fit_timestamp = datetime.now(timezone.utc)
        LOG.info(f"=== fit done in {time.perf_counter() - t0:.2f}s ===")
        return self

    def update(self, new_data: Dict[str, pd.DataFrame],
               benchmark: Optional[pd.DataFrame] = None) -> None:
        if not self._fitted or self.online is None:
            return
        try:
            new_fs = self.engine.build(new_data, benchmark)
        except Exception as e:
            LOG.warning(f"Update feature build failed: {e}")
            return
        primary = self.cfg.primary_horizon_hours
        ens = self.ensembles.get(primary)
        if ens is None:
            return
        df = new_fs.panel.dropna(subset=[f"target_{primary}"])
        if df.empty:
            return
        try:
            Xs = ens.preprocessor.transform(df[ens.feature_names_]).values
            Xs = ens.selector.transform(Xs)
        except Exception:
            return
        y = df[f"target_{primary}"].values.astype(int)
        for i in range(0, len(Xs), 64):
            self.online.partial_fit(Xs[i:i + 64], y[i:i + 64])
        LOG.info(f"Online update: {len(Xs)} new samples")

    def predict(self, price_data: Dict[str, pd.DataFrame],
                benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        fs = self.engine.build(price_data, benchmark)
        panel = fs.panel
        latest = (panel.reset_index()
                       .sort_values("timestamp")
                       .groupby("symbol").tail(1)
                       .set_index(["timestamp", "symbol"]))
        LOG.info(f"Predicting for {len(latest)} symbols")

        results = []
        regime_col = "regime" if "regime" in latest.columns else None

        for h, ens in self.ensembles.items():
            try:
                probs = ens.predict_proba(latest)
                means, stds = ens.predict_uncertainty(latest)
            except Exception as e:
                LOG.warning(f"[H={h}h] Predict failed: {e}")
                continue

            exp_ret = np.zeros(len(latest))
            lo_ret = np.zeros(len(latest))
            hi_ret = np.zeros(len(latest))
            if self.return_predictor is not None and h == self.cfg.primary_horizon_hours:
                try:
                    Xs = ens.preprocessor.transform(latest[ens.feature_names_]).values
                    Xs = ens.selector.transform(Xs)
                    q = self.return_predictor.predict(Xs)
                    exp_ret = q.get(0.5, np.zeros(len(latest)))
                    lo_ret = q.get(self.cfg.quantiles[0], exp_ret)
                    hi_ret = q.get(self.cfg.quantiles[-1], exp_ret)
                    if self.conformal is not None:
                        lo_ret, hi_ret = self.conformal.interval(exp_ret)
                except Exception:
                    pass

            for i, (idx, row) in enumerate(latest.iterrows()):
                sym = idx[1] if isinstance(idx, tuple) else "unknown"
                regime_val = int(row[regime_col]) if regime_col else -1
                conf = float(abs(probs[i] - 0.5) * 2) * (1.0 - min(0.9, stds[i]))
                results.append({
                    "symbol": sym,
                    "timestamp": idx[0] if isinstance(idx, tuple) else None,
                    "horizon_hours": h,
                    "probability_up": float(probs[i]),
                    "expected_return": float(exp_ret[i]),
                    "exp_return_lower": float(lo_ret[i]),
                    "exp_return_upper": float(hi_ret[i]),
                    "direction": int(1 if probs[i] > 0.5 else 0),
                    "confidence": float(conf),
                    "uncertainty": float(stds[i]),
                    "regime": REGIME_NAMES_CRYPTO.get(regime_val, "unknown"),
                })

        out = pd.DataFrame(results)
        if out.empty:
            return out

        if self.online is not None and self.cfg.primary_horizon_hours in self.ensembles:
            try:
                ens = self.ensembles[self.cfg.primary_horizon_hours]
                Xs = ens.preprocessor.transform(
                    latest[ens.feature_names_]).values
                Xs = ens.selector.transform(Xs)
                p_on = self.online.predict_proba(Xs)
                latest_syms = [i[1] if isinstance(i, tuple) else i
                               for i in latest.index]
                online_map = dict(zip(latest_syms, p_on))
                mask = out["horizon_hours"] == self.cfg.primary_horizon_hours
                out.loc[mask, "online_prob"] = out.loc[mask, "symbol"].map(online_map).values
                blended = (0.85 * out.loc[mask, "probability_up"].values +
                           0.15 * out.loc[mask, "online_prob"].fillna(0.5).values)
                out.loc[mask, "probability_up"] = blended
            except Exception as e:
                LOG.debug(f"Online overlay failed: {e}")
        return out.sort_values(["horizon_hours", "probability_up"],
                               ascending=[True, False])

    def explain(self, symbol: str, price_data: Dict[str, pd.DataFrame],
                horizon_hours: Optional[int] = None,
                top_k: int = 15) -> pd.DataFrame:
        h = horizon_hours or self.cfg.primary_horizon_hours
        if h not in self.ensembles:
            raise KeyError(f"No model for h={h}")
        ens = self.ensembles[h]
        fs = self.engine.build(price_data)
        sym_panel = fs.panel.xs(symbol, level="symbol")
        row = sym_panel.tail(1)
        Xs = ens.preprocessor.transform(row[ens.feature_names_]).values
        Xs = ens.selector.transform(Xs)
        p0 = ens.predict_proba(row)[0]
        names = [ens.feature_names_[i] for i in ens.selector.selected_]
        contribs = []
        rng = np.random.RandomState(0)
        for j in range(len(names)):
            X_pert = Xs.copy()
            col_std = Xs[:, j].std() if Xs.shape[0] > 1 else 1.0
            X_pert[:, j] = X_pert[:, j] + rng.randn(Xs.shape[0]) * (col_std + EPS)
            try:
                p1 = ens.predict_proba(row)[0]
            except Exception:
                continue
            contribs.append((names[j], float(p0 - p1)))
        df = pd.DataFrame(contribs, columns=["feature", "contribution"])
        df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index)
        return df.head(top_k)

    def metrics_report(self) -> pd.DataFrame:
        rows = []
        for h, ens in self.ensembles.items():
            if ens.metrics_ is None:
                continue
            m = ens.metrics_
            rows.append({
                "horizon_hours": h,
                "auc": m["auc"], "accuracy": m["accuracy"],
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "brier": m["brier"],
                "logloss": m["logloss"], "matthews": m["matthews"],
                "hit_rate": m["hit_rate"], "n_train": m["n_train"],
                "n_valid": m["n_valid"], "train_seconds": ens.training_time_s,
            })
        return pd.DataFrame(rows)

    def feature_importance_report(self, horizon_hours: Optional[int] = None,
                                  top_k: int = 30) -> pd.Series:
        h = horizon_hours or self.cfg.primary_horizon_hours
        if h not in self.ensembles:
            return pd.Series(dtype=float)
        return self.ensembles[h].feature_importance().head(top_k)

    def drift_report(self) -> pd.DataFrame:
        rows = []
        for i, entry in enumerate(self.drift_monitor.psi_history_):
            for name, val in entry.items():
                rows.append({"call": i, "feature": name, "psi": val})
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "version": self.version,
                "cfg": self.cfg,
                "ensembles": self.ensembles,
                "online": self.online,
                "feature_cols": self.feature_set.feature_cols
                                if self.feature_set else None,
                "fit_timestamp": self.fit_timestamp,
                "return_predictor": self.return_predictor,
                "conformal": self.conformal,
                "drift_monitor": self.drift_monitor,
            }, f)
        LOG.info(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CryptoPredictor":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        p = cls(blob["cfg"])
        p.version = blob["version"]
        p.ensembles = blob["ensembles"]
        p.online = blob["online"]
        p.fit_timestamp = blob["fit_timestamp"]
        p.return_predictor = blob.get("return_predictor")
        p.conformal = blob.get("conformal")
        p.drift_monitor = blob.get("drift_monitor", DriftMonitor(p.cfg))
        p._fitted = True
        LOG.info(f"Loaded {path} (v{p.version})")
        return p


class SyntheticCryptoMarket:
    def __init__(self, n_symbols: int = 12, n_hours: int = 6000,
                 random_state: int = 42):
        self.n_symbols = n_symbols
        self.n_hours = n_hours
        self.rs = np.random.RandomState(random_state)

    def _regime_path(self) -> np.ndarray:
        regimes = np.zeros(self.n_hours, dtype=int)
        n_switches = 12
        switch_points = np.sort(self.rs.choice(
            np.arange(200, self.n_hours - 200), size=n_switches, replace=False))
        current = self.rs.choice([0, 1, 2, 3, 4, 5, 6], p=[0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.10])
        cursor = 0
        for sp in switch_points:
            regimes[cursor:sp] = current
            current = self.rs.choice([0, 1, 2, 3, 4, 5, 6],
                                     p=[0.05, 0.15, 0.25, 0.25, 0.15, 0.05, 0.10])
            cursor = sp
        regimes[cursor:] = current
        return regimes

    def _regime_params(self, regime: int) -> Tuple[float, float]:
        return {
            0: (-0.00040, 0.030), 1: (-0.00015, 0.022), 2: (0.00000, 0.015),
            3: (0.00020, 0.018), 4: (0.00045, 0.025), 5: (-0.00150, 0.055),
            6: (0.00080, 0.030),
        }.get(regime, (0.0, 0.02))

    def generate(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        regimes = self._regime_path()
        vol_factor = np.zeros(self.n_hours)
        vol_factor[0] = 1.0
        omega, alpha, beta = 0.00001, 0.10, 0.87
        for t in range(1, self.n_hours):
            vol_factor[t] = np.sqrt(omega + alpha * vol_factor[t-1]**2 +
                                    beta * vol_factor[t-1]**2)

        market_ret = np.zeros(self.n_hours)
        for t in range(self.n_hours):
            drift, vol = self._regime_params(regimes[t])
            base_vol = vol * vol_factor[t]
            innov = student_t.rvs(df=5, random_state=self.rs) / np.sqrt(5/3)
            if self.rs.rand() < 0.002:
                innov += self.rs.randn() * 0.05
            market_ret[t] = drift + base_vol * innov

        market_price = 30000 * np.exp(np.cumsum(market_ret))

        start = pd.Timestamp("2023-01-01", tz="UTC")
        ts = pd.date_range(start, periods=self.n_hours, freq="h")

        season = 1.0 + 0.15 * np.sin(2 * np.pi * ts.hour / 24)

        prices: Dict[str, pd.DataFrame] = {}
        for i in range(self.n_symbols):
            beta = 0.5 + self.rs.rand() * 1.8
            idio_vol = 0.010 + self.rs.rand() * 0.030
            idio = self.rs.randn(self.n_hours) * idio_vol * season
            cycle = 0.0003 * np.sin(2 * np.pi * np.arange(self.n_hours) /
                                    (150 + 50 * (i % 5)))
            ret = beta * market_ret + idio + cycle
            base_price = 100 + 500 * self.rs.rand()
            price = base_price * np.exp(np.cumsum(ret))
            high = price * (1 + np.abs(self.rs.randn(self.n_hours)) * 0.003)
            low = price * (1 - np.abs(self.rs.randn(self.n_hours)) * 0.003)
            open_ = price * (1 + self.rs.randn(self.n_hours) * 0.002)
            base_vol = np.abs(self.rs.randn(self.n_hours)) * 1e6 + 5e5
            volume = base_vol * (1 + 0.5 * (regimes != 2).astype(float))
            volume = pd.Series(volume).ewm(alpha=0.3).mean().values
            df = pd.DataFrame({
                "open": open_,
                "high": np.maximum.reduce([open_, price, high]),
                "low": np.minimum.reduce([open_, price, low]),
                "close": price,
                "volume": volume,
            }, index=ts)
            sym = f"COIN{i:02d}USDT"
            prices[sym] = df

        bench_df = pd.DataFrame({
            "open": market_price,
            "high": market_price * 1.004,
            "low": market_price * 0.996,
            "close": market_price,
            "volume": np.abs(self.rs.randn(self.n_hours)) * 1e8 + 5e7,
        }, index=ts)

        return prices, bench_df


class CryptoDiagnostics:
    @staticmethod
    def information_coefficient(predictions: pd.Series,
                                forwards: pd.Series) -> float:
        df = pd.concat([predictions, forwards], axis=1).dropna()
        if len(df) < 20:
            return 0.0
        return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))

    @staticmethod
    def decile_analysis(scores: pd.Series, forwards: pd.Series,
                        n_deciles: int = 10) -> pd.DataFrame:
        df = pd.concat([scores, forwards], axis=1).dropna()
        df.columns = ["score", "fwd"]
        if len(df) < n_deciles * 3:
            return pd.DataFrame()
        try:
            df["decile"] = pd.qcut(df["score"], n_deciles, labels=False,
                                   duplicates="drop")
        except Exception:
            return pd.DataFrame()
        g = df.groupby("decile")["fwd"].agg(["mean", "std", "count"])
        g["sharpe"] = g["mean"] / (g["std"] + EPS)
        return g

    @staticmethod
    def monte_carlo_sharpe(returns: np.ndarray, n_paths: int = 1000,
                           random_state: int = 42) -> Dict[str, float]:
        rng = np.random.RandomState(random_state)
        n = len(returns)
        if n < 30:
            return {}
        sharpes = []
        for _ in range(n_paths):
            idx = rng.choice(n, size=n, replace=True)
            r = returns[idx]
            if r.std() < EPS:
                continue
            sharpes.append(r.mean() / r.std() * np.sqrt(HOURS_PER_YEAR))
        if not sharpes:
            return {}
        return {
            "mean": float(np.mean(sharpes)),
            "median": float(np.median(sharpes)),
            "std": float(np.std(sharpes)),
            "p05": float(np.percentile(sharpes, 5)),
            "p95": float(np.percentile(sharpes, 95)),
        }

    @staticmethod
    def max_drawdown(equity: pd.Series) -> float:
        return float((equity / equity.cummax() - 1).min())

    @staticmethod
    def value_at_risk(rets: pd.Series, confidence: float = 0.95) -> float:
        return float(np.percentile(rets.dropna(), (1 - confidence) * 100))

    @staticmethod
    def conditional_var(rets: pd.Series, confidence: float = 0.95) -> float:
        var = CryptoDiagnostics.value_at_risk(rets, confidence)
        return float(rets[rets <= var].mean())

    @staticmethod
    def tail_ratio(rets: pd.Series) -> float:
        q = rets.quantile([0.95, 0.05])
        return float(abs(q.iloc[0]) / (abs(q.iloc[1]) + EPS))


class CryptoSignalFusion:
    def __init__(self, horizon_weights: Optional[Dict[int, float]] = None):
        self.horizon_weights = horizon_weights or {
            1: 0.05, 4: 0.10, 12: 0.15, 24: 0.25, 72: 0.20, 168: 0.15, 720: 0.10,
        }

    def fuse(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return predictions
        rows = []
        for sym, group in predictions.groupby("symbol"):
            total_w = 0.0
            weighted_p = 0.0
            weighted_r = 0.0
            weighted_c = 0.0
            for _, row in group.iterrows():
                h = int(row["horizon_hours"])
                w = self.horizon_weights.get(h, 0.0)
                if w <= 0:
                    continue
                total_w += w
                weighted_p += w * row["probability_up"]
                weighted_r += w * row["expected_return"]
                weighted_c += w * row["confidence"]
            if total_w > 0:
                rows.append({
                    "symbol": sym,
                    "fused_prob": weighted_p / total_w,
                    "fused_expected_return": weighted_r / total_w,
                    "fused_confidence": weighted_c / total_w,
                })
        fused = pd.DataFrame(rows)
        if not fused.empty:
            fused = fused.sort_values("fused_prob", ascending=False)
        return fused


class CryptoBacktest:
    def __init__(self, cost_bps: float = 10.0, funding_bps_per_8h: float = 1.0):
        self.cost_bps = cost_bps
        self.funding_bps = funding_bps_per_8h

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame,
            top_k: int = 5, long_short: bool = True) -> pd.DataFrame:
        signals = signals.reindex(prices.index).fillna(0.0)
        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for dt in signals.index:
            row = signals.loc[dt].sort_values(ascending=False)
            longs = row.head(top_k).index
            shorts = row.tail(top_k).index if long_short else []
            for s in longs:
                positions.loc[dt, s] = 1.0 / max(1, len(longs))
            for s in shorts:
                positions.loc[dt, s] = -1.0 / max(1, len(shorts))
        rets = prices.pct_change().fillna(0.0)
        pnl = (positions.shift(1) * rets).sum(axis=1)
        turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover * self.cost_bps / 1e4
        bars_per_8h = 8
        funding = positions.abs().sum(axis=1) * self.funding_bps / 1e4 / bars_per_8h
        pnl_net = pnl - cost - funding
        equity = (1 + pnl_net).cumprod()
        return pd.DataFrame({
            "pnl_gross": pnl, "cost": cost, "funding": funding,
            "pnl_net": pnl_net, "equity": equity, "turnover": turnover,
        })


def _self_test() -> None:
    _configure_logging("INFO")
    LOG.info("=" * 72)
    LOG.info("CRYPTO_PREDICTOR.PY SELF-TEST")
    LOG.info("=" * 72)

    gen = SyntheticCryptoMarket(n_symbols=10, n_hours=5500, random_state=11)
    prices, benchmark = gen.generate()
    LOG.info(f"Generated {len(prices)} crypto pairs x "
             f"{len(next(iter(prices.values())))} hours")

    cfg = CryptoPredictorConfig(
        horizons_hours=(4, 24, 72, 168),
        primary_horizon_hours=24,
        target_type=CryptoPredictionTarget.DIRECTION,
        min_history=800,
        n_bootstrap=15,
        xgb_n_estimators=400,
        xgb_max_depth=5,
        feature_selection_enabled=True,
        max_features=100,
        log_level="INFO",
    )

    predictor = CryptoPredictor(cfg)
    predictor.fit(prices, benchmark)

    LOG.info("\n--- MODEL METRICS ---")
    metrics_df = predictor.metrics_report()
    LOG.info("\n" + metrics_df.to_string(index=False))

    imp = predictor.feature_importance_report(top_k=20)
    LOG.info("\n--- TOP-20 FEATURES (primary horizon) ---")
    LOG.info("\n" + imp.to_string())

    preds = predictor.predict(prices, benchmark)
    LOG.info("\n--- SAMPLE PREDICTIONS ---")
    if not preds.empty:
        sample = preds.head(20)[[
            "symbol", "horizon_hours", "probability_up",
            "expected_return", "confidence", "regime",
        ]]
        LOG.info("\n" + sample.to_string(index=False))

    fusion = CryptoSignalFusion()
    fused = fusion.fuse(preds)
    LOG.info("\n--- FUSED SIGNALS (top 10) ---")
    if not fused.empty:
        LOG.info("\n" + fused.head(10).to_string(index=False))

    sym = list(prices.keys())[0]
    try:
        expl = predictor.explain(sym, prices, horizon_hours=cfg.primary_horizon_hours,
                                 top_k=10)
        LOG.info(f"\n--- EXPLANATION FOR {sym} ---")
        LOG.info("\n" + expl.to_string(index=False))
    except Exception as e:
        LOG.warning(f"Explain failed: {e}")

    drift = predictor.drift_report()
    LOG.info(f"\n--- DRIFT REPORT (last call) ---")
    if not drift.empty:
        last = drift[drift["call"] == drift["call"].max()].sort_values(
            "psi", ascending=False).head(10)
        LOG.info("\n" + last.to_string(index=False))

    try:
        if not preds.empty:
            primary_preds = preds[
                preds["horizon_hours"] == cfg.primary_horizon_hours].set_index("symbol")
            forward_map = {}
            for s, d in prices.items():
                if len(d) >= cfg.primary_horizon_hours:
                    fwd = (d["close"].iloc[-1] /
                           d["close"].iloc[-cfg.primary_horizon_hours - 1] - 1)
                    forward_map[s] = fwd
            fwd_series = pd.Series(forward_map)
            ic = CryptoDiagnostics.information_coefficient(
                primary_preds["expected_return"], fwd_series)
            LOG.info(f"\n--- IC (expected vs realized) = {ic:.4f} ---")
            decile = CryptoDiagnostics.decile_analysis(
                primary_preds["probability_up"], fwd_series)
            if not decile.empty:
                LOG.info("\n--- DECILE ANALYSIS ---")
                LOG.info("\n" + decile.to_string())
    except Exception as e:
        LOG.warning(f"Diagnostics failed: {e}")

    try:
        p = Path("./crypto_predictor_selftest.pkl")
        predictor.save(str(p))
        loaded = CryptoPredictor.load(str(p))
        preds2 = loaded.predict(prices, benchmark)
        LOG.info(f"\n--- SAVE/LOAD round-trip OK: {len(preds2)} preds ---")
        p.unlink(missing_ok=True)
    except Exception as e:
        LOG.warning(f"Save/load failed: {e}")

    try:
        predictor.update(prices, benchmark)
        LOG.info("\n--- ONLINE UPDATE OK ---")
    except Exception as e:
        LOG.warning(f"Update failed: {e}")

    try:
        primary = cfg.primary_horizon_hours
        ens = predictor.ensembles.get(primary)
        if ens is not None:
            fs = predictor.engine.build(prices, benchmark)
            panel = fs.panel
            recent = panel.tail(3000)
            if len(recent) > 200:
                probs = ens.predict_proba(recent)
                scores = pd.DataFrame({
                    "timestamp": recent.index.get_level_values("timestamp"),
                    "symbol": recent.index.get_level_values("symbol"),
                    "score": probs - 0.5,
                }).pivot(index="timestamp", columns="symbol", values="score")
                close_prices = recent["close"].unstack("symbol")
                bt = CryptoBacktest(cost_bps=10.0, funding_bps_per_8h=1.0)
                results = bt.run(scores, close_prices, top_k=3, long_short=True)
                if len(results) > 10:
                    total = results["pnl_net"].sum()
                    sharpe = (results["pnl_net"].mean() /
                              (results["pnl_net"].std() + EPS) *
                              np.sqrt(HOURS_PER_YEAR))
                    mdd = CryptoDiagnostics.max_drawdown(results["equity"])
                    LOG.info(f"\n--- BACKTEST SUMMARY ---")
                    LOG.info(f"Total return: {total:.4f}")
                    LOG.info(f"Sharpe (hourly -> annual): {sharpe:.3f}")
                    LOG.info(f"Max drawdown: {mdd:.2%}")
                    var95 = CryptoDiagnostics.value_at_risk(results["pnl_net"], 0.95)
                    cvar95 = CryptoDiagnostics.conditional_var(results["pnl_net"], 0.95)
                    LOG.info(f"VaR 95%: {var95:.4f} | CVaR 95%: {cvar95:.4f}")
                    mc = CryptoDiagnostics.monte_carlo_sharpe(
                        results["pnl_net"].values, n_paths=500)
                    if mc:
                        LOG.info(f"MC Sharpe 5%/95%: {mc['p05']:.3f} / {mc['p95']:.3f}")
    except Exception as e:
        LOG.warning(f"Backtest failed: {e}")

    LOG.info("=" * 72)
    LOG.info("SELF-TEST COMPLETE")
    LOG.info("=" * 72)


if __name__ == "__main__":
    _self_test()
