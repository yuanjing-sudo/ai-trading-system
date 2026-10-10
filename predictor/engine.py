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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence,
    Set, Tuple, Union, NamedTuple,
)
from collections import defaultdict, deque, OrderedDict
from pathlib import Path
from enum import Enum, IntEnum, auto

import numpy as np
import pandas as pd
from scipy import stats, signal as sp_signal, optimize as sp_optimize
from scipy.stats import norm, t as student_t, skew, kurtosis

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import (
    Ridge, LogisticRegression, SGDClassifier, HuberRegressor,
    QuantileRegressor, BayesianRidge,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score,
    precision_recall_curve, average_precision_score, matthews_corrcoef,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LOG = logging.getLogger("predictor")


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


class PredictionTarget(Enum):
    DIRECTION = "direction"
    RETURN = "return"
    EXCESS_RETURN = "excess"
    VOLATILITY = "volatility"
    RANK = "rank"
    TRIPLE_BARRIER = "triple"


class RegimeLabel(IntEnum):
    CRISIS = 0
    HIGH_VOL = 1
    TRENDING_DOWN = 2
    RANGING = 3
    TRENDING_UP = 4
    LOW_VOL = 5


REGIME_NAMES: Dict[int, str] = {
    0: "crisis",
    1: "high_vol",
    2: "trending_down",
    3: "ranging",
    4: "trending_up",
    5: "low_vol",
}

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 5, 10, 20, 60)
DEFAULT_MA_WINDOWS: Tuple[int, ...] = (5, 10, 20, 50, 100, 200)
DEFAULT_VOL_WINDOWS: Tuple[int, ...] = (5, 10, 20, 60, 120)
DEFAULT_RSI_WINDOWS: Tuple[int, ...] = (7, 14, 21)
DEFAULT_ATR_WINDOWS: Tuple[int, ...] = (7, 14, 21)
EPS = 1e-9


def safe_div(a, b, fill: float = 0.0):
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b_arr) > EPS, a_arr / b_arr, fill)
    if isinstance(a, pd.Series):
        return pd.Series(out, index=a.index)
    return out if out.shape else float(out)


def zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    q1 = np.nanpercentile(x, 25, axis=axis, keepdims=True)
    q3 = np.nanpercentile(x, 75, axis=axis, keepdims=True)
    iqr = q3 - q1
    scale = np.where(iqr > EPS, iqr, 1.0)
    return (x - med) / scale


def cross_section_rank(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    return df.rank(axis=1, pct=True, ascending=ascending) - 0.5


def ewma_volatility(returns: pd.Series, halflife: int = 20) -> pd.Series:
    return returns.ewm(halflife=halflife, adjust=False).std() * np.sqrt(252)


def rolling_entropy(series: pd.Series, window: int = 20, bins: int = 10) -> pd.Series:
    def ent(x):
        if len(x) < 3:
            return np.nan
        h, _ = np.histogram(x, bins=bins)
        p = h / (h.sum() + EPS)
        p = p[p > 0]
        return -np.sum(p * np.log(p + EPS))
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
        slope, _ = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[:2]
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
        out[i] = np.dot(weights, window)
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
    a_counts, _ = np.histogram(a, bins=edges)
    e_counts, _ = np.histogram(e, bins=edges)
    a_pct = a_counts / (a_counts.sum() + EPS)
    e_pct = e_counts / (e_counts.sum() + EPS)
    a_pct = np.where(a_pct == 0, 1e-4, a_pct)
    e_pct = np.where(e_pct == 0, 1e-4, e_pct)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def timed(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        LOG.debug(f"{func.__qualname__} took {dt*1000:.2f} ms")
        return result
    return wrapper


def memoize_on_first_arg(func: Callable) -> Callable:
    cache: Dict[Any, Any] = {}

    @functools.wraps(func)
    def wrapper(first, *args, **kwargs):
        key = (id(first), args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(first, *args, **kwargs)
        return cache[key]
    return wrapper


@dataclass
class PredictorConfig:
    horizons: Tuple[int, ...] = DEFAULT_HORIZONS
    primary_horizon: int = 5
    target_type: PredictionTarget = PredictionTarget.DIRECTION
    return_threshold: float = 0.0
    benchmark_symbol: Optional[str] = None

    ma_windows: Tuple[int, ...] = DEFAULT_MA_WINDOWS
    vol_windows: Tuple[int, ...] = DEFAULT_VOL_WINDOWS
    rsi_windows: Tuple[int, ...] = DEFAULT_RSI_WINDOWS
    atr_windows: Tuple[int, ...] = DEFAULT_ATR_WINDOWS
    macd_params: Tuple[int, int, int] = (12, 26, 9)
    bb_window: int = 20
    bb_std: float = 2.0
    use_fractional_diff: bool = True
    fractional_d: float = 0.4
    use_kalman: bool = True
    kalman_process_var: float = 1e-4
    kalman_obs_var: float = 1e-3
    use_cross_sectional: bool = True
    use_regime_features: bool = True
    use_pattern_features: bool = True
    use_volume_features: bool = True
    use_entropy_features: bool = True
    use_hurst: bool = True

    winsorize_quantile: float = 0.01
    feature_clip_sigma: float = 5.0
    min_history: int = 260

    n_splits: int = 6
    embargo_days: int = 5
    purge_days: int = 5
    train_window: int = 504
    refit_every: int = 20
    random_state: int = 42
    n_jobs: int = -1

    xgb_n_estimators: int = 1200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.025
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.85
    xgb_min_child_weight: int = 5
    xgb_gamma: float = 0.1
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.5
    xgb_early_stop: int = 80

    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgb": 0.45, "extratrees": 0.20, "histgbm": 0.20,
        "ridge_meta": 0.15,
    })

    n_bootstrap: int = 40
    conformal_alpha: float = 0.10
    quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    online_enabled: bool = True
    online_alpha: float = 1e-4
    online_learning_rate: str = "optimal"

    feature_selection_enabled: bool = True
    max_features: int = 120
    min_mutual_info: float = 1e-4

    drift_psi_threshold: float = 0.25
    log_level: str = "INFO"

    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"


@dataclass
class PredictionResult:
    symbol: str
    as_of: pd.Timestamp
    horizon: int
    probability_up: float
    expected_return: float
    expected_return_lower: float
    expected_return_upper: float
    direction: int
    confidence: float
    regime: str
    feature_contributions: Dict[str, float]
    model_agreement: float
    uncertainty: float


@dataclass
class ModelMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    logloss: float
    matthews: float
    n_train: int
    n_valid: int
    hit_rate: float


@dataclass
class FeatureSet:
    panel: pd.DataFrame
    feature_cols: List[str]
    target_cols: Dict[int, str]
    symbol_index: List[str]
    date_index: List[pd.Timestamp]


class TechnicalIndicators:
    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(2, window // 2)).mean()

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()

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
        macd = ef - es
        sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        hist = macd - sig
        return macd, sig, hist

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
        atr_v = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_v
        minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_v
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
        return dx.ewm(alpha=1 / window, adjust=False).mean()

    @staticmethod
    def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3,
                   smooth: int = 3):
        low_k = df["low"].rolling(k, min_periods=k).min()
        high_k = df["high"].rolling(k, min_periods=k).max()
        k_pct = 100 * (df["close"] - low_k) / (high_k - low_k + EPS)
        k_smooth = k_pct.rolling(smooth).mean()
        return k_smooth, k_smooth.rolling(d).mean()

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
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
              (df["high"] - df["low"] + EPS)
        mfv = mfm * df["volume"]
        return mfv.rolling(window, min_periods=window).sum() / \
               (df["volume"].rolling(window, min_periods=window).sum() + EPS)

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


class KalmanFilter1D:
    def __init__(self, process_var: float = 1e-4, obs_var: float = 1e-3):
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
            self.P = np.eye(2) * 1.0
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
            if np.isnan(v):
                levels.append(np.nan)
                slopes.append(np.nan)
                continue
            lv, sl = self.update(v)
            levels.append(lv)
            slopes.append(sl)
        return (pd.Series(levels, index=series.index),
                pd.Series(slopes, index=series.index))


class RegimeDetector:
    def __init__(self, vol_window: int = 20, trend_window: int = 60,
                 vol_thresholds: Tuple[float, float, float] = (0.10, 0.22, 0.40),
                 trend_threshold: float = 0.05):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_low, self.vol_mid, self.vol_high = vol_thresholds
        self.trend_threshold = trend_threshold

    def classify_single(self, prices: pd.Series) -> int:
        if len(prices) < self.trend_window + 1:
            return int(RegimeLabel.RANGING)
        rets = prices.pct_change().dropna()
        if len(rets) < self.vol_window:
            return int(RegimeLabel.RANGING)
        vol = float(rets.tail(self.vol_window).std() * np.sqrt(252))
        trend = float(prices.iloc[-1] / prices.iloc[-self.trend_window] - 1)
        return self._classify(vol, trend)

    def _classify(self, vol: float, trend: float) -> int:
        if vol > self.vol_high:
            return int(RegimeLabel.CRISIS)
        if vol > self.vol_mid:
            if trend < -self.trend_threshold:
                return int(RegimeLabel.TRENDING_DOWN)
            return int(RegimeLabel.HIGH_VOL)
        if vol < self.vol_low:
            return int(RegimeLabel.LOW_VOL)
        if trend > self.trend_threshold:
            return int(RegimeLabel.TRENDING_UP)
        if trend < -self.trend_threshold:
            return int(RegimeLabel.TRENDING_DOWN)
        return int(RegimeLabel.RANGING)

    def classify_universe(self, prices: pd.DataFrame) -> pd.Series:
        rets = prices.pct_change()
        uni_vol = rets.rolling(self.vol_window).std().median(axis=1) * np.sqrt(252)
        idx_trend = prices.mean(axis=1)
        trend = idx_trend / idx_trend.shift(self.trend_window) - 1
        regimes = []
        for v, t in zip(uni_vol.values, trend.values):
            if np.isnan(v) or np.isnan(t):
                regimes.append(int(RegimeLabel.RANGING))
            else:
                regimes.append(self._classify(float(v), float(t)))
        return pd.Series(regimes, index=prices.index)

    @staticmethod
    def regime_risk_multiplier(regime: int) -> float:
        return {
            int(RegimeLabel.CRISIS): 0.35,
            int(RegimeLabel.HIGH_VOL): 0.65,
            int(RegimeLabel.TRENDING_DOWN): 0.85,
            int(RegimeLabel.RANGING): 0.95,
            int(RegimeLabel.TRENDING_UP): 1.10,
            int(RegimeLabel.LOW_VOL): 1.15,
        }.get(regime, 1.0)


class FeatureEngine:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.ti = TechnicalIndicators()
        self.kf = KalmanFilter1D(cfg.kalman_process_var, cfg.kalman_obs_var)
        self.feature_cols_: List[str] = []

    def build(self, price_data: Dict[str, pd.DataFrame],
              benchmark: Optional[pd.DataFrame] = None) -> FeatureSet:
        LOG.info(f"Building features for {len(price_data)} symbols")
        frames = []
        for sym, df in price_data.items():
            if df is None or df.empty or len(df) < self.cfg.min_history:
                LOG.debug(f"Skipping {sym}: insufficient history ({0 if df is None else len(df)})")
                continue
            try:
                f = self._build_per_symbol(df, sym)
                frames.append(f)
            except Exception as e:
                LOG.warning(f"Feature build failed for {sym}: {e}")
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
        target_cols = {h: f"target_{h}" for h in self.cfg.horizons}
        dates = panel.index.get_level_values("date").unique().tolist()
        symbols = panel.index.get_level_values("symbol").unique().tolist()
        LOG.info(f"Feature panel: {panel.shape[0]} rows, "
                 f"{len(self.feature_cols_)} features, "
                 f"{len(symbols)} symbols, {len(dates)} dates")
        return FeatureSet(panel, self.feature_cols_, target_cols, symbols, dates)

    def _build_per_symbol(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = df.copy().sort_index()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(-1)
        d.columns = [str(c).lower().replace(" ", "_") for c in d.columns]
        rename = {"adj_close": "close", "adj close": "close"}
        d = d.rename(columns=rename)
        needed = ["open", "high", "low", "close", "volume"]
        for col in needed:
            if col not in d.columns:
                d[col] = np.nan
        d = d.loc[:, ~d.columns.duplicated(keep="first")]
        d = d[needed].astype(float)
        d.index = pd.to_datetime(d.index)
        d = d[~d.index.duplicated(keep="last")].sort_index()
        d = d.dropna(subset=["close"])
        if "close" not in d.columns:
            raise ValueError(f"{symbol}: no 'close' column")
        d = pd.DataFrame({
            "open": pd.Series(np.asarray(d["open"].values, dtype=float), index=d.index),
            "high": pd.Series(np.asarray(d["high"].values, dtype=float), index=d.index),
            "low": pd.Series(np.asarray(d["low"].values, dtype=float), index=d.index),
            "close": pd.Series(np.asarray(d["close"].values, dtype=float), index=d.index),
            "volume": pd.Series(np.asarray(d["volume"].values, dtype=float), index=d.index),
        })
        close = d["close"]
        feat = pd.DataFrame(index=d.index)
        feat["symbol"] = symbol
        feat["close"] = close
        feat["volume"] = d.get("volume", pd.Series(0.0, index=d.index))
        feat["returns_1"] = close.pct_change()
        feat["log_returns_1"] = np.log(close / close.shift(1))

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

        for w in (2, 3, 5, 10, 20, 60, 120):
            feat[f"ret_{w}"] = close.pct_change(w)
            feat[f"logret_{w}"] = np.log(close / close.shift(w))

        for w in self.cfg.vol_windows:
            feat[f"vol_{w}"] = feat["log_returns_1"].rolling(w).std() * np.sqrt(252)
            feat[f"realized_vol_{w}"] = np.sqrt((feat["log_returns_1"] ** 2).rolling(w).sum()) * np.sqrt(252)
        for (a, b) in itertools.combinations(self.cfg.vol_windows, 2):
            feat[f"vol_ratio_{a}_{b}"] = safe_div(feat[f"vol_{a}"], feat[f"vol_{b}"], np.nan)

        for hl in (10, 20, 60):
            feat[f"ewma_vol_{hl}"] = ewma_volatility(feat["returns_1"], hl)

        for w in self.cfg.ma_windows:
            ma = self.ti.sma(close, w)
            ema = self.ti.ema(close, w)
            feat[f"sma_{w}"] = ma
            feat[f"ema_{w}"] = ema
            feat[f"dist_sma_{w}"] = (close - ma) / (ma.abs() + EPS)
            feat[f"dist_ema_{w}"] = (close - ema) / (ema.abs() + EPS)
            feat[f"slope_sma_{w}"] = ma.diff(5) / (ma.abs() + EPS)
        for (a, b) in [(5, 20), (10, 50), (20, 50), (50, 200)]:
            if a in self.cfg.ma_windows and b in self.cfg.ma_windows:
                ma_a = feat[f"sma_{a}"]
                ma_b = feat[f"sma_{b}"]
                feat[f"cross_{a}_{b}"] = (ma_a - ma_b) / (ma_b.abs() + EPS)

        for w in self.cfg.rsi_windows:
            rsi_v = self.ti.rsi(close, w)
            feat[f"rsi_{w}"] = rsi_v
            feat[f"rsi_slope_{w}"] = rsi_v.diff(5)

        f_, s_, sig_ = self.cfg.macd_params
        macd_line, macd_sig, macd_hist = self.ti.macd(close, f_, s_, sig_)
        feat["macd"] = macd_line / (close + EPS)
        feat["macd_signal"] = macd_sig / (close + EPS)
        feat["macd_hist"] = macd_hist / (close + EPS)
        feat["macd_slope"] = macd_line.diff(3) / (close + EPS)

        mid, up, lo = self.ti.bollinger(close, self.cfg.bb_window, self.cfg.bb_std)
        feat["bb_position"] = (close - lo) / (up - lo + EPS)
        feat["bb_width"] = (up - lo) / (mid.abs() + EPS)
        feat["bb_squeeze"] = feat["bb_width"] / (feat["bb_width"].rolling(60).mean() + EPS)

        for w in self.cfg.atr_windows:
            atr_v = self.ti.atr(d, w)
            feat[f"atr_{w}"] = atr_v / (close + EPS)
            feat[f"atr_pct_{w}"] = atr_v / (close + EPS)
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

        if self.cfg.use_volume_features and "volume" in d.columns:
            vol = d["volume"].astype(float)
            for w in (5, 10, 20, 60):
                vma = vol.rolling(w, min_periods=w // 2).mean()
                feat[f"vol_ratio_{w}"] = vol / (vma + EPS)
                feat[f"dollar_vol_{w}"] = (vol * close).rolling(w).mean()
            obv = self.ti.obv(d)
            feat["obv_z"] = (obv - obv.rolling(60).mean()) / (obv.rolling(60).std() + EPS)
            feat["obv_slope"] = obv.diff(10) / (obv.abs().rolling(60).mean() + EPS)
            feat["mfi_14"] = self.ti.mfi(d, 14)
            feat["cmf_20"] = self.ti.cmf(d, 20)
            feat["vwap_20"] = self.ti.vwap(d, 20) / (close + EPS)

        rng = (d["high"] - d["low"]).replace(0, np.nan)
        feat["upper_wick"] = (d["high"] - np.maximum(d["open"], close)) / (rng + EPS)
        feat["lower_wick"] = (np.minimum(d["open"], close) - d["low"]) / (rng + EPS)
        feat["body_pct"] = (close - d["open"]) / (rng + EPS)
        feat["gap"] = (d["open"] - close.shift(1)) / (close.shift(1).abs() + EPS)
        feat["range_pct"] = rng / (close + EPS)

        for w in (20, 60, 120):
            feat[f"skew_{w}"] = feat["log_returns_1"].rolling(w).skew()
            feat[f"kurt_{w}"] = feat["log_returns_1"].rolling(w).kurt()
            feat[f"up_days_{w}"] = (feat["returns_1"] > 0).rolling(w).mean()
            feat[f"ret_std_{w}"] = feat["returns_1"].rolling(w).std()
            feat[f"ret_mean_{w}"] = feat["returns_1"].rolling(w).mean()

        if self.cfg.use_entropy_features:
            feat["entropy_20"] = rolling_entropy(feat["log_returns_1"], 20)
            feat["entropy_60"] = rolling_entropy(feat["log_returns_1"], 60)

        if self.cfg.use_hurst:
            feat["hurst_60"] = feat["log_returns_1"].rolling(60).apply(
                lambda x: hurst_exponent(x, max_lag=15), raw=True)
            feat["hurst_120"] = feat["log_returns_1"].rolling(120).apply(
                lambda x: hurst_exponent(x, max_lag=20), raw=True)

        running_max = close.cummax()
        feat["dd_from_peak"] = (close / running_max) - 1
        feat["time_since_peak"] = (
            (close == running_max).astype(int).groupby(
                (close == running_max).cumsum()).cumcount()
        )

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

        feat["price_volume_corr_20"] = close.pct_change().rolling(20).corr(
            feat["volume"].pct_change())
        feat["price_volume_corr_60"] = close.pct_change().rolling(60).corr(
            feat["volume"].pct_change())

        return feat

    def _add_cross_sectional(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        rank_candidates = [
            "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "macd_hist",
            "atr_14", "adx_14", "vol_ratio_20", "mfi_14", "bb_position",
            "kalman_slope", "hurst_60", "entropy_20",
        ]
        if "date" not in panel.index.names:
            panel = panel.reset_index().rename(columns={"index": "date"})
            if "date" not in panel.columns:
                panel["date"] = panel.index if hasattr(panel.index, "date") else panel.iloc[:, 0]
            panel = panel.set_index(["date", "symbol"])

        by_date = panel.groupby(level="date", group_keys=False)
        for col in rank_candidates:
            if col in panel.columns:
                try:
                    ranked = by_date[col].rank(pct=True) - 0.5
                    panel[f"cs_rank_{col}"] = ranked
                    panel[f"cs_z_{col}"] = by_date[col].transform(
                        lambda s: (s - s.mean()) / (s.std() + EPS))
                except Exception:
                    continue
        return panel

    def _add_benchmark_relative(self, panel: pd.DataFrame,
                                benchmark: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        bench_ret = benchmark["close"].pct_change().rename("bench_ret")
        if not isinstance(panel.index.get_level_values("date"), pd.DatetimeIndex):
            try:
                dates = pd.to_datetime(panel.index.get_level_values("date"))
                panel.index = pd.MultiIndex.from_arrays(
                    [dates, panel.index.get_level_values("symbol")],
                    names=["date", "symbol"])
            except Exception:
                pass
        rets_df = panel["returns_1"].unstack("symbol")
        excess = rets_df.sub(bench_ret, axis=0)
        panel["excess_ret_1"] = excess.stack().reindex(panel.index)
        try:
            beta_60 = rets_df.rolling(60).cov(bench_ret).div(
                bench_ret.rolling(60).var() + EPS, axis=0)
            panel["beta_60"] = beta_60.stack().reindex(panel.index)
            beta_120 = rets_df.rolling(120).cov(bench_ret).div(
                bench_ret.rolling(120).var() + EPS, axis=0)
            panel["beta_120"] = beta_120.stack().reindex(panel.index)
        except Exception:
            panel["beta_60"] = np.nan
            panel["beta_120"] = np.nan
        try:
            corr_60 = rets_df.rolling(60).corr(bench_ret)
            panel["corr_bench_60"] = corr_60.stack().reindex(panel.index)
        except Exception:
            panel["corr_bench_60"] = np.nan
        return panel

    def _add_regime_features(self, panel: pd.DataFrame,
                             price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        panel = panel.copy()
        try:
            prices = pd.DataFrame({s: d["close"] for s, d in price_data.items()})
            detector = RegimeDetector()
            regimes = detector.classify_universe(prices)
            panel["regime"] = regimes.reindex(
                panel.index.get_level_values("date")).values
            panel["regime_risk_mult"] = panel["regime"].apply(
                RegimeDetector.regime_risk_multiplier)
            for r in RegimeLabel:
                panel[f"regime_is_{r.value}"] = (panel["regime"] == r.value).astype(int)
        except Exception as e:
            LOG.warning(f"Regime feature computation failed: {e}")
            panel["regime"] = int(RegimeLabel.RANGING)
            panel["regime_risk_mult"] = 1.0
        return panel

    def _add_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.copy()
        close_df = panel["close"].unstack("symbol")
        for h in self.cfg.horizons:
            fwd = close_df.shift(-h) / close_df - 1.0
            fwd_stacked = fwd.stack().reindex(panel.index)
            panel[f"fwd_ret_{h}"] = fwd_stacked
            if self.cfg.target_type == PredictionTarget.DIRECTION:
                panel[f"target_{h}"] = (fwd_stacked > self.cfg.return_threshold).astype(int)
            elif self.cfg.target_type == PredictionTarget.RETURN:
                panel[f"target_{h}"] = fwd_stacked
            elif self.cfg.target_type == PredictionTarget.EXCESS_RETURN:
                mean_fwd = fwd.mean(axis=1)
                excess = fwd.sub(mean_fwd, axis=0)
                panel[f"target_{h}"] = excess.stack().reindex(panel.index)
            elif self.cfg.target_type == PredictionTarget.RANK:
                ranks = fwd.rank(axis=1, pct=True)
                panel[f"target_{h}"] = ranks.stack().reindex(panel.index)
            elif self.cfg.target_type == PredictionTarget.TRIPLE_BARRIER:
                vol_series = panel["atr_14"] if "atr_14" in panel.columns else 0.02
                target = pd.Series(0, index=panel.index)
                threshold = vol_series.fillna(0.02) * np.sqrt(h)
                for h_step in range(1, h + 1):
                    r_step = close_df.shift(-h_step) / close_df - 1.0
                    r_step = r_step.stack().reindex(panel.index)
                    up_hit = (r_step > threshold) & (target == 0)
                    dn_hit = (r_step < -threshold) & (target == 0)
                    target = target.mask(up_hit, 1)
                    target = target.mask(dn_hit, -1)
                panel[f"target_{h}"] = target
        return panel

    @staticmethod
    def _infer_feature_cols(panel: pd.DataFrame) -> List[str]:
        drop = {"symbol"}
        drop |= {c for c in panel.columns if c.startswith("fwd_ret_")}
        drop |= {c for c in panel.columns if c.startswith("target_")}
        drop |= {"regime"}
        cols = [c for c in panel.columns
                if c not in drop and panel[c].dtype.kind in "if"]
        return cols


class Preprocessor:
    def __init__(self, cfg: PredictorConfig):
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
                self.winsor_bounds_[col] = (-5.0, 5.0)
                self.means_[col] = 0.0
                self.stds_[col] = 1.0
                continue
            s = df[col].replace([np.inf, -np.inf], np.nan)
            self.medians_[col] = float(s.median()) if s.notna().any() else 0.0
            filled = s.fillna(self.medians_[col])
            lo = float(filled.quantile(self.cfg.winsorize_quantile))
            hi = float(filled.quantile(1 - self.cfg.winsorize_quantile))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = -5.0, 5.0
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
            lo, hi = self.winsor_bounds_.get(col, (-5.0, 5.0))
            s = s.clip(lo, hi)
            mu = self.means_.get(col, 0.0)
            sd = self.stds_.get(col, 1.0)
            out[col] = (s - mu) / (sd + EPS)
        out = out.clip(-self.cfg.feature_clip_sigma, self.cfg.feature_clip_sigma)
        return out

    def fit_transform(self, df: pd.DataFrame,
                      feature_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, feature_cols).transform(df)


class PurgedEmbargoedSplit:
    def __init__(self, n_splits: int = 6, embargo_days: int = 5,
                 purge_days: int = 5):
        self.n_splits = n_splits
        self.embargo = embargo_days
        self.purge = purge_days

    def split(self, dates: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        unique_dates = np.sort(dates.unique())
        n = len(unique_dates)
        if n < self.n_splits * 3:
            fold = max(1, n // self.n_splits)
            for i in range(self.n_splits):
                start = i * fold
                end = min(start + fold, n)
                train_dates = unique_dates[:start]
                val_dates = unique_dates[start:end]
                tr = dates.isin(train_dates).values
                va = dates.isin(val_dates).values
                if tr.sum() > 0 and va.sum() > 0:
                    yield np.where(tr)[0], np.where(va)[0]
            return

        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else n
            val_dates = unique_dates[val_start:val_end]
            purge_start = val_start - self.purge
            train_dates_all = unique_dates[:max(0, purge_start)]
            embargo_start = min(val_end + self.embargo, n)
            tr_mask = dates.isin(train_dates_all).values
            va_mask = dates.isin(val_dates).values
            if tr_mask.sum() < 30 or va_mask.sum() < 5:
                continue
            yield np.where(tr_mask)[0], np.where(va_mask)[0]


class BaseLearner:
    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "BaseLearner":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def feature_importances(self) -> Optional[np.ndarray]:
        return None


class XGBLearner(BaseLearner):
    name = "xgb"

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.model: Optional[Any] = None
        self.use_gpu = self._detect_gpu()

    def _detect_gpu(self) -> bool:
        try:
            import subprocess
            _ = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
            return True
        except Exception:
            return False

    def fit(self, X, y, X_val=None, y_val=None):
        if not HAS_XGB:
            raise RuntimeError("xgboost not installed")
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
        if X_val is not None and y_val is not None and len(y_val) > 0:
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

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=20,
            min_samples_split=40,
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

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.model = HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=6,
            learning_rate=0.035,
            max_bins=255,
            min_samples_leaf=30,
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

    def feature_importances(self):
        return None


class RidgeMetaLearner(BaseLearner):
    name = "ridge_meta"

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.model = LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            random_state=cfg.random_state,
        )

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class QuantileReturnPredictor:
    def __init__(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)):
        self.quantiles = tuple(quantiles)
        self.models: Dict[float, QuantileRegressor] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileReturnPredictor":
        for q in self.quantiles:
            if q == 0.5:
                m = BayesianRidge()
            else:
                m = QuantileRegressor(
                    quantile=q, alpha=1e-3, solver="highs",
                    fit_intercept=True,
                )
            try:
                m.fit(X, y)
                self.models[q] = m
            except Exception as e:
                LOG.debug(f"Quantile fit failed for q={q}: {e}")
                m = Ridge(alpha=1.0)
                m.fit(X, y)
                self.models[q] = m
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
        self.residual_q_: float = 0.0

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
    def __init__(self, n_features: int, cfg: PredictorConfig):
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
        self._classes_ = np.array([0, 1])
        self.weights = np.zeros(n_features)

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self._fitted:
            self.model.partial_fit(X, y, classes=self._classes_)
            self._fitted = True
        else:
            try:
                self.model.partial_fit(X, y)
            except Exception:
                pass
        if hasattr(self.model, "coef_") and self.model.coef_ is not None:
            self.weights = self.model.coef_.ravel()[:self.n_features]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.full(X.shape[0], 0.5)
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return np.full(X.shape[0], 0.5)


class FeatureSelector:
    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg
        self.selected_: List[int] = []
        self.importances_: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> "FeatureSelector":
        if not self.cfg.feature_selection_enabled:
            self.selected_ = list(range(X.shape[1]))
            return self
        if X.shape[1] <= self.cfg.max_features:
            self.selected_ = list(range(X.shape[1]))
            return self

        n = X.shape[0]
        idx = np.random.RandomState(self.cfg.random_state).choice(
            n, size=min(n, 8000), replace=False)
        X_s = X[idx]
        y_s = y[idx]
        try:
            mi = mutual_info_classif(X_s, y_s, random_state=self.cfg.random_state)
        except Exception:
            mi = np.zeros(X.shape[1])
        try:
            clf = ExtraTreesClassifier(
                n_estimators=120, max_depth=6, n_jobs=self.cfg.n_jobs,
                random_state=self.cfg.random_state)
            clf.fit(X_s, y_s)
            fi = clf.feature_importances_
        except Exception:
            fi = np.zeros(X.shape[1])
        combined = 0.5 * zscore(mi) + 0.5 * zscore(fi)
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
        self.calibrator: Optional[Any] = None

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
        self.preds_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            base_factory: Callable[[], Any]) -> "BootstrapUncertainty":
        rng = np.random.RandomState(self.random_state)
        self.models = []
        n = X.shape[0]
        for b in range(self.n_bootstrap):
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
        LOG.info(f"Bootstrap uncertainty: {len(self.models)} models trained")
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
    def __init__(self, cfg: PredictorConfig):
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
            LOG.warning(f"Feature drift detected: {n_drift} features with PSI > "
                        f"{self.cfg.drift_psi_threshold}")
        return psi_vals

    def top_drifting(self, k: int = 10) -> List[Tuple[str, float]]:
        if not self.psi_history_:
            return []
        last = self.psi_history_[-1]
        return sorted(last.items(), key=lambda x: -x[1])[:k]


class HorizonEnsemble:
    def __init__(self, cfg: PredictorConfig, horizon: int):
        self.cfg = cfg
        self.horizon = horizon
        self.xgb = XGBLearner(cfg) if HAS_XGB else None
        self.extratrees = ExtraTreesLearner(cfg)
        self.histgbm = HistGBMLearner(cfg)
        self.meta = RidgeMetaLearner(cfg)
        self.selector = FeatureSelector(cfg)
        self.preprocessor = Preprocessor(cfg)
        self.calibrator = ProbabilityCalibrator(cfg.calibration_method)
        self.uncertainty: Optional[BootstrapUncertainty] = None
        self.metrics_: Optional[ModelMetrics] = None
        self.feature_names_: List[str] = []
        self.is_fitted = False
        self.training_time_s: float = 0.0
        self._base_weights: Dict[str, float] = dict(cfg.ensemble_weights)

    def fit(self, panel: pd.DataFrame, feature_cols: List[str],
            target_col: str) -> "HorizonEnsemble":
        t0 = time.perf_counter()
        df = panel.dropna(subset=[target_col]).copy()
        df = df[df[feature_cols].notna().sum(axis=1) >= int(0.7 * len(feature_cols))]
        if len(df) < 200:
            raise RuntimeError(f"Horizon {self.horizon}: insufficient samples "
                               f"({len(df)})")
        X_df = df[feature_cols]
        y = df[target_col].values.astype(int)
        self.preprocessor.fit(X_df, feature_cols)
        X_scaled = self.preprocessor.transform(X_df).values
        self.feature_names_ = list(feature_cols)

        self.selector.fit(X_scaled, y, self.feature_names_)
        X_sel = self.selector.transform(X_scaled)

        splitter = PurgedEmbargoedSplit(
            n_splits=min(self.cfg.n_splits, 5),
            embargo_days=self.cfg.embargo_days,
            purge_days=self.cfg.purge_days,
        )
        date_series = pd.Series(df.index.get_level_values("date"))
        cv_aucs, cv_metrics = [], []

        folds = list(splitter.split(date_series))
        if folds:
            tr_idx, va_idx = folds[-1]
            X_tr, y_tr = X_sel[tr_idx], y[tr_idx]
            X_va, y_va = X_sel[va_idx], y[va_idx]
        else:
            cut = int(len(X_sel) * 0.8)
            X_tr, y_tr = X_sel[:cut], y[:cut]
            X_va, y_va = X_sel[cut:], y[cut:]

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
        LOG.info(f"[H={self.horizon}] CV AUC mean={np.mean(cv_aucs) if cv_aucs else float('nan'):.4f} "
                 f"std={np.std(cv_aucs) if cv_aucs else float('nan'):.4f}")

        base_probs_val: List[np.ndarray] = []
        base_probs_tr: List[np.ndarray] = []

        if self.xgb is not None:
            try:
                self.xgb.fit(X_tr, y_tr, X_va, y_va)
                base_probs_val.append(self.xgb.predict_proba(X_va))
                base_probs_tr.append(self.xgb.predict_proba(X_tr))
            except Exception as e:
                LOG.warning(f"[H={self.horizon}] XGB fit failed: {e}")
                base_probs_val.append(np.full(len(X_va), 0.5))
                base_probs_tr.append(np.full(len(X_tr), 0.5))
        else:
            base_probs_val.append(np.full(len(X_va), 0.5))
            base_probs_tr.append(np.full(len(X_tr), 0.5))

        try:
            self.extratrees.fit(X_tr, y_tr)
            base_probs_val.append(self.extratrees.predict_proba(X_va))
            base_probs_tr.append(self.extratrees.predict_proba(X_tr))
        except Exception:
            base_probs_val.append(np.full(len(X_va), 0.5))
            base_probs_tr.append(np.full(len(X_tr), 0.5))

        try:
            self.histgbm.fit(X_tr, y_tr)
            base_probs_val.append(self.histgbm.predict_proba(X_va))
            base_probs_tr.append(self.histgbm.predict_proba(X_tr))
        except Exception:
            base_probs_val.append(np.full(len(X_va), 0.5))
            base_probs_tr.append(np.full(len(X_tr), 0.5))

        X_meta_val = np.column_stack(base_probs_val)
        X_meta_tr = np.column_stack(base_probs_tr)
        try:
            self.meta.fit(X_meta_tr, y_tr)
            blended_val = self.meta.predict_proba(X_meta_val)
            blended_tr = self.meta.predict_proba(X_meta_tr)
        except Exception:
            blended_val = base_probs_val[0]
            blended_tr = base_probs_tr[0]

        if self.cfg.calibrate_probabilities:
            try:
                self.calibrator.fit(blended_val, y_va)
                blended_val_cal = self.calibrator.transform(blended_val)
                blended_tr_cal = self.calibrator.transform(blended_tr)
            except Exception:
                blended_val_cal = blended_val
                blended_tr_cal = blended_tr
        else:
            blended_val_cal = blended_val
            blended_tr_cal = blended_tr

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
        except Exception as e:
            LOG.debug(f"Bootstrap uncertainty failed: {e}")
            self.uncertainty = None

        self.is_fitted = True
        self.training_time_s = time.perf_counter() - t0
        LOG.info(f"[H={self.horizon}] trained in {self.training_time_s:.2f}s | "
                 f"AUC={self.metrics_.auc:.4f} | Acc={self.metrics_.accuracy:.4f} | "
                 f"Brier={self.metrics_.brier:.4f} | n_train={self.metrics_.n_train}")
        return self

    def _compute_metrics(self, y: np.ndarray, p: np.ndarray,
                         n_tr: int, n_va: int) -> ModelMetrics:
        if len(y) == 0 or len(np.unique(y)) < 2:
            return ModelMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_tr, n_va, 0.0)
        labels = (p > 0.5).astype(int)
        tp = ((labels == 1) & (y == 1)).sum()
        fp = ((labels == 1) & (y == 0)).sum()
        fn = ((labels == 0) & (y == 1)).sum()
        tn = ((labels == 0) & (y == 0)).sum()
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
        hit_rate = float((np.sign(p - 0.5) == np.sign(y - 0.5)).mean())
        return ModelMetrics(
            auc=float(auc), accuracy=float(acc),
            precision=float(precision), recall=float(recall),
            f1=float(f1), brier=float(brier), logloss=float(ll),
            matthews=float(mcc), n_train=n_tr, n_valid=n_va,
            hit_rate=float(hit_rate),
        )

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
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
        for learner in [self.xgb, self.extratrees, self.histgbm]:
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


class Predictor:
    def __init__(self, cfg: Optional[PredictorConfig] = None):
        self.cfg = cfg or PredictorConfig()
        _configure_logging(self.cfg.log_level)
        self.engine = FeatureEngine(self.cfg)
        self.ensembles: Dict[int, HorizonEnsemble] = {}
        self.online: Optional[OnlineLearner] = None
        self.drift_monitor = DriftMonitor(self.cfg)
        self.feature_set: Optional[FeatureSet] = None
        self.return_predictor: Optional[QuantileReturnPredictor] = None
        self.conformal: Optional[ConformalPredictor] = None
        self._fitted = False
        self.fit_timestamp: Optional[datetime] = None
        self.version: str = f"1.0.0-{uuid.uuid4().hex[:8]}"

    def fit(self, price_data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None) -> "Predictor":
        LOG.info(f"=== Predictor.fit() starting [{self.version}] ===")
        t0 = time.perf_counter()

        self.feature_set = self.engine.build(price_data, benchmark)
        panel = self.feature_set.panel
        feature_cols = self.feature_set.feature_cols

        feature_array = panel[feature_cols].fillna(0.0).values
        self.drift_monitor.set_reference(feature_array, feature_cols)

        for h in self.cfg.horizons:
            target_col = f"target_{h}"
            if target_col not in panel.columns:
                LOG.warning(f"Target {target_col} not in panel; skipping h={h}")
                continue
            try:
                ens = HorizonEnsemble(self.cfg, horizon=h)
                ens.fit(panel, feature_cols, target_col)
                self.ensembles[h] = ens
            except Exception as e:
                LOG.error(f"[H={h}] Training failed: {e}")

        primary = self.cfg.primary_horizon
        fwd_col = f"fwd_ret_{primary}"
        if fwd_col in panel.columns:
            df_ret = panel.dropna(subset=[fwd_col])
            if len(df_ret) >= 200:
                X_ret = df_ret[feature_cols].fillna(0.0).values
                pre = Preprocessor(self.cfg)
                X_scaled = pre.fit_transform(
                    df_ret[feature_cols], feature_cols).values
                y_ret = df_ret[fwd_col].values
                try:
                    self.return_predictor = QuantileReturnPredictor(
                        quantiles=self.cfg.quantiles)
                    self.return_predictor.fit(X_scaled, y_ret)
                except Exception as e:
                    LOG.warning(f"Return quantile model failed: {e}")
                    self.return_predictor = None

        if self.return_predictor is not None and fwd_col in panel.columns:
            try:
                df_ret = panel.dropna(subset=[fwd_col]).tail(300)
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
                LOG.debug(f"Conformal fit failed: {e}")

        if self.cfg.online_enabled:
            n_feat = len(feature_cols)
            self.online = OnlineLearner(n_feat, self.cfg)
            tail = panel.dropna(subset=[f"target_{primary}"]).tail(500)
            if len(tail) > 100:
                X_tail = tail[feature_cols].fillna(0.0).values
                y_tail = tail[f"target_{primary}"].values.astype(int)
                if primary in self.ensembles:
                    pre = self.ensembles[primary].preprocessor
                    X_tail_s = pre.transform(tail[feature_cols]).values
                    for i in range(0, len(X_tail_s), 50):
                        self.online.partial_fit(
                            X_tail_s[i:i + 50], y_tail[i:i + 50])

        self._fitted = True
        self.fit_timestamp = datetime.utcnow()
        LOG.info(f"=== Predictor.fit() done in {time.perf_counter() - t0:.2f}s ===")
        return self

    def update(self, new_data: Dict[str, pd.DataFrame],
               benchmark: Optional[pd.DataFrame] = None) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        if self.online is None:
            return
        try:
            new_fs = self.engine.build(new_data, benchmark)
        except Exception as e:
            LOG.warning(f"Incremental feature build failed: {e}")
            return
        primary = self.cfg.primary_horizon
        if primary not in self.ensembles:
            return
        ens = self.ensembles[primary]
        df = new_fs.panel.dropna(subset=[f"target_{primary}"])
        if df.empty:
            return
        try:
            Xs = ens.preprocessor.transform(df[ens.feature_names_]).values
        except Exception:
            return
        y = df[f"target_{primary}"].values.astype(int)
        for i in range(0, len(Xs), 32):
            self.online.partial_fit(Xs[i:i + 32], y[i:i + 32])
        LOG.info(f"Online update with {len(Xs)} new labeled samples")

    def predict(self, price_data: Dict[str, pd.DataFrame],
                benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        fs = self.engine.build(price_data, benchmark)
        panel = fs.panel

        latest = (panel.reset_index()
                       .sort_values("date")
                       .groupby("symbol").tail(1)
                       .set_index(["date", "symbol"]))
        LOG.info(f"Predicting for {len(latest)} symbols")

        regime_col = "regime" if "regime" in latest.columns else None

        results = []
        for h, ens in self.ensembles.items():
            try:
                probs = ens.predict_proba(latest)
                means, stds = ens.predict_uncertainty(latest)
            except Exception as e:
                LOG.warning(f"[H={h}] Prediction failed: {e}")
                continue

            exp_ret = np.zeros(len(latest))
            lo_ret = np.zeros(len(latest))
            hi_ret = np.zeros(len(latest))
            if self.return_predictor is not None and h == self.cfg.primary_horizon:
                try:
                    Xs = ens.preprocessor.transform(latest[ens.feature_names_]).values
                    q = self.return_predictor.predict(Xs)
                    exp_ret = q.get(0.5, np.zeros(len(latest)))
                    lo_ret = q.get(self.cfg.quantiles[0], exp_ret)
                    hi_ret = q.get(self.cfg.quantiles[-1], exp_ret)
                    if self.conformal is not None:
                        lo_ret, hi_ret = self.conformal.interval(exp_ret)
                except Exception as e:
                    LOG.debug(f"Return prediction failed: {e}")

            for i, (idx, row) in enumerate(latest.iterrows()):
                sym = idx[1] if isinstance(idx, tuple) else "unknown"
                regime_val = int(row[regime_col]) if regime_col else -1
                conf = float(abs(probs[i] - 0.5) * 2) * (1.0 - min(0.9, stds[i]))
                results.append({
                    "symbol": sym,
                    "date": idx[0] if isinstance(idx, tuple) else None,
                    "horizon": h,
                    "probability_up": float(probs[i]),
                    "expected_return": float(exp_ret[i]),
                    "exp_return_lower": float(lo_ret[i]),
                    "exp_return_upper": float(hi_ret[i]),
                    "direction": int(1 if probs[i] > 0.5 else 0),
                    "confidence": float(conf),
                    "uncertainty": float(stds[i]),
                    "regime": REGIME_NAMES.get(regime_val, "unknown"),
                })

        out = pd.DataFrame(results)
        if out.empty:
            return out
        if self.online is not None and self.cfg.primary_horizon in self.ensembles:
            try:
                ens = self.ensembles[self.cfg.primary_horizon]
                Xs = ens.preprocessor.transform(
                    latest[ens.feature_names_]).values
                p_on = self.online.predict_proba(Xs)
                mask = out["horizon"] == self.cfg.primary_horizon
                primary_syms = out.loc[mask, "symbol"].values
                latest_syms = [i[1] if isinstance(i, tuple) else i
                               for i in latest.index]
                online_map = dict(zip(latest_syms, p_on))
                online_series = out.loc[mask, "symbol"].map(online_map)
                out.loc[mask, "online_prob"] = online_series.values
                out.loc[mask, "probability_up"] = (
                    0.85 * out.loc[mask, "probability_up"].values +
                    0.15 * online_series.values
                )
            except Exception as e:
                LOG.debug(f"Online overlay failed: {e}")
        return out.sort_values(["horizon", "probability_up"], ascending=[True, False])

    def explain(self, symbol: str, price_data: Dict[str, pd.DataFrame],
                horizon: Optional[int] = None,
                top_k: int = 15) -> pd.DataFrame:
        h = horizon or self.cfg.primary_horizon
        if h not in self.ensembles:
            raise KeyError(f"No model for horizon {h}")
        ens = self.ensembles[h]
        fs = self.engine.build(price_data)
        panel = fs.panel
        sym_panel = panel.xs(symbol, level="symbol")
        row = sym_panel.tail(1)
        Xs = ens.preprocessor.transform(row[ens.feature_names_]).values
        p0 = ens.predict_proba(row)[0]

        names = [ens.feature_names_[i] for i in ens.selector.selected_]
        contribs = []
        for j in range(len(names)):
            X_pert = Xs.copy()
            col_std = Xs[:, j].std() if Xs.shape[0] > 1 else 1.0
            X_pert[:, j] = X_pert[:, j] + np.random.RandomState(0).randn(
                Xs.shape[0]) * (col_std + EPS)
            try:
                p1 = ens.predict_proba(row)[0]
            except Exception:
                continue
            contribs.append((names[j], p0 - p1))
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
                "horizon": h,
                "auc": m.auc,
                "accuracy": m.accuracy,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "brier": m.brier,
                "logloss": m.logloss,
                "matthews": m.matthews,
                "hit_rate": m.hit_rate,
                "n_train": m.n_train,
                "n_valid": m.n_valid,
                "train_seconds": ens.training_time_s,
            })
        return pd.DataFrame(rows)

    def feature_importance_report(self, horizon: Optional[int] = None,
                                  top_k: int = 30) -> pd.Series:
        h = horizon or self.cfg.primary_horizon
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
        LOG.info(f"Predictor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Predictor":
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
        LOG.info(f"Predictor loaded from {path} (v{p.version})")
        return p


class SyntheticMarketGenerator:
    def __init__(self, n_symbols: int = 20, n_days: int = 1200,
                 random_state: int = 42):
        self.n_symbols = n_symbols
        self.n_days = n_days
        self.rs = np.random.RandomState(random_state)

    def generate(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        t = np.arange(self.n_days)
        vol_regimes = np.ones(self.n_days) * 0.012
        n_breaks = 8
        break_points = np.sort(self.rs.choice(
            np.arange(50, self.n_days - 50), size=n_breaks, replace=False))
        current_vol = 0.010
        vol_profile = np.zeros(self.n_days)
        for i in range(self.n_days):
            if i in break_points:
                current_vol = self.rs.choice([0.008, 0.012, 0.018, 0.030])
            vol_profile[i] = current_vol
        drift = 0.0003 * np.sin(2 * np.pi * t / 250) + 0.0001
        market_ret = self.rs.randn(self.n_days) * vol_profile + drift
        market_price = 100 * np.exp(np.cumsum(market_ret))

        dates = pd.bdate_range("2019-01-01", periods=self.n_days)
        symbols = [f"SYN{i:03d}" for i in range(self.n_symbols)]

        prices: Dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            beta = 0.5 + self.rs.rand() * 1.5
            idio_vol = 0.008 + self.rs.rand() * 0.020
            idio = self.rs.randn(self.n_days) * idio_vol
            cycle = 0.0004 * np.sin(2 * np.pi * t / (60 + 20 * (i % 5)))
            ret = beta * market_ret + idio + cycle
            price = 50 + 50 * self.rs.rand() * np.exp(np.cumsum(ret))
            high = price * (1 + np.abs(self.rs.randn(self.n_days)) * 0.005)
            low = price * (1 - np.abs(self.rs.randn(self.n_days)) * 0.005)
            open_ = price * (1 + self.rs.randn(self.n_days) * 0.003)
            vol = np.abs(self.rs.randn(self.n_days)) * 1e6 + 5e5
            df = pd.DataFrame({
                "open": open_,
                "high": np.maximum.reduce([open_, price, high]),
                "low": np.minimum.reduce([open_, price, low]),
                "close": price,
                "volume": vol,
            }, index=dates)
            prices[sym] = df

        bench = pd.DataFrame({
            "open": market_price,
            "high": market_price * 1.005,
            "low": market_price * 0.995,
            "close": market_price,
            "volume": np.abs(self.rs.randn(self.n_days)) * 5e7 + 1e7,
        }, index=dates)
        return prices, bench


class Diagnostics:
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
            sharpes.append(r.mean() / r.std() * np.sqrt(252))
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
    def turnover(weights: pd.DataFrame) -> pd.Series:
        if weights.empty:
            return pd.Series(dtype=float)
        diff = weights.diff().abs().sum(axis=1)
        return diff.fillna(0.0) / 2.0


class SignalFusion:
    def __init__(self, horizon_weights: Optional[Dict[int, float]] = None):
        self.horizon_weights = horizon_weights or {1: 0.1, 5: 0.4, 10: 0.3, 20: 0.2}

    def fuse(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            return predictions
        out_rows = []
        for sym, group in predictions.groupby("symbol"):
            total_w = 0.0
            weighted_p = 0.0
            weighted_r = 0.0
            weighted_c = 0.0
            for _, row in group.iterrows():
                h = int(row["horizon"])
                w = self.horizon_weights.get(h, 0.0)
                if w <= 0:
                    continue
                total_w += w
                weighted_p += w * row["probability_up"]
                weighted_r += w * row["expected_return"]
                weighted_c += w * row["confidence"]
            if total_w > 0:
                out_rows.append({
                    "symbol": sym,
                    "fused_prob": weighted_p / total_w,
                    "fused_expected_return": weighted_r / total_w,
                    "fused_confidence": weighted_c / total_w,
                })
        fused = pd.DataFrame(out_rows)
        if not fused.empty:
            fused = fused.sort_values("fused_prob", ascending=False)
        return fused


class SimpleBacktest:
    def __init__(self, cost_bps: float = 5.0):
        self.cost_bps = cost_bps

    def run(self, signals: pd.DataFrame, prices: pd.DataFrame,
            top_k: int = 10, long_short: bool = True) -> pd.DataFrame:
        signals = signals.reindex(prices.index).fillna(0.0)
        positions = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for dt in signals.index:
            row = signals.loc[dt].sort_values(ascending=False)
            longs = row.head(top_k).index
            if long_short:
                shorts = row.tail(top_k).index
            else:
                shorts = []
            for s in longs:
                positions.loc[dt, s] = 1.0 / max(1, len(longs))
            for s in shorts:
                positions.loc[dt, s] = -1.0 / max(1, len(shorts))

        rets = prices.pct_change().fillna(0.0)
        pnl = (positions.shift(1) * rets).sum(axis=1)
        turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover * self.cost_bps / 1e4
        pnl_net = pnl - cost
        equity = (1 + pnl_net).cumprod()
        return pd.DataFrame({
            "pnl_gross": pnl,
            "cost": cost,
            "pnl_net": pnl_net,
            "equity": equity,
            "turnover": turnover,
        })


class WalkForwardTrainer:
    def __init__(self, cfg: PredictorConfig,
                 refit_frequency_days: int = 20):
        self.cfg = cfg
        self.refit_freq = refit_frequency_days
        self.predictor: Optional[Predictor] = None
        self.snapshots: List[Dict[str, Any]] = []

    def run(self, price_data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None,
            start_after: int = 500) -> pd.DataFrame:
        all_dates = sorted(set().union(*[d.index for d in price_data.values()]))
        if len(all_dates) < start_after + self.refit_freq:
            LOG.warning("Insufficient history for walk-forward")
            return pd.DataFrame()

        refit_dates = all_dates[start_after::self.refit_freq]
        all_preds = []

        for i, refit_date in enumerate(refit_dates[:-1]):
            LOG.info(f"[WF {i+1}/{len(refit_dates)-1}] refit as-of {refit_date.date()}")
            slice_data = {}
            for s, d in price_data.items():
                d_slice = d.loc[:refit_date]
                if len(d_slice) >= self.cfg.min_history:
                    slice_data[s] = d_slice
            if len(slice_data) < 5:
                continue
            b_slice = benchmark.loc[:refit_date] if benchmark is not None else None
            try:
                p = Predictor(self.cfg)
                p.fit(slice_data, b_slice)
                self.predictor = p
                next_date = refit_dates[i + 1]
                pred_data = {}
                for s, d in price_data.items():
                    d_pred = d.loc[refit_date:next_date]
                    if not d_pred.empty:
                        pred_data[s] = d_pred
                if not pred_data:
                    continue
                preds = p.predict(pred_data, benchmark)
                preds["refit_date"] = refit_date
                all_preds.append(preds)
                self.snapshots.append({
                    "refit_date": refit_date,
                    "metrics": p.metrics_report().to_dict("records"),
                })
            except Exception as e:
                LOG.warning(f"Walk-forward refit failed at {refit_date}: {e}")
                continue

        if not all_preds:
            return pd.DataFrame()
        return pd.concat(all_preds, ignore_index=True)


def _self_test() -> None:
    _configure_logging("INFO")
    LOG.info("=" * 70)
    LOG.info("PREDICTOR.PY SELF-TEST")
    LOG.info("=" * 70)

    gen = SyntheticMarketGenerator(n_symbols=15, n_days=900, random_state=7)
    prices, benchmark = gen.generate()
    LOG.info(f"Generated {len(prices)} synthetic symbols, "
             f"{len(next(iter(prices.values())))} days each")

    cfg = PredictorConfig(
        horizons=(1, 5, 10, 20),
        primary_horizon=5,
        target_type=PredictionTarget.DIRECTION,
        min_history=260,
        n_bootstrap=15,
        xgb_n_estimators=350,
        xgb_max_depth=5,
        feature_selection_enabled=True,
        max_features=80,
        log_level="INFO",
    )

    predictor = Predictor(cfg)
    predictor.fit(prices, benchmark)

    metrics_df = predictor.metrics_report()
    LOG.info("\n--- MODEL METRICS ---")
    LOG.info("\n" + metrics_df.to_string(index=False))

    imp = predictor.feature_importance_report(top_k=15)
    LOG.info("\n--- TOP-15 FEATURES (primary horizon) ---")
    LOG.info("\n" + imp.to_string())

    preds = predictor.predict(prices, benchmark)
    LOG.info("\n--- SAMPLE PREDICTIONS ---")
    if not preds.empty:
        sample = preds.head(20)[[
            "symbol", "horizon", "probability_up", "expected_return",
            "confidence", "regime",
        ]]
        LOG.info("\n" + sample.to_string(index=False))

    fusion = SignalFusion()
    fused = fusion.fuse(preds)
    LOG.info("\n--- FUSED SIGNALS (top 10) ---")
    if not fused.empty:
        LOG.info("\n" + fused.head(10).to_string(index=False))

    sym = list(prices.keys())[0]
    try:
        expl = predictor.explain(sym, prices, horizon=cfg.primary_horizon, top_k=10)
        LOG.info(f"\n--- EXPLANATION FOR {sym} ---")
        LOG.info("\n" + expl.to_string(index=False))
    except Exception as e:
        LOG.warning(f"Explain failed: {e}")

    drift = predictor.drift_report()
    LOG.info(f"\n--- DRIFT REPORT (last) ---")
    if not drift.empty:
        last = drift[drift["call"] == drift["call"].max()].sort_values(
            "psi", ascending=False).head(10)
        LOG.info("\n" + last.to_string(index=False))

    try:
        if not preds.empty:
            primary_preds = preds[preds["horizon"] == cfg.primary_horizon].set_index("symbol")
            forward_map = {}
            for s, d in prices.items():
                if len(d) >= cfg.primary_horizon:
                    fwd = d["close"].iloc[-1] / d["close"].iloc[-cfg.primary_horizon - 1] - 1
                    forward_map[s] = fwd
            fwd_series = pd.Series(forward_map)
            ic = Diagnostics.information_coefficient(
                primary_preds["expected_return"], fwd_series)
            LOG.info(f"\n--- IC (expected return vs realized) = {ic:.4f} ---")

            decile = Diagnostics.decile_analysis(
                primary_preds["probability_up"], fwd_series)
            if not decile.empty:
                LOG.info("\n--- DECILE ANALYSIS ---")
                LOG.info("\n" + decile.to_string())
    except Exception as e:
        LOG.warning(f"Diagnostics failed: {e}")

    try:
        p = Path("./predictor_selftest.pkl")
        predictor.save(str(p))
        loaded = Predictor.load(str(p))
        preds2 = loaded.predict(prices, benchmark)
        LOG.info(f"\n--- SAVE/LOAD round-trip OK: {len(preds2)} preds ---")
        p.unlink(missing_ok=True)
    except Exception as e:
        LOG.warning(f"Save/load failed: {e}")

    LOG.info("\n--- WALK-FORWARD (short) ---")
    wf_cfg = PredictorConfig(
        horizons=(5,),
        primary_horizon=5,
        min_history=260,
        xgb_n_estimators=200,
        n_bootstrap=5,
        max_features=60,
        log_level="WARNING",
    )
    try:
        wf = WalkForwardTrainer(wf_cfg, refit_frequency_days=60)
        wf_preds = wf.run(prices, benchmark, start_after=550)
        LOG.info(f"Walk-forward produced {len(wf_preds)} out-of-sample predictions")
    except Exception as e:
        LOG.warning(f"Walk-forward failed: {e}")

    LOG.info("=" * 70)
    LOG.info("SELF-TEST COMPLETE")
    LOG.info("=" * 70)


if __name__ == "__main__":
    _self_test()
