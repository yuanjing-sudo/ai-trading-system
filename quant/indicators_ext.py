"""Indicator extension pack: 137 additional vectorized technical indicators.

Complements :mod:`quant.indicators` with moving-average variants, momentum
oscillators, volatility estimators, volume analytics, trend-quality scores,
price transforms, pivot families, and rolling statistics.

Conventions
-----------
- ``close``, ``high``, ``low``, ``open``, ``volume`` are pandas Series.
- Functions never mutate their inputs.
- Every function accepts plain arrays/lists where convenient via ``_as_series``.
- Adaptive recursions (VIDYA, McGinley, JMA approximation, zigzag) are the only
  loop-based implementations; everything else is fully vectorized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import (
    _as_series,
    _safe_div,
    _true_range,
    _wilder,
    adx,
    atr,
    bollinger_bands,
    dmi,
    ema,
    keltner_channels,
    obv,
    rsi,
    sma,
)
from quant.indicators import aroon as _aroon
from quant.indicators import linreg_slope as _linreg_slope
from quant.indicators import rolling_corr as _rolling_corr
from quant.indicators import zscore as _zscore

__all__ = [
    # moving averages
    "smma", "zlema", "lsma", "tsf", "frama", "vidya", "mcginley_dynamic",
    "t3", "dswma", "hamming_ma", "raised_cosine_ma", "jma_approx",
    "ma_slope", "ma_distance", "ma_ratio", "price_above_ma", "crossovers",
    "has_crossed", "golden_cross", "death_cross", "ma_ribbon",
    "ribbon_width", "guppy_mma", "vwema", "vwma_typical",
    "ema_ribbon_score",
    # momentum / oscillators
    "momentum", "roc", "apo", "tsi", "connors_rsi", "rmi", "rsi_ma",
    "rsi_slope", "divergence", "rsi_divergence", "tmo", "smi", "bop",
    "cti", "vhf", "efficiency_ratio", "dynamic_momentum_index", "pgo",
    "qstick", "fisher_rsi", "stoch_slow_rsi", "sma_oscillator", "imp",
    "intraday_intensity", "roc_ma", "momentum_score",
    "price_momentum_percentile", "gain_loss_ratio", "up_down_volume_ratio",
    "hist_vol_osc", "rvi_volatility", "atr_roc",
    # volatility
    "stddev", "variance", "mean_absolute_deviation", "semideviation",
    "upside_deviation", "chaikin_volatility", "adr", "adr_percent",
    "atr_percent", "bb_width", "bb_percent_b", "kc_width", "atr_bands",
    "bipower_variation", "jump_variation", "realized_semivariance",
    "vol_of_vol", "volatility_ratio", "true_range_pct",
    "range_expansion_index", "gap_volatility",
    # volume
    "ad_line", "vpt", "wad", "relative_volume", "volume_oscillator", "vzo",
    "twiggs_money_flow", "volume_rsi", "obv_ema", "dollar_volume",
    "volume_zscore", "force_ratio", "buy_pressure", "volume_breakout",
    "volume_dryup", "volume_trend", "volume_price_corr", "turnover_ratio",
    "relative_strength",
    # trend quality
    "trend_quality", "trend_score", "trend_intensity", "velocity",
    "acceleration", "curvature", "directional_bias", "aroon_oscillator",
    "dmi_spread", "adxr", "curve_smoothness", "range_position",
    "distance_from_high", "distance_from_low", "yearly_position",
    # price transforms & pivots
    "hl2", "hlc3", "ohlc4", "weighted_close", "mid_point", "hl_range",
    "hl_range_pct", "close_location_value", "gap", "true_high", "true_low",
    "camarilla_pivots", "woodie_pivots", "demark_pivots",
    "fibonacci_pivots", "zigzag", "swing_high", "swing_low",
    # rolling statistics
    "rolling_median", "rolling_quantile", "iqr", "trimmed_mean",
    "winsorized_mean", "robust_zscore",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adaptive_ema(series: pd.Series, alphas: pd.Series, seed_ma: pd.Series) -> pd.Series:
    """Recursive EMA with a time-varying smoothing factor.

    ``alphas`` is clipped to [0, 1]; the recursion seeds from ``seed_ma`` so
    warm-up values are not dominated by an arbitrary first observation.
    """
    x = series.to_numpy(dtype="float64")
    a = np.clip(alphas.to_numpy(dtype="float64"), 0.0, 1.0)
    seed = seed_ma.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    prev = seed[0] if len(seed) else x[0]
    for i in range(len(x)):
        ai = a[i]
        xi = x[i]
        if not np.isfinite(xi):
            out[i] = prev
            continue
        if not np.isfinite(ai):
            ai = 0.0
        if not np.isfinite(prev):
            prev = xi
        prev = prev + ai * (xi - prev)
        out[i] = prev
    return pd.Series(out, index=series.index, name=series.name)


def _rolling_apply(series: pd.Series, period: int, func) -> pd.Series:
    return series.rolling(period, min_periods=period).apply(func, raw=True)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def smma(series: pd.Series, period: int = 14) -> pd.Series:
    """Smoothed moving average (Wilder)."""
    return _wilder(_as_series(series), period)


def zlema(series: pd.Series, period: int = 20) -> pd.Series:
    """Zero-lag exponential moving average: EMA of de-lagged input."""
    s = _as_series(series)
    lag = max((period - 1) // 2, 0)
    de_lagged = 2.0 * s - s.shift(lag)
    return ema(de_lagged, period)


def lsma(series: pd.Series, period: int = 25) -> pd.Series:
    """Least squares moving average (linear regression endpoint)."""
    s = _as_series(series)
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    denom = float(((x - x_mean) ** 2).sum())

    def _endpoint(w: np.ndarray) -> float:
        y = w
        slope = float(((x - x_mean) * (y - y.mean())).sum() / denom)
        return y.mean() + slope * (period - 1 - x_mean)

    return _rolling_apply(s, period, _endpoint)


def tsf(series: pd.Series, period: int = 14, offset: int = 0) -> pd.Series:
    """Time series forecast: linear regression projected ``offset`` bars ahead."""
    s = _as_series(series)
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    denom = float(((x - x_mean) ** 2).sum())

    def _forecast(w: np.ndarray) -> float:
        slope = float(((x - x_mean) * (w - w.mean())).sum() / denom)
        return w.mean() + slope * (period - 1 + offset - x_mean)

    return _rolling_apply(s, period, _forecast)


def frama(series: pd.Series, period: int = 16) -> pd.Series:
    """Fractal adaptive moving average (Ehlers)."""
    s = _as_series(series)
    half = max(period // 2, 2)
    roll_max = s.rolling(half, min_periods=half).max()
    roll_min = s.rolling(half, min_periods=half).min()
    n1 = (roll_max - roll_min) / half
    roll_max2 = s.rolling(period, min_periods=period).max()
    roll_min2 = s.rolling(period, min_periods=period).min()
    n2 = (roll_max2 - roll_min2) / float(period)
    n3 = _safe_div(n1, n2)
    dim = (
        np.log(n3.where(n3 > 0)) / np.log(2.0)
    ).clip(lower=1.0)
    alpha = (np.exp(-4.6 * (dim - 1.0))).clip(0.01, 1.0)
    return _adaptive_ema(s, alpha, s)


def vidya(series: pd.Series, period: int = 14, cmo_period: int = 9) -> pd.Series:
    """Variable index dynamic average (Chande): EMA speed follows |CMO|."""
    s = _as_series(series)
    delta = s.diff()
    up = delta.clip(lower=0.0).rolling(cmo_period, min_periods=cmo_period).sum()
    down = (-delta.clip(upper=0.0)).rolling(cmo_period, min_periods=cmo_period).sum()
    cmo_val = 100.0 * _safe_div(up - down, up + down).abs()
    alpha = (2.0 / (period + 1.0)) * cmo_val / 100.0
    return _adaptive_ema(s, alpha, s)


def mcginley_dynamic(series: pd.Series, period: int = 14, k: float = 0.6) -> pd.Series:
    """McGinley dynamic average: self-adjusting to market speed."""
    s = _as_series(series)
    x = s.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    prev = x[0]
    denom = k * period * period
    for i in range(len(x)):
        xi = x[i]
        if not np.isfinite(xi):
            out[i] = prev
            continue
        ratio = xi / prev if prev > 0 else 1.0
        prev = prev + (xi - prev) / max(denom * ratio ** 4, 1e-12)
        out[i] = prev
    return pd.Series(out, index=s.index, name=s.name)


def t3(series: pd.Series, period: int = 10, vfactor: float = 0.7) -> pd.Series:
    """T3 moving average (Tillson): triple-smoothed EMA with volume factor."""
    s = _as_series(series)
    v = float(vfactor)
    e1 = ema(s, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)
    c1 = -v ** 3
    c2 = 3.0 * v ** 2 + 3.0 * v ** 3
    c3 = -6.0 * v ** 2 - 3.0 * v - 3.0 * v ** 3
    c4 = 1.0 + 3.0 * v + 3.0 * v ** 2 + v ** 3
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def dswma(series: pd.Series, period: int = 20) -> pd.Series:
    """Double-smoothed weighted moving average."""
    from quant.indicators import wma

    return wma(wma(_as_series(series), period), period)


def hamming_ma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hamming-window weighted moving average."""
    s = _as_series(series)
    idx = np.arange(period, dtype="float64")
    weights = 0.54 - 0.46 * np.cos(2.0 * np.pi * idx / max(period - 1, 1))
    wsum = weights.sum()
    return _rolling_apply(s, period, lambda w: float(np.dot(w, weights) / wsum))


def raised_cosine_ma(series: pd.Series, period: int = 20) -> pd.Series:
    """Raised-cosine weighted moving average (tapered edges)."""
    s = _as_series(series)
    idx = np.arange(period, dtype="float64")
    weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * idx / max(period - 1, 1)))
    wsum = weights.sum()
    return _rolling_apply(s, period, lambda w: float(np.dot(w, weights) / wsum))


def jma_approx(series: pd.Series, period: int = 14, phase: float = 0.0) -> pd.Series:
    """Approximation of Jurik's adaptive moving average.

    Uses a volatility-modulated EMA speed around the JMA default smoothing
    constant. Documented as an approximation: exact JMA (Jurik Research) is
    proprietary; behavior follows the same adaptivity principle.
    """
    s = _as_series(series)
    base = 2.0 / (period + 1.0)
    vol_ratio = _safe_div(
        s.diff().abs().rolling(period, min_periods=period).mean(),
        s.diff().abs().rolling(period * 4, min_periods=period).mean(),
    ).fillna(1.0)
    phase_adj = 1.0 + float(phase) / 100.0
    alpha = (base * vol_ratio * phase_adj).clip(0.01, 1.0)
    return _adaptive_ema(s, alpha, s)


def ma_slope(series: pd.Series, period: int = 20, lookback: int = 1) -> pd.Series:
    """Normalized slope of a moving average."""
    s = _as_series(series)
    m = sma(s, period)
    return _safe_div(m - m.shift(lookback), m.shift(lookback)) * 100.0


def ma_distance(series: pd.Series, period: int = 20) -> pd.Series:
    """Percent distance of price from its simple moving average."""
    s = _as_series(series)
    m = sma(s, period)
    return _safe_div(s - m, m) * 100.0


def ma_ratio(series: pd.Series, period: int = 20) -> pd.Series:
    """Ratio of price to its simple moving average."""
    s = _as_series(series)
    return _safe_div(s, sma(s, period))


def price_above_ma(series: pd.Series, period: int = 20) -> pd.Series:
    """Boolean: price above its simple moving average."""
    s = _as_series(series)
    return (s > sma(s, period)).fillna(False)


def crossovers(a: pd.Series, b: pd.Series) -> pd.Series:
    """Detect crosses between two lines: +1 golden, -1 death, 0 otherwise."""
    s1 = _as_series(a)
    s2 = _as_series(b).reindex_like(s1)
    diff = s1 - s2
    prev = diff.shift(1)
    out = pd.Series(0.0, index=s1.index)
    out[(diff > 0) & (prev <= 0)] = 1.0
    out[(diff < 0) & (prev >= 0)] = -1.0
    return out


def has_crossed(a: pd.Series, b: pd.Series, lookback: int = 5, direction: int = 1) -> pd.Series:
    """Boolean: a crossed b in the last ``lookback`` bars in ``direction``."""
    crosses = crossovers(a, b)
    target = 1.0 if direction >= 0 else -1.0
    hit = (crosses == target).rolling(lookback, min_periods=1).max()
    return hit.fillna(0.0).astype(bool)


def golden_cross(series: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Boolean: fast SMA crossed above slow SMA on this bar."""
    s = _as_series(series)
    return crossovers(sma(s, fast), sma(s, slow)) == 1.0


def death_cross(series: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Boolean: fast SMA crossed below slow SMA on this bar."""
    s = _as_series(series)
    return crossovers(sma(s, fast), sma(s, slow)) == -1.0


def ma_ribbon(series: pd.Series, periods: tuple[int, ...] = (10, 20, 30, 40, 50, 60)) -> pd.DataFrame:
    """DataFrame of simple moving averages across ``periods``."""
    s = _as_series(series)
    return pd.DataFrame({f"ma_{p}": sma(s, p) for p in periods}, index=s.index)


def ribbon_width(series: pd.Series, periods: tuple[int, ...] = (10, 20, 30, 40, 50, 60)) -> pd.Series:
    """Percent spread between the widest and narrowest ribbon averages."""
    ribbon = ma_ribbon(series, periods)
    return _safe_div(ribbon.max(axis=1) - ribbon.min(axis=1), ribbon.mean(axis=1)) * 100.0


def guppy_mma(close: pd.Series) -> pd.DataFrame:
    """Guppy multiple moving average: short (trader) and long (investor) sets."""
    s = _as_series(close)
    short = (3, 5, 8, 10, 12, 15)
    long_ = (30, 35, 40, 45, 50, 60)
    out = {}
    for p in short:
        out[f"short_{p}"] = sma(s, p)
    for p in long_:
        out[f"long_{p}"] = sma(s, p)
    return pd.DataFrame(out, index=s.index)


def vwema(series: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume-weighted exponential moving average."""
    s = _as_series(series)
    v = _as_series(volume).reindex_like(s).fillna(0.0)
    num = ema(s * v, period)
    den = ema(v, period)
    return _safe_div(num, den, fill=np.nan)


def vwma_typical(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume-weighted moving average of the typical price (HLC/3)."""
    from quant.indicators import vwma

    return vwma(hlc3(high, low, close), volume, period)


def ema_ribbon_score(series: pd.Series, periods: tuple[int, ...] = (5, 10, 20, 50, 100)) -> pd.Series:
    """Composite trend score in [-1, 1] from the alignment of an EMA ribbon.

    +1 when all EMAs are stacked in bullish order with rising slopes, -1 when
    fully inverted; the score is the average pairwise alignment.
    """
    s = _as_series(series)
    emas = [ema(s, p) for p in periods]
    score = pd.Series(0.0, index=s.index)
    n_pairs = 0
    for i in range(len(emas)):
        for j in range(i + 1, len(emas)):
            score = score + np.sign(emas[i] - emas[j])
            n_pairs += 1
    return (score / max(n_pairs, 1)).astype("float64")


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Absolute price momentum: close minus close ``period`` bars ago."""
    return _as_series(series).diff(period)


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change in percent."""
    s = _as_series(series)
    return _safe_div(s - s.shift(period), s.shift(period)) * 100.0


def apo(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Absolute price oscillator: fast EMA minus slow EMA."""
    s = _as_series(series)
    return ema(s, fast) - ema(s, slow)


def tsi(series: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    """True strength index: double-smoothed momentum over its absolute value."""
    s = _as_series(series)
    m = s.diff()
    smooth = ema(ema(m, long), short)
    smooth_abs = ema(ema(m.abs(), long), short)
    return 100.0 * _safe_div(smooth, smooth_abs)


def connors_rsi(series: pd.Series, rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> pd.Series:
    """Connors RSI: blend of short RSI, streak RSI, and percent rank."""
    s = _as_series(series)
    rsi_close = rsi(s, rsi_period)
    up = s.diff() > 0
    raw_streak = pd.Series(np.where(up, 1.0, -1.0), index=s.index)
    acc = 0.0
    prev = 0.0
    values = raw_streak.to_numpy()
    out = np.empty(len(values), dtype="float64")
    for i, v in enumerate(values):
        if not np.isfinite(v):
            out[i] = np.nan
            continue
        if v == prev:
            acc += v
        else:
            acc = v
        prev = v
        out[i] = acc
    streak = pd.Series(out, index=s.index)
    rsi_streak = rsi(streak, streak_period)
    pct = s.pct_change(fill_method=None).rolling(rank_period, min_periods=rank_period).rank(pct=True) * 100.0
    return (rsi_close + rsi_streak + pct) / 3.0


def rmi(series: pd.Series, period: int = 14, momentum_period: int = 4) -> pd.Series:
    """Relative momentum index: RSI applied to momentum instead of price."""
    return rsi(momentum(_as_series(series), momentum_period), period)


def rsi_ma(series: pd.Series, rsi_period: int = 14, ma_period: int = 7) -> pd.Series:
    """Moving average of RSI (smooths the oscillator)."""
    return sma(rsi(_as_series(series), rsi_period), ma_period)


def rsi_slope(series: pd.Series, period: int = 14, slope: int = 1) -> pd.Series:
    """Bar-over-bar change of RSI."""
    return rsi(_as_series(series), period).diff(slope)


def divergence(prices: pd.Series, indicator: pd.Series, pivot_window: int = 3) -> pd.Series:
    """Generic bullish/bearish divergence detector between price and an indicator.

    A bullish divergence (+1) marks a bar where price prints a lower pivot low
    while the indicator prints a higher pivot low; bearish (-1) is symmetric on
    pivot highs.
    """
    p = _as_series(prices)
    ind = _as_series(indicator).reindex_like(p)
    w = 2 * pivot_window + 1
    win_min = p.rolling(w, center=True, min_periods=w).min()
    win_max = p.rolling(w, center=True, min_periods=w).max()
    is_pl = p <= win_min
    is_ph = p >= win_max

    prev_pl_p = p.where(is_pl).ffill().shift(1)
    prev_pl_i = ind.where(is_pl).ffill().shift(1)
    prev_ph_p = p.where(is_ph).ffill().shift(1)
    prev_ph_i = ind.where(is_ph).ffill().shift(1)

    bull = is_pl & (p < prev_pl_p) & (ind > prev_pl_i)
    bear = is_ph & (p > prev_ph_p) & (ind < prev_ph_i)

    out = pd.Series(0.0, index=p.index)
    out[bull] = 1.0
    out[bear] = -1.0
    return out


def rsi_divergence(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, pivot_window: int = 3) -> pd.Series:
    """RSI divergence detector: +1 bullish, -1 bearish, 0 elsewhere."""
    c = _as_series(close)
    h = _as_series(high).reindex_like(c)
    l = _as_series(low).reindex_like(c)
    r = rsi(c, period)
    piv = h.rolling(2 * pivot_window + 1, center=True, min_periods=1).max()
    piv_low = l.rolling(2 * pivot_window + 1, center=True, min_periods=1).min()
    # use blend of high/low pivots for a more robust price series
    proxy = (h.where(h == piv) + l.where(l == piv_low)).bfill()
    proxy = proxy.fillna(c)
    return divergence(proxy, r, pivot_window)


def tmo(close: pd.Series, long: int = 14, short: int = 5) -> pd.Series:
    """True momentum oscillator (Chande): normalized sum of momentum."""
    s = _as_series(close)
    m = s.diff()
    up = m.clip(lower=0.0).rolling(long, min_periods=long).sum()
    down = m.clip(upper=0.0).rolling(long, min_periods=long).sum()
    return 100.0 * _safe_div(up, up - down)


def smi(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 10, d: int = 3, ema_period: int = 3) -> pd.DataFrame:
    """Stochastic momentum index with signal line."""
    h = _as_series(high)
    l = _as_series(low)
    c = _as_series(close).reindex_like(h)
    hh = h.rolling(k, min_periods=k).max()
    ll = l.rolling(k, min_periods=k).min()
    mid = (hh + ll) / 2.0
    distance = c - mid
    rng = hh - ll
    ds = ema(ema(distance, ema_period), ema_period)
    rs = ema(ema(rng, ema_period), ema_period) / 2.0
    smi_line = 100.0 * _safe_div(ds, rs)
    signal = sma(smi_line, d)
    return pd.DataFrame({"smi": smi_line, "signal": signal}, index=h.index)


def bop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, smooth: int = 14) -> pd.Series:
    """Balance of power: (close - open) / (high - low) with SMA smoothing."""
    o = _as_series(open_)
    h = _as_series(high).reindex_like(o)
    l = _as_series(low).reindex_like(o)
    c = _as_series(close).reindex_like(o)
    raw = _safe_div(c - o, h - l)
    return sma(raw, smooth) if smooth > 1 else raw


def cti(close: pd.Series, period: int = 12) -> pd.Series:
    """Correlation trend indicator: corr(price, time) over ``period`` bars."""
    s = _as_series(close)
    x = np.arange(period, dtype="float64")
    x_std = x.std()
    x_mean = x.mean()

    def _corr(w: np.ndarray) -> float:
        y_std = float(w.std())
        if y_std == 0:
            return 0.0
        return float(((x - x_mean) * (w - w.mean())).sum() / (len(w) * x_std * y_std))

    return _rolling_apply(s, period, _corr)


def vhf(close: pd.Series, period: int = 28) -> pd.Series:
    """Vertical horizontal filter: trending vs ranging markets."""
    s = _as_series(close)
    num = s.rolling(period, min_periods=period).max() - s.rolling(period, min_periods=period).min()
    den = s.diff().abs().rolling(period, min_periods=period).sum()
    return _safe_div(num, den)


def efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman efficiency ratio: directional move over path length."""
    s = _as_series(close)
    direction = (s - s.shift(period)).abs()
    volatility = s.diff().abs().rolling(period, min_periods=period).sum()
    return _safe_div(direction, volatility)


def dynamic_momentum_index(close: pd.Series, min_period: int = 5, max_period: int = 30, smooth: int = 14) -> pd.Series:
    """RSI whose lookback adapts to volatility: more speed in calm markets."""
    s = _as_series(close)
    fast = s.pct_change(fill_method=None).rolling(min_period, min_periods=min_period).std()
    slow = s.pct_change(fill_method=None).rolling(max_period, min_periods=max_period).std()
    ratio = _safe_div(fast, slow).clip(0.0, 1.0)
    period = (ratio * (max_period - min_period) + min_period).round().clip(min_period, max_period)
    candidates = {}
    for p in range(min_period, max_period + 1):
        candidates[p] = rsi(s, int(p)).to_numpy(dtype="float64")
    cols = np.vstack([candidates[p] for p in range(min_period, max_period + 1)])
    idx = (period.fillna(min_period).to_numpy(dtype="float64") - min_period).astype(int)
    idx = np.clip(idx, 0, max_period - min_period)
    values = cols[idx, np.arange(len(s))]
    out = pd.Series(values, index=s.index)
    return sma(out, smooth) if smooth > 1 else out


def pgo(close: pd.Series, period: int = 14) -> pd.Series:
    """Pretty good oscillator: deviation of price from its average, ATR-scaled."""
    s = _as_series(close)
    base = sma(s, period)
    scale = s.diff().abs().rolling(period, min_periods=period).mean()
    return _safe_div(s - base, scale)


def qstick(open_: pd.Series, close: pd.Series, period: int = 8) -> pd.Series:
    """Qstick: moving average of candle bodies (close - open)."""
    o = _as_series(open_)
    c = _as_series(close).reindex_like(o)
    return sma(c - o, period)


def fisher_rsi(series: pd.Series, rsi_period: int = 14, fisher_period: int = 10) -> pd.Series:
    """Fisher transform of RSI: sharpens oscillator turning points."""
    r = rsi(_as_series(series), rsi_period)
    lo = r.rolling(fisher_period, min_periods=fisher_period).min()
    hi = r.rolling(fisher_period, min_periods=fisher_period).max()
    scaled = (2.0 * _safe_div(r - lo, hi - lo) - 1.0).fillna(0.0).clip(-0.999, 0.999)
    out = np.empty(len(scaled), dtype="float64")
    prev = 0.0
    values = scaled.to_numpy()
    for i, v in enumerate(values):
        v = 0.0 if not np.isfinite(v) else v
        val = 0.33 * v + 0.67 * prev
        val = float(np.clip(val, -0.999, 0.999))
        out[i] = 0.5 * np.log((1.0 + val) / (1.0 - val)) + 0.5 * prev
        prev = val
    return pd.Series(out, index=scaled.index)


def stoch_slow_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    """Smoothed stochastic RSI with %K and %D lines."""
    from quant.indicators import stoch_rsi

    raw = stoch_rsi(_as_series(series), rsi_period, 1, 1)
    k_line = sma(raw["k"], k)
    d_line = sma(k_line, d)
    return pd.DataFrame({"k": k_line, "d": d_line}, index=raw.index)


def sma_oscillator(series: pd.Series, fast: int = 10, slow: int = 20) -> pd.Series:
    """Percent spread between a fast and slow SMA."""
    s = _as_series(series)
    f = sma(s, fast)
    sl = sma(s, slow)
    return 100.0 * _safe_div(f - sl, sl)


def imp(open_: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Intraday momentum index: RSI-style ratio of body gains to total bodies."""
    o = _as_series(open_)
    c = _as_series(close).reindex_like(o)
    body = c - o
    up = body.clip(lower=0.0).rolling(period, min_periods=period).sum()
    dn = (-body.clip(upper=0.0)).rolling(period, min_periods=period).sum()
    return 100.0 * _safe_div(up, up + dn)


def intraday_intensity(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """Intraday intensity index: closing location weighted by volume, summed."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    v = _as_series(volume).reindex_like(h)
    ii = _safe_div(2.0 * c - h - l, h - l) * v
    return ii.rolling(period, min_periods=period).sum()


def roc_ma(series: pd.Series, roc_period: int = 12, ma_period: int = 5) -> pd.Series:
    """Smoothed rate of change."""
    return sma(roc(_as_series(series), roc_period), ma_period)


def momentum_score(close: pd.Series, periods: tuple[int, ...] = (5, 10, 21, 63, 126, 252)) -> pd.Series:
    """Composite multi-horizon momentum score: mean of rolling z-scored ROC."""
    s = _as_series(close)
    parts = []
    for p in periods:
        r = roc(s, min(p, max(len(s) - 1, 1)))
        z = (r - r.expanding(min_periods=2).mean()) / r.expanding(min_periods=2).std().replace(0, np.nan)
        parts.append(z)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1)


def price_momentum_percentile(close: pd.Series, period: int = 60, lookback: int = 252) -> pd.Series:
    """Percentile rank of N-bar price momentum over a longer lookback."""
    m = momentum(_as_series(close), period)
    return m.rolling(lookback, min_periods=max(period + 1, 20)).rank(pct=True) * 100.0


def gain_loss_ratio(series: pd.Series, period: int = 14) -> pd.Series:
    """Average gain divided by average loss (the RS inside RSI)."""
    s = _as_series(series)
    d = s.diff()
    gain = d.clip(lower=0.0).rolling(period, min_periods=period).mean()
    loss = (-d.clip(upper=0.0)).rolling(period, min_periods=period).mean()
    return _safe_div(gain, loss)


def up_down_volume_ratio(close: pd.Series, volume: pd.Series, period: int = 50) -> pd.Series:
    """Ratio of up-day volume to down-day volume over a rolling window."""
    c = _as_series(close)
    v = _as_series(volume).reindex_like(c)
    up = (c.diff() > 0).astype("float64")
    dn = (c.diff() < 0).astype("float64")
    up_vol = (up * v).rolling(period, min_periods=period).sum()
    dn_vol = (dn * v).rolling(period, min_periods=period).sum()
    return _safe_div(up_vol, dn_vol)


def hist_vol_osc(close: pd.Series, fast: int = 10, slow: int = 30) -> pd.Series:
    """Volatility oscillator: percent spread between short and long realized vol."""
    s = _as_series(close)
    r = s.pct_change(fill_method=None)
    vol_f = r.rolling(fast, min_periods=fast).std()
    vol_s = r.rolling(slow, min_periods=slow).std()
    return 100.0 * _safe_div(vol_f - vol_s, vol_s)


def rvi_volatility(close: pd.Series, period: int = 10, smooth: int = 14) -> pd.Series:
    """Relative volatility index (Dorsey): up-vol share of total volatility."""
    s = _as_series(close)
    sd = s.rolling(period, min_periods=period).std()
    up = (s.diff() > 0)
    up_vol = sd.where(up, 0.0).ewm(span=smooth, adjust=False, min_periods=smooth).mean()
    dn_vol = sd.where(~up, 0.0).ewm(span=smooth, adjust=False, min_periods=smooth).mean()
    return 100.0 * _safe_div(up_vol, up_vol + dn_vol)


def atr_roc(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, roc_period: int = 5) -> pd.Series:
    """Rate of change of ATR: accelerating or decelerating volatility."""
    return roc(atr(high, low, close, period), roc_period)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def stddev(series: pd.Series, period: int = 20, ddof: int = 0) -> pd.Series:
    """Rolling standard deviation."""
    return _as_series(series).rolling(period, min_periods=period).std(ddof=ddof)


def variance(series: pd.Series, period: int = 20, ddof: int = 1) -> pd.Series:
    """Rolling variance."""
    return _as_series(series).rolling(period, min_periods=period).var(ddof=ddof)


def mean_absolute_deviation(series: pd.Series, period: int = 20) -> pd.Series:
    """Mean absolute deviation from the rolling mean."""
    s = _as_series(series)
    m = s.rolling(period, min_periods=period).mean()
    return (s - m).abs().rolling(period, min_periods=period).mean()


def semideviation(series: pd.Series, period: int = 20, annualize: int = 252) -> pd.Series:
    """Annualized downside deviation of returns."""
    r = _as_series(series)
    neg = r.clip(upper=0.0)
    return np.sqrt((neg ** 2).rolling(period, min_periods=period).mean()) * np.sqrt(annualize)


def upside_deviation(series: pd.Series, period: int = 20, annualize: int = 252) -> pd.Series:
    """Annualized upside deviation of returns."""
    r = _as_series(series)
    pos = r.clip(lower=0.0)
    return np.sqrt((pos ** 2).rolling(period, min_periods=period).mean()) * np.sqrt(annualize)


def chaikin_volatility(high: pd.Series, low: pd.Series, period: int = 10, roc_period: int = 10) -> pd.Series:
    """Chaikin volatility: ROC of the EMA of the high-low range."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    hl_ema = ema(h - l, period)
    return roc(hl_ema, roc_period)


def adr(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """Average daily range (absolute)."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    return (h - l).rolling(period, min_periods=period).mean()


def adr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average daily range as percent of close."""
    c = _as_series(close)
    return 100.0 * _safe_div(adr(high, low, period), c)


def atr_percent(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR as percent of close."""
    c = _as_series(close)
    return 100.0 * _safe_div(atr(high, low, close, period), c)


def bb_width(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger band width: (upper - lower) / middle."""
    bb = bollinger_bands(_as_series(series), period, num_std)
    return _safe_div(bb["upper"] - bb["lower"], bb["mid"])


def bb_percent_b(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Bollinger %B: position of price within the bands (0 = lower, 1 = upper)."""
    return bollinger_bands(_as_series(series), period, num_std)["pct_b"]


def kc_width(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Keltner channel width relative to its middle band."""
    kc = keltner_channels(high, low, close, period, mult)
    return _safe_div(kc["upper"] - kc["lower"], kc["mid"])


def atr_bands(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """SMA middle band with ATR-scaled upper/lower bands."""
    c = _as_series(close)
    mid = sma(c, period)
    a = atr(high, low, close, period)
    return pd.DataFrame({"upper": mid + mult * a, "middle": mid, "lower": mid - mult * a}, index=c.index)


def bipower_variation(close: pd.Series, period: int = 20) -> pd.Series:
    """Bipower variation: jump-robust realized variance estimator."""
    r = _as_series(close).pct_change(fill_method=None)
    prod = (r.abs() * r.abs().shift(1)) * (np.pi / 2.0)
    return prod.rolling(period, min_periods=period).sum()


def jump_variation(close: pd.Series, period: int = 20) -> pd.Series:
    """Jump variation: realized variance minus bipower variation (floored at 0)."""
    r = _as_series(close).pct_change(fill_method=None)
    rv = (r ** 2).rolling(period, min_periods=period).sum()
    return (rv - bipower_variation(close, period)).clip(lower=0.0)


def realized_semivariance(close: pd.Series, period: int = 20, annualize: int = 252) -> pd.DataFrame:
    """Annualized upside/downside semivariance of returns."""
    r = _as_series(close).pct_change(fill_method=None)
    down = (r.clip(upper=0.0) ** 2).rolling(period, min_periods=period).sum() * annualize
    up = (r.clip(lower=0.0) ** 2).rolling(period, min_periods=period).sum() * annualize
    return pd.DataFrame({"downside": down, "upside": up}, index=r.index)


def vol_of_vol(close: pd.Series, vol_period: int = 20, vov_period: int = 20) -> pd.Series:
    """Volatility of volatility: rolling std of realized volatility."""
    r = _as_series(close).pct_change(fill_method=None)
    rv = r.rolling(vol_period, min_periods=vol_period).std()
    return rv.rolling(vov_period, min_periods=vov_period).std()


def volatility_ratio(close: pd.Series, short: int = 10, long_: int = 60) -> pd.Series:
    """Ratio of short-window to long-window realized volatility."""
    r = _as_series(close).pct_change(fill_method=None)
    return _safe_div(
        r.rolling(short, min_periods=short).std(),
        r.rolling(long_, min_periods=long_).std(),
    )


def true_range_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range as percent of close."""
    c = _as_series(close)
    return 100.0 * _safe_div(_true_range(_as_series(high), _as_series(low).reindex_like(c), c), c)


def range_expansion_index(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 5, period: int = 20) -> pd.Series:
    """Ratio of the current true range to its recent average."""
    tr = _true_range(_as_series(high), _as_series(low), _as_series(close))
    return _safe_div(tr.rolling(lookback, min_periods=1).mean(), tr.rolling(period, min_periods=period).mean())


def gap_volatility(open_: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling volatility of overnight gaps."""
    o = _as_series(open_)
    c = _as_series(close).reindex_like(o)
    gaps = _safe_div(o - c.shift(1), c.shift(1))
    return gaps.rolling(period, min_periods=period).std()


# ---------------------------------------------------------------------------
# Volume analytics
# ---------------------------------------------------------------------------

def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/distribution line (Chaikin): CLV-weighted cumulative volume."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    v = _as_series(volume).reindex_like(h)
    clv = _safe_div(2.0 * c - h - l, h - l)
    return (clv * v).cumsum()


def vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume price trend: cumulative volume scaled by percent returns."""
    c = _as_series(close)
    v = _as_series(volume).reindex_like(c)
    change = _safe_div(c - c.shift(1), c.shift(1))
    return (v * change).fillna(0.0).cumsum()


def wad(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Williams accumulation/distribution: bar-by-bar accumulation measure."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    prev_c = c.shift(1)
    up = c > prev_c
    dn = c < prev_c
    add = c - np.minimum(l, prev_c)
    sub = c - np.maximum(h, prev_c)
    step = pd.Series(0.0, index=c.index)
    step[up] = add[up]
    step[dn] = sub[dn]
    return step.fillna(0.0).cumsum()


def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume relative to its simple moving average."""
    v = _as_series(volume)
    return _safe_div(v, sma(v, period))


def volume_oscillator(volume: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Percent spread between fast and slow volume SMAs."""
    v = _as_series(volume)
    return 100.0 * _safe_div(sma(v, fast) - sma(v, slow), sma(v, slow))


def vzo(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Volume zone oscillator: signed-volume share of total volume."""
    c = _as_series(close)
    v = _as_series(volume).reindex_like(c)
    signed = v * np.sign(c.diff())
    num = ema(signed.fillna(0.0), period)
    den = ema(v, period)
    return 100.0 * _safe_div(num, den)


def twiggs_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """Twiggs money flow: closing location within the true range, volume weighted."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    v = _as_series(volume).reindex_like(h)
    true_low = np.minimum(l, c.shift(1))
    tr = _true_range(h, l, c)
    raw = _safe_div(c - true_low, tr) * v
    return ema(raw, period)


def volume_rsi(volume: pd.Series, period: int = 14) -> pd.Series:
    """RSI applied to volume: detects volume thrusts."""
    return rsi(_as_series(volume), period)


def obv_ema(close: pd.Series, volume: pd.Series, period: int = 21) -> pd.Series:
    """EMA of on-balance volume."""
    return ema(obv(_as_series(close), _as_series(volume)), period)


def dollar_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Dollar (turnover) volume."""
    return _as_series(close) * _as_series(volume).reindex_like(_as_series(close))


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    """Z-score of volume versus its rolling distribution."""
    return _zscore(_as_series(volume), period)


def force_ratio(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Force ratio: smoothed price-change force divided by smoothed volume."""
    c = _as_series(close)
    v = _as_series(volume).reindex_like(c)
    force = c.diff() * v
    return _safe_div(ema(force, period), ema(v, period))


def buy_pressure(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Buying pressure: volume weighted by close position in the bar, summed."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    v = _as_series(volume).reindex_like(h)
    bp = _safe_div(c - l, h - l) * v
    return bp.rolling(period, min_periods=period).sum()


def volume_breakout(volume: pd.Series, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Boolean: volume above ``mult`` times its rolling average."""
    v = _as_series(volume)
    avg = sma(v, period)
    return ((v > mult * avg) & avg.notna()).fillna(False)


def volume_dryup(volume: pd.Series, period: int = 20, ratio: float = 0.5) -> pd.Series:
    """Boolean: volume below ``ratio`` times its rolling average."""
    v = _as_series(volume)
    avg = sma(v, period)
    return ((v < ratio * avg) & avg.notna()).fillna(False)


def volume_trend(volume: pd.Series, period: int = 20) -> pd.Series:
    """Normalized slope of volume (positive = expanding participation)."""
    v = _as_series(volume)
    slope = _linreg_slope(v, period)
    return _safe_div(slope * period, v.rolling(period, min_periods=period).mean())


def volume_price_corr(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling correlation between price level and volume."""
    return _rolling_corr(_as_series(close), _as_series(volume), period)


def turnover_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Raw volume divided by its rolling average (alias of relative volume)."""
    return relative_volume(volume, period)


def relative_strength(close: pd.Series, benchmark_close: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength versus a benchmark: smoothed ratio of prices."""
    c = _as_series(close)
    b = _as_series(benchmark_close).reindex_like(c)
    ratio = _safe_div(c, b)
    return sma(ratio, period)


# ---------------------------------------------------------------------------
# Trend quality
# ---------------------------------------------------------------------------

def trend_quality(close: pd.Series, period: int = 20) -> pd.Series:
    """R-squared of a rolling linear fit: how clean the trend is (0..1)."""
    s = _as_series(close)
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    x_ss = float(((x - x_mean) ** 2).sum())

    def _r2(w: np.ndarray) -> float:
        y = w
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 0:
            return 0.0
        slope = float(((x - x_mean) * (y - y.mean())).sum() / x_ss)
        intercept = y.mean() - slope * x_mean
        resid = y - (intercept + slope * x)
        ss_res = float((resid ** 2).sum())
        return max(0.0, 1.0 - ss_res / ss_tot)

    return _rolling_apply(s, period, _r2)


def trend_score(close: pd.Series, period: int = 20) -> pd.Series:
    """Composite trend strength: signed slope scaled by R-squared and volatility.

    Output roughly in [-3, 3]: positive = strong clean uptrend.
    """
    s = _as_series(close)
    slope = _linreg_slope(s, period)
    r2 = trend_quality(s, period)
    vol = s.pct_change(fill_method=None).rolling(period, min_periods=period).std()
    return _safe_div(slope * period, s * vol) * r2


def trend_intensity(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Signed ADX-like intensity in [-1, 1]: trend strength with direction."""
    c = _as_series(close)
    strength = adx(high, low, close, period) / 100.0
    direction = np.sign(_linreg_slope(c, period * 2))
    return (strength * direction).clip(-1.0, 1.0)


def velocity(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change of a smoothed series (first derivative)."""
    s = _as_series(series)
    e = ema(s, period)
    return 100.0 * _safe_div(e - e.shift(period), e.shift(period))


def acceleration(series: pd.Series, period: int = 10) -> pd.Series:
    """Change in velocity (second derivative of the smoothed series)."""
    return velocity(series, period).diff(period)


def curvature(series: pd.Series, period: int = 20, step: int = 5) -> pd.Series:
    """Discrete second derivative of a smoothed series."""
    s = _as_series(series)
    e = ema(s, period)
    return _safe_div(e - 2.0 * e.shift(step) + e.shift(2 * step), e.shift(step).abs(), fill=np.nan)


def directional_bias(high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    """Wilder directional bias: net directional movement share in [-1, 1]."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    up = (h - h.shift(1)).clip(lower=0.0)
    down = (l.shift(1) - l).clip(lower=0.0)
    plus = _wilder(up.where(up > down, 0.0), period)
    minus = _wilder(down.where(down > up, 0.0), period)
    return _safe_div(plus - minus, plus + minus).clip(-1.0, 1.0)


def aroon_oscillator(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    """Aroon oscillator: Aroon up minus Aroon down, in [-100, 100]."""
    return _aroon(high, low, period)["osc"]


def dmi_spread(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Plus DI minus minus DI: directional dominance in [-100, 100]."""
    d = dmi(high, low, close, period)
    plus = d["plus_di"] if "plus_di" in d.columns else d.iloc[:, 0]
    minus = d["minus_di"] if "minus_di" in d.columns else d.iloc[:, 1]
    return plus - minus


def adxr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX rating: average of ADX and its value ``period`` bars ago."""
    a = adx(high, low, close, period)
    return (a + a.shift(period)) / 2.0


def curve_smoothness(series: pd.Series, period: int = 20) -> pd.Series:
    """Smoothness score in [0, 1]: low path curvature relative to range."""
    s = _as_series(series)
    second = s.diff().diff().abs().rolling(period, min_periods=period).mean()
    rng = (s.rolling(period, min_periods=period).max() - s.rolling(period, min_periods=period).min())
    return 1.0 / (1.0 + _safe_div(second, rng))


def range_position(close: pd.Series, period: int = 20) -> pd.Series:
    """Position of price within its rolling range, scaled to [-1, 1]."""
    s = _as_series(close)
    hi = s.rolling(period, min_periods=period).max()
    lo = s.rolling(period, min_periods=period).min()
    return 2.0 * _safe_div(s - lo, hi - lo) - 1.0


def distance_from_high(close: pd.Series, period: int = 252) -> pd.Series:
    """Percent drawdown from the rolling high (<= 0)."""
    s = _as_series(close)
    hi = s.rolling(period, min_periods=min(period, 20)).max()
    return _safe_div(s - hi, hi) * 100.0


def distance_from_low(close: pd.Series, period: int = 252) -> pd.Series:
    """Percent premium to the rolling low (>= 0)."""
    s = _as_series(close)
    lo = s.rolling(period, min_periods=min(period, 20)).min()
    return _safe_div(s - lo, lo) * 100.0


def yearly_position(close: pd.Series, period: int = 252) -> pd.Series:
    """Percentile position within the trailing year's range (0..100)."""
    s = _as_series(close)
    lo = s.rolling(period, min_periods=min(period, 20)).min()
    hi = s.rolling(period, min_periods=min(period, 20)).max()
    return 100.0 * _safe_div(s - lo, hi - lo)


# ---------------------------------------------------------------------------
# Price transforms & pivots
# ---------------------------------------------------------------------------

def hl2(high: pd.Series, low: pd.Series) -> pd.Series:
    """Midpoint of high and low."""
    return (_as_series(high) + _as_series(low).reindex_like(_as_series(high))) / 2.0


def hlc3(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Typical price: (high + low + close) / 3."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    return (h + l + c) / 3.0


def ohlc4(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Average price: (open + high + low + close) / 4."""
    o = _as_series(open_)
    h = _as_series(high).reindex_like(o)
    l = _as_series(low).reindex_like(o)
    c = _as_series(close).reindex_like(o)
    return (o + h + l + c) / 4.0


def weighted_close(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Weighted close: (high + low + 2 * close) / 4."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    return (h + l + 2.0 * c) / 4.0


def mid_point(series: pd.Series, period: int = 14) -> pd.Series:
    """Midpoint of the rolling N-bar range."""
    s = _as_series(series)
    return (s.rolling(period, min_periods=period).max() + s.rolling(period, min_periods=period).min()) / 2.0


def hl_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bar range: high minus low."""
    return _as_series(high) - _as_series(low).reindex_like(_as_series(high))


def hl_range_pct(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Bar range as percent of close."""
    c = _as_series(close)
    return 100.0 * _safe_div(hl_range(high, low), c)


def close_location_value(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Close location value in [-1, 1]: where the close sits inside the bar."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    return _safe_div((c - l) - (h - c), h - l)


def gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap in percent: open versus previous close."""
    o = _as_series(open_)
    c = _as_series(close).reindex_like(o)
    return 100.0 * _safe_div(o - c.shift(1), c.shift(1))


def true_high(high: pd.Series, close: pd.Series) -> pd.Series:
    """True high: max(high, previous close)."""
    h = _as_series(high)
    c = _as_series(close).reindex_like(h)
    return np.maximum(h, c.shift(1))


def true_low(low: pd.Series, close: pd.Series) -> pd.Series:
    """True low: min(low, previous close)."""
    l = _as_series(low)
    c = _as_series(close).reindex_like(l)
    return np.minimum(l, c.shift(1))


def camarilla_pivots(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Camarilla pivot levels derived from the current bar's HLC."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    rng = h - l
    r4 = rng * 0.55 + c
    r3 = rng * 0.275 + c
    r2 = rng * (1.1 / 6.0) + c
    r1 = rng * (1.1 / 12.0) + c
    s1 = c - rng * (1.1 / 12.0)
    s2 = c - rng * (1.1 / 6.0)
    s3 = c - rng * 0.275
    s4 = c - rng * 0.55
    return pd.DataFrame({"h4": r4, "h3": r3, "h2": r2, "h1": r1, "l1": s1, "l2": s2, "l3": s3, "l4": s4}, index=h.index)


def woodie_pivots(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Woodie pivot points (close weighted 2x in the pivot)."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    p = (h + l + 2.0 * c) / 4.0
    r1 = 2.0 * p - l
    s1 = 2.0 * p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    return pd.DataFrame({"pivot": p, "r1": r1, "r2": r2, "s1": s1, "s2": s2}, index=h.index)


def demark_pivots(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """DeMark pivot points with conditional X calculation."""
    o = _as_series(open_)
    h = _as_series(high).reindex_like(o)
    l = _as_series(low).reindex_like(o)
    c = _as_series(close).reindex_like(o)
    x = pd.Series(np.where(c < o, h + 2.0 * l + c, np.where(c > o, 2.0 * h + l + c, h + l + 2.0 * c)), index=o.index)
    p = x / 4.0
    r1 = x / 2.0 - l
    s1 = x / 2.0 - h
    return pd.DataFrame({"pivot": p, "r1": r1, "s1": s1}, index=o.index)


def fibonacci_pivots(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Fibonacci pivot points (38.2% / 61.8% / 100% levels)."""
    h = _as_series(high)
    l = _as_series(low).reindex_like(h)
    c = _as_series(close).reindex_like(h)
    p = (h + l + c) / 3.0
    rng = h - l
    r1 = p + 0.382 * rng
    r2 = p + 0.618 * rng
    r3 = p + 1.000 * rng
    s1 = p - 0.382 * rng
    s2 = p - 0.618 * rng
    s3 = p - 1.000 * rng
    return pd.DataFrame({"pivot": p, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}, index=h.index)


def zigzag(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """ZigZag swing labels: +1 at confirmed peaks, -1 at confirmed troughs.

    A swing point is confirmed once price retraces ``pct`` (e.g. 0.05 = 5%)
    against the prevailing move. Non-pivot bars are NaN.
    """
    h = _as_series(high).to_numpy(dtype="float64")
    l = _as_series(low).to_numpy(dtype="float64")
    n = len(h)
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return pd.Series(out, index=_as_series(high).index)
    trend = 1
    ext_price = h[0]
    ext_idx = 0
    for i in range(1, n):
        if not (np.isfinite(h[i]) and np.isfinite(l[i])):
            continue
        if trend == 1:
            if h[i] > ext_price:
                ext_price = h[i]
                ext_idx = i
            elif l[i] <= ext_price * (1.0 - pct):
                out[ext_idx] = 1.0
                trend = -1
                ext_price = l[i]
                ext_idx = i
        else:
            if l[i] < ext_price:
                ext_price = l[i]
                ext_idx = i
            elif h[i] >= ext_price * (1.0 + pct):
                out[ext_idx] = -1.0
                trend = 1
                ext_price = h[i]
                ext_idx = i
    return pd.Series(out, index=_as_series(high).index)


def swing_high(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean: bar is a fractal swing high (higher than ``left``/``right`` neighbors)."""
    h = _as_series(high)
    w = left + right + 1
    roll_max = h.rolling(w, center=True, min_periods=w).max()
    return (h >= roll_max).fillna(False)


def swing_low(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean: bar is a fractal swing low (lower than ``left``/``right`` neighbors)."""
    l = _as_series(low)
    w = left + right + 1
    roll_min = l.rolling(w, center=True, min_periods=w).min()
    return (l <= roll_min).fillna(False)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_median(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling median."""
    return _as_series(series).rolling(period, min_periods=period).median()


def rolling_quantile(series: pd.Series, q: float = 0.5, period: int = 20) -> pd.Series:
    """Rolling quantile at level ``q`` in [0, 1]."""
    return _as_series(series).rolling(period, min_periods=period).quantile(q)


def iqr(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling interquartile range (75th minus 25th percentile)."""
    s = _as_series(series)
    return s.rolling(period, min_periods=period).quantile(0.75) - s.rolling(period, min_periods=period).quantile(0.25)


def trimmed_mean(series: pd.Series, period: int = 20, fraction: float = 0.1) -> pd.Series:
    """Rolling trimmed mean dropping ``fraction`` from each tail."""
    s = _as_series(series)
    k = int(max(1, np.floor(period * fraction)))

    def _trim(w: np.ndarray) -> float:
        if len(w) <= 2 * k:
            return float(np.mean(w))
        return float(np.mean(np.sort(w)[k:-k]))

    return _rolling_apply(s, period, _trim)


def winsorized_mean(series: pd.Series, period: int = 20, fraction: float = 0.1) -> pd.Series:
    """Rolling winsorized mean clipping ``fraction`` from each tail."""
    s = _as_series(series)
    k = int(max(1, np.floor(period * fraction)))

    def _wins(w: np.ndarray) -> float:
        if len(w) <= 2 * k:
            return float(np.mean(w))
        ws = np.sort(w)
        lo = ws[k]
        hi = ws[-k - 1]
        return float(np.clip(w, lo, hi).mean())

    return _rolling_apply(s, period, _wins)


def robust_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Z-score built on median and MAD (outlier resistant)."""
    s = _as_series(series)
    med = s.rolling(period, min_periods=period).median()
    mad = (s - med).abs().rolling(period, min_periods=period).median()
    return _safe_div(s - med, 1.4826 * mad)

