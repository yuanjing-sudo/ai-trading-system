"""Comprehensive vectorized technical indicator library.

All functions operate on pandas Series/DataFrames and return pandas objects
aligned to the input index. Implementations favor vectorized NumPy operations
so a 20-year daily bar series computes in milliseconds.

Conventions
-----------
- ``close``, ``high``, ``low``, ``open``, ``volume`` are pandas Series.
- Wilders smoothing uses ``alpha = 1 / period`` with ``adjust=False``.
- Functions never mutate their inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "wma", "dema", "tema", "hma", "kama", "vwma", "trima", "alma",
    "rsi", "stochastic", "stoch_rsi", "macd", "cci", "williams_r",
    "ultimate_oscillator", "aroon", "adx", "dmi", "atr", "natr", "true_range",
    "bollinger_bands", "keltner_channels", "donchian_channels", "envelope",
    "obv", "mfi", "cmf", "vwap", "anchored_vwap", "chaikin_oscillator",
    "force_index", "ease_of_movement", "volume_profile", "klinger",
    "hurst_exponent", "realized_volatility", "ewma_volatility",
    "parkinson_volatility", "garman_klass_volatility",
    "rogers_satchell_volatility", "yang_zhang_volatility", "volatility_cone",
    "skewness", "kurtosis", "zscore", "percentile_rank",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown",
    "drawdown_series", "omega_ratio", "ulcer_index",
    "linreg_slope", "linreg_channel", "linreg_forecast", "rolling_corr",
    "rolling_cov", "beta", "alpha", "tracking_error", "information_ratio",
    "entropy", "fractdim", "hurst_rs",
    "mass_index", "vortex", "trix", "dpo", "cmo", "coppock_curve",
    "ppo", "pvo", "rvi", "elder_ray", "choppiness_index", "relative_vigor",
    "know_sure_thing", "schaff_trend", "fisher_transform",
    "awesome_oscillator", "accelerator_oscillator", "squeeze_momentum",
    "supertrend", "parabolic_sar", "ichimoku", "pivot_points",
    "fib_retracements", "heikin_ashi", "renko", "price_channels",
    "historical_volatility_ratio", "amihud_illiquidity", "roll_spread",
    "turnover", "money_flow", "negative_volume_index", "positive_volume_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, dtype="float64")


def _safe_div(a, b, fill: float = 0.0):
    out = a / b.replace(0, np.nan)
    if isinstance(out, pd.Series):
        return out.fillna(fill)
    return np.where(b == 0, fill, out)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range (Wilder)."""
    return _true_range(high, low, close)


def _wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple moving average."""
    return _as_series(series).rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20, adjust: bool = False) -> pd.Series:
    """Exponential moving average."""
    return _as_series(series).ewm(span=period, adjust=adjust, min_periods=period).mean()


def wma(series: pd.Series, period: int = 20) -> pd.Series:
    """Weighted moving average (linear weights)."""
    s = _as_series(series)
    weights = np.arange(1, period + 1, dtype="float64")
    return s.rolling(period).apply(
        lambda w: np.dot(w, weights) / weights.sum(), raw=True
    )


def dema(series: pd.Series, period: int = 20) -> pd.Series:
    """Double exponential moving average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def tema(series: pd.Series, period: int = 20) -> pd.Series:
    """Triple exponential moving average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def trima(series: pd.Series, period: int = 20) -> pd.Series:
    """Triangular moving average (double smoothed SMA)."""
    s = _as_series(series)
    n = (period + 1) // 2
    return s.rolling(n, min_periods=n).mean().rolling(n, min_periods=n).mean()


def hma(series: pd.Series, period: int = 20) -> pd.Series:
    """Hull moving average."""
    s = _as_series(series)
    half = max(1, period // 2)
    sqrt_p = max(1, int(round(np.sqrt(period))))
    w1 = wma(s, half)
    w2 = wma(s, period)
    return wma(2 * w1 - w2, sqrt_p)


def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive moving average."""
    s = _as_series(series)
    change = (s - s.shift(period)).abs()
    volatility = s.diff().abs().rolling(period).sum()
    er = _safe_div(change, volatility, fill=0.0)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    result = np.full(len(s), np.nan)
    values = s.to_numpy(dtype="float64")
    sc_values = sc.to_numpy(dtype="float64")
    start = period
    if start < len(s):
        result[start] = values[start]
        for i in range(start + 1, len(s)):
            prev = result[i - 1]
            if np.isnan(prev):
                prev = values[i - 1] if not np.isnan(values[i - 1]) else values[i]
            a = sc_values[i]
            if np.isnan(a):
                a = slow_sc ** 2
            result[i] = prev + a * (values[i] - prev)
    return pd.Series(result, index=s.index)


def vwma(series: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume weighted moving average."""
    s = _as_series(series)
    v = _as_series(volume)
    return _safe_div((s * v).rolling(period).sum(), v.rolling(period).sum())


def alma(series: pd.Series, period: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux moving average."""
    s = _as_series(series).to_numpy(dtype="float64")
    m = offset * (period - 1)
    sd = period / sigma
    w = np.exp(-((np.arange(period) - m) ** 2) / (2 * sd * sd))
    w /= w.sum()
    out = np.full(len(s), np.nan)
    for i in range(period - 1, len(s)):
        out[i] = np.dot(s[i - period + 1 : i + 1], w)
    return pd.Series(out, index=_as_series(series).index)


# ---------------------------------------------------------------------------
# Momentum oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index."""
    s = _as_series(series)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = _safe_div(avg_gain, avg_loss, fill=100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3, smooth_k: int = 3,
) -> pd.DataFrame:
    """Stochastic oscillator (%K, %D)."""
    hh = _as_series(high).rolling(k_period).max()
    ll = _as_series(low).rolling(k_period).min()
    raw_k = 100.0 * _safe_div(_as_series(close) - ll, hh - ll)
    k = raw_k.rolling(smooth_k).mean() if smooth_k > 1 else raw_k
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def stoch_rsi(series: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> pd.DataFrame:
    """Stochastic RSI."""
    r = rsi(series, period)
    ll = r.rolling(period).min()
    hh = r.rolling(period).max()
    raw = 100.0 * _safe_div(r - ll, hh - ll)
    k = raw.rolling(k_period).mean()
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    s = _as_series(series)
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (_as_series(high) + _as_series(low) + _as_series(close)) / 3.0
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return _safe_div(tp - ma, 0.015 * md)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    hh = _as_series(high).rolling(period).max()
    ll = _as_series(low).rolling(period).min()
    return -100.0 * _safe_div(hh - _as_series(close), hh - ll)


def ultimate_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series,
    p1: int = 7, p2: int = 14, p3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator (Williams)."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    prev_close = c.shift(1)
    bp = c - pd.concat([l, prev_close], axis=1).min(axis=1)
    tr = _true_range(h, l, c)
    avg1 = _safe_div(bp.rolling(p1).sum(), tr.rolling(p1).sum())
    avg2 = _safe_div(bp.rolling(p2).sum(), tr.rolling(p2).sum())
    avg3 = _safe_div(bp.rolling(p3).sum(), tr.rolling(p3).sum())
    return 100.0 * (4 * avg1 + 2 * avg2 + avg3) / 7.0


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon Up/Down."""
    h, l = _as_series(high), _as_series(low)
    up = h.rolling(period + 1).apply(lambda x: float(np.argmax(x)), raw=True)
    down = l.rolling(period + 1).apply(lambda x: float(np.argmin(x)), raw=True)
    aroon_up = 100.0 * (period - (period - up)) / period
    aroon_down = 100.0 * down / period
    return pd.DataFrame({"up": aroon_up, "down": aroon_down, "osc": aroon_up - aroon_down})


def dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Directional Movement Index components (+DI, -DI)."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=h.index)
    atr_ = _wilder(_true_range(h, l, c), period)
    plus_di = 100.0 * _safe_div(_wilder(plus_dm, period), atr_)
    minus_di = 100.0 * _safe_div(_wilder(minus_dm, period), atr_)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    dm = dmi(high, low, close, period)
    dx = 100.0 * _safe_div((dm["plus_di"] - dm["minus_di"]).abs(), dm["plus_di"] + dm["minus_di"])
    return _wilder(dx, period)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return _wilder(_true_range(high, low, close), period)


def natr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Normalized ATR in percent."""
    return 100.0 * _safe_div(atr(high, low, close, period), _as_series(close))


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands with %B and bandwidth."""
    s = _as_series(series)
    mid = sma(s, period)
    sd = s.rolling(period).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    pct_b = _safe_div(s - lower, upper - lower, fill=0.5)
    width = _safe_div(upper - lower, mid)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "pct_b": pct_b, "width": width}
    )


def keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 20, mult: float = 2.0, atr_period: int = 10,
) -> pd.DataFrame:
    """Keltner Channels based on EMA and ATR."""
    mid = ema(close, period)
    band = mult * atr(high, low, close, atr_period)
    return pd.DataFrame({"mid": mid, "upper": mid + band, "lower": mid - band})


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Donchian Channels."""
    upper = _as_series(high).rolling(period).max()
    lower = _as_series(low).rolling(period).min()
    mid = (upper + lower) / 2.0
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": mid})


def envelope(series: pd.Series, period: int = 20, pct: float = 0.05) -> pd.DataFrame:
    """Moving average envelope."""
    mid = sma(series, period)
    return pd.DataFrame({"mid": mid, "upper": mid * (1 + pct), "lower": mid * (1 - pct)})


def realized_volatility(returns: pd.Series, period: int = 20, annualize: int = 252) -> pd.Series:
    """Close-to-close realized volatility (annualized by default)."""
    r = _as_series(returns)
    return r.rolling(period).std(ddof=0) * np.sqrt(annualize)


def ewma_volatility(returns: pd.Series, lam: float = 0.94, annualize: int = 252) -> pd.Series:
    """RiskMetrics EWMA volatility."""
    r = _as_series(returns)
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var) * np.sqrt(annualize)


def parkinson_volatility(
    high: pd.Series, low: pd.Series, period: int = 20, annualize: int = 252
) -> pd.Series:
    """Parkinson high-low range volatility estimator."""
    h, l = _as_series(high), _as_series(low)
    log_hl = np.log(_safe_div(h, l, fill=1.0)).abs()
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * log_hl.pow(2).rolling(period).mean()) * np.sqrt(annualize)


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 20, annualize: int = 252,
) -> pd.Series:
    """Garman-Klass volatility estimator."""
    o, h, l, c = map(_as_series, (open_, high, low, close))
    hl = np.log(_safe_div(h, l, fill=1.0)) ** 2
    co = np.log(_safe_div(c, o, fill=1.0)) ** 2
    est = 0.5 * hl - (2 * np.log(2.0) - 1.0) * co
    return np.sqrt(est.rolling(period).mean().clip(lower=0.0)) * np.sqrt(annualize)


def rogers_satchell_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 20, annualize: int = 252,
) -> pd.Series:
    """Rogers-Satchell volatility estimator (drift independent)."""
    o, h, l, c = map(_as_series, (open_, high, low, close))
    term = np.log(_safe_div(h, c, fill=1.0)) * np.log(_safe_div(h, o, fill=1.0)) + \
           np.log(_safe_div(l, c, fill=1.0)) * np.log(_safe_div(l, o, fill=1.0))
    return np.sqrt(term.rolling(period).mean().clip(lower=0.0)) * np.sqrt(annualize)


def yang_zhang_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 20, annualize: int = 252,
) -> pd.Series:
    """Yang-Zhang volatility estimator (handles overnight gaps)."""
    o, h, l, c = map(_as_series, (open_, high, low, close))
    log_oc = np.log(_safe_div(c, o, fill=1.0))
    log_co = np.log(_safe_div(o, c.shift(1), fill=1.0))
    rs = np.log(_safe_div(h, c, fill=1.0)) * np.log(_safe_div(h, o, fill=1.0)) + \
         np.log(_safe_div(l, c, fill=1.0)) * np.log(_safe_div(l, o, fill=1.0))
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    var_o = log_co.rolling(period).var(ddof=0)
    var_c = log_oc.rolling(period).var(ddof=0)
    var_rs = rs.rolling(period).mean()
    return np.sqrt((var_o + k * var_c + (1 - k) * var_rs).clip(lower=0.0)) * np.sqrt(annualize)


def volatility_cone(
    returns: pd.Series, windows=(5, 10, 20, 30, 60, 90, 120, 252), quantiles=(0.1, 0.5, 0.9)
) -> pd.DataFrame:
    """Volatility cone: realized vol quantiles across lookback windows."""
    r = _as_series(returns)
    rows = {}
    for w in windows:
        rv = r.rolling(w).std(ddof=0) * np.sqrt(252)
        rows[w] = rv.quantile(list(quantiles)).to_dict()
    df = pd.DataFrame(rows).T
    df.columns = [f"q{int(q * 100)}" for q in quantiles]
    df["current"] = [r.iloc[-w:].std(ddof=0) * np.sqrt(252) for w in windows]
    return df


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    c, v = _as_series(close), _as_series(volume)
    direction = np.sign(c.diff()).fillna(0.0)
    return (direction * v).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (_as_series(high) + _as_series(low) + _as_series(close)) / 3.0
    raw = tp * _as_series(volume)
    delta = tp.diff()
    pos = raw.where(delta > 0, 0.0).rolling(period).sum()
    neg = raw.where(delta < 0, 0.0).rolling(period).sum()
    return 100.0 - 100.0 / (1.0 + _safe_div(pos, neg, fill=1.0))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    h, l, c, v = map(_as_series, (high, low, close, volume))
    mfm = _safe_div((c - l) - (h - c), h - l)
    mfv = mfm * v
    return _safe_div(mfv.rolling(period).sum(), v.rolling(period).sum())


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Cumulative VWAP."""
    tp = (_as_series(high) + _as_series(low) + _as_series(close)) / 3.0
    v = _as_series(volume)
    return _safe_div((tp * v).cumsum(), v.cumsum())


def anchored_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, anchor_iloc: int = 0
) -> pd.Series:
    """Anchored VWAP from a given bar index."""
    tp = (_as_series(high) + _as_series(low) + _as_series(close)) / 3.0
    v = _as_series(volume)
    pv = (tp * v).iloc[anchor_iloc:].cumsum()
    vv = v.iloc[anchor_iloc:].cumsum()
    return _safe_div(pv, vv)


def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                       fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Oscillator (EMA of A/D line)."""
    h, l, c, v = map(_as_series, (high, low, close, volume))
    mfm = _safe_div((c - l) - (h - c), h - l)
    ad = (mfm * v).cumsum()
    return ema(ad, fast) - ema(ad, slow)


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Elder Force Index."""
    c, v = _as_series(close), _as_series(volume)
    return ema(c.diff() * v, period)


def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Ease of Movement."""
    h, l, v = map(_as_series, (high, low, volume))
    distance = ((h + l) / 2.0) - ((h.shift(1) + l.shift(1)) / 2.0)
    box_ratio = _safe_div(v, (h - l) * 1e8, fill=1.0)
    emv = _safe_div(distance, box_ratio)
    return emv.rolling(period).mean()


def volume_profile(close: pd.Series, volume: pd.Series, bins: int = 20) -> pd.DataFrame:
    """Static volume-at-price histogram."""
    c, v = _as_series(close), _as_series(volume)
    hist, edges = np.histogram(c.dropna(), bins=bins, weights=v.reindex(c.dropna().index))
    mid = (edges[:-1] + edges[1:]) / 2.0
    return pd.DataFrame({"price": mid, "volume": hist})


def klinger(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
            fast: int = 34, slow: int = 55) -> pd.DataFrame:
    """Klinger Volume Oscillator."""
    h, l, c, v = map(_as_series, (high, low, close, volume))
    trend = np.sign((h + l + c) - (h.shift(1) + l.shift(1) + c.shift(1))).fillna(0.0)
    dm = h - l
    cumulative = dm.cumsum()
    same_trend = trend == trend.shift(1)
    cm = dm.where(~same_trend, cumulative)
    vf = v * trend.abs() * _safe_div(dm, cm, fill=0.0) * 100.0
    kvo = ema(vf, fast) - ema(vf, slow)
    return pd.DataFrame({"kvo": kvo, "signal": ema(kvo, 13)})


def negative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Negative Volume Index (starts at 1000)."""
    c, v = _as_series(close), _as_series(volume)
    roc = c.pct_change().fillna(0.0)
    nvi = pd.Series(1000.0, index=c.index)
    mask = v.diff() < 0
    factor = (1.0 + roc.where(mask, 0.0)).cumprod()
    return nvi * factor


def positive_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Positive Volume Index (starts at 1000)."""
    c, v = _as_series(close), _as_series(volume)
    roc = c.pct_change().fillna(0.0)
    pvi = pd.Series(1000.0, index=c.index)
    mask = v.diff() > 0
    factor = (1.0 + roc.where(mask, 0.0)).cumprod()
    return pvi * factor


def turnover(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Average dollar turnover."""
    return (_as_series(close) * _as_series(volume)).rolling(period).mean()


def money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Raw money flow (typical price x volume)."""
    tp = (_as_series(high) + _as_series(low) + _as_series(close)) / 3.0
    return tp * _as_series(volume)


# ---------------------------------------------------------------------------
# Statistical / fractal
# ---------------------------------------------------------------------------

def hurst_exponent(series: pd.Series, max_lag: int = 100) -> float:
    """Hurst exponent via variance of lagged differences (R/S style estimate)."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    if len(s) < max_lag + 2:
        max_lag = max(2, len(s) - 2)
    lags = np.arange(2, max_lag)
    tau = []
    valid_lags = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        std = np.std(diff)
        if std > 0:
            tau.append(std)
            valid_lags.append(lag)
    if len(tau) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(valid_lags), np.log(tau), 1)
    return float(slope)


def hurst_rs(series: pd.Series, min_window: int = 10, max_window: int = 100, step: int = 10) -> float:
    """Hurst exponent via classic rescaled range analysis."""
    s = _as_series(series).dropna().to_numpy(dtype="float64")
    windows = list(range(min_window, min(max_window, len(s) // 2) + 1, step))
    rs_values = []
    for w in windows:
        chunks = len(s) // w
        if chunks < 1:
            continue
        rs = []
        for i in range(chunks):
            chunk = s[i * w : (i + 1) * w]
            mean = chunk.mean()
            dev = np.cumsum(chunk - mean)
            r = dev.max() - dev.min()
            sd = chunk.std()
            if sd > 0:
                rs.append(r / sd)
        if rs:
            rs_values.append((w, np.mean(rs)))
    if len(rs_values) < 2:
        return np.nan
    x = np.log([v[0] for v in rs_values])
    y = np.log([v[1] for v in rs_values])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def entropy(returns: pd.Series, bins: int = 10, period: int = 60) -> pd.Series:
    """Rolling Shannon entropy of return distribution."""
    r = _as_series(returns)

    def _shannon(x):
        hist, _ = np.histogram(x, bins=bins)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    return r.rolling(period).apply(_shannon, raw=True)


def fractdim(series: pd.Series, period: int = 30) -> pd.Series:
    """Katz fractal dimension."""
    s = _as_series(series)

    def _fd(x):
        x = np.asarray(x, dtype="float64")
        diffs = np.abs(np.diff(x))
        length = diffs.sum()
        max_dist = np.abs(x - x[0]).max()
        if max_dist <= 0 or length <= 0:
            return 1.0
        return float(np.log(length / max_dist) / np.log(len(x) - 1) + 1.0)

    return s.rolling(period).apply(_fd, raw=True)


def skewness(returns: pd.Series, period: int = 20) -> pd.Series:
    """Rolling skewness."""
    return _as_series(returns).rolling(period).skew()


def kurtosis(returns: pd.Series, period: int = 20) -> pd.Series:
    """Rolling excess kurtosis."""
    return _as_series(returns).rolling(period).kurt()


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score."""
    s = _as_series(series)
    mu = s.rolling(period).mean()
    sd = s.rolling(period).std(ddof=0)
    return _safe_div(s - mu, sd)


def percentile_rank(series: pd.Series, period: int = 252) -> pd.Series:
    """Rolling percentile rank in [0, 1]."""
    return _as_series(series).rolling(period).apply(
        lambda x: float((x <= x[-1]).mean()), raw=True
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown series from an equity curve."""
    e = _as_series(equity)
    peak = e.cummax()
    return _safe_div(e - peak, peak)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd.dropna()) else np.nan


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, annualize: int = 252) -> float:
    """Annualized Sharpe ratio."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf / annualize
    sd = excess.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(annualize))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, annualize: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf / annualize
    downside = excess[excess < 0].std(ddof=1)
    if not downside or np.isnan(downside):
        return np.nan
    return float(excess.mean() / downside * np.sqrt(annualize))


def calmar_ratio(equity: pd.Series, annualize: int = 252) -> float:
    """Calmar ratio: CAGR / |MaxDD|."""
    e = _as_series(equity).dropna()
    if len(e) < 2:
        return np.nan
    years = len(e) / annualize
    if years <= 0:
        return np.nan
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if e.iloc[0] > 0 else np.nan
    mdd = abs(max_drawdown(e))
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr / mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0, annualize: int = 252) -> float:
    """Omega ratio at a return threshold."""
    r = _as_series(returns).dropna() - threshold / annualize
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer index (RMS drawdown)."""
    dd = drawdown_series(equity).dropna() * 100.0
    if dd.empty:
        return np.nan
    return float(np.sqrt((dd ** 2).mean()))


# ---------------------------------------------------------------------------
# Regression / relationship
# ---------------------------------------------------------------------------

def linreg_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling OLS slope of price versus time."""
    s = _as_series(series)
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y):
        y = np.asarray(y, dtype="float64")
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return s.rolling(period).apply(_slope, raw=True)


def linreg_channel(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Linear regression channel with residual bands."""
    s = _as_series(series)
    x = np.arange(period, dtype="float64")
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _intercept(y):
        y = np.asarray(y, dtype="float64")
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        return float(y.mean() - slope * x_mean)

    def _resid_std(y):
        y = np.asarray(y, dtype="float64")
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        intercept = y.mean() - slope * x_mean
        resid = y - (slope * x + intercept)
        return float(resid.std(ddof=0))

    intercept = s.rolling(period).apply(_intercept, raw=True)
    slope = linreg_slope(s, period)
    x_last = period - 1
    mid = intercept + slope * x_last
    sd = s.rolling(period).apply(_resid_std, raw=True)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": mid + num_std * sd,
            "lower": mid - num_std * sd,
            "slope": slope,
        }
    )


def linreg_forecast(series: pd.Series, period: int = 20, forecast: int = 1) -> pd.Series:
    """One-step-ahead OLS forecast within a rolling window."""
    s = _as_series(series)
    slope = linreg_slope(s, period)
    x_last = period - 1
    x_target = x_last + forecast
    intercept = s.rolling(period).mean() - slope * (x_last / 2.0)
    return intercept + slope * x_target


def rolling_corr(a: pd.Series, b: pd.Series, period: int = 60) -> pd.Series:
    """Rolling Pearson correlation."""
    return _as_series(a).rolling(period).corr(_as_series(b))


def rolling_cov(a: pd.Series, b: pd.Series, period: int = 60) -> pd.Series:
    """Rolling covariance."""
    return _as_series(a).rolling(period).cov(_as_series(b))


def beta(asset_returns: pd.Series, benchmark_returns: pd.Series, period: int = 60) -> pd.Series:
    """Rolling CAPM beta."""
    cov = rolling_cov(asset_returns, benchmark_returns, period)
    var = _as_series(benchmark_returns).rolling(period).var(ddof=0)
    return _safe_div(cov, var)


def alpha(asset_returns: pd.Series, benchmark_returns: pd.Series, period: int = 60, annualize: int = 252) -> pd.Series:
    """Rolling Jensen's alpha (annualized)."""
    a = _as_series(asset_returns)
    b = _as_series(benchmark_returns)
    beta_ = beta(a, b, period)
    alpha_daily = a.rolling(period).mean() - beta_ * b.rolling(period).mean()
    return alpha_daily * annualize


def tracking_error(asset_returns: pd.Series, benchmark_returns: pd.Series, period: int = 60, annualize: int = 252) -> pd.Series:
    """Rolling tracking error of active returns."""
    active = _as_series(asset_returns) - _as_series(benchmark_returns)
    return active.rolling(period).std(ddof=0) * np.sqrt(annualize)


def information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series, period: int = 60) -> pd.Series:
    """Rolling information ratio."""
    active = _as_series(asset_returns) - _as_series(benchmark_returns)
    mean = active.rolling(period).mean()
    te = active.rolling(period).std(ddof=0)
    return _safe_div(mean, te)


# ---------------------------------------------------------------------------
# Oscillators (extended set)
# ---------------------------------------------------------------------------

def mass_index(high: pd.Series, low: pd.Series, fast: int = 9, slow: int = 25) -> pd.Series:
    """Mass Index (Dorsey)."""
    rng = _as_series(high) - _as_series(low)
    e1 = ema(rng, fast)
    e2 = ema(e1, fast)
    ratio = _safe_div(e1, e2, fill=1.0)
    return ratio.rolling(slow).sum()


def vortex(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Vortex Indicator (+VI, -VI)."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    vm_plus = (h - l.shift(1)).abs()
    vm_minus = (l - h.shift(1)).abs()
    tr_sum = _true_range(h, l, c).rolling(period).sum()
    return pd.DataFrame(
        {
            "vi_plus": _safe_div(vm_plus.rolling(period).sum(), tr_sum),
            "vi_minus": _safe_div(vm_minus.rolling(period).sum(), tr_sum),
        }
    )


def trix(series: pd.Series, period: int = 15) -> pd.Series:
    """TRIX: 1-day ROC of triple EMA."""
    e3 = ema(ema(ema(series, period), period), period)
    return 100.0 * e3.pct_change()


def dpo(series: pd.Series, period: int = 20) -> pd.Series:
    """Detrended Price Oscillator."""
    s = _as_series(series)
    shift = period // 2 + 1
    return s - sma(s, period).shift(shift)


def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator."""
    s = _as_series(series)
    delta = s.diff()
    up = delta.clip(lower=0.0).rolling(period).sum()
    down = (-delta.clip(upper=0.0)).rolling(period).sum()
    return 100.0 * _safe_div(up - down, up + down)


def coppock_curve(series: pd.Series, roc1: int = 14, roc2: int = 11, wma_period: int = 10) -> pd.Series:
    """Coppock Curve."""
    s = _as_series(series)
    roc_sum = 100.0 * (s / s.shift(roc1) - 1.0) + 100.0 * (s / s.shift(roc2) - 1.0)
    return wma(roc_sum, wma_period)


def ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Percentage Price Oscillator."""
    s = _as_series(series)
    fast_e, slow_e = ema(s, fast), ema(s, slow)
    line = 100.0 * _safe_div(fast_e - slow_e, slow_e)
    sig = ema(line, signal)
    return pd.DataFrame({"ppo": line, "signal": sig, "hist": line - sig})


def pvo(volume: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Percentage Volume Oscillator."""
    v = _as_series(volume)
    fast_e, slow_e = ema(v, fast), ema(v, slow)
    line = 100.0 * _safe_div(fast_e - slow_e, slow_e)
    sig = ema(line, signal)
    return pd.DataFrame({"pvo": line, "signal": sig, "hist": line - sig})


def rvi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.DataFrame:
    """Relative Vigor Index."""
    o, h, l, c = map(_as_series, (open_, high, low, close))
    num = (c - o)
    den = (h - l)
    num_s = (num + 2 * num.shift(1) + 2 * num.shift(2) + num.shift(3)) / 6.0
    den_s = (den + 2 * den.shift(1) + 2 * den.shift(2) + den.shift(3)) / 6.0
    line = _safe_div(num_s.rolling(period).sum(), den_s.rolling(period).sum())
    signal = (line + 2 * line.shift(1) + 2 * line.shift(2) + line.shift(3)) / 6.0
    return pd.DataFrame({"rvi": line, "signal": signal})


def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13) -> pd.DataFrame:
    """Elder Ray Bull/Bear Power."""
    e = ema(close, period)
    return pd.DataFrame({"bull": _as_series(high) - e, "bear": _as_series(low) - e})


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Choppiness Index: high = choppy, low = trending."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    tr_sum = _true_range(h, l, c).rolling(period).sum()
    rng = h.rolling(period).max() - l.rolling(period).min()
    ratio = _safe_div(tr_sum, rng, fill=1.0)
    return 100.0 * np.log10(ratio.clip(lower=1e-9)) / np.log10(period)


def relative_vigor(close: pd.Series, period: int = 10) -> pd.Series:
    """Relative Vigor Index (close-based variant)."""
    c = _as_series(close)
    num = c.diff()
    den = c.rolling(2).max() - c.rolling(2).min()
    return _safe_div(num.rolling(period).mean(), den.rolling(period).mean())


def know_sure_thing(
    close: pd.Series, roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
    sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15, signal: int = 9,
) -> pd.DataFrame:
    """Know Sure Thing momentum oscillator."""
    c = _as_series(close)
    r1 = 100.0 * (c / c.shift(roc1) - 1.0)
    r2 = 100.0 * (c / c.shift(roc2) - 1.0)
    r3 = 100.0 * (c / c.shift(roc3) - 1.0)
    r4 = 100.0 * (c / c.shift(roc4) - 1.0)
    kst = sma(r1, sma1) + 2 * sma(r2, sma2) + 3 * sma(r3, sma3) + 4 * sma(r4, sma4)
    return pd.DataFrame({"kst": kst, "signal": sma(kst, signal)})


def schaff_trend(close: pd.Series, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.Series:
    """Schaff Trend Cycle."""
    m = macd(close, fast, slow, 9)["macd"]
    def _stoch(x, p):
        ll = x.rolling(p).min()
        hh = x.rolling(p).max()
        return 100.0 * _safe_div(x - ll, hh - ll, fill=50.0)
    k = _stoch(m, cycle).ewm(alpha=0.5, adjust=False).mean()
    d = _stoch(k, cycle).ewm(alpha=0.5, adjust=False).mean()
    return d


def fisher_transform(high: pd.Series, low: pd.Series, period: int = 10) -> pd.DataFrame:
    """Ehlers Fisher Transform."""
    h, l = _as_series(high), _as_series(low)
    mid = (h + l) / 2.0
    ll = mid.rolling(period).min()
    hh = mid.rolling(period).max()
    val = 2.0 * _safe_div(mid - ll, hh - ll) - 0.5
    val = val.clip(-0.999, 0.999)
    val = 0.33 * 2.0 * val + 0.67 * val.shift(1)
    fisher = 0.5 * np.log((1 + val) / (1 - val))
    return pd.DataFrame({"fisher": fisher, "signal": fisher.shift(1)})


def awesome_oscillator(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """Bill Williams Awesome Oscillator."""
    mid = (_as_series(high) + _as_series(low)) / 2.0
    return sma(mid, fast) - sma(mid, slow)


def accelerator_oscillator(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bill Williams Accelerator Oscillator."""
    ao = awesome_oscillator(high, low)
    return ao - sma(ao, 5)


def squeeze_momentum(
    high: pd.Series, low: pd.Series, close: pd.Series,
    bb_period: int = 20, bb_mult: float = 2.0, kc_period: int = 20, kc_mult: float = 1.5,
) -> pd.DataFrame:
    """TTM Squeeze Momentum."""
    bb = bollinger_bands(close, bb_period, bb_mult)
    kc = keltner_channels(high, low, close, kc_period, kc_mult)
    squeeze_on = (bb["lower"] > kc["lower"]) & (bb["upper"] < kc["upper"])
    mid = sma(close, bb_period)
    delta = _as_series(close) - ((sma(close, bb_period) + sma(close, bb_period)) / 2.0 + (sma(high, bb_period) + sma(low, bb_period)) / 2.0) / 2.0
    momentum = delta - delta.rolling(bb_period).mean()
    return pd.DataFrame({"momentum": momentum, "squeeze_on": squeeze_on})


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, mult: float = 3.0
) -> pd.DataFrame:
    """Supertrend indicator with direction."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    atr_ = atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    upper = (hl2 + mult * atr_).to_numpy(dtype="float64")
    lower = (hl2 - mult * atr_).to_numpy(dtype="float64")
    close_v = c.to_numpy(dtype="float64")
    n = len(c)
    trend = np.zeros(n)
    st = np.full(n, np.nan)
    direction = np.ones(n)
    for i in range(1, n):
        if np.isnan(upper[i]):
            continue
        if np.isnan(st[i - 1]):
            st[i] = upper[i]
            direction[i] = -1
            continue
        if close_v[i] > st[i - 1]:
            direction[i] = 1
            st[i] = max(lower[i], st[i - 1]) if direction[i - 1] == 1 else lower[i]
        else:
            direction[i] = -1
            st[i] = min(upper[i], st[i - 1]) if direction[i - 1] == -1 else upper[i]
    return pd.DataFrame({"supertrend": st, "direction": direction}, index=c.index)


def parabolic_sar(
    high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2
) -> pd.Series:
    """Parabolic SAR (Wilder)."""
    h = _as_series(high).to_numpy(dtype="float64")
    l = _as_series(low).to_numpy(dtype="float64")
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=high.index)
    bull = True
    af = step
    ep = h[0]
    sar[0] = l[0]
    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
        if bull:
            if l[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = l[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + step, max_step)
        else:
            if h[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = h[i]
                af = step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + step, max_step)
    return pd.Series(sar, index=high.index)


# ---------------------------------------------------------------------------
# Chart overlays / structural
# ---------------------------------------------------------------------------

def ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo cloud."""
    h, l = _as_series(high), _as_series(low)
    tenkan_line = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2.0
    kijun_line = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2.0
    senkou_a = ((tenkan_line + kijun_line) / 2.0).shift(kijun)
    senkou_b_line = ((h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2.0).shift(kijun)
    chikou = _as_series(close).shift(-kijun)
    return pd.DataFrame(
        {
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b_line,
            "chikou": chikou,
        }
    )


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Classic floor trader pivot points."""
    h, l, c = _as_series(high), _as_series(low), _as_series(close)
    p = (h + l + c) / 3.0
    r1 = 2 * p - l
    s1 = 2 * p - h
    r2 = p + (h - l)
    s2 = p - (h - l)
    r3 = h + 2 * (p - l)
    s3 = l - 2 * (h - p)
    return pd.DataFrame({"pivot": p, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3})


def fib_retracements(high: pd.Series, low: pd.Series, lookback: int = 100) -> pd.DataFrame:
    """Rolling Fibonacci retracement levels."""
    hh = _as_series(high).rolling(lookback).max()
    ll = _as_series(low).rolling(lookback).min()
    diff = hh - ll
    return pd.DataFrame(
        {
            "level_0": hh,
            "level_236": hh - 0.236 * diff,
            "level_382": hh - 0.382 * diff,
            "level_500": hh - 0.500 * diff,
            "level_618": hh - 0.618 * diff,
            "level_786": hh - 0.786 * diff,
            "level_1000": ll,
        }
    )


def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Heikin-Ashi smoothed candles."""
    o, h, l, c = map(_as_series, (open_, high, low, close))
    ha_close = (o + h + l + c) / 4.0
    ha_close_values = np.array(ha_close.to_numpy(dtype="float64"), copy=True)
    ha_open_values = np.array(ha_close_values, copy=True)
    ha_open_values[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    for i in range(1, len(ha_open_values)):
        ha_open_values[i] = (ha_open_values[i - 1] + ha_close_values[i - 1]) / 2.0
    ha_open = pd.Series(ha_open_values, index=h.index)
    ha_high = pd.concat([h, ha_open, pd.Series(ha_close_values, index=h.index)], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, pd.Series(ha_close_values, index=h.index)], axis=1).min(axis=1)
    return pd.DataFrame({"open": ha_open_values, "high": ha_high, "low": ha_low, "close": ha_close_values}, index=h.index)


def renko(close: pd.Series, brick_size: float | None = None) -> pd.DataFrame:
    """Renko brick series. Brick size defaults to ATR-like mean abs move."""
    c = _as_series(close).dropna().to_numpy(dtype="float64")
    if brick_size is None:
        brick_size = float(np.nanmean(np.abs(np.diff(c))))
        if not brick_size or np.isnan(brick_size):
            brick_size = 1.0
    bricks = []
    last = c[0]
    for price in c[1:]:
        while price - last >= brick_size:
            bricks.append((last, last + brick_size, 1))
            last += brick_size
        while last - price >= brick_size:
            bricks.append((last, last - brick_size, -1))
            last -= brick_size
    return pd.DataFrame(bricks, columns=["open", "close", "direction"])


def price_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Simple highest-high / lowest-low channels."""
    upper = _as_series(high).rolling(period).max()
    lower = _as_series(low).rolling(period).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2.0})


def historical_volatility_ratio(returns: pd.Series, short: int = 10, long_: int = 60) -> pd.Series:
    """Ratio of short-term to long-term realized volatility (vol regime signal)."""
    short_vol = realized_volatility(returns, short)
    long_vol = realized_volatility(returns, long_)
    return _safe_div(short_vol, long_vol, fill=1.0)


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, period: int = 20) -> pd.Series:
    """Amihud illiquidity: |return| / dollar volume, rolling mean."""
    illiq = _safe_div(_as_series(returns).abs(), _as_series(dollar_volume), fill=0.0)
    return illiq.rolling(period).mean() * 1e6


def roll_spread(close: pd.Series, period: int = 20) -> pd.Series:
    """Roll's implied bid-ask spread estimator."""
    c = _as_series(close)
    dp = c.diff()
    cov = dp.rolling(period).cov(dp.shift(1))
    spread = 2.0 * np.sqrt(cov.clip(lower=0.0))
    return spread
