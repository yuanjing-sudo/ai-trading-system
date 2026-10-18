"""Strategy library: 60+ rules-based strategies with a shared signal API.

Every strategy takes an OHLCV DataFrame (columns ``open, high, low, close,
volume``, case-insensitive) plus optional parameters, and returns a pandas
Series aligned to the index holding the desired position: ``+1`` long, ``-1``
short, ``0`` flat (a few volatility-targeted strategies return fractional
weights in ``[-1, 1]``, noted in their docstrings).

The module also exposes a registry (:data:`STRATEGIES`) with categories and
default parameters, plus helpers to run one strategy, run many at once, build
ensemble votes, and compare strategies by the correlation of their returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import (
    _as_series,
    adx,
    aroon,
    atr,
    bollinger_bands,
    cci,
    cmf,
    donchian_channels,
    ema,
    hma,
    ichimoku,
    kama,
    keltner_channels,
    macd,
    mfi,
    rsi,
    sma,
    squeeze_momentum,
    stochastic,
    stoch_rsi,
    supertrend,
    tema,
    trix,
    vortex,
    williams_r,
)
from quant.indicators_ext import efficiency_ratio, trend_score, velocity, volume_zscore, relative_volume, obv_ema
from quant.candlesticks import pattern_score
from quant.patterns_ext import break_of_structure, detect_patterns, range_position, rolling_support, rolling_resistance, rsi_divergence, structure_score

__all__ = [
    # utilities
    "ohlc_parts", "hold_positions", "strategy_returns", "long_only",
    # trend
    "sma_cross", "ema_cross", "triple_ma", "price_vs_sma", "ma_slope",
    "donchian_breakout", "turtle_system", "macd_trend", "adx_trend",
    "supertrend_strategy", "keltner_breakout", "bollinger_breakout",
    "ichimoku_trend", "aroon_trend", "vortex_trend", "hma_trend",
    "trend_quality_filter", "efficiency_trend", "tema_trend", "kama_trend",
    # momentum
    "momentum_signal", "momentum_3m", "risk_adjusted_momentum_strategy",
    "dual_momentum", "breakout_momentum", "momentum_vol_filter",
    "macd_momentum", "trix_signal", "momentum_acceleration", "faber_timing",
    # mean reversion
    "rsi_reversal", "rsi2_reversal", "bollinger_reversion",
    "percent_b_reversion", "zscore_reversion", "cci_reversal",
    "williams_reversal", "stoch_reversal", "stoch_rsi_reversal",
    "keltner_reversion", "vwap_reversion", "gap_reversion", "range_fade",
    "connors_rsi",
    # volatility
    "vol_regime_switch", "atr_channel_breakout", "squeeze_release",
    "vol_target_trend", "bollinger_squeeze_breakout", "vol_breakout_momentum",
    # volume
    "obv_trend", "mfi_reversal", "cmf_trend", "volume_breakout",
    "vwap_cross", "relative_volume_trend",
    # patterns
    "candle_pattern_score", "chart_pattern_structure", "divergence_strategy",
    "pivot_structure_trade", "support_resistance_trade", "engulfing_proxy",
    # composites
    "ensemble_vote", "regime_adaptive", "multi_timeframe_trend",
    "alpha_combo", "all_strategy_signals", "strategy_correlation",
    # registry / runners
    "STRATEGIES", "list_strategies", "get_strategy", "strategy_positions",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ohlc_parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (open, high, low, close, volume) with a volume fallback."""
    lookup = {str(c).lower(): c for c in df.columns}
    o = _as_series(df[lookup["open"]])
    h = _as_series(df[lookup["high"]])
    l = _as_series(df[lookup["low"]])
    c = _as_series(df[lookup["close"]])
    v = _as_series(df[lookup["volume"]]) if "volume" in lookup else pd.Series(np.nan, index=c.index)
    return o, h, l, c, v


def _flat(signal) -> pd.Series:
    s = _as_series(signal).astype("float64").fillna(0.0)
    return s.replace([np.inf, -np.inf], 0.0)


def _sign(series, deadzone: float = 0.0) -> pd.Series:
    s = _as_series(series).astype("float64")
    out = np.sign(s.where(s.abs() > deadzone))
    return out.fillna(0.0)


def hold_positions(enter_long, enter_short, exit_long=None, exit_short=None) -> pd.Series:
    """State machine: hold +1/-1 between entry and exit events.

    Entries are event Series that are truthy on the bar the position opens;
    exits close the corresponding side. Positions persist until an exit fires.
    """
    el = _as_series(enter_long).fillna(False).astype(bool).to_numpy()
    es = _as_series(enter_short).fillna(False).astype(bool).to_numpy()
    if exit_long is None:
        xl = np.zeros(len(el), dtype=bool)
    else:
        xl = _as_series(exit_long).fillna(False).astype(bool).to_numpy()
    if exit_short is None:
        xs = np.zeros(len(el), dtype=bool)
    else:
        xs = _as_series(exit_short).fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(el), dtype="float64")
    pos = 0.0
    for i in range(len(el)):
        if pos > 0 and xl[i]:
            pos = 0.0
        elif pos < 0 and xs[i]:
            pos = 0.0
        if pos == 0.0:
            if el[i]:
                pos = 1.0
            elif es[i]:
                pos = -1.0
        out[i] = pos
    return pd.Series(out, index=_as_series(enter_long).index, name="position")


def long_only(signal) -> pd.Series:
    """Clip a position series to long/flat."""
    return _flat(signal).clip(lower=0.0).rename("position")


def strategy_returns(df: pd.DataFrame, positions, lag: int = 1) -> pd.Series:
    """Bar returns of a position series (positions applied with ``lag`` bars)."""
    _, _, _, c, _ = ohlc_parts(df)
    pos = _flat(positions).reindex(c.index)
    r = c.pct_change(fill_method=None)
    return (pos.shift(lag) * r).rename("strategy_return")


# ---------------------------------------------------------------------------
# Trend strategies
# ---------------------------------------------------------------------------

def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast SMA is above the slow SMA, short otherwise."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(sma(c, fast) - sma(c, slow)).rename("position")


def ema_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """Long when the fast EMA is above the slow EMA, short otherwise."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(ema(c, fast) - ema(c, slow)).rename("position")


def triple_ma(df: pd.DataFrame, fast: int = 10, mid: int = 30, slow: int = 60) -> pd.Series:
    """Long when fast > mid > slow, short when fast < mid < slow, else flat."""
    _, _, _, c, _ = ohlc_parts(df)
    f, m, s = sma(c, fast), sma(c, mid), sma(c, slow)
    out = pd.Series(0.0, index=c.index)
    out[(f > m) & (m > s)] = 1.0
    out[(f < m) & (m < s)] = -1.0
    return out.rename("position")


def price_vs_sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Long above the SMA, short below it."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(c - sma(c, period)).rename("position")


def ma_slope(df: pd.DataFrame, period: int = 50, lookback: int = 10) -> pd.Series:
    """Sign of the SMA slope over ``lookback`` bars."""
    _, _, _, c, _ = ohlc_parts(df)
    m = sma(c, period)
    return _sign(m - m.shift(lookback)).rename("position")


def donchian_breakout(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Long on a breakout above the rolling high, short below the rolling low."""
    _, h, l, c, _ = ohlc_parts(df)
    dc = donchian_channels(h, l, period)
    return hold_positions(
        c > dc["upper"].shift(1), c < dc["lower"].shift(1),
        c < dc["lower"].shift(1), c > dc["upper"].shift(1),
    )


def turtle_system(df: pd.DataFrame, entry: int = 20, exit: int = 10) -> pd.Series:
    """Turtle-style breakout entries with shorter opposite exit channels."""
    _, h, l, c, _ = ohlc_parts(df)
    e = donchian_channels(h, l, entry)
    x = donchian_channels(h, l, exit)
    return hold_positions(
        c > e["upper"].shift(1), c < e["lower"].shift(1),
        c < x["lower"].shift(1), c > x["upper"].shift(1),
    )


def macd_trend(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Long when the MACD histogram is positive, short when negative."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(macd(c, fast, slow, signal)["hist"]).rename("position")


def adx_trend(df: pd.DataFrame, period: int = 14, threshold: float = 25.0) -> pd.Series:
    """Directional trade only when ADX confirms a trend above ``threshold``."""
    _, h, l, c, _ = ohlc_parts(df)
    strength = adx(h, l, c, period)
    direction = _sign(c - sma(c, period))
    out = direction.where(strength >= threshold, 0.0)
    return out.fillna(0.0).rename("position")


def supertrend_strategy(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Follow the Supertrend direction flip."""
    _, h, l, c, _ = ohlc_parts(df)
    st = supertrend(h, l, c, period, mult)
    return st["direction"].fillna(0.0).rename("position")


def keltner_breakout(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Breakout of Keltner channels: long above the upper band, short below."""
    _, h, l, c, _ = ohlc_parts(df)
    kc = keltner_channels(h, l, c, period, mult)
    out = pd.Series(0.0, index=c.index)
    out[c > kc["upper"]] = 1.0
    out[c < kc["lower"]] = -1.0
    return out.rename("position")


def bollinger_breakout(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Bollinger breakout: enter on band pierce, hold until the mid cross."""
    _, _, _, c, _ = ohlc_parts(df)
    bb = bollinger_bands(c, period, mult)
    return hold_positions(
        c > bb["upper"], c < bb["lower"],
        c < bb["mid"], c > bb["mid"],
    )


def ichimoku_trend(df: pd.DataFrame) -> pd.Series:
    """Ichimoku: long when the conversion line leads the base line, short otherwise."""
    _, h, l, c, _ = ohlc_parts(df)
    ichi = ichimoku(h, l, c)
    kijun = ichi.iloc[:, 1]
    tenkan = ichi.iloc[:, 0]
    return _sign(tenkan - kijun).rename("position")


def aroon_trend(df: pd.DataFrame, period: int = 25) -> pd.Series:
    """Aroon trend: long when Aroon-Up leads, short when Aroon-Down leads."""
    _, h, l, c, _ = ohlc_parts(df)
    ar = aroon(h, l, period)
    return _sign(ar["up"] - ar["down"]).rename("position")


def vortex_trend(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Vortex trend: long when +VI leads -VI, short otherwise."""
    _, h, l, c, _ = ohlc_parts(df)
    vx = vortex(h, l, c, period)
    plus, minus = vx.iloc[:, 0], vx.iloc[:, 1]
    return _sign(plus - minus).rename("position")


def hma_trend(df: pd.DataFrame, period: int = 20, lookback: int = 5) -> pd.Series:
    """Sign of the Hull moving average slope."""
    _, _, _, c, _ = ohlc_parts(df)
    h = hma(c, period)
    return _sign(h - h.shift(lookback)).rename("position")


def tema_trend(df: pd.DataFrame, period: int = 20, lookback: int = 5) -> pd.Series:
    """Sign of the TEMA slope."""
    _, _, _, c, _ = ohlc_parts(df)
    t = tema(c, period)
    return _sign(t - t.shift(lookback)).rename("position")


def kama_trend(df: pd.DataFrame, period: int = 10, lookback: int = 5) -> pd.Series:
    """Sign of the Kaufman adaptive MA slope."""
    _, _, _, c, _ = ohlc_parts(df)
    k = kama(c, period)
    return _sign(k - k.shift(lookback)).rename("position")


def trend_quality_filter(df: pd.DataFrame, period: int = 20, min_score: float = 0.3) -> pd.Series:
    """Trade the trend only when the trend-quality score is decisive."""
    _, _, _, c, _ = ohlc_parts(df)
    score = trend_score(c, period)
    out = pd.Series(0.0, index=c.index)
    out[score > min_score] = 1.0
    out[score < -min_score] = -1.0
    return out.rename("position")


def efficiency_trend(df: pd.DataFrame, period: int = 10, threshold: float = 0.3) -> pd.Series:
    """Kaufman efficiency ratio gate around a price/SMA direction."""
    _, _, _, c, _ = ohlc_parts(df)
    er = efficiency_ratio(c, period)
    direction = _sign(c - sma(c, period))
    return direction.where(er >= threshold, 0.0).fillna(0.0).rename("position")


# ---------------------------------------------------------------------------
# Momentum strategies
# ---------------------------------------------------------------------------

def _roc(series: pd.Series, lookback: int) -> pd.Series:
    return series / series.shift(lookback) - 1.0


def momentum_signal(df: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.Series:
    """12-1 style momentum: sign of the return over ``lookback`` skipping ``skip``."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(c.shift(skip) / c.shift(lookback + skip) - 1.0).rename("position")


def momentum_3m(df: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Sign of the 3-month (default) price momentum."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(_roc(c, lookback)).rename("position")


def risk_adjusted_momentum_strategy(df: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """Sign of momentum divided by realized volatility."""
    _, _, _, c, _ = ohlc_parts(df)
    r = c.pct_change(fill_method=None)
    vol = r.rolling(lookback, min_periods=lookback).std(ddof=1)
    return _sign(_roc(c, lookback) / vol.replace(0, np.nan)).rename("position")


def dual_momentum(df: pd.DataFrame, lookback: int = 252, benchmark_df: pd.DataFrame | None = None) -> pd.Series:
    """Absolute momentum with an optional relative leg against a benchmark."""
    _, _, _, c, _ = ohlc_parts(df)
    abs_mom = _roc(c, lookback)
    pos = _sign(abs_mom)
    if benchmark_df is not None:
        _, _, _, b, _ = ohlc_parts(benchmark_df)
        b = b.reindex(c.index).ffill()
        rel = _roc(c, lookback) - _roc(b, lookback)
        pos = pos.where(rel > 0, -1.0)
    return pos.rename("position")


def breakout_momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Enter on a fresh rolling-high breakout, flip on a rolling-low breakdown."""
    _, h, l, c, _ = ohlc_parts(df)
    up = (c > h.rolling(period, min_periods=period).max().shift(1))
    dn = (c < l.rolling(period, min_periods=period).min().shift(1))
    return hold_positions(up, dn, dn, up)


def momentum_vol_filter(df: pd.DataFrame, lookback: int = 63, vol_window: int = 21, vol_cap: float = 0.03) -> pd.Series:
    """Momentum only when recent realized volatility is below ``vol_cap``."""
    _, _, _, c, _ = ohlc_parts(df)
    r = c.pct_change(fill_method=None)
    vol = r.rolling(vol_window, min_periods=vol_window).std(ddof=1)
    mom = _sign(_roc(c, lookback))
    return mom.where(vol <= vol_cap, 0.0).fillna(0.0).rename("position")


def macd_momentum(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, threshold: float = 0.0) -> pd.Series:
    """MACD histogram momentum with a dead zone around zero."""
    _, _, _, c, _ = ohlc_parts(df)
    hist = macd(c, fast, slow, signal)["hist"] / c
    return _sign(hist, deadzone=threshold).rename("position")


def trix_signal(df: pd.DataFrame, period: int = 15) -> pd.Series:
    """Sign of the TRIX rate of change."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(trix(c, period)).rename("position")


def momentum_acceleration(df: pd.DataFrame, short: int = 21, long: int = 63) -> pd.Series:
    """Sign of the difference between short and long momentum."""
    _, _, _, c, _ = ohlc_parts(df)
    return _sign(_roc(c, short) - _roc(c, long)).rename("position")


def faber_timing(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Faber-style long-only timing: invested above the 10-month SMA."""
    _, _, _, c, _ = ohlc_parts(df)
    return long_only(_sign(c - sma(c, period))).rename("position")


# ---------------------------------------------------------------------------
# Mean reversion strategies
# ---------------------------------------------------------------------------

def rsi_reversal(df: pd.DataFrame, period: int = 14, lower: float = 30.0, upper: float = 70.0, exit_mid: float = 50.0) -> pd.Series:
    """Long oversold RSI, short overbought; exit at the midline."""
    _, _, _, c, _ = ohlc_parts(df)
    r = rsi(c, period)
    return hold_positions(r < lower, r > upper, r > exit_mid, r < exit_mid)


def rsi2_reversal(df: pd.DataFrame, period: int = 2, lower: float = 10.0, upper: float = 90.0) -> pd.Series:
    """Connors-style ultra-short RSI(2) reversal entries."""
    _, _, _, c, _ = ohlc_parts(df)
    r = rsi(c, period)
    return hold_positions(r < lower, r > upper, r > upper, r < lower)


def bollinger_reversion(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Fade band pierces and exit on the mid-band cross."""
    _, _, _, c, _ = ohlc_parts(df)
    bb = bollinger_bands(c, period, mult)
    return hold_positions(c < bb["lower"], c > bb["upper"], c > bb["mid"], c < bb["mid"])


def percent_b_reversion(df: pd.DataFrame, period: int = 20, mult: float = 2.0, deadzone: float = 0.1) -> pd.Series:
    """Trade against Bollinger %B extremes with a dead zone around 0.5."""
    _, _, _, c, _ = ohlc_parts(df)
    pct_b = bollinger_bands(c, period, mult)["pct_b"]
    return _sign(0.5 - pct_b, deadzone=deadzone).rename("position")


def zscore_reversion(df: pd.DataFrame, period: int = 20, entry: float = 2.0, exit_level: float = 0.5) -> pd.Series:
    """Enter on extreme price z-scores, exit back toward the mean."""
    _, _, _, c, _ = ohlc_parts(df)
    mean = c.rolling(period, min_periods=period).mean()
    std = c.rolling(period, min_periods=period).std(ddof=1)
    z = (c - mean) / std.replace(0, np.nan)
    return hold_positions(z < -entry, z > entry, z > -exit_level, z < exit_level)


def cci_reversal(df: pd.DataFrame, period: int = 20, lower: float = -100.0, upper: float = 100.0) -> pd.Series:
    """CCI reversal around the +/-100 thresholds."""
    _, h, l, c, _ = ohlc_parts(df)
    v = cci(h, l, c, period)
    return hold_positions(v < lower, v > upper, v > 0.0, v < 0.0)


def williams_reversal(df: pd.DataFrame, period: int = 14, lower: float = -80.0, upper: float = -20.0, exit_mid: float = -50.0) -> pd.Series:
    """Williams %R reversal at overbought/oversold extremes."""
    _, h, l, c, _ = ohlc_parts(df)
    w = williams_r(h, l, c, period)
    return hold_positions(w < lower, w > upper, w > exit_mid, w < exit_mid)


def stoch_reversal(df: pd.DataFrame, k_period: int = 14, lower: float = 20.0, upper: float = 80.0, exit_mid: float = 50.0) -> pd.Series:
    """Stochastic oscillator reversal around 20/80."""
    _, h, l, c, _ = ohlc_parts(df)
    k = stochastic(h, l, c, k_period, 3, 3)["k"]
    return hold_positions(k < lower, k > upper, k > exit_mid, k < exit_mid)


def stoch_rsi_reversal(df: pd.DataFrame, period: int = 14, lower: float = 20.0, upper: float = 80.0) -> pd.Series:
    """Stochastic-RSI reversal entries."""
    _, _, _, c, _ = ohlc_parts(df)
    sr = stoch_rsi(c, period)
    k = sr.iloc[:, 0]
    return hold_positions(k < lower, k > upper, k > 50.0, k < 50.0)


def keltner_reversion(df: pd.DataFrame, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Fade Keltner band pierces and exit at the channel mid."""
    _, h, l, c, _ = ohlc_parts(df)
    kc = keltner_channels(h, l, c, period, mult)
    return hold_positions(c < kc["lower"], c > kc["upper"], c > kc["mid"], c < kc["mid"])


def vwap_reversion(df: pd.DataFrame, window: int = 20, band: float = 0.02) -> pd.Series:
    """Fade deviations from a rolling volume-weighted average price."""
    _, h, l, c, v = ohlc_parts(df)
    typical = (h + l + c) / 3.0
    if v.isna().all():
        v = pd.Series(1.0, index=c.index)
    tpv = (typical * v).rolling(window, min_periods=window).sum()
    vol = v.rolling(window, min_periods=window).sum()
    rvwap = tpv / vol.replace(0, np.nan)
    return hold_positions(c < rvwap * (1.0 - band), c > rvwap * (1.0 + band), c > rvwap, c < rvwap)


def gap_reversion(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    """Fade overnight gaps and exit when the gap is filled."""
    o, _, _, c, _ = ohlc_parts(df)
    prev_close = c.shift(1)
    gap = o / prev_close - 1.0
    return hold_positions(gap <= -threshold, gap >= threshold, c >= prev_close, c <= prev_close)


def range_fade(df: pd.DataFrame, period: int = 20, entry: float = 0.9, exit_level: float = 0.5) -> pd.Series:
    """Fade the extremes of the rolling high-low range."""
    _, h, l, c, _ = ohlc_parts(df)
    rp = range_position(c, h, l, period)
    return hold_positions(rp <= 1.0 - entry, rp >= entry, rp >= exit_level, rp <= exit_level)


def connors_rsi(df: pd.DataFrame, period: int = 3, lower: float = 10.0, upper: float = 70.0) -> pd.Series:
    """Connors long-only RSI(3) dip buyer."""
    _, _, _, c, _ = ohlc_parts(df)
    r = rsi(c, period)
    return long_only(hold_positions(r < lower, pd.Series(False, index=c.index), r > upper))


# ---------------------------------------------------------------------------
# Volatility strategies
# ---------------------------------------------------------------------------

def vol_regime_switch(df: pd.DataFrame, fast: int = 21, slow: int = 63) -> pd.Series:
    """Trend-follow in calm regimes, mean-revert in turbulent regimes."""
    _, _, _, c, _ = ohlc_parts(df)
    r = c.pct_change(fill_method=None)
    vol_fast = r.rolling(fast, min_periods=fast).std(ddof=1)
    vol_slow = r.rolling(slow, min_periods=slow).std(ddof=1)
    calm = vol_fast < vol_slow
    trend = sma_cross(df, 20, 50)
    revert = bollinger_reversion(df, 20, 2.0)
    return trend.where(calm, revert).fillna(0.0).rename("position")


def atr_channel_breakout(df: pd.DataFrame, period: int = 20, mult: float = 1.5) -> pd.Series:
    """Breakout of an SMA +/- ATR channel."""
    _, h, l, c, _ = ohlc_parts(df)
    mid = sma(c, period)
    band = mult * atr(h, l, c, period)
    return _sign(c - mid, deadzone=0.0).where((c - mid).abs() > band, 0.0).fillna(0.0).rename("position")


def squeeze_release(df: pd.DataFrame) -> pd.Series:
    """TTM squeeze: take the momentum side only after the squeeze releases."""
    _, h, l, c, _ = ohlc_parts(df)
    sq = squeeze_momentum(h, l, c)
    released = (~sq["squeeze_on"].fillna(True))
    return _sign(sq["momentum"]).where(released, 0.0).fillna(0.0).rename("position")


def vol_target_trend(df: pd.DataFrame, period: int = 50, target_vol: float = 0.20, vol_window: int = 21) -> pd.Series:
    """Trend positions scaled to a volatility target (returns weights in [-1, 1])."""
    _, _, _, c, _ = ohlc_parts(df)
    r = c.pct_change(fill_method=None)
    realized = r.rolling(vol_window, min_periods=vol_window).std(ddof=1) * np.sqrt(252)
    scale = (target_vol / realized.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    direction = _sign(c - sma(c, period))
    return (direction * scale).clip(-1.0, 1.0).rename("position")


def bollinger_squeeze_breakout(df: pd.DataFrame, period: int = 20, mult: float = 2.0, lookback: int = 126) -> pd.Series:
    """Enter Bollinger breakouts that occur from a low-width (squeeze) base."""
    _, _, _, c, _ = ohlc_parts(df)
    bb = bollinger_bands(c, period, mult)
    width = bb["width"]
    squeeze = width <= width.rolling(lookback, min_periods=lookback // 3).quantile(0.25)
    up = (c > bb["upper"]) & squeeze.shift(1).fillna(False)
    dn = (c < bb["lower"]) & squeeze.shift(1).fillna(False)
    return hold_positions(up, dn, c < bb["mid"], c > bb["mid"])


def vol_breakout_momentum(df: pd.DataFrame, period: int = 20, expansion: float = 1.2) -> pd.Series:
    """Follow price velocity when volatility expands above its recent norm."""
    _, h, l, c, _ = ohlc_parts(df)
    a = atr(h, l, c, period)
    expanding = a > a.rolling(period * 3, min_periods=period).mean() * expansion
    return _sign(velocity(c, 10)).where(expanding, 0.0).fillna(0.0).rename("position")


# ---------------------------------------------------------------------------
# Volume strategies
# ---------------------------------------------------------------------------

def obv_trend(df: pd.DataFrame, period: int = 21, lookback: int = 5) -> pd.Series:
    """Sign of the smoothed On-Balance Volume slope."""
    _, _, _, c, v = ohlc_parts(df)
    if v.isna().all():
        return pd.Series(0.0, index=c.index, name="position")
    smoothed = obv_ema(c, v, period)
    return _sign(smoothed - smoothed.shift(lookback)).rename("position")


def mfi_reversal(df: pd.DataFrame, period: int = 14, lower: float = 20.0, upper: float = 80.0, exit_mid: float = 50.0) -> pd.Series:
    """Money Flow Index reversal around 20/80."""
    _, h, l, c, v = ohlc_parts(df)
    m = mfi(h, l, c, v, period)
    return hold_positions(m < lower, m > upper, m > exit_mid, m < exit_mid)


def cmf_trend(df: pd.DataFrame, period: int = 20, deadzone: float = 0.0) -> pd.Series:
    """Sign of the Chaikin Money Flow."""
    _, h, l, c, v = ohlc_parts(df)
    return _sign(cmf(h, l, c, v, period), deadzone=deadzone).rename("position")


def volume_breakout(df: pd.DataFrame, period: int = 20, vol_z: float = 1.0) -> pd.Series:
    """Breakout with volume confirmation; exit on the SMA cross."""
    _, h, l, c, v = ohlc_parts(df)
    hi = h.rolling(period, min_periods=period).max().shift(1)
    lo = l.rolling(period, min_periods=period).min().shift(1)
    strong = volume_zscore(v, period) >= vol_z if not v.isna().all() else pd.Series(True, index=c.index)
    mid = sma(c, period)
    return hold_positions((c > hi) & strong, (c < lo) & strong, c < mid, c > mid)


def vwap_cross(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Long above the rolling VWAP, short below it."""
    _, h, l, c, v = ohlc_parts(df)
    typical = (h + l + c) / 3.0
    if v.isna().all():
        v = pd.Series(1.0, index=c.index)
    tpv = (typical * v).rolling(window, min_periods=window).sum()
    vol = v.rolling(window, min_periods=window).sum()
    rvwap = tpv / vol.replace(0, np.nan)
    return _sign(c - rvwap).rename("position")


def relative_volume_trend(df: pd.DataFrame, period: int = 20, threshold: float = 1.2) -> pd.Series:
    """Price-vs-SMA direction, gated to days with elevated relative volume."""
    _, _, _, c, v = ohlc_parts(df)
    if v.isna().all():
        return pd.Series(0.0, index=c.index, name="position")
    rv = relative_volume(v, period)
    direction = _sign(c - sma(c, period))
    return direction.where(rv >= threshold, 0.0).fillna(0.0).rename("position")


# ---------------------------------------------------------------------------
# Pattern strategies
# ---------------------------------------------------------------------------

def candle_pattern_score(df: pd.DataFrame, window: int = 5, threshold: float = 0.2) -> pd.Series:
    """Trade the aggregated candlestick pattern score."""
    o, h, l, c, _ = ohlc_parts(df)
    return _sign(pattern_score(o, h, l, c, window), deadzone=threshold).rename("position")


def engulfing_proxy(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    """Body-engulfing proxy: reversal candle bodies with opposing follow-through."""
    o, h, l, c, _ = ohlc_parts(df)
    body = c - o
    prev = body.shift(1)
    engulf_up = (body > 0) & (body.abs() > prev.abs()) & (c > o.shift(1))
    engulf_dn = (body < 0) & (body.abs() > prev.abs()) & (c < o.shift(1))
    base = _sign(body, deadzone=threshold * (h - l).replace(0, np.nan))
    return base.where(engulf_up | engulf_dn, 0.0).fillna(0.0).rename("position")


def chart_pattern_structure(df: pd.DataFrame, window: int = 60, threshold: float = 0.1) -> pd.Series:
    """Trade the composite chart-formation score."""
    _, _, _, c, _ = ohlc_parts(df)
    detections = detect_patterns(df, window)
    return _sign(structure_score(detections), deadzone=threshold).rename("position")


def divergence_strategy(df: pd.DataFrame, period: int = 14, lookback: int = 100, hold: int = 10) -> pd.Series:
    """Hold a position for ``hold`` bars after an RSI divergence confirms."""
    _, _, _, c, _ = ohlc_parts(df)
    sig = rsi_divergence(c, period, lookback=lookback)
    out = np.zeros(len(c), dtype="float64")
    pos = 0.0
    remaining = 0
    for i in range(len(c)):
        v = float(sig.iloc[i])
        if v != 0.0:
            pos = v
            remaining = hold
        elif remaining > 0:
            remaining -= 1
        else:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=c.index, name="position")


def pivot_structure_trade(df: pd.DataFrame, pct: float = 0.05) -> pd.Series:
    """Ride break-of-structure events until the structure flips."""
    _, h, l, c, _ = ohlc_parts(df)
    bos = break_of_structure(c, h, l, pct)
    out = bos.replace(0.0, np.nan).ffill().fillna(0.0)
    return out.rename("position")


def support_resistance_trade(df: pd.DataFrame, period: int = 20, tol: float = 0.02) -> pd.Series:
    """Buy near rolling support and sell near rolling resistance."""
    _, h, l, c, _ = ohlc_parts(df)
    sup = rolling_support(l, period)
    res = rolling_resistance(h, period)
    near_sup = c <= sup * (1.0 + tol)
    near_res = c >= res * (1.0 - tol)
    return hold_positions(near_sup, near_res, near_res, near_sup)


# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------

_DEFAULT_ENSEMBLE = (
    "sma_cross", "macd_trend", "donchian_breakout", "momentum_signal",
    "rsi_reversal", "bollinger_reversion", "cmf_trend", "chart_pattern_structure",
    "supertrend_strategy",
)


def ensemble_vote(df: pd.DataFrame, members=None, min_votes: int = 3) -> pd.Series:
    """Majority vote across member strategies: +1/-1 when votes clear the threshold."""
    _, _, _, c, _ = ohlc_parts(df)
    names = list(members) if members is not None else list(_DEFAULT_ENSEMBLE)
    votes = pd.Series(0.0, index=c.index)
    for name in names:
        entry = STRATEGIES.get(name)
        if entry is None:
            raise KeyError(f"Unknown strategy in ensemble: {name}")
        votes = votes + _flat(entry["func"](df))
    out = pd.Series(0.0, index=c.index)
    out[votes >= min_votes] = 1.0
    out[votes <= -min_votes] = -1.0
    return out.rename("position")


def regime_adaptive(df: pd.DataFrame, efficiency_period: int = 20, threshold: float = 0.35) -> pd.Series:
    """Switch between trend following and mean reversion by the efficiency ratio."""
    _, _, _, c, _ = ohlc_parts(df)
    er = efficiency_ratio(c, efficiency_period)
    trending = er >= threshold
    return sma_cross(df, 20, 50).where(trending, zscore_reversion(df, 20, 2.0, 0.5)).fillna(0.0).rename("position")


def multi_timeframe_trend(df: pd.DataFrame, short: int = 20, long: int = 100) -> pd.Series:
    """Long/short only when the short and long trend filters agree."""
    _, _, _, c, _ = ohlc_parts(df)
    fast = _sign(c - sma(c, short))
    slow = _sign(c - sma(c, long))
    return fast.where(fast == slow, 0.0).fillna(0.0).rename("position")


def alpha_combo(df: pd.DataFrame, weights=None) -> pd.Series:
    """Weighted combination of trend, momentum, reversion, volume, and pattern signals."""
    names = ("sma_cross", "macd_trend", "momentum_signal", "rsi_reversal", "cmf_trend", "chart_pattern_structure")
    if weights is None:
        w = np.full(len(names), 1.0 / len(names))
    else:
        w = np.asarray(weights, dtype="float64")
        if len(w) != len(names):
            raise ValueError("weights length must match the combo members")
    _, _, _, c, _ = ohlc_parts(df)
    combined = pd.Series(0.0, index=c.index)
    for name, weight in zip(names, w):
        combined = combined + weight * _flat(STRATEGIES[name]["func"](df))
    return _sign(combined, deadzone=0.25).rename("position")


def all_strategy_signals(df: pd.DataFrame, names=None) -> pd.DataFrame:
    """Position frame with one column per strategy (composites excluded by default)."""
    if names is None:
        names = [n for n, meta in STRATEGIES.items() if meta["category"] != "composite"]
    data = {}
    for name in names:
        entry = STRATEGIES.get(name)
        if entry is None:
            raise KeyError(f"Unknown strategy: {name}")
        data[name] = _flat(entry["func"](df))
    return pd.DataFrame(data)


def strategy_correlation(df: pd.DataFrame, names=None) -> pd.DataFrame:
    """Correlation matrix of strategy bar returns."""
    signals = all_strategy_signals(df, names)
    returns = pd.DataFrame(
        {col: strategy_returns(df, signals[col]) for col in signals.columns}
    )
    return returns.corr()


# ---------------------------------------------------------------------------
# Registry / runners
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "sma_cross": "trend", "ema_cross": "trend", "triple_ma": "trend",
    "price_vs_sma": "trend", "ma_slope": "trend", "donchian_breakout": "trend",
    "turtle_system": "trend", "macd_trend": "trend", "adx_trend": "trend",
    "supertrend_strategy": "trend", "keltner_breakout": "trend",
    "bollinger_breakout": "trend", "ichimoku_trend": "trend",
    "aroon_trend": "trend", "vortex_trend": "trend", "hma_trend": "trend",
    "tema_trend": "trend", "kama_trend": "trend",
    "trend_quality_filter": "trend", "efficiency_trend": "trend",
    "momentum_signal": "momentum", "momentum_3m": "momentum",
    "risk_adjusted_momentum_strategy": "momentum", "dual_momentum": "momentum",
    "breakout_momentum": "momentum", "momentum_vol_filter": "momentum",
    "macd_momentum": "momentum", "trix_signal": "momentum",
    "momentum_acceleration": "momentum", "faber_timing": "momentum",
    "rsi_reversal": "mean_reversion", "rsi2_reversal": "mean_reversion",
    "bollinger_reversion": "mean_reversion", "percent_b_reversion": "mean_reversion",
    "zscore_reversion": "mean_reversion", "cci_reversal": "mean_reversion",
    "williams_reversal": "mean_reversion", "stoch_reversal": "mean_reversion",
    "stoch_rsi_reversal": "mean_reversion", "keltner_reversion": "mean_reversion",
    "vwap_reversion": "mean_reversion", "gap_reversion": "mean_reversion",
    "range_fade": "mean_reversion", "connors_rsi": "mean_reversion",
    "vol_regime_switch": "volatility", "atr_channel_breakout": "volatility",
    "squeeze_release": "volatility", "vol_target_trend": "volatility",
    "bollinger_squeeze_breakout": "volatility", "vol_breakout_momentum": "volatility",
    "obv_trend": "volume", "mfi_reversal": "volume", "cmf_trend": "volume",
    "volume_breakout": "volume", "vwap_cross": "volume",
    "relative_volume_trend": "volume",
    "candle_pattern_score": "pattern", "engulfing_proxy": "pattern",
    "chart_pattern_structure": "pattern", "divergence_strategy": "pattern",
    "pivot_structure_trade": "pattern", "support_resistance_trade": "pattern",
    "ensemble_vote": "composite", "regime_adaptive": "composite",
    "multi_timeframe_trend": "composite", "alpha_combo": "composite",
}


def _build_registry() -> dict:
    import inspect

    funcs = {
        "sma_cross": sma_cross, "ema_cross": ema_cross, "triple_ma": triple_ma,
        "price_vs_sma": price_vs_sma, "ma_slope": ma_slope,
        "donchian_breakout": donchian_breakout, "turtle_system": turtle_system,
        "macd_trend": macd_trend, "adx_trend": adx_trend,
        "supertrend_strategy": supertrend_strategy, "keltner_breakout": keltner_breakout,
        "bollinger_breakout": bollinger_breakout, "ichimoku_trend": ichimoku_trend,
        "aroon_trend": aroon_trend, "vortex_trend": vortex_trend, "hma_trend": hma_trend,
        "tema_trend": tema_trend, "kama_trend": kama_trend,
        "trend_quality_filter": trend_quality_filter, "efficiency_trend": efficiency_trend,
        "momentum_signal": momentum_signal, "momentum_3m": momentum_3m,
        "risk_adjusted_momentum_strategy": risk_adjusted_momentum_strategy,
        "dual_momentum": dual_momentum, "breakout_momentum": breakout_momentum,
        "momentum_vol_filter": momentum_vol_filter, "macd_momentum": macd_momentum,
        "trix_signal": trix_signal, "momentum_acceleration": momentum_acceleration,
        "faber_timing": faber_timing,
        "rsi_reversal": rsi_reversal, "rsi2_reversal": rsi2_reversal,
        "bollinger_reversion": bollinger_reversion, "percent_b_reversion": percent_b_reversion,
        "zscore_reversion": zscore_reversion, "cci_reversal": cci_reversal,
        "williams_reversal": williams_reversal, "stoch_reversal": stoch_reversal,
        "stoch_rsi_reversal": stoch_rsi_reversal, "keltner_reversion": keltner_reversion,
        "vwap_reversion": vwap_reversion, "gap_reversion": gap_reversion,
        "range_fade": range_fade, "connors_rsi": connors_rsi,
        "vol_regime_switch": vol_regime_switch, "atr_channel_breakout": atr_channel_breakout,
        "squeeze_release": squeeze_release, "vol_target_trend": vol_target_trend,
        "bollinger_squeeze_breakout": bollinger_squeeze_breakout,
        "vol_breakout_momentum": vol_breakout_momentum,
        "obv_trend": obv_trend, "mfi_reversal": mfi_reversal, "cmf_trend": cmf_trend,
        "volume_breakout": volume_breakout, "vwap_cross": vwap_cross,
        "relative_volume_trend": relative_volume_trend,
        "candle_pattern_score": candle_pattern_score, "engulfing_proxy": engulfing_proxy,
        "chart_pattern_structure": chart_pattern_structure,
        "divergence_strategy": divergence_strategy,
        "pivot_structure_trade": pivot_structure_trade,
        "support_resistance_trade": support_resistance_trade,
        "ensemble_vote": ensemble_vote, "regime_adaptive": regime_adaptive,
        "multi_timeframe_trend": multi_timeframe_trend, "alpha_combo": alpha_combo,
    }
    registry = {}
    for name, fn in funcs.items():
        sig = inspect.signature(fn)
        defaults = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name != "df" and p.default is not inspect.Parameter.empty
        }
        registry[name] = {"func": fn, "category": _CATEGORIES.get(name, "other"), "defaults": defaults}
    return registry


STRATEGIES = _build_registry()


def list_strategies(category: str | None = None) -> list[str]:
    """Names of registered strategies, optionally filtered by category."""
    if category is None:
        return sorted(STRATEGIES)
    return sorted(n for n, meta in STRATEGIES.items() if meta["category"] == category)


def get_strategy(name: str) -> dict:
    """Registry entry for a strategy (raises KeyError with a hint if unknown)."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy '{name}'. Available: {', '.join(list_strategies())}")
    return STRATEGIES[name]


def strategy_positions(name: str, df: pd.DataFrame, **overrides) -> pd.Series:
    """Run a registered strategy by name, overriding default parameters."""
    entry = get_strategy(name)
    params = {**entry["defaults"], **overrides}
    return _flat(entry["func"](df, **params)).rename(name)
