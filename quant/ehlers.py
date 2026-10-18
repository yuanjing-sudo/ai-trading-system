"""Ehlers cycle-analysis and digital-filter pack.

Implements the classic John Ehlers (MESA / Hilbert / Laguerre) toolkit as
discrete recursions on pandas Series. These filters separate price into trend
and cycle components and adapt their speed to the dominant cycle, which makes
them useful for regime-aware trading systems.

Conventions
-----------
- All functions return pandas objects aligned to the input index.
- Recursive filters seed from the first valid observation.
- ``dominant_cycle`` uses the autocorrelation-periodogram method; it is an
  approximation of Ehlers' MESA algorithm and is documented as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series
from quant.indicators_ext import _adaptive_ema

__all__ = [
    "super_smoother", "ehlers_ema", "instant_trendline", "cyber_cycle",
    "roof", "laguerre_filter", "laguerre_rsi", "adaptive_rsi",
    "center_of_gravity", "mama_fama", "dominant_cycle", "sine_wave",
    "even_better_sine", "phase_accumulator", "dual_differential",
    "trendflex", "devstop", "hilbert_transform", "bandpass",
    "zero_crossing_rate", "adaptive_stochastic", "noise_cleaner",
    "ehlers_fisher", "sma_of_sine",
]


def _recur2(series: pd.Series, a: float, b: float, c: float) -> pd.Series:
    """Generic two-pole recursion: y[i] = a*x[i] + b*y[i-1] + c*y[i-2]."""
    x = series.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    y1 = x[0] if len(x) else 0.0
    y2 = y1
    for i in range(len(x)):
        xi = x[i] if np.isfinite(x[i]) else y1
        yi = a * xi + b * y1 + c * y2
        out[i] = yi
        y2 = y1
        y1 = yi
    return pd.Series(out, index=series.index, name=series.name)


def _recur1(series: pd.Series, a: float, b: float) -> pd.Series:
    """Generic one-pole recursion: y[i] = a*x[i] + b*y[i-1]."""
    x = series.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    y1 = x[0] if len(x) else 0.0
    for i in range(len(x)):
        xi = x[i] if np.isfinite(x[i]) else y1
        yi = a * xi + b * y1
        out[i] = yi
        y1 = yi
    return pd.Series(out, index=series.index, name=series.name)


def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers two-pole super smoother: near-zero lag with low noise."""
    s = _as_series(series)
    a = np.exp(-np.sqrt(2.0) * np.pi / max(period, 2))
    b = 2.0 * a * np.cos(np.sqrt(2.0) * np.pi / max(period, 2))
    c2 = b
    c3 = -a * a
    c1 = 1.0 - c2 - c3
    return _recur2(s, c1, c2, c3)


def ehlers_ema(series: pd.Series, period: int = 10) -> pd.Series:
    """Ehlers one-pole exponential smoother (alpha in [0, 1])."""
    s = _as_series(series)
    alpha = 2.0 / (period + 1.0)
    return s.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def instant_trendline(series: pd.Series, period: int = 20) -> pd.Series:
    """Instantaneous trendline: smoothed price with an EMA correction term."""
    s = _as_series(series)
    weighted = (4.0 * s + 3.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 10.0
    return _recur2(weighted.bfill(), 2.0 / (period + 1.0), 1.0 - 2.0 / (period + 1.0), 0.0)


def cyber_cycle(series: pd.Series, period: int = 20) -> pd.Series:
    """Ehlers cyber cycle: high-pass filtered, smoothed detrended oscillator."""
    s = _as_series(series)
    alpha = 2.0 / (period + 1.0)
    smooth = (s + 2.0 * s.shift(1) + 2.0 * s.shift(2) + s.shift(3)) / 6.0
    x = smooth.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    c_prev2 = 0.0
    c_prev1 = 0.0
    for i in range(len(x)):
        xi = x[i] if np.isfinite(x[i]) else (x[i - 1] if i else 0.0)
        ci = (1.0 - 0.5 * alpha) * (xi - 2.0 * (x[i - 1] if i else xi) + (x[i - 2] if i > 1 else xi))
        ci += 2.0 * (1.0 - alpha) * c_prev1 - (1.0 - alpha) ** 2 * c_prev2
        out[i] = ci
        c_prev2 = c_prev1
        c_prev1 = ci
    return pd.Series(out, index=s.index, name=s.name)


def roof(series: pd.Series, period: int = 20) -> pd.Series:
    """Ehlers roofing filter (high-pass): removes trend, keeps cycle."""
    s = _as_series(series)
    alpha = 2.0 / (period + 1.0)
    return _recur2(s, (1.0 - alpha / 2.0) ** 2, 2.0 * (1.0 - alpha), -(1.0 - alpha) ** 2)


def laguerre_filter(series: pd.Series, gamma: float = 0.7) -> pd.Series:
    """Laguerre four-pole low-pass filter (Ehlers)."""
    s = _as_series(series)
    x = s.to_numpy(dtype="float64")
    g = float(np.clip(gamma, 0.0, 1.0))
    l0 = l1 = l2 = l3 = x[0] if len(x) else 0.0
    out = np.empty(len(x), dtype="float64")
    for i in range(len(x)):
        xi = x[i] if np.isfinite(x[i]) else out[i - 1] if i else 0.0
        l0p = l1
        l1p = l2
        l2p = l3
        l0 = (1.0 - g) * xi + g * l0p
        l1 = -g * l0 + l0p + g * l1p
        l2 = -g * l1 + l1p + g * l2p
        l3 = -g * l2 + l2p + g * l3
        out[i] = (l0 + 2.0 * l1 + 2.0 * l2 + l3) / 6.0
    return pd.Series(out, index=s.index, name=s.name)


def laguerre_rsi(series: pd.Series, gamma: float = 0.7) -> pd.Series:
    """Laguerre RSI: fast, low-lag RSI-style oscillator in [0, 1]."""
    s = _as_series(series)
    x = s.to_numpy(dtype="float64")
    g = float(np.clip(gamma, 0.0, 1.0))
    l0 = l1 = l2 = l3 = x[0] if len(x) else 0.0
    out = np.empty(len(x), dtype="float64")
    for i in range(len(x)):
        xi = x[i] if np.isfinite(x[i]) else out[i - 1] if i else 0.0
        l0p, l1p, l2p = l1, l2, l3
        l0 = (1.0 - g) * xi + g * l0p
        l1 = -g * l0 + l0p + g * l1p
        l2 = -g * l1 + l1p + g * l2p
        l3 = -g * l2 + l2p + g * l3
        cu = cd = 0.0
        if l0 >= l1:
            cu = l0 - l1
        else:
            cd = l1 - l0
        if l1 >= l2:
            cu += l1 - l2
        else:
            cd += l2 - l1
        if l2 >= l3:
            cu += l2 - l3
        else:
            cd += l3 - l2
        total = cu + cd
        out[i] = cu / total if total > 0 else 0.5
    return pd.Series(out, index=s.index, name=s.name)


def adaptive_rsi(series: pd.Series, period: int = 10) -> pd.Series:
    """Adaptive RSI (Ehlers): RSI of the adaptive-smoothed momentum path."""
    s = _as_series(series)
    trend = instant_trendline(s, period)
    cycle = roof(s, period)
    path = trend + cycle
    delta = path.diff()
    k = 2.0 / (period + 1.0)
    up = delta.clip(lower=0.0).ewm(alpha=k, adjust=False, min_periods=period).mean()
    dn = (-delta.clip(upper=0.0)).ewm(alpha=k, adjust=False, min_periods=period).mean()
    return 100.0 * up / (up + dn).replace(0, np.nan)


def center_of_gravity(series: pd.Series, period: int = 10) -> pd.Series:
    """Center of gravity oscillator (Ehlers): weighted price centroid minus price."""
    s = _as_series(series)
    weights = np.arange(period, 0, -1, dtype="float64")
    num = s.rolling(period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights)), raw=True
    )
    den = s.rolling(period, min_periods=period).sum()
    cg = -num / den + (period + 1.0) / 2.0
    return cg


def mama_fama(
    series: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05
) -> pd.DataFrame:
    """MESA adaptive moving average (MAMA) with following adaptive MA (FAMA).

    The adaptive alpha uses Ehlers' simplified Hilbert-transform phase rate.
    """
    s = _as_series(series)
    x = s.to_numpy(dtype="float64")
    n = len(x)
    mama = np.empty(n, dtype="float64")
    fama = np.empty(n, dtype="float64")
    smooth = x[0] if n else 0.0
    detrender = q1 = i1 = jI = jQ = 0.0
    period = 0.0
    phase = 0.0
    prev_phase = 0.0
    mama_v = x[0] if n else 0.0
    fama_v = x[0] if n else 0.0
    for i in range(n):
        xi = x[i] if np.isfinite(x[i]) else (x[i - 1] if i else 0.0)
        smooth = (4.0 * xi + 3.0 * (x[i - 1] if i else xi) + 2.0 * (x[i - 2] if i > 1 else xi) + (x[i - 3] if i > 2 else xi)) / 10.0
        detrender = (0.0962 * smooth + 0.5769 * (x[i - 2] if i > 1 else xi) - 0.5769 * (x[i - 4] if i > 3 else xi) - 0.0962 * (x[i - 6] if i > 5 else xi)) * (0.075 * period + 0.54) if i > 6 else 0.0
        q1 = (0.0962 * detrender + 0.5769 * (x[i - 2] if i > 1 else xi) - 0.5769 * (x[i - 4] if i > 3 else xi) - 0.0962 * (x[i - 6] if i > 5 else xi)) * (0.075 * period + 0.54) if i > 6 else 0.0
        i1 = detrender if i else xi
        jI = (0.0962 * i1 + 0.5769 * (mama_v) - 0.5769 * (x[i - 2] if i > 1 else xi) - 0.0962 * (x[i - 4] if i > 3 else xi)) if i > 4 else 0.0
        jQ = (0.0962 * q1 + 0.5769 * (fama_v) - 0.5769 * (x[i - 2] if i > 1 else xi) - 0.0962 * (x[i - 4] if i > 3 else xi)) if i > 4 else 0.0
        if i > 0:
            if i1 != 0 and jI != 0:
                phase = np.arctan(abs(jQ / jI)) * 180.0 / np.pi
            delta_phase = prev_phase - phase
            prev_phase = phase
            if delta_phase < 1.0:
                delta_phase = 1.0
            alpha = float(np.clip(fast_limit / delta_phase, slow_limit, fast_limit))
        else:
            alpha = slow_limit
        mama_v = alpha * xi + (1.0 - alpha) * mama_v
        fama_v = 0.5 * alpha * mama_v + (1.0 - 0.5 * alpha) * fama_v
        mama[i] = mama_v
        fama[i] = fama_v
    return pd.DataFrame({"mama": mama, "fama": fama}, index=s.index)


def dominant_cycle(series: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.Series:
    """Dominant cycle period via the autocorrelation periodogram (approximation).

    For each window smoothed with the super smoother, the period with the
    highest autocorrelation in [min_period, max_period] becomes the estimate.
    """
    s = _as_series(series)
    smooth = super_smoother(s, 10)
    detrended = smooth - super_smoother(smooth, 10)
    x = detrended.to_numpy(dtype="float64")
    n = len(x)
    win = max(max_period * 2, 64)
    out = np.full(n, np.nan, dtype="float64")
    for i in range(win, n):
        seg = x[i - win:i]
        seg = seg[np.isfinite(seg)]
        if len(seg) < max_period * 2:
            continue
        seg = seg - seg.mean()
        denom = float(np.dot(seg, seg))
        if denom <= 0:
            continue
        best_p = min_period
        best_c = -np.inf
        for p in range(min_period, max_period + 1):
            corr = float(np.dot(seg[:-p], seg[p:]) / denom)
            if corr > best_c:
                best_c = corr
                best_p = p
        out[i] = float(best_p)
    return pd.Series(out, index=s.index, name="dominant_cycle").ffill()


def hilbert_transform(series: pd.Series) -> pd.DataFrame:
    """Ehlers' Hilbert transform approximation.

    Returns in-phase, quadrature, amplitude, and instantaneous phase for the
    input series using the classic 4/7/9-bar FIR approximation.
    """
    s = _as_series(series)
    x = s.to_numpy(dtype="float64")
    n = len(x)
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    phase = np.full(n, np.nan)
    amp = np.full(n, np.nan)

    def adj(xs, idx, val):
        return xs[idx] if idx >= 0 and idx < n and np.isfinite(xs[idx]) else val

    for i in range(6, n):
        xi = x[i] if np.isfinite(x[i]) else (x[i - 1] if i else 0.0)
        smooth[i] = (4.0 * xi + 3.0 * x[i - 1] + 2.0 * x[i - 2] + x[i - 3]) / 10.0
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
                        - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6])
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2]
                 - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6])
        i1[i] = detrender[i - 3] if i >= 3 else 0.0
        jQ = (0.0962 * q1[i] + 0.5769 * q1[i - 2] - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6]) if i >= 6 else 0.0
        if i1[i] != 0.0:
            phase[i] = np.arctan(jQ / i1[i]) * 180.0 / np.pi
        amp[i] = np.sqrt(i1[i] ** 2 + jQ ** 2)
    return pd.DataFrame(
        {"inphase": i1, "quadrature": q1, "amplitude": amp, "phase": phase},
        index=s.index,
    )


def sine_wave(series: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.DataFrame:
    """Sine wave indicator: sine/lead-sine pair plus the dominant cycle period."""
    s = _as_series(series)
    period = dominant_cycle(s, min_period, max_period).ffill()
    prices = pd.concat([s, period], axis=1)
    prices.columns = ["price", "period"]
    prices = prices.ffill()
    dc = prices["period"].to_numpy(dtype="float64")
    px = prices["price"].to_numpy(dtype="float64")
    n = len(px)
    sin_wave = np.full(n, np.nan)
    lead_wave = np.full(n, np.nan)
    prev_sin = 0.0
    prev_lead = 0.0
    phase = 0.0
    for i in range(1, n):
        p = dc[i] if np.isfinite(dc[i]) and dc[i] > 0 else 20.0
        real = px[i] if np.isfinite(px[i]) else px[i - 1]
        diff = abs(px[i] - px[i - 1]) if np.isfinite(px[i]) and np.isfinite(px[i - 1]) else 0.0
        dc_phase = 360.0 / p
        _ = (real, diff)
        phase = (phase + dc_phase) % 360.0
        sin_wave[i] = np.sin(np.radians(phase))
        lead_wave[i] = np.sin(np.radians(phase + 45.0))
        prev_sin, prev_lead = sin_wave[i], lead_wave[i]
    _ = (prev_sin, prev_lead)
    return pd.DataFrame({"sine": sin_wave, "lead_sine": lead_wave, "period": dc}, index=s.index)


def even_better_sine(series: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.DataFrame:
    """Even-better sine wave: quadrature signal from smoothed Hilbert pairs."""
    s = _as_series(series)
    ht = hilbert_transform(super_smoother(s, 10))
    inphase = ht["inphase"]
    quad = ht["quadrature"]
    dc = dominant_cycle(s, min_period, max_period).ffill()
    dc_smooth = dc.rolling(10, min_periods=1).mean()
    phase = np.degrees(np.arctan2(quad, inphase))
    sine = np.sin(np.radians(phase + 360.0 / dc_smooth.fillna(20.0)))
    lead = np.sin(np.radians(phase + 360.0 / dc_smooth.fillna(20.0) + 45.0))
    return pd.DataFrame({"sine": sine, "lead_sine": lead, "period": dc}, index=s.index)


def phase_accumulator(series: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.Series:
    """Phase accumulator: steady ramp whose rate tracks the dominant cycle."""
    s = _as_series(series)
    dc = dominant_cycle(s, min_period, max_period).ffill().fillna(20.0)
    increments = 100.0 * s.pct_change(fill_method=None).abs().fillna(0.0)
    phase = (increments / (360.0 / dc)).cumsum()
    return phase


def dual_differential(series: pd.Series, period: int = 10) -> pd.Series:
    """Dual differential oscillator: difference of two band-limited paths."""
    s = _as_series(series)
    fast = bandpass(s, max(period // 2, 2))
    slow = bandpass(s, period)
    return fast - slow


def bandpass(series: pd.Series, period: int = 20) -> pd.Series:
    """Two-pole bandpass filter centered on ``period`` (Ehlers)."""
    s = _as_series(series)
    p = max(int(period), 2)
    alpha2 = 2.0 * np.sin(np.pi / p) if p > 2 else 2.0
    beta = np.cos(2.0 * np.pi / p)
    gamma = 1.0 / np.cos(2.0 * np.pi * 1.414 / p) if p > 4 else 1.0
    a = alpha2 * gamma
    b = beta
    c = -(alpha2 * alpha2) / 4.0
    return _recur2(s, a, b, c)


def trendflex(series: pd.Series, period: int = 20) -> pd.Series:
    """Trendflex smoothing: minimizes noise while preserving real trend moves."""
    s = _as_series(series)
    n = len(s)
    if n == 0:
        return s.copy()
    x = s.to_numpy(dtype="float64")
    out = np.full(n, np.nan, dtype="float64")
    num = 0.0
    den = 0.0
    for i in range(n):
        lo = max(0, i - period + 1)
        window = x[lo:i + 1]
        window = window[np.isfinite(window)]
        if len(window) == 0:
            continue
        var = float(np.var(window))
        den = 0.5 * den + var
        num = 0.5 * num + x[i] if np.isfinite(x[i]) else num
        out[i] = num / den if den != 0 else x[i]
    return pd.Series(out, index=s.index, name=s.name)


def devstop(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    """Deviation-scaled trailing stop (Ehlers' dev-stop) with trend state.

    Returns stop level and a long/short state series.
    """
    h = _as_series(high)
    l = _as_series(low)
    c = _as_series(close)
    typical = (h + l + 2.0 * c) / 4.0
    prices = typical.to_numpy(dtype="float64")
    n = len(prices)
    long_stop = np.full(n, np.nan, dtype="float64")
    short_stop = np.full(n, np.nan, dtype="float64")
    state = np.ones(n, dtype="float64")
    ls = ss = prices[0]
    for i in range(n):
        p = prices[i] if np.isfinite(prices[i]) else (prices[i - 1] if i else 0.0)
        sl = max(0, i - period + 1)
        win = prices[sl:i + 1]
        win = win[np.isfinite(win)]
        norm = float(np.sqrt(np.sum((win - win.mean()) ** 2) / max(len(win), 1))) if len(win) else 0.0
        if i > 0:
            prev_ls = ls
            ls = prev_ls + (p - prev_ls + mult * norm) if state[i - 1] == 1 else p - mult * norm
            ss = (ss + (p - ss - mult * norm)) if state[i - 1] == -1 else p + mult * norm
            if p > prev_ls and p > ls:
                state[i] = 1
            elif p < ss and p < (short_stop[i - 1] if np.isfinite(short_stop[i - 1]) else ss):
                state[i] = -1
            else:
                state[i] = state[i - 1]
        long_stop[i] = ls
        short_stop[i] = ss
    return pd.DataFrame({"long_stop": long_stop, "short_stop": short_stop, "state": state}, index=c.index)


def zero_crossing_rate(series: pd.Series, period: int = 20) -> pd.Series:
    """Rate of sign changes of the detrended series over a rolling window.

    Values near 0 imply persistent trend; near 1 imply choppy oscillation.
    """
    s = _as_series(series)
    detrended = roof(s, period)
    signs = np.sign(detrended).fillna(0.0)
    changes = (signs != signs.shift(1)).astype("float64")
    changes[detrended.isna() | detrended.shift(1).isna()] = np.nan
    return changes.rolling(period, min_periods=period).mean()


def adaptive_stochastic(series: pd.Series, period: int = 20, smooth: int = 3) -> pd.Series:
    """Stochastic oscillator on the adaptive price path (trend + cycle)."""
    s = _as_series(series)
    path = instant_trendline(s, period) + roof(s, period)
    lo = path.rolling(period, min_periods=period).min()
    hi = path.rolling(period, min_periods=period).max()
    k = 100.0 * (path - lo) / (hi - lo).replace(0, np.nan)
    return k.ewm(span=smooth, adjust=False, min_periods=1).mean()


def noise_cleaner(series: pd.Series, min_period: int = 10, max_period: int = 48) -> pd.Series:
    """Ehlers noise eliminator: adaptive smoother locked to the dominant cycle."""
    s = _as_series(series)
    dc = dominant_cycle(s, min_period, max_period).ffill().fillna(20.0)
    alpha = 2.0 / (dc + 1.0)
    return _adaptive_ema(s, alpha, s)


def ehlers_fisher(series: pd.Series, period: int = 10) -> pd.Series:
    """Fisher transform of the normalized median price (Ehlers' version)."""
    s = _as_series(series)
    lo = s.rolling(period, min_periods=period).min()
    hi = s.rolling(period, min_periods=period).max()
    norm = (2.0 * (s - lo) / (hi - lo).replace(0, np.nan) - 1.0).fillna(0.0).clip(-0.999, 0.999)
    x = norm.to_numpy(dtype="float64")
    out = np.empty(len(x), dtype="float64")
    prev = 0.0
    for i in range(len(x)):
        val = 0.33 * (0.0 if not np.isfinite(x[i]) else x[i]) + 0.67 * prev
        val = float(np.clip(val, -0.999, 0.999))
        out[i] = 0.5 * np.log((1.0 + val) / (1.0 - val)) + 0.5 * prev
        prev = val
    return pd.Series(out, index=s.index, name=s.name)


def sma_of_sine(series: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.DataFrame:
    """Sine wave with smoothed %D-style average for crossover signals."""
    sw = even_better_sine(series, min_period, max_period)
    return pd.DataFrame(
        {"sine": sw["sine"], "sine_avg": sw["sine"].rolling(5, min_periods=1).mean(), "period": sw["period"]},
        index=sw.index,
    )
