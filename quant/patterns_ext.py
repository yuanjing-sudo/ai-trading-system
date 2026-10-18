"""Chart pattern, market-structure, and divergence detection.

Modules in this family use swing pivots (rolling or zigzag based) to identify
classical chart formations - double/triple tops, head-and-shoulders, triangles,
wedges, channels, broadening, cup-and-handle, rounding and V formations, and
flags - plus structure events (breaks of structure, higher-high/lower-low
sequences), horizontal support/resistance levels with breakout flags, and
price/indicator divergences.

Every detector returns a boolean Series that is True on the bar where the
formation becomes identifiable (i.e. the confirming pivot bar), or a numeric
Series of states/distances. Inputs are never mutated and all outputs are
aligned to the input index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series, macd, obv, rsi

__all__ = [
    # pivots / swings
    "pivot_highs", "pivot_lows", "pivot_high_values", "pivot_low_values",
    "swing_points", "swing_series", "swing_count", "swing_levels",
    "swing_high_distance", "swing_low_distance",
    # structure events
    "higher_highs", "lower_lows", "higher_lows", "lower_highs",
    "trend_structure", "break_of_structure", "change_of_character",
    # levels / breakouts
    "rolling_support", "rolling_resistance", "support_distance",
    "resistance_distance", "breakout_up", "breakdown", "false_breakout",
    "false_breakdown", "range_position", "level_touches",
    # reversal formations
    "double_top", "double_bottom", "triple_top", "triple_bottom",
    "head_and_shoulders", "inverse_head_and_shoulders", "rounding_top",
    "rounding_bottom", "v_bottom", "v_top", "cup_and_handle",
    # continuation formations
    "ascending_triangle", "descending_triangle", "symmetric_triangle",
    "rising_wedge", "falling_wedge", "ascending_channel",
    "descending_channel", "horizontal_channel", "rectangle_pattern",
    "broadening_formation", "bull_flag", "bear_flag",
    # divergence
    "divergence_series", "bullish_divergence", "bearish_divergence",
    "hidden_bullish_divergence", "hidden_bearish_divergence",
    "rsi_divergence", "macd_divergence", "obv_divergence",
    # assembly
    "detect_patterns", "pattern_frequency", "structure_score", "chart_state",
]


# ---------------------------------------------------------------------------
# Pivot / swing infrastructure
# ---------------------------------------------------------------------------

def _rolling_pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    h = _as_series(high)
    window = left + right + 1
    wmax = h.rolling(window, center=True, min_periods=window).max()
    return (h >= wmax).fillna(False)


def _rolling_pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    l = _as_series(low)
    window = left + right + 1
    wmin = l.rolling(window, center=True, min_periods=window).min()
    return (l <= wmin).fillna(False)


def pivot_highs(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean flags where ``high`` is the maximum of a centered window."""
    return _rolling_pivot_high(high, left, right)


def pivot_lows(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Boolean flags where ``low`` is the minimum of a centered window."""
    return _rolling_pivot_low(low, left, right)


def pivot_high_values(high: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Pivot-high prices, NaN elsewhere."""
    h = _as_series(high)
    return h.where(_rolling_pivot_high(h, left, right))


def pivot_low_values(low: pd.Series, left: int = 2, right: int = 2) -> pd.Series:
    """Pivot-low prices, NaN elsewhere."""
    l = _as_series(low)
    return l.where(_rolling_pivot_low(l, left, right))


def _zigzag_records(high: pd.Series, low: pd.Series, pct: float = 0.05) -> list[tuple[int, float, str]]:
    """Zigzag pivot records as ``(bar_position, price, kind)`` with kind H/L."""
    h = _as_series(high).to_numpy(dtype="float64")
    l = _as_series(low).to_numpy(dtype="float64")
    n = len(h)
    if n < 2:
        return []
    records: list[tuple[int, float, str]] = []
    direction = 1
    ph_i, ph = 0, h[0]
    pl_i, pl = 0, l[0]
    for i in range(1, n):
        if not (np.isfinite(h[i]) and np.isfinite(l[i])):
            continue
        if direction == 1:
            if h[i] > ph:
                ph_i, ph = i, h[i]
            elif l[i] <= ph * (1.0 - pct):
                records.append((ph_i, ph, "H"))
                direction = -1
                pl_i, pl = i, l[i]
        else:
            if l[i] < pl:
                pl_i, pl = i, l[i]
            elif h[i] >= pl * (1.0 + pct):
                records.append((pl_i, pl, "L"))
                direction = 1
                ph_i, ph = i, h[i]
    if direction == 1:
        records.append((ph_i, ph, "H"))
    else:
        records.append((pl_i, pl, "L"))
    return records


def swing_points(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.DataFrame:
    """Zigzag swing pivots as a frame with ``bar``, ``price``, ``kind``."""
    h = _as_series(high)
    records = _zigzag_records(h, low, pct)
    return pd.DataFrame(
        {
            "bar": [r[0] for r in records],
            "price": [r[1] for r in records],
            "kind": [r[2] for r in records],
        }
    )


def swing_series(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Step-wise swing line: price at the latest confirmed pivot, forward-filled."""
    h = _as_series(high)
    records = _zigzag_records(h, low, pct)
    out = np.full(len(h), np.nan)
    for bar, price, _kind in records:
        if bar < len(out):
            out[bar] = price
    return pd.Series(out, index=h.index).ffill()


def swing_count(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05, kind: str = "both") -> pd.Series:
    """Rolling count of zigzag pivots within ``window`` bars."""
    h = _as_series(high)
    records = _zigzag_records(h, low, pct)
    bars = np.array([r[0] for r in records], dtype="int64")
    kinds = np.array([r[2] for r in records])
    flags = np.zeros(len(h), dtype="float64")
    for bar, k in zip(bars, kinds):
        if bar < len(flags) and (kind == "both" or k.lower().startswith(kind.lower())):
            flags[bar] = 1.0
    return pd.Series(flags, index=h.index).rolling(window, min_periods=1).sum()


def swing_levels(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.DataFrame:
    """Forward-filled latest swing-high and swing-low levels per bar."""
    h = _as_series(high)
    records = _zigzag_records(h, low, pct)
    h_lvl = np.full(len(h), np.nan)
    l_lvl = np.full(len(h), np.nan)
    for bar, price, k in records:
        if bar >= len(h):
            continue
        if k == "H":
            h_lvl[bar] = price
        else:
            l_lvl[bar] = price
    return pd.DataFrame(
        {"swing_high_level": pd.Series(h_lvl, index=h.index).ffill(), "swing_low_level": pd.Series(l_lvl, index=h.index).ffill()}
    )


def swing_high_distance(close: pd.Series, high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Relative distance from close to the latest swing-high level."""
    lvl = swing_levels(high, low, pct)["swing_high_level"]
    c = _as_series(close)
    return c / lvl - 1.0


def swing_low_distance(close: pd.Series, high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Relative distance from close to the latest swing-low level."""
    lvl = swing_levels(high, low, pct)["swing_low_level"]
    c = _as_series(close)
    return c / lvl - 1.0


# ---------------------------------------------------------------------------
# Structure events
# ---------------------------------------------------------------------------

def _pivot_pairs(high: pd.Series, low: pd.Series, pct: float) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    records = _zigzag_records(high, low, pct)
    highs = [(bar, price) for bar, price, k in records if k == "H"]
    lows = [(bar, price) for bar, price, k in records if k == "L"]
    return highs, lows


def _event_series(high: pd.Series, low: pd.Series, pct: float, kind: str) -> pd.Series:
    h = _as_series(high)
    highs, lows = _pivot_pairs(h, low, pct)
    flags = np.zeros(len(h), dtype="float64")
    if kind in ("hh", "lh"):
        pts = highs
        better = (lambda a, b: a > b) if kind == "hh" else (lambda a, b: a < b)
    else:
        pts = lows
        better = (lambda a, b: a > b) if kind == "hl" else (lambda a, b: a < b)
    for i in range(1, len(pts)):
        if better(pts[i][1], pts[i - 1][1]) and pts[i][0] < len(flags):
            flags[pts[i][0]] = 1.0
    return pd.Series(flags, index=h.index).astype(bool)


def higher_highs(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """True at swing highs that exceed the previous swing high."""
    return _event_series(high, low, pct, "hh")


def lower_lows(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """True at swing lows that undercut the previous swing low."""
    return _event_series(high, low, pct, "ll")


def higher_lows(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """True at swing lows that hold above the previous swing low."""
    return _event_series(high, low, pct, "hl")


def lower_highs(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """True at swing highs that fail below the previous swing high."""
    return _event_series(high, low, pct, "lh")


def trend_structure(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Market structure state: +1 uptrend, -1 downtrend, 0 mixed/undefined."""
    hh = higher_highs(high, low, pct)
    hl = higher_lows(high, low, pct)
    ll = lower_lows(high, low, pct)
    lh = lower_highs(high, low, pct)
    h = _as_series(high)
    state = np.zeros(len(h), dtype="float64")
    for i in range(len(h)):
        if hh.iloc[i] and hl.iloc[i]:
            state[i] = 1.0
        elif ll.iloc[i] and lh.iloc[i]:
            state[i] = -1.0
    return pd.Series(state, index=h.index).replace(0.0, np.nan).ffill().fillna(0.0)


def break_of_structure(close: pd.Series, high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Break-of-structure events: +1 close above last swing high, -1 below last swing low."""
    c = _as_series(close)
    lv = swing_levels(high, low, pct)
    up = (c > lv["swing_high_level"]) & (c.shift(1) <= lv["swing_high_level"].shift(1))
    dn = (c < lv["swing_low_level"]) & (c.shift(1) >= lv["swing_low_level"].shift(1))
    out = pd.Series(0.0, index=c.index)
    out[up.fillna(False)] = 1.0
    out[dn.fillna(False)] = -1.0
    return out


def change_of_character(high: pd.Series, low: pd.Series, pct: float = 0.05) -> pd.Series:
    """Change-of-character: +1 first higher low in a downtrend, -1 first lower high in an uptrend."""
    h = _as_series(high)
    hh = higher_highs(h, low, pct)
    hl = higher_lows(h, low, pct)
    ll = lower_lows(h, low, pct)
    lh = lower_highs(h, low, pct)
    out = np.zeros(len(h), dtype="float64")
    hs = hh.to_numpy()
    ls = hl.to_numpy()
    llv = ll.to_numpy()
    lhv = lh.to_numpy()
    prev_down = prev_up = False
    for i in range(len(h)):
        if hl.iloc[i] and prev_down:
            out[i] = 1.0
        elif lh.iloc[i] and prev_up:
            out[i] = -1.0
        if llv[i] and lhv[i]:
            prev_down, prev_up = True, False
        elif hs[i] and ls[i]:
            prev_up, prev_down = True, False
    return pd.Series(out, index=h.index)


# ---------------------------------------------------------------------------
# Levels / breakouts
# ---------------------------------------------------------------------------

def rolling_support(low: pd.Series, period: int = 20) -> pd.Series:
    """Rolling minimum low as dynamic support."""
    return _as_series(low).rolling(period, min_periods=max(2, period // 2)).min()


def rolling_resistance(high: pd.Series, period: int = 20) -> pd.Series:
    """Rolling maximum high as dynamic resistance."""
    return _as_series(high).rolling(period, min_periods=max(2, period // 2)).max()


def support_distance(close: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Relative distance of close above the rolling support."""
    c = _as_series(close)
    return c / rolling_support(low, period).replace(0, np.nan) - 1.0


def resistance_distance(close: pd.Series, high: pd.Series, period: int = 20) -> pd.Series:
    """Relative distance of close below the rolling resistance."""
    c = _as_series(close)
    return rolling_resistance(high, period).replace(0, np.nan) / c - 1.0


def breakout_up(close: pd.Series, high: pd.Series, period: int = 20, confirm_pct: float = 0.0) -> pd.Series:
    """True when close exceeds the prior rolling resistance by ``confirm_pct``."""
    c = _as_series(close)
    level = _as_series(high).rolling(period, min_periods=max(2, period // 2)).max().shift(1)
    return ((c > level * (1.0 + confirm_pct)) & (c.shift(1) <= level)).fillna(False)


def breakdown(close: pd.Series, low: pd.Series, period: int = 20, confirm_pct: float = 0.0) -> pd.Series:
    """True when close breaks below the prior rolling support by ``confirm_pct``."""
    c = _as_series(close)
    level = _as_series(low).rolling(period, min_periods=max(2, period // 2)).min().shift(1)
    return ((c < level * (1.0 - confirm_pct)) & (c.shift(1) >= level)).fillna(False)


def false_breakout(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20, tol: float = 0.005) -> pd.Series:
    """False upside breakouts: high pierces resistance but close falls back inside."""
    c = _as_series(close)
    h = _as_series(high)
    level = h.rolling(period, min_periods=max(2, period // 2)).max().shift(1)
    pierced = (h > level * (1.0 + tol)) & (c < level)
    return pierced.fillna(False)


def false_breakdown(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20, tol: float = 0.005) -> pd.Series:
    """False downside breaks: low pierces support but close recovers inside."""
    c = _as_series(close)
    l = _as_series(low)
    level = l.rolling(period, min_periods=max(2, period // 2)).min().shift(1)
    pierced = (l < level * (1.0 - tol)) & (c > level)
    return pierced.fillna(False)


def range_position(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    """Position of close within the rolling high-low range, in [0, 1]."""
    c = _as_series(close)
    hh = _as_series(high).rolling(period, min_periods=max(2, period // 2)).max()
    ll = _as_series(low).rolling(period, min_periods=max(2, period // 2)).min()
    return (c - ll) / (hh - ll).replace(0, np.nan)


def level_touches(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 60, tol: float = 0.01) -> pd.Series:
    """Rolling count of bars touching (within ``tol``) either range boundary."""
    c = _as_series(close)
    hh = _as_series(high).rolling(period, min_periods=max(2, period // 2)).max()
    ll = _as_series(low).rolling(period, min_periods=max(2, period // 2)).min()
    near_top = (hh - c) / c <= tol
    near_bot = (c - ll) / c <= tol
    return (near_top | near_bot).astype(float).rolling(period, min_periods=1).sum()


# ---------------------------------------------------------------------------
# Formation scanners (cores take precomputed zigzag records)
# ---------------------------------------------------------------------------

def _same_kind(records: list[tuple[int, float, str]], kind: str) -> list[tuple[int, float]]:
    return [(b, p) for b, p, k in records if k == kind]


def _double_flags(index: pd.Index, records: list[tuple[int, float, str]], kind: str, tol: float, min_gap: int) -> pd.Series:
    pts = _same_kind(records, kind)
    flags = np.zeros(len(index), dtype=bool)
    for i in range(1, len(pts)):
        b1, p1 = pts[i - 1]
        b2, p2 = pts[i]
        if b2 - b1 < min_gap or p1 <= 0:
            continue
        if abs(p2 / p1 - 1.0) <= tol and b2 < len(flags):
            flags[b2] = True
    return pd.Series(flags, index=index)


def _triple_flags(index: pd.Index, records: list[tuple[int, float, str]], kind: str, tol: float, min_gap: int) -> pd.Series:
    pts = _same_kind(records, kind)
    flags = np.zeros(len(index), dtype=bool)
    for i in range(2, len(pts)):
        (b1, p1), (b2, p2), (b3, p3) = pts[i - 2], pts[i - 1], pts[i]
        if min(b2 - b1, b3 - b2) < min_gap or p1 <= 0:
            continue
        if abs(p2 / p1 - 1.0) <= tol and abs(p3 / p1 - 1.0) <= tol and b3 < len(flags):
            flags[b3] = True
    return pd.Series(flags, index=index)


def _hs_flags(index: pd.Index, records: list[tuple[int, float, str]], inverse: bool, shoulder_min: float, tol: float) -> pd.Series:
    pts = _same_kind(records, "L" if inverse else "H")
    flags = np.zeros(len(index), dtype=bool)
    for i in range(2, len(pts)):
        (_, p1), (_, p2), (b3, p3) = pts[i - 2], pts[i - 1], pts[i]
        if p1 <= 0:
            continue
        if inverse:
            head_ok = p2 <= p1 * (1.0 - shoulder_min) and p2 <= p3 * (1.0 - shoulder_min)
        else:
            head_ok = p2 >= p1 * (1.0 + shoulder_min) and p2 >= p3 * (1.0 + shoulder_min)
        if head_ok and abs(p3 / p1 - 1.0) <= tol and b3 < len(flags):
            flags[b3] = True
    return pd.Series(flags, index=index)


def _slope_flags(
    index: pd.Index,
    records: list[tuple[int, float, str]],
    kind: str,
    window: int,
    flat: float,
    min_slope: float,
    parallel_tol: float = 0.35,
) -> pd.Series:
    hs = _same_kind(records, "H")
    ls = _same_kind(records, "L")
    hb = np.array([b for b, _ in hs], dtype="int64")
    lb = np.array([b for b, _ in ls], dtype="int64")
    flags = np.zeros(len(index), dtype=bool)
    for i in range(len(index)):
        kh = int(np.searchsorted(hb, i))
        kl = int(np.searchsorted(lb, i))
        if kh < 2 or kl < 2:
            continue
        (bh1, ph1), (bh2, ph2) = hs[kh - 2], hs[kh - 1]
        (bl1, pl1), (bl2, pl2) = ls[kl - 2], ls[kl - 1]
        if i - bh2 > window or i - bl2 > window:
            continue
        if ph1 <= 0 or pl1 <= 0:
            continue
        rel_h = ph2 / ph1 - 1.0
        rel_l = pl2 / pl1 - 1.0
        sh = (ph2 - ph1) / max(bh2 - bh1, 1)
        sl = (pl2 - pl1) / max(bl2 - bl1, 1)
        ok = False
        if kind == "ascending_triangle":
            ok = abs(rel_h) <= flat and rel_l >= min_slope
        elif kind == "descending_triangle":
            ok = rel_h <= -min_slope and abs(rel_l) <= flat
        elif kind == "symmetric_triangle":
            ok = rel_h <= -min_slope and rel_l >= min_slope
        elif kind == "rising_wedge":
            ok = sh > 0.0 and sl > 0.0 and sh < sl
        elif kind == "falling_wedge":
            ok = sh < 0.0 and sl < 0.0 and sh < sl
        elif kind == "broadening_formation":
            ok = rel_h >= min_slope and rel_l <= -min_slope
        elif kind in ("ascending_channel", "descending_channel"):
            same_dir = (sh > 0.0 and sl > 0.0) if kind == "ascending_channel" else (sh < 0.0 and sl < 0.0)
            scale = max(abs(sh), abs(sl), 1e-12)
            parallel = abs(sh - sl) <= parallel_tol * scale
            neck1 = max(ph1 - pl1, 1e-12)
            neck2 = max(ph2 - pl2, 1e-12)
            stable = abs(neck2 / neck1 - 1.0) <= parallel_tol
            ok = same_dir and parallel and stable
        elif kind == "horizontal_channel":
            ok = abs(rel_h) <= flat and abs(rel_l) <= flat
        if ok:
            flags[i] = True
    return pd.Series(flags, index=index)


def _window_loop(index: pd.Index, values: np.ndarray, window: int, criterion) -> pd.Series:
    flags = np.zeros(len(values), dtype=bool)
    for i in range(window - 1, len(values)):
        w = values[i - window + 1: i + 1]
        if not np.isfinite(w).all():
            continue
        try:
            if criterion(w):
                flags[i] = True
        except Exception:
            continue
    return pd.Series(flags, index=index)


def _quad_r2(w: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(w), dtype="float64")
    coef = np.polyfit(x, w, 2)
    fit = np.polyval(coef, x)
    ss_res = float(((w - fit) ** 2).sum())
    ss_tot = float(((w - w.mean()) ** 2).sum()) or 1e-12
    return float(coef[0]), 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Reversal formations
# ---------------------------------------------------------------------------

def double_top(high: pd.Series, low: pd.Series, tol: float = 0.03, pct: float = 0.05, min_gap: int = 5) -> pd.Series:
    """Double-top: two swing highs within ``tol`` with a trough between them."""
    h = _as_series(high)
    return _double_flags(h.index, _zigzag_records(h, low, pct), "H", tol, min_gap)


def double_bottom(high: pd.Series, low: pd.Series, tol: float = 0.03, pct: float = 0.05, min_gap: int = 5) -> pd.Series:
    """Double-bottom: two swing lows within ``tol`` with a peak between them."""
    h = _as_series(high)
    return _double_flags(h.index, _zigzag_records(h, low, pct), "L", tol, min_gap)


def triple_top(high: pd.Series, low: pd.Series, tol: float = 0.03, pct: float = 0.05, min_gap: int = 5) -> pd.Series:
    """Triple-top: three swing highs within ``tol``."""
    h = _as_series(high)
    return _triple_flags(h.index, _zigzag_records(h, low, pct), "H", tol, min_gap)


def triple_bottom(high: pd.Series, low: pd.Series, tol: float = 0.03, pct: float = 0.05, min_gap: int = 5) -> pd.Series:
    """Triple-bottom: three swing lows within ``tol``."""
    h = _as_series(high)
    return _triple_flags(h.index, _zigzag_records(h, low, pct), "L", tol, min_gap)


def head_and_shoulders(high: pd.Series, low: pd.Series, shoulder_min: float = 0.02, tol: float = 0.05, pct: float = 0.05) -> pd.Series:
    """Head-and-shoulders top: higher middle peak with matching shoulders."""
    h = _as_series(high)
    return _hs_flags(h.index, _zigzag_records(h, low, pct), False, shoulder_min, tol)


def inverse_head_and_shoulders(high: pd.Series, low: pd.Series, shoulder_min: float = 0.02, tol: float = 0.05, pct: float = 0.05) -> pd.Series:
    """Inverse head-and-shoulders: lower middle trough with matching shoulders."""
    h = _as_series(high)
    return _hs_flags(h.index, _zigzag_records(h, low, pct), True, shoulder_min, tol)


def rounding_top(close: pd.Series, window: int = 60, min_r2: float = 0.5) -> pd.Series:
    """Rounding top: concave-down quadratic fit with the peak mid-window."""
    c = _as_series(close)

    def crit(w: np.ndarray) -> bool:
        coef, r2 = _quad_r2(w)
        mid = len(w) // 3
        peak = int(np.argmax(w))
        return coef < 0 and r2 >= min_r2 and mid <= peak <= len(w) - mid

    return _window_loop(c.index, c.to_numpy(dtype="float64"), window, crit)


def rounding_bottom(close: pd.Series, window: int = 60, min_r2: float = 0.5) -> pd.Series:
    """Rounding bottom: concave-up quadratic fit with the trough mid-window."""
    c = _as_series(close)

    def crit(w: np.ndarray) -> bool:
        coef, r2 = _quad_r2(w)
        mid = len(w) // 3
        trough = int(np.argmin(w))
        return coef > 0 and r2 >= min_r2 and mid <= trough <= len(w) - mid

    return _window_loop(c.index, c.to_numpy(dtype="float64"), window, crit)


def v_bottom(close: pd.Series, window: int = 30, min_move: float = 0.03) -> pd.Series:
    """V-bottom: sharp trough with both legs moving at least ``min_move``."""
    c = _as_series(close)

    def crit(w: np.ndarray) -> bool:
        low = w.min()
        if low <= 0:
            return False
        left = w[0] / low - 1.0
        right = w[-1] / low - 1.0
        trough = int(np.argmin(w))
        return left >= min_move and right >= min_move and len(w) // 4 <= trough <= 3 * len(w) // 4

    return _window_loop(c.index, c.to_numpy(dtype="float64"), window, crit)


def v_top(close: pd.Series, window: int = 30, min_move: float = 0.03) -> pd.Series:
    """V-top: sharp peak with both legs falling at least ``min_move``."""
    c = _as_series(close)

    def crit(w: np.ndarray) -> bool:
        high = w.max()
        if high <= 0:
            return False
        left = 1.0 - w[0] / high
        right = 1.0 - w[-1] / high
        peak = int(np.argmax(w))
        return left >= min_move and right >= min_move and len(w) // 4 <= peak <= 3 * len(w) // 4

    return _window_loop(c.index, c.to_numpy(dtype="float64"), window, crit)


def cup_and_handle(close: pd.Series, window: int = 80, tol: float = 0.05, handle_frac: float = 0.25, min_depth: float = 0.06) -> pd.Series:
    """Cup-and-handle: rounded base with rims within ``tol`` and a shallow handle."""
    c = _as_series(close)
    n = len(c)
    handle = max(int(window * handle_frac), 3)
    cup_len = window - handle
    if cup_len < 20:
        return pd.Series(False, index=c.index)
    flags = np.zeros(n, dtype=bool)
    cv = c.to_numpy(dtype="float64")
    for i in range(window - 1, n):
        w = cv[i - window + 1: i + 1]
        if not np.isfinite(w).all():
            continue
        cup = w[:cup_len]
        hand = w[cup_len:]
        rim = max(cup[0], cup[-1])
        if rim <= 0:
            continue
        depth = rim - cup.min()
        if depth / rim < min_depth:
            continue
        if abs(cup[-1] / cup[0] - 1.0) > tol:
            continue
        trough = int(np.argmin(cup))
        if not (len(cup) // 4 <= trough <= 3 * len(cup) // 4):
            continue
        if hand.min() < rim - depth / 3.0:
            continue
        if (hand.max() - hand.min()) / rim > depth * 0.8:
            continue
        flags[i] = True
    return pd.Series(flags, index=c.index)


# ---------------------------------------------------------------------------
# Continuation formations
# ---------------------------------------------------------------------------

def ascending_triangle(high: pd.Series, low: pd.Series, window: int = 60, flat: float = 0.02, min_slope: float = 0.015, pct: float = 0.05) -> pd.Series:
    """Ascending triangle: flat resistance with rising support pivots."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "ascending_triangle", window, flat, min_slope)


def descending_triangle(high: pd.Series, low: pd.Series, window: int = 60, flat: float = 0.02, min_slope: float = 0.015, pct: float = 0.05) -> pd.Series:
    """Descending triangle: flat support with falling resistance pivots."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "descending_triangle", window, flat, min_slope)


def symmetric_triangle(high: pd.Series, low: pd.Series, window: int = 60, flat: float = 0.02, min_slope: float = 0.015, pct: float = 0.05) -> pd.Series:
    """Symmetric triangle: falling highs into rising lows (converging)."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "symmetric_triangle", window, flat, min_slope)


def rising_wedge(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05) -> pd.Series:
    """Rising wedge: converging upward lines (bearish bias)."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "rising_wedge", window, 0.02, 0.015)


def falling_wedge(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05) -> pd.Series:
    """Falling wedge: converging downward lines (bullish bias)."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "falling_wedge", window, 0.02, 0.015)


def ascending_channel(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05) -> pd.Series:
    """Ascending parallel channel."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "ascending_channel", window, 0.02, 0.015)


def descending_channel(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05) -> pd.Series:
    """Descending parallel channel."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "descending_channel", window, 0.02, 0.015)


def horizontal_channel(high: pd.Series, low: pd.Series, window: int = 60, flat: float = 0.02, pct: float = 0.05) -> pd.Series:
    """Horizontal range: flat highs and flat lows."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "horizontal_channel", window, flat, 0.015)


def rectangle_pattern(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 40, tol: float = 0.03) -> pd.Series:
    """Rectangle: tight rolling range with repeated boundary touches."""
    c = _as_series(close)
    hh = _as_series(high).rolling(period, min_periods=max(2, period // 2)).max()
    ll = _as_series(low).rolling(period, min_periods=max(2, period // 2)).min()
    tight = (hh - ll) / c <= 2.0 * tol
    touches = level_touches(c, high, low, period, tol)
    return (tight & (touches >= 2)).fillna(False)


def broadening_formation(high: pd.Series, low: pd.Series, window: int = 60, pct: float = 0.05) -> pd.Series:
    """Broadening formation: rising highs with falling lows (diverging)."""
    h = _as_series(high)
    return _slope_flags(h.index, _zigzag_records(h, low, pct), "broadening_formation", window, 0.02, 0.015)


def _flag_pattern(close: pd.Series, high: pd.Series, low: pd.Series, bearish: bool, pole_window: int, flag_window: int, pole_min: float) -> pd.Series:
    c = _as_series(close)
    hv = _as_series(high).to_numpy(dtype="float64")
    lv = _as_series(low).to_numpy(dtype="float64")
    cv = c.to_numpy(dtype="float64")
    n = len(cv)
    flags = np.zeros(n, dtype=bool)
    span = pole_window + flag_window
    for i in range(span, n):
        base = cv[i - span]
        pole_end = cv[i - flag_window] if flag_window > 0 else cv[i]
        if not (np.isfinite(base) and np.isfinite(pole_end)) or base <= 0:
            continue
        pole_ret = pole_end / base - 1.0
        fh = hv[i - flag_window + 1: i + 1]
        fl = lv[i - flag_window + 1: i + 1]
        if not (np.isfinite(fh).all() and np.isfinite(fl).all()):
            continue
        if bearish:
            if pole_ret > -pole_min:
                continue
            if (fh.max() - fl.min()) / pole_end > 0.5 * abs(pole_ret):
                continue
            if fh.max() > base + 0.5 * (pole_end - base):
                continue
        else:
            if pole_ret < pole_min:
                continue
            if (fh.max() - fl.min()) / pole_end > 0.5 * pole_ret:
                continue
            if fl.min() < base + 0.5 * (pole_end - base):
                continue
        flags[i] = True
    return pd.Series(flags, index=c.index)


def bull_flag(close: pd.Series, high: pd.Series, low: pd.Series, pole_window: int = 20, flag_window: int = 10, pole_min: float = 0.05) -> pd.Series:
    """Bull flag: sharp advance followed by a shallow consolidation."""
    return _flag_pattern(close, high, low, False, pole_window, flag_window, pole_min)


def bear_flag(close: pd.Series, high: pd.Series, low: pd.Series, pole_window: int = 20, flag_window: int = 10, pole_min: float = 0.05) -> pd.Series:
    """Bear flag: sharp decline followed by a shallow consolidation."""
    return _flag_pattern(close, high, low, True, pole_window, flag_window, pole_min)


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------

def _divergence_core(
    close: pd.Series,
    indicator: pd.Series,
    left: int,
    right: int,
    lookback: int,
    mode: str,
) -> pd.Series:
    c = _as_series(close)
    ind = _as_series(indicator).reindex(c.index)
    ph_bars = np.flatnonzero(_rolling_pivot_high(c, left, right).to_numpy())
    pl_bars = np.flatnonzero(_rolling_pivot_low(c, left, right).to_numpy())
    c_arr = c.to_numpy(dtype="float64")
    ind_arr = ind.to_numpy(dtype="float64")
    out = np.zeros(len(c), dtype="float64")

    def scan(bars: np.ndarray, is_high: bool) -> None:
        prev: tuple[int, float, float] | None = None
        for b in bars:
            if b + right >= len(c_arr):
                continue
            pv = c_arr[b]
            iv = ind_arr[b]
            if not (np.isfinite(pv) and np.isfinite(iv)):
                continue
            if prev is not None:
                b0, p0, i0 = prev
                if b - b0 <= lookback:
                    price_up = pv > p0
                    ind_down = iv < i0
                    if is_high:
                        if mode == "regular" and price_up and ind_down:
                            out[min(b + right, len(out) - 1)] = -1.0
                        elif mode == "hidden" and not price_up and not ind_down:
                            out[min(b + right, len(out) - 1)] = -1.0
                    else:
                        if mode == "regular" and not price_up and not ind_down:
                            out[min(b + right, len(out) - 1)] = 1.0
                        elif mode == "hidden" and price_up and ind_down:
                            out[min(b + right, len(out) - 1)] = 1.0
            prev = (b, pv, iv)

    scan(ph_bars, True)
    scan(pl_bars, False)
    return pd.Series(out, index=c.index)


def divergence_series(close: pd.Series, indicator: pd.Series, left: int = 3, right: int = 3, lookback: int = 100, mode: str = "regular") -> pd.Series:
    """Divergence state: +1 bullish, -1 bearish, 0 none (marked at confirmation bar)."""
    return _divergence_core(close, indicator, left, right, lookback, mode)


def bullish_divergence(close: pd.Series, indicator: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """True where a regular bullish price/indicator divergence confirms."""
    return (_divergence_core(close, indicator, left, right, lookback, "regular") == 1.0)


def bearish_divergence(close: pd.Series, indicator: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """True where a regular bearish price/indicator divergence confirms."""
    return (_divergence_core(close, indicator, left, right, lookback, "regular") == -1.0)


def hidden_bullish_divergence(close: pd.Series, indicator: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """True where a hidden bullish divergence confirms (trend continuation)."""
    return (_divergence_core(close, indicator, left, right, lookback, "hidden") == 1.0)


def hidden_bearish_divergence(close: pd.Series, indicator: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """True where a hidden bearish divergence confirms (trend continuation)."""
    return (_divergence_core(close, indicator, left, right, lookback, "hidden") == -1.0)


def rsi_divergence(close: pd.Series, period: int = 14, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """Divergence between price and RSI."""
    c = _as_series(close)
    return _divergence_core(c, rsi(c, period), left, right, lookback, "regular")


def macd_divergence(close: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """Divergence between price and the MACD histogram."""
    c = _as_series(close)
    return _divergence_core(c, macd(c)["hist"], left, right, lookback, "regular")


def obv_divergence(close: pd.Series, volume: pd.Series, left: int = 3, right: int = 3, lookback: int = 100) -> pd.Series:
    """Divergence between price and On-Balance Volume."""
    c = _as_series(close)
    return _divergence_core(c, obv(c, volume), left, right, lookback, "regular")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_BULLISH_FORMATIONS = (
    "double_bottom", "triple_bottom", "inverse_head_and_shoulders",
    "rounding_bottom", "v_bottom", "cup_and_handle", "ascending_triangle",
    "falling_wedge", "bull_flag",
)
_BEARISH_FORMATIONS = (
    "double_top", "triple_top", "head_and_shoulders", "rounding_top", "v_top",
    "descending_triangle", "rising_wedge", "bear_flag",
)


def _ohlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    lookup = {str(col).lower(): col for col in df.columns}
    high = _as_series(df[lookup["high"]])
    low = _as_series(df[lookup["low"]])
    close = _as_series(df[lookup["close"]])
    volume = _as_series(df[lookup["volume"]]) if "volume" in lookup else pd.Series(np.nan, index=close.index)
    return high, low, close, volume


def detect_patterns(df: pd.DataFrame, window: int = 60, tol: float = 0.03, pct: float = 0.05) -> pd.DataFrame:
    """Run every chart-formation detector on an OHLCV frame."""
    high, low, close, _ = _ohlc(df)
    checks = {
        "double_top": lambda: double_top(high, low, tol=tol, pct=pct),
        "double_bottom": lambda: double_bottom(high, low, tol=tol, pct=pct),
        "triple_top": lambda: triple_top(high, low, tol=tol, pct=pct),
        "triple_bottom": lambda: triple_bottom(high, low, tol=tol, pct=pct),
        "head_and_shoulders": lambda: head_and_shoulders(high, low, pct=pct),
        "inverse_head_and_shoulders": lambda: inverse_head_and_shoulders(high, low, pct=pct),
        "rounding_top": lambda: rounding_top(close, window),
        "rounding_bottom": lambda: rounding_bottom(close, window),
        "v_bottom": lambda: v_bottom(close, max(window // 2, 20)),
        "v_top": lambda: v_top(close, max(window // 2, 20)),
        "cup_and_handle": lambda: cup_and_handle(close, max(window, 40)),
        "ascending_triangle": lambda: ascending_triangle(high, low, window, pct=pct),
        "descending_triangle": lambda: descending_triangle(high, low, window, pct=pct),
        "symmetric_triangle": lambda: symmetric_triangle(high, low, window, pct=pct),
        "rising_wedge": lambda: rising_wedge(high, low, window, pct=pct),
        "falling_wedge": lambda: falling_wedge(high, low, window, pct=pct),
        "ascending_channel": lambda: ascending_channel(high, low, window, pct=pct),
        "descending_channel": lambda: descending_channel(high, low, window, pct=pct),
        "horizontal_channel": lambda: horizontal_channel(high, low, window, pct=pct),
        "rectangle_pattern": lambda: rectangle_pattern(close, high, low, max(window // 2, 20), tol),
        "broadening_formation": lambda: broadening_formation(high, low, window, pct=pct),
        "bull_flag": lambda: bull_flag(close, high, low),
        "bear_flag": lambda: bear_flag(close, high, low),
    }
    data = {name: fn().fillna(False).astype(bool) for name, fn in checks.items()}
    return pd.DataFrame(data, index=close.index)


def pattern_frequency(detections: pd.DataFrame) -> pd.Series:
    """Count firings of each chart formation, sorted descending."""
    return detections.sum().sort_values(ascending=False)


def structure_score(detections: pd.DataFrame, window: int = 5) -> pd.Series:
    """Net formation score in [-1, 1]: bullish minus bearish firings, smoothed."""
    bull_cols = [c for c in _BULLISH_FORMATIONS if c in detections.columns]
    bear_cols = [c for c in _BEARISH_FORMATIONS if c in detections.columns]
    bull = detections[bull_cols].sum(axis=1) if bull_cols else 0.0
    bear = detections[bear_cols].sum(axis=1) if bear_cols else 0.0
    net = bull - bear
    if isinstance(net, pd.Series):
        return net.rolling(window, min_periods=1).mean().clip(-1.0, 1.0)
    return pd.Series(0.0, index=detections.index)


def chart_state(df: pd.DataFrame, window: int = 20, pct: float = 0.05) -> pd.DataFrame:
    """One-frame market-structure summary: trend, levels, breaks, swings."""
    high, low, close, _ = _ohlc(df)
    return pd.DataFrame(
        {
            "trend_structure": trend_structure(high, low, pct),
            "break_of_structure": break_of_structure(close, high, low, pct),
            "range_position_20": range_position(close, high, low, window),
            "support_distance_20": support_distance(close, low, window),
            "resistance_distance_20": resistance_distance(close, high, window),
            "breakout_up_20": breakout_up(close, high, window).astype(int),
            "breakdown_20": breakdown(close, low, window).astype(int),
            "false_breakout_20": false_breakout(close, high, low, window).astype(int),
            "false_breakdown_20": false_breakdown(close, high, low, window).astype(int),
            "swing_high_distance": swing_high_distance(close, high, low, pct),
            "swing_low_distance": swing_low_distance(close, high, low, pct),
        },
        index=close.index,
    )
