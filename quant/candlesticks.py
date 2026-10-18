"""Candlestick pattern library: 88 classical detectors (Nison taxonomy).

Every detector accepts ``(open, high, low, close)`` as pandas Series (or
arrays) and returns a boolean Series that is True on bars where the pattern is
present. Reversal patterns include their traditional trend precondition
(e.g. hammers only fire after a decline); pure shape detectors are documented
as such.

Aggregation helpers at the bottom of the module convert the library into
trading signals:

- :func:`detect_all` - DataFrame with one boolean column per pattern
- :func:`pattern_counts` - how often each pattern fired
- :func:`pattern_score` - weighted bull/bear score in [-1, 1]
- ``BULLISH_PATTERNS`` / ``BEARISH_PATTERNS`` - named registries

Conventions
-----------
- All comparisons use small tolerances derived from the bar range, so the
  detectors are robust to floating point noise.
- Trend preconditions look back 5 bars by default (``trend_lookback``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series

__all__ = [
    # single bar
    "doji", "long_legged_doji", "dragonfly_doji", "gravestone_doji",
    "four_price_doji", "rickshaw_man", "takuri", "hammer",
    "inverted_hammer", "hanging_man", "shooting_star", "spinning_top",
    "high_wave", "long_lower_shadow", "long_upper_shadow",
    "white_marubozu", "black_marubozu", "closing_marubozu_white",
    "closing_marubozu_black", "belt_hold_bull", "belt_hold_bear",
    "shaven_head", "shaven_bottom", "wide_range_bar", "narrow_range_bar",
    "up_bar", "down_bar",
    # two bar
    "inside_bar", "outside_bar", "bullish_engulfing", "bearish_engulfing",
    "bullish_harami", "bearish_harami", "bullish_harami_cross",
    "bearish_harami_cross", "piercing_line", "dark_cloud_cover",
    "bullish_counterattack", "bearish_counterattack", "matching_low",
    "matching_high", "kicking_bull", "kicking_bear", "meeting_lines_bull",
    "meeting_lines_bear", "separating_lines_bull", "separating_lines_bear",
    "thrusting", "in_neck", "on_neck", "two_crows_gapping",
    # three bar
    "morning_star", "morning_doji_star", "evening_star",
    "evening_doji_star", "three_white_soldiers", "three_black_crows",
    "three_inside_up", "three_inside_down", "three_outside_up",
    "three_outside_down", "three_stars_in_the_south",
    "three_line_strike_bull", "three_line_strike_bear",
    "abandoned_baby_bull", "abandoned_baby_bear", "advance_block",
    "deliberation", "identical_three_crows", "tasuki_gap_up",
    "tasuki_gap_down", "upside_gap_three_methods",
    "downside_gap_three_methods", "unique_three_river",
    "stick_sandwich", "tri_star_bull", "tri_star_bear",
    # multi bar
    "rising_three_methods", "falling_three_methods", "mat_hold_bull",
    "mat_hold_bear", "breakaway_bull", "breakaway_bear",
    "three_drives_bull", "three_drives_bear", "side_by_side_white_gapup",
    "side_by_side_white_gapdown", "concealing_baby_swallow",
    "ladder_bottom", "ladder_top",
    # aggregation
    "detect_all", "pattern_counts", "pattern_score", "ALL_PATTERNS",
    "BULLISH_PATTERNS", "BEARISH_PATTERNS",
]

TREND_LOOKBACK = 5


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

class _Bars:
    """Lightweight container of aligned bar-shape series."""

    __slots__ = ("o", "h", "l", "c", "body", "body_abs", "rng", "upper", "lower", "bull", "bear")

    def __init__(self, open_, high, low, close):
        o = _as_series(open_)
        h = _as_series(high).reindex_like(o)
        l = _as_series(low).reindex_like(o)
        c = _as_series(close).reindex_like(o)
        self.o, self.h, self.l, self.c = o, h, l, c
        self.body = c - o
        self.body_abs = self.body.abs()
        self.rng = (h - l).replace(0, np.nan)
        self.upper = h - np.maximum(o, c)
        self.lower = np.minimum(o, c) - l
        self.bull = (c > o).fillna(False)
        self.bear = (c < o).fillna(False)

    def shift(self, n: int) -> "_Bars":
        """Return the same bar set shifted ``n`` bars forward."""
        return _Bars(self.o.shift(n), self.h.shift(n), self.l.shift(n), self.c.shift(n))


def _bars(open_, high, low, close) -> _Bars:
    return _Bars(open_, high, low, close)


def _prior_down(close: pd.Series, lookback: int = TREND_LOOKBACK) -> pd.Series:
    """True when the bar before the pattern sat below the trend reference."""
    c = _as_series(close)
    return (c.shift(1) < c.shift(1 + lookback)).fillna(False)


def _prior_up(close: pd.Series, lookback: int = TREND_LOOKBACK) -> pd.Series:
    c = _as_series(close)
    return (c.shift(1) > c.shift(1 + lookback)).fillna(False)


def _is_doji(b: _Bars, max_body: float) -> pd.Series:
    return (b.body_abs <= max_body * b.rng).fillna(False)


def _all(*conds) -> pd.Series:
    out = None
    for cond in conds:
        s = _as_series(cond)
        out = s if out is None else (out & s)
    return out.fillna(False)


# ---------------------------------------------------------------------------
# Single-bar patterns
# ---------------------------------------------------------------------------

def doji(open_, high, low, close, body_pct: float = 0.1) -> pd.Series:
    """Doji: open and close nearly equal (body <= 10% of range)."""
    return _is_doji(_bars(open_, high, low, close), body_pct)


def long_legged_doji(open_, high, low, close, body_pct: float = 0.1, shadow_pct: float = 0.3) -> pd.Series:
    """Long-legged doji: doji with long upper and lower shadows."""
    b = _bars(open_, high, low, close)
    return _all(_is_doji(b, body_pct), b.upper >= shadow_pct * b.rng, b.lower >= shadow_pct * b.rng)


def dragonfly_doji(open_, high, low, close, body_pct: float = 0.1, lower_pct: float = 0.6) -> pd.Series:
    """Dragonfly doji: doji opening/closing at the high with a long lower shadow."""
    b = _bars(open_, high, low, close)
    return _all(_is_doji(b, body_pct), b.lower >= lower_pct * b.rng, b.upper <= 0.1 * b.rng)


def gravestone_doji(open_, high, low, close, body_pct: float = 0.1, upper_pct: float = 0.6) -> pd.Series:
    """Gravestone doji: doji at the low with a long upper shadow."""
    b = _bars(open_, high, low, close)
    return _all(_is_doji(b, body_pct), b.upper >= upper_pct * b.rng, b.lower <= 0.1 * b.rng)


def four_price_doji(open_, high, low, close, tol_pct: float = 1e-6) -> pd.Series:
    """Four-price doji: open, high, low, and close all equal."""
    b = _bars(open_, high, low, close)
    ref = b.c.abs().replace(0, np.nan)
    spread = (b.h - b.l) / ref
    same = (b.body_abs / ref) <= tol_pct
    return _all(b.h >= b.l, spread.fillna(0.0) <= tol_pct, same)


def rickshaw_man(open_, high, low, close, body_pct: float = 0.1) -> pd.Series:
    """Rickshaw man: long-legged doji with the tiny body centered in the range."""
    b = _bars(open_, high, low, close)
    center = (b.o + b.c) / 2.0
    mid = (b.h + b.l) / 2.0
    return _all(
        _is_doji(b, body_pct),
        b.upper >= 0.3 * b.rng,
        b.lower >= 0.3 * b.rng,
        (center - mid).abs() <= 0.15 * b.rng,
    )


def takuri(open_, high, low, close, lower_pct: float = 0.6, body_pct: float = 0.2) -> pd.Series:
    """Takuri: dragonfly variant with a small body at the top of the range."""
    b = _bars(open_, high, low, close)
    return _all(
        b.body_abs <= body_pct * b.rng,
        b.lower >= lower_pct * b.rng,
        b.upper <= 0.15 * b.rng,
        b.bull | b.bear,
    )


def hammer(open_, high, low, close, shadow_mult: float = 2.0) -> pd.Series:
    """Hammer: small body on top of a long lower shadow after a decline."""
    b = _bars(open_, high, low, close)
    body = b.body_abs.clip(lower=1e-12)
    return _all(
        _prior_down(b.c),
        b.lower >= shadow_mult * body,
        b.upper <= 0.3 * b.rng,
        body <= 0.6 * b.rng,
    )


def inverted_hammer(open_, high, low, close, shadow_mult: float = 2.0) -> pd.Series:
    """Inverted hammer: small body above a long upper shadow after a decline."""
    b = _bars(open_, high, low, close)
    body = b.body_abs.clip(lower=1e-12)
    return _all(
        _prior_down(b.c),
        b.upper >= shadow_mult * body,
        b.lower <= 0.3 * b.rng,
        body <= 0.6 * b.rng,
    )


def hanging_man(open_, high, low, close, shadow_mult: float = 2.0) -> pd.Series:
    """Hanging man: hammer shape appearing after an advance."""
    b = _bars(open_, high, low, close)
    body = b.body_abs.clip(lower=1e-12)
    return _all(
        _prior_up(b.c),
        b.lower >= shadow_mult * body,
        b.upper <= 0.3 * b.rng,
        body <= 0.6 * b.rng,
    )


def shooting_star(open_, high, low, close, shadow_mult: float = 2.0) -> pd.Series:
    """Shooting star: inverted-hammer shape appearing after an advance."""
    b = _bars(open_, high, low, close)
    body = b.body_abs.clip(lower=1e-12)
    return _all(
        _prior_up(b.c),
        b.upper >= shadow_mult * body,
        b.lower <= 0.3 * b.rng,
        body <= 0.6 * b.rng,
    )


def spinning_top(open_, high, low, close, max_body: float = 0.3) -> pd.Series:
    """Spinning top: small body with shadows on both sides."""
    b = _bars(open_, high, low, close)
    return _all(
        b.body_abs <= max_body * b.rng,
        b.upper > b.body_abs,
        b.lower > b.body_abs,
        b.rng > 0,
    )


def high_wave(open_, high, low, close, body_pct: float = 0.2, range_mult: float = 1.5, period: int = 20) -> pd.Series:
    """High-wave bar: unusually wide range with a small body and long shadows."""
    b = _bars(open_, high, low, close)
    avg = b.rng.rolling(period, min_periods=period).mean()
    return _all(
        b.rng >= range_mult * avg,
        b.body_abs <= body_pct * b.rng,
        b.upper >= 0.3 * b.rng,
        b.lower >= 0.3 * b.rng,
    )


def long_lower_shadow(open_, high, low, close, lower_pct: float = 0.6, upper_pct: float = 0.15) -> pd.Series:
    """Bullish pin bar: dominant lower shadow, negligible upper shadow."""
    b = _bars(open_, high, low, close)
    return _all(b.lower >= lower_pct * b.rng, b.upper <= upper_pct * b.rng)


def long_upper_shadow(open_, high, low, close, upper_pct: float = 0.6, lower_pct: float = 0.15) -> pd.Series:
    """Bearish pin bar: dominant upper shadow, negligible lower shadow."""
    b = _bars(open_, high, low, close)
    return _all(b.upper >= upper_pct * b.rng, b.lower <= lower_pct * b.rng)


def white_marubozu(open_, high, low, close, body_pct: float = 0.9, shadow_pct: float = 0.05) -> pd.Series:
    """White marubozu: full-length bullish body with no shadows."""
    b = _bars(open_, high, low, close)
    return _all(b.bull, b.body_abs >= body_pct * b.rng, b.upper <= shadow_pct * b.rng, b.lower <= shadow_pct * b.rng)


def black_marubozu(open_, high, low, close, body_pct: float = 0.9, shadow_pct: float = 0.05) -> pd.Series:
    """Black marubozu: full-length bearish body with no shadows."""
    b = _bars(open_, high, low, close)
    return _all(b.bear, b.body_abs >= body_pct * b.rng, b.upper <= shadow_pct * b.rng, b.lower <= shadow_pct * b.rng)


def closing_marubozu_white(open_, high, low, close, body_pct: float = 0.85, shadow_pct: float = 0.05) -> pd.Series:
    """Closing marubozu (white): bullish bar closing at the high (no upper shadow)."""
    b = _bars(open_, high, low, close)
    return _all(b.bull, b.upper <= shadow_pct * b.rng, b.body_abs >= body_pct * b.rng)


def closing_marubozu_black(open_, high, low, close, body_pct: float = 0.85, shadow_pct: float = 0.05) -> pd.Series:
    """Closing marubozu (black): bearish bar closing at the low (no lower shadow)."""
    b = _bars(open_, high, low, close)
    return _all(b.bear, b.lower <= shadow_pct * b.rng, b.body_abs >= body_pct * b.rng)


def belt_hold_bull(open_, high, low, close, body_pct: float = 0.7, shadow_pct: float = 0.05) -> pd.Series:
    """Bullish belt hold: opens at the low and runs higher."""
    b = _bars(open_, high, low, close)
    return _all(b.bull, b.lower <= shadow_pct * b.rng, b.body_abs >= body_pct * b.rng)


def belt_hold_bear(open_, high, low, close, body_pct: float = 0.7, shadow_pct: float = 0.05) -> pd.Series:
    """Bearish belt hold: opens at the high and runs lower."""
    b = _bars(open_, high, low, close)
    return _all(b.bear, b.upper <= shadow_pct * b.rng, b.body_abs >= body_pct * b.rng)


def shaven_head(open_, high, low, close, shadow_pct: float = 0.05) -> pd.Series:
    """Shaven head: close equals the bar high (no upper shadow)."""
    b = _bars(open_, high, low, close)
    return _all(b.upper <= shadow_pct * b.rng)


def shaven_bottom(open_, high, low, close, shadow_pct: float = 0.05) -> pd.Series:
    """Shaven bottom: open equals the bar low (no lower shadow)."""
    b = _bars(open_, high, low, close)
    return _all(b.lower <= shadow_pct * b.rng)


def wide_range_bar(open_, high, low, close, period: int = 20, mult: float = 2.0) -> pd.Series:
    """Wide range bar: true range at least ``mult`` times its recent average."""
    b = _bars(open_, high, low, close)
    avg = b.rng.rolling(period, min_periods=period).mean()
    return _all(b.rng >= mult * avg)


def narrow_range_bar(open_, high, low, close, period: int = 20, mult: float = 0.5) -> pd.Series:
    """Narrow range bar: true range at most ``mult`` times its recent average."""
    b = _bars(open_, high, low, close)
    avg = b.rng.rolling(period, min_periods=period).mean()
    return _all(b.rng <= mult * avg)


def up_bar(open_, high, low, close) -> pd.Series:
    """Up bar: close above open."""
    return _bars(open_, high, low, close).bull


def down_bar(open_, high, low, close) -> pd.Series:
    """Down bar: close below open."""
    return _bars(open_, high, low, close).bear


# ---------------------------------------------------------------------------
# Two-bar patterns
# ---------------------------------------------------------------------------

def inside_bar(open_, high, low, close) -> pd.Series:
    """Inside bar: range contained within the previous bar's range."""
    b = _bars(open_, high, low, close)
    return _all(b.h < b.h.shift(1), b.l > b.l.shift(1))


def outside_bar(open_, high, low, close) -> pd.Series:
    """Outside bar: range engulfs the previous bar's range."""
    b = _bars(open_, high, low, close)
    return _all(b.h > b.h.shift(1), b.l < b.l.shift(1))


def bullish_engulfing(open_, high, low, close) -> pd.Series:
    """Bullish engulfing: up body fully engulfs the prior down body after a decline."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o <= c1,
        b.c >= o1,
    )


def bearish_engulfing(open_, high, low, close) -> pd.Series:
    """Bearish engulfing: down body fully engulfs the prior up body after an advance."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        b.o >= c1,
        b.c <= o1,
    )


def bullish_harami(open_, high, low, close) -> pd.Series:
    """Bullish harami: small up body contained in the prior down body after a decline."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    body1 = (o1 - c1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o >= c1,
        b.c <= o1,
        b.body_abs < body1,
    )


def bearish_harami(open_, high, low, close) -> pd.Series:
    """Bearish harami: small down body contained in the prior up body after an advance."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    body1 = (c1 - o1)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        b.o <= c1,
        b.c >= o1,
        b.body_abs < body1,
    )


def bullish_harami_cross(open_, high, low, close) -> pd.Series:
    """Bullish harami cross: bullish harami whose second bar is a doji."""
    b = _bars(open_, high, low, close)
    return _all(bullish_harami(open_, high, low, close), _is_doji(b, 0.1))


def bearish_harami_cross(open_, high, low, close) -> pd.Series:
    """Bearish harami cross: bearish harami whose second bar is a doji."""
    b = _bars(open_, high, low, close)
    return _all(bearish_harami(open_, high, low, close), _is_doji(b, 0.1))


def piercing_line(open_, high, low, close) -> pd.Series:
    """Piercing line: down gap open then a close above the midpoint of the prior body."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    mid = (o1 + c1) / 2.0
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o < c1,
        b.c > mid,
        b.c < o1,
    )


def dark_cloud_cover(open_, high, low, close) -> pd.Series:
    """Dark cloud cover: up gap open then a close below the midpoint of the prior body."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    mid = (o1 + c1) / 2.0
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        b.o > c1,
        b.c < mid,
        b.c > o1,
    )


def bullish_counterattack(open_, high, low, close, tol_pct: float = 0.1) -> pd.Series:
    """Bullish counterattack: gap down then close back at the prior close."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o < c1,
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def bearish_counterattack(open_, high, low, close, tol_pct: float = 0.1) -> pd.Series:
    """Bearish counterattack: gap up then close back at the prior close."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        b.o > c1,
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def matching_low(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Matching low: two consecutive bearish closes at the same level after a decline."""
    b = _bars(open_, high, low, close)
    c1 = b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        b.bear,
        b.c.shift(1) < b.o.shift(1),
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def matching_high(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Matching high: two consecutive bullish closes at the same level after an advance."""
    b = _bars(open_, high, low, close)
    c1 = b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_up(b.c),
        b.bull,
        b.c.shift(1) > b.o.shift(1),
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def kicking_bull(open_, high, low, close) -> pd.Series:
    """Bullish kicking: black marubozu followed by a gapped white marubozu."""
    b = _bars(open_, high, low, close)
    return _all(
        black_marubozu(b.o.shift(1), b.h.shift(1), b.l.shift(1), b.c.shift(1)),
        white_marubozu(b.o, b.h, b.l, b.c),
        b.o > b.h.shift(1),
    )


def kicking_bear(open_, high, low, close) -> pd.Series:
    """Bearish kicking: white marubozu followed by a gapped black marubozu."""
    b = _bars(open_, high, low, close)
    return _all(
        white_marubozu(b.o.shift(1), b.h.shift(1), b.l.shift(1), b.c.shift(1)),
        black_marubozu(b.o, b.h, b.l, b.c),
        b.o < b.l.shift(1),
    )


def meeting_lines_bull(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Bullish meeting lines: long down bar then a long up bar closing at the same price."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        (o1 - c1) >= 0.5 * rng1,
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def meeting_lines_bear(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Bearish meeting lines: long up bar then a long down bar closing at the same price."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        (c1 - o1) >= 0.5 * rng1,
        (b.c - c1).abs() <= tol_pct * rng1,
    )


def separating_lines_bull(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Bullish separating lines: down bar then an up bar opening at the same level."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        (b.o - o1).abs() <= tol_pct * rng1,
    )


def separating_lines_bear(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Bearish separating lines: up bar then a down bar opening at the same level."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.bear,
        (b.o - o1).abs() <= tol_pct * rng1,
    )


def thrusting(open_, high, low, close) -> pd.Series:
    """Thrusting: down bar, gap-down up bar closing into the prior body below its midpoint."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    mid = (o1 + c1) / 2.0
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o < c1,
        b.c > c1,
        b.c < mid,
    )


def in_neck(open_, high, low, close, tol_pct: float = 0.1) -> pd.Series:
    """In-neck: up bar closing just barely above the prior bearish close."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    body1 = (o1 - c1).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o < c1,
        b.c >= c1,
        b.c <= c1 + tol_pct * body1,
    )


def on_neck(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """On-neck: up bar closing at (or marginally below) the prior bearish close."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    body1 = (o1 - c1).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.bull,
        b.o < c1,
        (b.c - c1).abs() <= tol_pct * body1,
    )


def two_crows_gapping(open_, high, low, close) -> pd.Series:
    """Two crows gapping: white bar, gapped-up black bar, then a lower black bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(1), b.c.shift(1)
    o2, c2 = b.o.shift(2), b.c.shift(2)
    return _all(
        _prior_up(b.c),
        c2 > o2,
        o1 > c2,
        c1 < o1,
        b.bear,
        b.c < c1,
    )


# ---------------------------------------------------------------------------
# Three-bar patterns
# ---------------------------------------------------------------------------

def morning_star(open_, high, low, close, long_pct: float = 0.5, small_pct: float = 0.5) -> pd.Series:
    """Morning star: long down bar, small gapped-down star, strong up bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    body1 = (o1 - c1)
    body2 = b.body_abs.shift(1)
    body_top2 = np.maximum(b.o.shift(1), b.c.shift(1))
    mid1 = (o1 + c1) / 2.0
    return _all(
        _prior_down(b.c),
        c1 < o1,
        body1 >= long_pct * (b.h.shift(2) - b.l.shift(2)).replace(0, np.nan),
        body2 <= small_pct * body1,
        body_top2 < c1,
        b.bull,
        b.c > mid1,
    )


def morning_doji_star(open_, high, low, close) -> pd.Series:
    """Morning doji star: morning star whose middle bar is a doji."""
    b = _bars(open_, high, low, close)
    return _all(morning_star(open_, high, low, close), _is_doji(b.shift(1), 0.1))


def evening_star(open_, high, low, close, long_pct: float = 0.5, small_pct: float = 0.5) -> pd.Series:
    """Evening star: long up bar, small gapped-up star, strong down bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    body1 = (c1 - o1)
    body2 = b.body_abs.shift(1)
    body_bottom2 = np.minimum(b.o.shift(1), b.c.shift(1))
    mid1 = (o1 + c1) / 2.0
    return _all(
        _prior_up(b.c),
        c1 > o1,
        body1 >= long_pct * (b.h.shift(2) - b.l.shift(2)).replace(0, np.nan),
        body2 <= small_pct * body1,
        body_bottom2 > c1,
        b.bear,
        b.c < mid1,
    )


def evening_doji_star(open_, high, low, close) -> pd.Series:
    """Evening doji star: evening star whose middle bar is a doji."""
    b = _bars(open_, high, low, close)
    return _all(evening_star(open_, high, low, close), _is_doji(b.shift(1), 0.1))


def three_white_soldiers(open_, high, low, close, shadow_pct: float = 0.3) -> pd.Series:
    """Three white soldiers: three rising bullish bars with small upper shadows."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    upper_ok1 = (b.h.shift(2) - c1) <= shadow_pct * (b.h.shift(2) - b.l.shift(2))
    upper_ok2 = (b.h.shift(1) - c2) <= shadow_pct * (b.h.shift(1) - b.l.shift(1))
    upper_ok3 = (b.h - b.c) <= shadow_pct * b.rng
    return _all(
        _prior_down(b.c),
        c1 > o1,
        c2 > o2,
        b.bull,
        c2 > c1,
        b.c > c2,
        o2 > o1,
        o2 < c1,
        b.o > o2,
        b.o < c2,
        upper_ok1,
        upper_ok2,
        upper_ok3,
    )


def three_black_crows(open_, high, low, close, shadow_pct: float = 0.3) -> pd.Series:
    """Three black crows: three falling bearish bars with small lower shadows."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    lower_ok1 = (c1 - b.l.shift(2)) <= shadow_pct * (b.h.shift(2) - b.l.shift(2))
    lower_ok2 = (c2 - b.l.shift(1)) <= shadow_pct * (b.h.shift(1) - b.l.shift(1))
    lower_ok3 = (b.c - b.l) <= shadow_pct * b.rng
    return _all(
        _prior_up(b.c),
        c1 < o1,
        c2 < o2,
        b.bear,
        c2 < c1,
        b.c < c2,
        o2 < o1,
        o2 > c1,
        b.o < o2,
        b.o > c2,
        lower_ok1,
        lower_ok2,
        lower_ok3,
    )


def three_inside_up(open_, high, low, close) -> pd.Series:
    """Three inside up: bullish harami followed by a confirming up bar."""
    b = _bars(open_, high, low, close)
    o1 = b.o.shift(2)
    return _all(bullish_harami(open_, high, low, close), b.bull, b.c > o1)


def three_inside_down(open_, high, low, close) -> pd.Series:
    """Three inside down: bearish harami followed by a confirming down bar."""
    b = _bars(open_, high, low, close)
    o1 = b.o.shift(2)
    return _all(bearish_harami(open_, high, low, close), b.bear, b.c < o1)


def three_outside_up(open_, high, low, close) -> pd.Series:
    """Three outside up: bullish engulfing followed by a higher close."""
    b = _bars(open_, high, low, close)
    return _all(bullish_engulfing(open_, high, low, close), b.bull, b.c > b.c.shift(1))


def three_outside_down(open_, high, low, close) -> pd.Series:
    """Three outside down: bearish engulfing followed by a lower close."""
    b = _bars(open_, high, low, close)
    return _all(bearish_engulfing(open_, high, low, close), b.bear, b.c < b.c.shift(1))


def three_stars_in_the_south(open_, high, low, close) -> pd.Series:
    """Three stars in the south: three declining dojis, the middle one gapping lower."""
    b = _bars(open_, high, low, close)
    d1 = _is_doji(b.shift(2), 0.1)
    d2 = _is_doji(b.shift(1), 0.1)
    d3 = _is_doji(b, 0.1)
    c1, c2 = b.c.shift(2), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        d1, d2, d3,
        c1 > c2,
        c2 > b.c,
        b.l.shift(1) < b.l.shift(2),
    )


def three_line_strike_bull(open_, high, low, close) -> pd.Series:
    """Bullish three-line strike: three falling bars erased by one long up bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(3), b.c.shift(3)
    c2, c3 = b.c.shift(2), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        c2 < c1,
        c3 < c2,
        b.bull,
        b.o < c3,
        b.c > o1,
    )


def three_line_strike_bear(open_, high, low, close) -> pd.Series:
    """Bearish three-line strike: three rising bars erased by one long down bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(3), b.c.shift(3)
    c2, c3 = b.c.shift(2), b.c.shift(1)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        c2 > c1,
        c3 > c2,
        b.bear,
        b.o > c3,
        b.c < o1,
    )


def abandoned_baby_bull(open_, high, low, close) -> pd.Series:
    """Bullish abandoned baby: down bar, gapped-down doji, gapped-up up bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    mid1 = (o1 + c1) / 2.0
    return _all(
        _prior_down(b.c),
        c1 < o1,
        _is_doji(b.shift(1), 0.1),
        b.h.shift(1) < b.l.shift(2),
        b.l > b.h.shift(1),
        b.bull,
        b.c > mid1,
    )


def abandoned_baby_bear(open_, high, low, close) -> pd.Series:
    """Bearish abandoned baby: up bar, gapped-up doji, gapped-down down bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    mid1 = (o1 + c1) / 2.0
    return _all(
        _prior_up(b.c),
        c1 > o1,
        _is_doji(b.shift(1), 0.1),
        b.l.shift(1) > b.h.shift(2),
        b.h < b.l.shift(1),
        b.bear,
        b.c < mid1,
    )


def advance_block(open_, high, low, close) -> pd.Series:
    """Advance block: three rising bars whose bodies shrink (weakening advance)."""
    b = _bars(open_, high, low, close)
    body1, body2, body3 = b.body_abs.shift(2), b.body_abs.shift(1), b.body_abs
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 > o1,
        c2 > o2,
        b.bull,
        c2 > c1,
        b.c > c2,
        o2 > o1,
        o2 < c1,
        b.o > o2,
        b.o < c2,
        body2 < body1,
        body3 < body2,
    )


def deliberation(open_, high, low, close) -> pd.Series:
    """Deliberation: two long up bars followed by a tiny-bodied star."""
    b = _bars(open_, high, low, close)
    body2 = b.body_abs.shift(1)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 > o1,
        c2 > o2,
        c2 > c1,
        body2 >= 0.5 * b.body_abs.shift(2),
        b.body_abs <= 0.3 * body2,
        b.o > c2,
    )


def identical_three_crows(open_, high, low, close, tol_pct: float = 0.05) -> pd.Series:
    """Identical three crows: three down bars each opening at the prior close."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    rng1 = (b.h.shift(2) - b.l.shift(2)).replace(0, np.nan)
    rng2 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        _prior_up(b.c),
        c1 < o1,
        c2 < o2,
        b.bear,
        c2 < c1,
        b.c < c2,
        (o2 - c1).abs() <= tol_pct * rng1,
        (b.o - c2).abs() <= tol_pct * rng2,
    )


def tasuki_gap_up(open_, high, low, close) -> pd.Series:
    """Upside tasuki gap: gap up held partially unfilled by the third bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        c2 > o2,
        b.l.shift(1) > b.h.shift(2),
        b.bear,
        b.o < o2,
        b.c > c1,
        b.c < o2,
    )


def tasuki_gap_down(open_, high, low, close) -> pd.Series:
    """Downside tasuki gap: gap down held partially unfilled by the third bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        c2 < o2,
        b.h.shift(1) < b.l.shift(2),
        b.bull,
        b.o > o2,
        b.c < c1,
        b.c > o2,
    )


def upside_gap_three_methods(open_, high, low, close) -> pd.Series:
    """Upside gap three methods: two advancing gapped bars, third pulls back into the gap."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        c1 > o1,
        c2 > o2,
        b.l.shift(1) > b.h.shift(2),
        b.bear,
        b.c < o2,
        b.c > c1,
    )


def downside_gap_three_methods(open_, high, low, close) -> pd.Series:
    """Downside gap three methods: two falling gapped bars, third pulls back into the gap."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        c1 < o1,
        c2 < o2,
        b.h.shift(1) < b.l.shift(2),
        b.bull,
        b.c > o2,
        b.c < c1,
    )


def unique_three_river(open_, high, low, close) -> pd.Series:
    """Unique three river: down bar, lower-low small down bar, recovering up bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2, c2 = b.o.shift(1), b.c.shift(1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        c2 < o2,
        b.body_abs.shift(1) < 0.5 * b.body_abs.shift(2),
        b.l.shift(1) < b.l.shift(2),
        b.bull,
        b.c > c2,
    )


def stick_sandwich(open_, high, low, close) -> pd.Series:
    """Stick sandwich: two identical bearish closes around an up bar."""
    b = _bars(open_, high, low, close)
    c1 = b.c.shift(2)
    rng1 = (b.h.shift(2) - b.l.shift(2)).replace(0, np.nan)
    return _all(
        _prior_down(b.c),
        b.c.shift(2) < b.o.shift(2),
        b.c.shift(1) > b.o.shift(1),
        b.bear,
        (b.c - c1).abs() <= 0.05 * rng1,
    )


def tri_star_bull(open_, high, low, close) -> pd.Series:
    """Bullish tri-star: three dojis with the middle star gapping lower."""
    b = _bars(open_, high, low, close)
    return _all(
        _is_doji(b.shift(2), 0.1),
        _is_doji(b.shift(1), 0.1),
        _is_doji(b, 0.1),
        b.h.shift(1) < b.l.shift(2),
        b.l > b.l.shift(1),
    )


def tri_star_bear(open_, high, low, close) -> pd.Series:
    """Bearish tri-star: three dojis with the middle star gapping higher."""
    b = _bars(open_, high, low, close)
    return _all(
        _is_doji(b.shift(2), 0.1),
        _is_doji(b.shift(1), 0.1),
        _is_doji(b, 0.1),
        b.l.shift(1) > b.h.shift(2),
        b.h < b.h.shift(1),
    )


# ---------------------------------------------------------------------------
# Multi-bar patterns
# ---------------------------------------------------------------------------

def rising_three_methods(open_, high, low, close) -> pd.Series:
    """Rising three methods: long up bar, three contained pullback bars, breakout bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(4), b.c.shift(4)
    h1, l1 = b.h.shift(4), b.l.shift(4)
    contained = [
        _all(b.h.shift(i) < h1, b.l.shift(i) > l1, b.c.shift(i) < c1)
        for i in (3, 2, 1)
    ]
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.body_abs.shift(4) >= 0.5 * (h1 - l1).replace(0, np.nan),
        *contained,
        b.bull,
        b.c > h1,
    )


def falling_three_methods(open_, high, low, close) -> pd.Series:
    """Falling three methods: long down bar, three contained bounce bars, breakdown bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(4), b.c.shift(4)
    h1, l1 = b.h.shift(4), b.l.shift(4)
    contained = [
        _all(b.h.shift(i) < h1, b.l.shift(i) > l1, b.c.shift(i) > c1)
        for i in (3, 2, 1)
    ]
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.body_abs.shift(4) >= 0.5 * (h1 - l1).replace(0, np.nan),
        *contained,
        b.bear,
        b.c < l1,
    )


def mat_hold_bull(open_, high, low, close) -> pd.Series:
    """Bullish mat hold: rising three methods with a gap up on the second bar."""
    b = _bars(open_, high, low, close)
    o1, c1, h1, l1 = b.o.shift(4), b.c.shift(4), b.h.shift(4), b.l.shift(4)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        b.l.shift(3) > h1,
        b.l.shift(2) > l1,
        b.l.shift(1) > l1,
        b.bull,
        b.c > b.h.shift(3),
    )


def mat_hold_bear(open_, high, low, close) -> pd.Series:
    """Bearish mat hold: falling three methods with a gap down on the second bar."""
    b = _bars(open_, high, low, close)
    o1, c1, h1, l1 = b.o.shift(4), b.c.shift(4), b.h.shift(4), b.l.shift(4)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        b.h.shift(3) < l1,
        b.h.shift(2) < h1,
        b.h.shift(1) < h1,
        b.bear,
        b.c < b.l.shift(3),
    )


def breakaway_bull(open_, high, low, close) -> pd.Series:
    """Bullish breakaway: gap-down exhaustion relieved by a strong reversing bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(4), b.c.shift(4)
    o2, c2 = b.o.shift(3), b.c.shift(3)
    h2 = b.h.shift(3)
    o3 = b.o.shift(2)
    o4 = b.o.shift(1)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        c2 < o2,
        h2 < b.l.shift(4),
        b.body_abs.shift(2) <= 0.5 * b.body_abs.shift(3),
        b.body_abs.shift(1) <= 0.5 * b.body_abs.shift(3),
        b.h.shift(2) <= h2,
        b.h.shift(1) <= h2,
        _all(o3 < c2, o4 < c2),
        b.bull,
        b.c > o2,
    )


def breakaway_bear(open_, high, low, close) -> pd.Series:
    """Bearish breakaway: gap-up exhaustion relieved by a strong reversing bar."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(4), b.c.shift(4)
    o2, c2 = b.o.shift(3), b.c.shift(3)
    l2 = b.l.shift(3)
    o3 = b.o.shift(2)
    o4 = b.o.shift(1)
    return _all(
        _prior_up(b.c),
        c1 > o1,
        c2 > o2,
        l2 > b.h.shift(4),
        b.body_abs.shift(2) <= 0.5 * b.body_abs.shift(3),
        b.body_abs.shift(1) <= 0.5 * b.body_abs.shift(3),
        b.l.shift(2) >= l2,
        b.l.shift(1) >= l2,
        _all(o3 > c2, o4 > c2),
        b.bear,
        b.c < o2,
    )


def three_drives_bull(open_, high, low, close) -> pd.Series:
    """Bullish three drives (heuristic): three stair-stepped declines reversing up.

    Recognized over a 7-bar window: close[t-6] > close[t-4] > close[t-2] with
    intervening bounces, and today's bar closes back above the last drive high.
    """
    b = _bars(open_, high, low, close)
    c0, c2, c4, c6 = b.c.shift(6), b.c.shift(4), b.c.shift(2), b.c
    return _all(
        _prior_down(b.c),
        c0 > c2,
        c2 > c4,
        b.c.shift(5) > c2,
        b.c.shift(3) > c4,
        b.h.shift(1) > b.h.shift(3),
        c6 > b.h.shift(1),
    )


def three_drives_bear(open_, high, low, close) -> pd.Series:
    """Bearish three drives (heuristic): three stair-stepped advances reversing down."""
    b = _bars(open_, high, low, close)
    c0, c2, c4, c6 = b.c.shift(6), b.c.shift(4), b.c.shift(2), b.c
    return _all(
        _prior_up(b.c),
        c0 < c2,
        c2 < c4,
        b.c.shift(5) < c2,
        b.c.shift(3) < c4,
        b.l.shift(1) < b.l.shift(3),
        c6 < b.l.shift(1),
    )


def side_by_side_white_gapup(open_, high, low, close) -> pd.Series:
    """Upside gap side-by-side white lines: gapped-up pair of near-identical up bars."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2 = b.o.shift(1)
    rng2 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        c1 > o1,
        b.l.shift(1) > b.h.shift(2),
        b.c.shift(1) > o2,
        b.bull,
        (b.o - o2).abs() <= 0.2 * rng2,
        b.c >= b.c.shift(1),
    )


def side_by_side_white_gapdown(open_, high, low, close) -> pd.Series:
    """Downside gap side-by-side white lines: gapped-down pair of near-identical up bars."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(2), b.c.shift(2)
    o2 = b.o.shift(1)
    rng2 = (b.h.shift(1) - b.l.shift(1)).replace(0, np.nan)
    return _all(
        c1 < o1,
        b.h.shift(1) < b.l.shift(2),
        b.c.shift(1) > o2,
        b.bull,
        (b.o - o2).abs() <= 0.2 * rng2,
        b.c >= b.c.shift(1),
    )


def concealing_baby_swallow(open_, high, low, close) -> pd.Series:
    """Concealing baby swallow: two down bars, an engulfing down bar, then a hammer.

    The final hammer's lower shadow "conceals" the prior low, signaling that
    sellers are exhausted.
    """
    b = _bars(open_, high, low, close)
    o1, c1, l1 = b.o.shift(3), b.c.shift(3), b.l.shift(3)
    o2, c2 = b.o.shift(2), b.c.shift(2)
    o3, c3 = b.o.shift(1), b.c.shift(1)
    body4 = b.body_abs.clip(lower=1e-12)
    return _all(
        _prior_down(b.c),
        c1 < o1,
        (b.h.shift(3) - c1) <= 0.1 * (b.h.shift(3) - l1),
        o2 < l1,
        c2 < o2,
        o3 > o2,
        c3 < c2,
        b.lower >= 2.0 * body4,
        b.upper <= 0.3 * b.rng,
        b.l <= b.l.shift(1),
        b.c >= o3,
    )


def ladder_bottom(open_, high, low, close) -> pd.Series:
    """Ladder bottom: three stepped-down bars with small upper shadows, then a gap up."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(3), b.c.shift(3)
    c2, c3 = b.c.shift(2), b.c.shift(1)
    small_upper = [_all((b.h.shift(i) - b.c.shift(i)) <= 0.2 * (b.h.shift(i) - b.l.shift(i))) for i in (3, 2, 1)]
    return _all(
        _prior_down(b.c),
        c1 < o1,
        c2 < c1,
        c3 < c2,
        *small_upper,
        b.bull,
        b.o > c3,
        b.c > c1,
    )


def ladder_top(open_, high, low, close) -> pd.Series:
    """Ladder top: three stepped-up bars with small lower shadows, then a gap down."""
    b = _bars(open_, high, low, close)
    o1, c1 = b.o.shift(3), b.c.shift(3)
    c2, c3 = b.c.shift(2), b.c.shift(1)
    small_lower = [_all((b.c.shift(i) - b.l.shift(i)) <= 0.2 * (b.h.shift(i) - b.l.shift(i))) for i in (3, 2, 1)]
    return _all(
        _prior_up(b.c),
        c1 > o1,
        c2 > c1,
        c3 > c2,
        *small_lower,
        b.bear,
        b.o < c3,
        b.c < c1,
    )


# ---------------------------------------------------------------------------
# Aggregation: registries, batch detection, scoring
# ---------------------------------------------------------------------------

BULLISH_PATTERNS: dict[str, object] = {
    "dragonfly_doji": dragonfly_doji,
    "takuri": takuri,
    "hammer": hammer,
    "inverted_hammer": inverted_hammer,
    "long_lower_shadow": long_lower_shadow,
    "belt_hold_bull": belt_hold_bull,
    "closing_marubozu_white": closing_marubozu_white,
    "white_marubozu": white_marubozu,
    "shaven_bottom": shaven_bottom,
    "bullish_engulfing": bullish_engulfing,
    "bullish_harami": bullish_harami,
    "bullish_harami_cross": bullish_harami_cross,
    "piercing_line": piercing_line,
    "bullish_counterattack": bullish_counterattack,
    "matching_low": matching_low,
    "kicking_bull": kicking_bull,
    "meeting_lines_bull": meeting_lines_bull,
    "separating_lines_bull": separating_lines_bull,
    "in_neck": in_neck,
    "on_neck": on_neck,
    "morning_star": morning_star,
    "morning_doji_star": morning_doji_star,
    "three_white_soldiers": three_white_soldiers,
    "three_inside_up": three_inside_up,
    "three_outside_up": three_outside_up,
    "three_stars_in_the_south": three_stars_in_the_south,
    "three_line_strike_bull": three_line_strike_bull,
    "abandoned_baby_bull": abandoned_baby_bull,
    "tasuki_gap_up": tasuki_gap_up,
    "upside_gap_three_methods": upside_gap_three_methods,
    "unique_three_river": unique_three_river,
    "stick_sandwich": stick_sandwich,
    "tri_star_bull": tri_star_bull,
    "rising_three_methods": rising_three_methods,
    "mat_hold_bull": mat_hold_bull,
    "breakaway_bull": breakaway_bull,
    "three_drives_bull": three_drives_bull,
    "side_by_side_white_gapup": side_by_side_white_gapup,
    "concealing_baby_swallow": concealing_baby_swallow,
    "ladder_bottom": ladder_bottom,
}

BEARISH_PATTERNS: dict[str, object] = {
    "gravestone_doji": gravestone_doji,
    "shooting_star": shooting_star,
    "hanging_man": hanging_man,
    "long_upper_shadow": long_upper_shadow,
    "belt_hold_bear": belt_hold_bear,
    "closing_marubozu_black": closing_marubozu_black,
    "black_marubozu": black_marubozu,
    "shaven_head": shaven_head,
    "bearish_engulfing": bearish_engulfing,
    "bearish_harami": bearish_harami,
    "bearish_harami_cross": bearish_harami_cross,
    "dark_cloud_cover": dark_cloud_cover,
    "bearish_counterattack": bearish_counterattack,
    "matching_high": matching_high,
    "kicking_bear": kicking_bear,
    "meeting_lines_bear": meeting_lines_bear,
    "separating_lines_bear": separating_lines_bear,
    "thrusting": thrusting,
    "two_crows_gapping": two_crows_gapping,
    "evening_star": evening_star,
    "evening_doji_star": evening_doji_star,
    "three_black_crows": three_black_crows,
    "three_inside_down": three_inside_down,
    "three_outside_down": three_outside_down,
    "three_line_strike_bear": three_line_strike_bear,
    "abandoned_baby_bear": abandoned_baby_bear,
    "advance_block": advance_block,
    "deliberation": deliberation,
    "identical_three_crows": identical_three_crows,
    "tasuki_gap_down": tasuki_gap_down,
    "downside_gap_three_methods": downside_gap_three_methods,
    "tri_star_bear": tri_star_bear,
    "falling_three_methods": falling_three_methods,
    "mat_hold_bear": mat_hold_bear,
    "breakaway_bear": breakaway_bear,
    "three_drives_bear": three_drives_bear,
    "side_by_side_white_gapdown": side_by_side_white_gapdown,
    "ladder_top": ladder_top,
}

NEUTRAL_PATTERNS: dict[str, object] = {
    "doji": doji,
    "long_legged_doji": long_legged_doji,
    "four_price_doji": four_price_doji,
    "rickshaw_man": rickshaw_man,
    "spinning_top": spinning_top,
    "high_wave": high_wave,
    "wide_range_bar": wide_range_bar,
    "narrow_range_bar": narrow_range_bar,
    "up_bar": up_bar,
    "down_bar": down_bar,
    "inside_bar": inside_bar,
    "outside_bar": outside_bar,
}

ALL_PATTERNS: dict[str, object] = {
    **BULLISH_PATTERNS,
    **BEARISH_PATTERNS,
    **NEUTRAL_PATTERNS,
}


def detect_all(
    open_, high, low, close, patterns: dict[str, object] | None = None
) -> pd.DataFrame:
    """Run every registered pattern detector; returns a boolean DataFrame."""
    registry = patterns if patterns is not None else ALL_PATTERNS
    data = {
        name: _as_series(fn(open_, high, low, close)).fillna(False).astype(bool)
        for name, fn in registry.items()
    }
    return pd.DataFrame(data)


def pattern_counts(detections: pd.DataFrame) -> pd.Series:
    """Count how often each pattern fired in a ``detect_all`` frame."""
    return detections.sum().sort_values(ascending=False)


def pattern_score(
    open_, high, low, close, window: int = 5, bull_weight: float = 1.0, bear_weight: float = 1.0
) -> pd.Series:
    """Aggregate candle signal in [-1, 1]: weighted bull/bear hits, smoothed.

    Each bullish pattern adds ``bull_weight`` and each bearish pattern subtracts
    ``bear_weight`` on the bars where it fires; the net score is averaged over
    ``window`` bars and clipped to [-1, 1].
    """
    bull = pd.DataFrame(
        {name: _as_series(fn(open_, high, low, close)).fillna(False) for name, fn in BULLISH_PATTERNS.items()}
    ).sum(axis=1)
    bear = pd.DataFrame(
        {name: _as_series(fn(open_, high, low, close)).fillna(False) for name, fn in BEARISH_PATTERNS.items()}
    ).sum(axis=1)
    net = bull_weight * bull - bear_weight * bear
    return (net.rolling(window, min_periods=1).mean()).clip(-1.0, 1.0)
