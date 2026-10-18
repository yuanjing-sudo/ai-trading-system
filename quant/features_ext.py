"""Feature engineering library for alpha research and ML pipelines.

Two layers:

1. **Feature builders** - composable functions that consume an OHLCV DataFrame
   (columns ``open, high, low, close, volume``, case-insensitive) and return
   DataFrames of engineered features across parameter grids: returns, ranges,
   volatility, momentum, mean reversion, volume/liquidity, trend, oscillators,
   candlestick aggregates, cycles, statistics, and microstructure proxies.
2. **Targets, transforms, and a leakage-safe pipeline** - forward-return,
   triple-barrier, and quantile labels; scalers; feature selection by IC or
   variance; and :class:`FeaturePipeline`, which fits transforms on the training
   slice only.

All features are computed with pandas/NumPy and never look ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import (
    _as_series,
    adx,
    bollinger_bands,
    cmf,
    ema,
    mfi,
    obv,
    rsi,
    sma,
    stochastic,
)
from quant.indicators_ext import (
    ad_line,
    atr_percent,
    chaikin_volatility,
    distance_from_high,
    distance_from_low,
    efficiency_ratio,
    ma_distance,
    momentum_score,
    obv_ema,
    relative_volume,
    relative_strength,
    trend_quality,
    trend_score,
    velocity,
    volume_price_corr,
    volume_zscore,
    wad,
    yearly_position,
)
from quant.candlesticks import BEARISH_PATTERNS, BULLISH_PATTERNS, detect_all, pattern_score
from quant.ehlers import sine_wave
from quant.statistics_ext import (
    half_life,
    kendall_tau,
    mutual_information,
    PCA,
    rolling_hurst,
    rolling_kurt,
    rolling_skew,
    spearman,
    variance_ratio,
)

__all__ = [
    # calendar
    "calendar_features", "encode_cyclical", "days_to_month_end",
    "turn_of_month", "intramonth_position",
    # returns / price
    "return_features", "overnight_feature", "intraday_feature",
    "range_features", "price_level_features", "price_zscore_features",
    "rolling_rank_features",
    # volatility
    "volatility_features", "vol_of_vol_features", "vol_percentile_features",
    "vol_regime_features", "semivariance_features",
    # momentum
    "momentum_features", "momentum_quality_features", "high_52w_features",
    "reversal_features", "risk_adjusted_momentum",
    # mean reversion
    "mean_reversion_features", "bollinger_features", "vwap_features",
    # volume / liquidity
    "volume_features", "volume_flow_features", "liquidity_features",
    # trend / oscillators
    "trend_features", "oscillator_features",
    # patterns / cycles
    "candlestick_features", "cycle_features",
    # statistics / microstructure
    "statistical_features", "entropy_features", "microstructure_features",
    "kyle_lambda", "corwin_schultz_spread",
    # cross-asset
    "cross_asset_features", "beta_features", "lead_lag_features",
    # generic builders
    "lag_features", "diff_features", "rolling_stat_features",
    "interaction_features", "ewma_features",
    # targets / labels
    "target_forward_returns", "target_binary_direction",
    "target_quantile_labels", "target_triple_barrier", "target_trend_labels",
    "target_volatility_labels", "meta_labels", "sample_weights_by_vol",
    # transforms / selection
    "winsorize_df", "scale_zscore_df", "scale_minmax_df", "rank_transform_df",
    "impute_median_df", "drop_low_variance", "drop_correlated",
    "select_by_ic", "select_by_mutual_info", "pca_features",
    # pipeline / summary
    "FeaturePipeline", "build_feature_matrix", "feature_summary",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, *names: str) -> pd.Series:
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in lookup:
            return df[lookup[name]]
    raise KeyError(f"Missing column among {names}; have {list(df.columns)}")


def _parts(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (open, high, low, close, volume) with graceful volume fallback."""
    o = _as_series(_col(df, "open"))
    h = _as_series(_col(df, "high", "adj_high"))
    l = _as_series(_col(df, "low", "adj_low"))
    c = _as_series(_col(df, "close", "adj_close"))
    try:
        v = _as_series(_col(df, "volume"))
    except KeyError:
        v = pd.Series(np.nan, index=c.index)
    return o, h, l, c, v


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def calendar_features(index: pd.Index) -> pd.DataFrame:
    """Calendar/seasonality features from a DatetimeIndex."""
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)
    out["day_of_week"] = idx.dayofweek
    out["day_of_month"] = idx.day
    out["day_of_year"] = idx.dayofyear
    out["week_of_year"] = idx.isocalendar().week.to_numpy()
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    out["is_quarter_end"] = idx.is_quarter_end.astype(int)
    out["is_year_end"] = idx.is_year_end.astype(int)
    return out


def encode_cyclical(series: pd.Series, period: float) -> pd.DataFrame:
    """Sine/cosine encoding of a cyclical integer feature."""
    s = _as_series(series).astype("float64")
    angle = 2.0 * np.pi * s / float(period)
    return pd.DataFrame({"sin": np.sin(angle), "cos": np.cos(angle)}, index=s.index)


def days_to_month_end(index: pd.Index) -> pd.Series:
    """Trading-day distance to the end of the calendar month."""
    idx = pd.DatetimeIndex(index)
    month_end = idx.to_period("M").to_timestamp(how="end").normalize()
    days = np.busday_count(idx.values.astype("datetime64[D]"), month_end.values.astype("datetime64[D]"))
    return pd.Series(np.minimum(days, 23), index=idx)


def turn_of_month(index: pd.Index, window: int = 3) -> pd.Series:
    """Boolean flag: within ``window`` trading days of month start or end."""
    dte = days_to_month_end(index)
    dom = pd.Series(pd.DatetimeIndex(index).day, index=index)
    return ((dte <= window) | (dom <= window)).fillna(False)


def intramonth_position(index: pd.Index) -> pd.Series:
    """Position within the month in [0, 1] (0 = month start, 1 = month end)."""
    dte = days_to_month_end(index)
    dom = pd.Series(pd.DatetimeIndex(index).day, index=index)
    total = dte + dom - 1
    return (dom - 1) / total.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Return / price features
# ---------------------------------------------------------------------------

def return_features(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 5, 10, 21, 63, 126, 252), log: bool = False) -> pd.DataFrame:
    """Simple (or log) returns across a grid of horizons."""
    _, _, _, c, _ = _parts(df)
    out = {}
    for h in horizons:
        if log:
            out[f"logret_{h}"] = np.log(c / c.shift(h))
        else:
            out[f"ret_{h}"] = c.pct_change(h, fill_method=None).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(out, index=c.index)


def overnight_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight gap and its rolling sums."""
    o, _, _, c, _ = _parts(df)
    gap = o / c.shift(1) - 1.0
    return pd.DataFrame(
        {
            "overnight_gap": gap,
            "overnight_gap_5": gap.rolling(5, min_periods=1).sum(),
            "overnight_gap_21": gap.rolling(21, min_periods=1).sum(),
        },
        index=c.index,
    )


def intraday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Intraday open-to-close returns and their rolling sums."""
    o, _, _, c, _ = _parts(df)
    intr = c / o - 1.0
    return pd.DataFrame(
        {
            "intraday_ret": intr,
            "intraday_ret_5": intr.rolling(5, min_periods=1).sum(),
            "intraday_ret_21": intr.rolling(21, min_periods=1).sum(),
        },
        index=c.index,
    )


def range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bar geometry: shadows, body, close location, and gaps."""
    o, h, l, c, _ = _parts(df)
    rng = (h - l).replace(0, np.nan)
    body = (c - o)
    out = pd.DataFrame(
        {
            "range_pct": rng / c,
            "body_pct": body / rng,
            "upper_shadow_pct": (h - np.maximum(o, c)) / rng,
            "lower_shadow_pct": (np.minimum(o, c) - l) / rng,
            "close_location": ((c - l) - (h - c)) / rng,
            "range_zscore": (rng - rng.rolling(20, min_periods=20).mean()) / rng.rolling(20, min_periods=20).std().replace(0, np.nan),
        },
        index=c.index,
    )
    return out


def price_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Typical/median/weighted price levels normalized by close."""
    o, h, l, c, _ = _parts(df)
    return pd.DataFrame(
        {
            "hlc3_rel": ((h + l + c) / 3.0) / c - 1.0,
            "ohlc4_rel": ((o + h + l + c) / 4.0) / c - 1.0,
            "hl2_rel": ((h + l) / 2.0) / c - 1.0,
            "weighted_close_rel": ((h + l + 2.0 * c) / 4.0) / c - 1.0,
        },
        index=c.index,
    )


def price_zscore_features(df: pd.DataFrame, windows: tuple[int, ...] = (20, 60, 252)) -> pd.DataFrame:
    """Z-score of price against its rolling mean for several windows."""
    _, _, _, c, _ = _parts(df)
    out = {}
    for w in windows:
        mean = c.rolling(w, min_periods=w).mean()
        std = c.rolling(w, min_periods=w).std()
        out[f"price_z_{w}"] = (c - mean) / std.replace(0, np.nan)
    return pd.DataFrame(out, index=c.index)


def rolling_rank_features(df: pd.DataFrame, windows: tuple[int, ...] = (21, 63, 252)) -> pd.DataFrame:
    """Percentile rank of price within its trailing window."""
    _, _, _, c, _ = _parts(df)
    out = {}
    for w in windows:
        out[f"price_rank_{w}"] = c.rolling(w, min_periods=w).rank(pct=True)
    return pd.DataFrame(out, index=c.index)


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------

def volatility_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 21, 63, 252), annualize: int = 252) -> pd.DataFrame:
    """Realized volatility grid plus ATR percent and range-based estimators."""
    _, h, l, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    out = {}
    for w in windows:
        out[f"vol_{w}"] = r.rolling(w, min_periods=w).std() * np.sqrt(annualize)
    out["atr_pct"] = atr_percent(h, l, c, 14)
    out["vol_ratio"] = out.get("vol_21", np.nan) / (out.get("vol_63", np.nan) + 1e-12) if "vol_21" in out and "vol_63" in out else np.nan
    return pd.DataFrame(out, index=c.index)


def vol_of_vol_features(df: pd.DataFrame, vol_window: int = 21, vov_windows: tuple[int, ...] = (21, 63)) -> pd.DataFrame:
    """Volatility-of-volatility and volatility momentum features."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    rv = r.rolling(vol_window, min_periods=vol_window).std()
    out = {}
    for w in vov_windows:
        out[f"vov_{w}"] = rv.rolling(w, min_periods=w).std()
    out["vol_momentum"] = rv / rv.shift(vol_window) - 1.0
    out["chaikin_vol"] = chaikin_volatility(_parts(df)[1], _parts(df)[2], 10, 10)
    return pd.DataFrame(out, index=c.index)


def vol_percentile_features(df: pd.DataFrame, vol_window: int = 21, lookback: int = 252) -> pd.DataFrame:
    """Percentile rank of current volatility within its trailing history."""
    _, _, _, c, _ = _parts(df)
    rv = c.pct_change(fill_method=None).rolling(vol_window, min_periods=vol_window).std()
    pct = rv.rolling(lookback, min_periods=max(lookback // 4, 20)).rank(pct=True)
    z = (rv - rv.rolling(lookback, min_periods=max(lookback // 4, 20)).mean()) / rv.rolling(lookback, min_periods=max(lookback // 4, 20)).std()
    return pd.DataFrame({"vol_percentile": pct, "vol_zscore": z}, index=c.index)


def vol_regime_features(df: pd.DataFrame, window: int = 21, threshold: float = 0.75) -> pd.DataFrame:
    """Coarse volatility regime flags (high when above the threshold percentile)."""
    feats = vol_percentile_features(df, vol_window=window)
    high = (feats["vol_percentile"] >= threshold).fillna(False)
    expanding = feats["vol_percentile"].diff() > 0.1
    return pd.DataFrame(
        {"vol_regime_high": high.astype(int), "vol_expanding": expanding.astype(int)},
        index=feats.index,
    )


def semivariance_features(df: pd.DataFrame, window: int = 21, annualize: int = 252) -> pd.DataFrame:
    """Downside and upside semivariance plus their ratio."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    down = np.sqrt((r.clip(upper=0.0) ** 2).rolling(window, min_periods=window).mean()) * np.sqrt(annualize)
    up = np.sqrt((r.clip(lower=0.0) ** 2).rolling(window, min_periods=window).mean()) * np.sqrt(annualize)
    return pd.DataFrame(
        {"semivol_down": down, "semivol_up": up, "semivol_ratio": down / up.replace(0, np.nan)},
        index=c.index,
    )


# ---------------------------------------------------------------------------
# Momentum features
# ---------------------------------------------------------------------------

def momentum_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 21, 63, 126, 252)) -> pd.DataFrame:
    """Return grid plus composite momentum score and efficiency ratio."""
    _, _, _, c, _ = _parts(df)
    out = return_features(df, windows)
    out["momentum_score"] = momentum_score(c)
    out["efficiency_ratio"] = efficiency_ratio(c, 10)
    out["velocity"] = velocity(c, 10)
    return out


def momentum_quality_features(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Momentum quality: hit rate, sign consistency, and reward/risk of trend."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    pos = (r > 0).rolling(window, min_periods=window).mean()
    consistency = r.rolling(window, min_periods=window).apply(
        lambda w: float(np.mean(np.sign(w) == np.sign(w.mean()))) if len(w) else np.nan, raw=True
    )
    mean_r = r.rolling(window, min_periods=window).mean()
    std_r = r.rolling(window, min_periods=window).std()
    return pd.DataFrame(
        {
            "mom_hit_rate": pos,
            "mom_consistency": consistency,
            "mom_reward_risk": mean_r / std_r.replace(0, np.nan),
        },
        index=c.index,
    )


def high_52w_features(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Distance to the 52-week high/low and the yearly position."""
    _, h, l, c, _ = _parts(df)
    return pd.DataFrame(
        {
            "dist_52w_high": distance_from_high(c, lookback),
            "dist_52w_low": distance_from_low(c, lookback),
            "yearly_position": yearly_position(c, lookback),
            "new_52w_high": (c >= c.rolling(lookback, min_periods=20).max()).astype(int),
            "new_52w_low": (c <= c.rolling(lookback, min_periods=20).min()).astype(int),
        },
        index=c.index,
    )


def reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal signals (negative of recent returns)."""
    _, _, _, c, _ = _parts(df)
    r1 = c.pct_change(1, fill_method=None)
    r5 = c.pct_change(5, fill_method=None)
    r21 = c.pct_change(21, fill_method=None)
    out = pd.DataFrame(
        {
            "reversal_1d": -r1,
            "reversal_5d": -r5,
            "reversal_21d": -r21,
            "overnight_reversal": -(_col(df, "open") / c.shift(1) - 1.0),
        },
        index=c.index,
    )
    return out


def risk_adjusted_momentum(df: pd.DataFrame, windows: tuple[int, ...] = (21, 63, 126)) -> pd.DataFrame:
    """Momentum divided by realized volatility for several windows."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    out = {}
    for w in windows:
        vol = r.rolling(w, min_periods=w).std()
        out[f"mom_sharpe_{w}"] = c.pct_change(w, fill_method=None) / (vol * np.sqrt(w))
    return pd.DataFrame(out, index=c.index)


# ---------------------------------------------------------------------------
# Mean reversion features
# ---------------------------------------------------------------------------

def mean_reversion_features(df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Half-life, variance ratio, Hurst exponent, and efficiency of mean reversion."""
    _, _, _, c, _ = _parts(df)
    log_c = np.log(c)
    dlog = log_c.diff()
    out = pd.DataFrame(index=c.index)
    out["half_life"] = log_c.rolling(window, min_periods=window).apply(
        lambda w: half_life(w) if len(w) else np.nan, raw=True
    )
    out["variance_ratio"] = dlog.rolling(window, min_periods=window).apply(
        lambda w: variance_ratio(w, q=5) if len(w) > 15 else np.nan, raw=True
    )
    out["hurst"] = rolling_hurst(log_c, period=window, min_window=8)
    out["mr_zscore"] = (c - sma(c, window)) / c.pct_change(fill_method=None).rolling(window, min_periods=window).std().replace(0, np.nan)
    return out


def bollinger_features(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger %B, width, and squeeze flags."""
    _, _, _, c, _ = _parts(df)
    bb = bollinger_bands(c, period, num_std)
    width = bb["width"]
    squeeze = (width <= width.rolling(126, min_periods=60).quantile(0.2)).astype(int)
    return pd.DataFrame(
        {
            "bb_pct_b": bb["pct_b"],
            "bb_width": width,
            "bb_squeeze": squeeze,
            "bb_width_z": (width - width.rolling(126, min_periods=60).mean()) / width.rolling(126, min_periods=60).std(),
        },
        index=c.index,
    )


def vwap_features(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling-VWAP deviation and z-score features."""
    o, h, l, c, v = _parts(df)
    typical = (h + l + c) / 3.0
    if v.isna().all():
        v = pd.Series(1.0, index=c.index)
    cum_tpv = (typical * v).rolling(window, min_periods=window).sum()
    cum_v = v.rolling(window, min_periods=window).sum()
    vwap_roll = cum_tpv / cum_v.replace(0, np.nan)
    dev = c / vwap_roll - 1.0
    z = (dev - dev.rolling(window, min_periods=window).mean()) / dev.rolling(window, min_periods=window).std().replace(0, np.nan)
    return pd.DataFrame({"vwap_dev": dev, "vwap_dev_z": z}, index=c.index)


# ---------------------------------------------------------------------------
# Volume / liquidity features
# ---------------------------------------------------------------------------

def volume_features(df: pd.DataFrame, windows: tuple[int, ...] = (5, 20, 60)) -> pd.DataFrame:
    """Volume z-scores, relative volume, dollar-volume z, and volume slope."""
    _, _, _, c, v = _parts(df)
    out = {}
    for w in windows:
        out[f"vol_z_{w}"] = volume_zscore(v, w)
        out[f"rel_vol_{w}"] = relative_volume(v, w)
    out["volume_slope_20"] = v.rolling(20, min_periods=20).apply(_linreg_slope, raw=True) / v.rolling(20, min_periods=20).mean()
    out["vol_price_corr_20"] = volume_price_corr(c, v, 20)
    dv = (c * v)
    out["dollar_vol_z_60"] = (np.log(dv.where(dv > 0)) - np.log(dv.where(dv > 0)).rolling(60, min_periods=60).mean()) / np.log(dv.where(dv > 0)).rolling(60, min_periods=60).std()
    return pd.DataFrame(out, index=c.index)


def volume_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """OBV/AD/CMF/MFI/WAD trend-flow features, normalized where possible."""
    o, h, l, c, v = _parts(df)
    obv_s = obv(c, v)
    obv_scale = obv_s.rolling(20, min_periods=20).std().replace(0, np.nan)
    return pd.DataFrame(
        {
            "obv_slope_20": obv_s.diff().rolling(20, min_periods=20).mean() / obv_scale,
            "obv_ema_ratio": obv_ema(c, v, 21) / obv_scale,
            "ad_slope_20": ad_line(h, l, c, v).diff().rolling(20, min_periods=20).mean() / (v.rolling(20, min_periods=20).mean() + 1e-12),
            "cmf_20": cmf(h, l, c, v, 20),
            "mfi_14": mfi(h, l, c, v, 14) / 100.0,
            "wad_pct": wad(h, l, c) / c,
        },
        index=c.index,
    )


def liquidity_features(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Liquidity proxies: Amihud illiquidity, turnover z, and spread estimates."""
    o, h, l, c, v = _parts(df)
    r = c.pct_change(fill_method=None).abs()
    dv = (c * v).replace(0, np.nan)
    amihud = (r / dv * 1e6).rolling(window, min_periods=window).mean()
    return pd.DataFrame(
        {
            "amihud_illiquidity": amihud,
            "dollar_volume_z": volume_zscore(np.log((c * v).where((c * v) > 0)), 60),
            "roll_spread": _roll_spread_series(c, window),
            "cs_spread": corwin_schultz_spread(h, l, window),
            "hl_spread_rel": ((h - l) / c).rolling(window, min_periods=window).mean(),
        },
        index=c.index,
    )


# ---------------------------------------------------------------------------
# Trend / oscillator features
# ---------------------------------------------------------------------------

def trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend state: MA distances, ADX, slope/R2 of log price, trend scores."""
    o, h, l, c, v = _parts(df)
    log_c = np.log(c)
    return pd.DataFrame(
        {
            "sma_dist_20": c / sma(c, 20) - 1.0,
            "sma_dist_50": c / sma(c, 50) - 1.0,
            "ema_dist_20": c / ema(c, 20) - 1.0,
            "ma_distance_20": ma_distance(c, 20),
            "adx_14": adx(h, l, c, 14) / 100.0,
            "trend_quality_20": trend_quality(c, 20),
            "trend_score_20": trend_score(c, 20),
            "log_slope_20": log_c.rolling(20, min_periods=20).apply(_linreg_slope, raw=True),
            "trend_r2_20": log_c.rolling(20, min_periods=20).apply(_linreg_r2, raw=True),
            "above_sma_200": (c > sma(c, 200)).astype(int),
        },
        index=c.index,
    )


def oscillator_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI/MACD/Stochastic/Williams %R/CCI grid on one frame."""
    o, h, l, c, v = _parts(df)
    out = pd.DataFrame(index=c.index)
    for w in (14, 28, 63):
        out[f"rsi_{w}"] = rsi(c, w) / 100.0
    macd_line = ema(c, 12) - ema(c, 26)
    signal = ema(macd_line, 9)
    out["macd"] = macd_line / c
    out["macd_hist"] = (macd_line - signal) / c
    st = stochastic(h, l, c, 14, 3, 3)
    out["stoch_k"] = st["k"] / 100.0
    out["stoch_d"] = st["d"] / 100.0
    hh = h.rolling(14, min_periods=14).max()
    ll = l.rolling(14, min_periods=14).min()
    out["williams_r"] = -100.0 * (hh - c) / (hh - ll).replace(0, np.nan) / 100.0
    tp = (h + l + c) / 3.0
    mad = tp.rolling(20, min_periods=20).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
    out["cci_20"] = (tp - sma(tp, 20)) / (0.015 * mad.replace(0, np.nan))
    return out


# ---------------------------------------------------------------------------
# Candlestick / cycle features
# ---------------------------------------------------------------------------

def candlestick_features(df: pd.DataFrame, score_window: int = 5) -> pd.DataFrame:
    """Aggregated bull/bear pattern counts and smoothed net candle score."""
    o, h, l, c, v = _parts(df)
    bull = detect_all(o, h, l, c, BULLISH_PATTERNS).sum(axis=1)
    bear = detect_all(o, h, l, c, BEARISH_PATTERNS).sum(axis=1)
    score = pattern_score(o, h, l, c, window=score_window)
    return pd.DataFrame(
        {
            "bull_patterns": bull,
            "bear_patterns": bear,
            "net_patterns": bull - bear,
            "candle_score": score,
            "candle_score_ema": ema(score, 5),
        },
        index=c.index,
    )


def cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ehlers dominant cycle, sine phase, and period-normalized lead."""
    _, _, _, c, _ = _parts(df)
    sw = sine_wave(c)
    period = sw["period"]
    return pd.DataFrame(
        {
            "dominant_cycle": period,
            "cycle_sine": sw["sine"],
            "cycle_lead": sw["lead_sine"],
            "cycle_period_z": (np.log(period.where(period > 0)) - np.log(period.where(period > 0)).rolling(60, min_periods=20).mean()) / np.log(period.where(period > 0)).rolling(60, min_periods=20).std(),
        },
        index=c.index,
    )


# ---------------------------------------------------------------------------
# Statistical / microstructure features
# ---------------------------------------------------------------------------

def statistical_features(df: pd.DataFrame, windows: tuple[int, ...] = (20, 60, 120)) -> pd.DataFrame:
    """Rolling skew, kurtosis, and lag-1 autocorrelation of returns."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    out = {}
    for w in windows:
        out[f"ret_skew_{w}"] = rolling_skew(r, w)
        out[f"ret_kurt_{w}"] = rolling_kurt(r, w)
        out[f"autocorr_1_{w}"] = r.rolling(w, min_periods=w).corr(r.shift(1))
    return pd.DataFrame(out, index=c.index)


def entropy_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling permutation/approximate/spectral entropy of returns."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "perm_entropy": r.rolling(window, min_periods=window // 2).apply(_perm_entropy, raw=True),
            "approx_entropy": r.rolling(window, min_periods=window // 2).apply(_approx_entropy, raw=True),
            "spectral_entropy": r.rolling(window, min_periods=window // 2).apply(_spectral_entropy, raw=True),
        },
        index=c.index,
    )


def kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Kyle's lambda: return sensitivity to signed volume flow."""
    c = _as_series(close)
    v = _as_series(volume)
    r = c.pct_change(fill_method=None)
    signed = np.sign(r) * v
    cov = r.rolling(window, min_periods=window).cov(signed)
    var = signed.rolling(window, min_periods=window).var()
    return cov / var.replace(0, np.nan)


def corwin_schultz_spread(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Corwin-Schultz high-low spread estimator (2-bar form, rolling mean)."""
    h = _as_series(high)
    l = _as_series(low)
    hl = np.log((h / l).where(h / l > 0)).abs()
    beta = (hl ** 2).rolling(2, min_periods=2).sum()
    gamma = (hl ** 2).shift(-1)
    root2 = np.sqrt(2.0)
    denom = 3.0 - 2.0 * root2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma.clip(lower=0.0).fillna(0.0) / denom)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return spread.clip(lower=0.0).rolling(window, min_periods=max(window // 2, 2)).mean()


def microstructure_features(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Microstructure pack: rolled spread, Corwin-Schultz, Kyle lambda, noise."""
    o, h, l, c, v = _parts(df)
    if v.isna().all():
        v = pd.Series(1.0, index=c.index)
    r = c.pct_change(fill_method=None)
    return pd.DataFrame(
        {
            "roll_spread": _roll_spread_series(c, window),
            "cs_spread": corwin_schultz_spread(h, l, window),
            "kyle_lambda": kyle_lambda(c, v, max(window, 20)),
            "noise_ratio": r.var() / (r.rolling(5, min_periods=5).var()).mean() if r.notna().sum() > 0 else np.nan,
        },
        index=c.index,
    )


# ---------------------------------------------------------------------------
# Cross-asset features
# ---------------------------------------------------------------------------

def cross_asset_features(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Relative strength, rolling correlation, and spread z vs a benchmark."""
    _, _, _, c, _ = _parts(df)
    _, _, _, b, _ = _parts(benchmark_df)
    b = b.reindex(c.index).ffill()
    r1 = c.pct_change(fill_method=None)
    r2 = b.pct_change(fill_method=None)
    ratio = np.log((c / b).where((c / b) > 0))
    return pd.DataFrame(
        {
            "rel_strength_14": relative_strength(c, b, 14),
            "corr_60": r1.rolling(60, min_periods=60).corr(r2),
            "spread_z_60": (ratio - ratio.rolling(60, min_periods=60).mean()) / ratio.rolling(60, min_periods=60).std(),
        },
        index=c.index,
    )


def beta_features(df: pd.DataFrame, benchmark_df: pd.DataFrame, windows: tuple[int, ...] = (60, 252)) -> pd.DataFrame:
    """Rolling market beta/alpha of the asset against a benchmark."""
    _, _, _, c, _ = _parts(df)
    _, _, _, b, _ = _parts(benchmark_df)
    b = b.reindex(c.index).ffill()
    r1 = c.pct_change(fill_method=None)
    r2 = b.pct_change(fill_method=None)
    out = {}
    for w in windows:
        cov = r1.rolling(w, min_periods=w).cov(r2)
        var = r2.rolling(w, min_periods=w).var()
        beta = cov / var.replace(0, np.nan)
        out[f"beta_{w}"] = beta
        out[f"alpha_{w}"] = r1.rolling(w, min_periods=w).mean() - beta * r2.rolling(w, min_periods=w).mean()
    return pd.DataFrame(out, index=c.index)


def lead_lag_features(series_a: pd.Series, series_b: pd.Series, lags: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3), window: int = 60) -> pd.DataFrame:
    """Rolling lead/lag correlations between two series."""
    a = _as_series(series_a)
    b = _as_series(series_b)
    out = {}
    for lag in lags:
        out[f"leadlag_{lag}"] = a.rolling(window, min_periods=window).corr(b.shift(lag))
    return pd.DataFrame(out, index=a.index)


# ---------------------------------------------------------------------------
# Generic builders
# ---------------------------------------------------------------------------

def lag_features(df: pd.DataFrame, lags: tuple[int, ...] = (1, 2, 3, 5, 10)) -> pd.DataFrame:
    """Lag every column at the given horizons, with suffix ``_lag{n}``."""
    frames = []
    for n in lags:
        shifted = df.shift(n)
        shifted.columns = [f"{col}_lag{n}" for col in df.columns]
        frames.append(shifted)
    return pd.concat(frames, axis=1) if frames else df.copy()


def diff_features(df: pd.DataFrame, orders: tuple[int, ...] = (1, 2, 5, 21)) -> pd.DataFrame:
    """N-th discrete differences of every column, suffix ``_diff{n}``."""
    frames = []
    for n in orders:
        d = df.diff(n)
        d.columns = [f"{col}_diff{n}" for col in df.columns]
        frames.append(d)
    return pd.concat(frames, axis=1) if frames else df.copy()


def rolling_stat_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (10, 20, 60),
    stats: tuple[str, ...] = ("mean", "std", "skew"),
) -> pd.DataFrame:
    """Rolling mean/std/skew/min/max/median for every numeric column."""
    out = {}
    for col in df.columns:
        s = df[col]
        if not np.issubdtype(np.asarray(s).dtype, np.number):
            continue
        for w in windows:
            roll = s.rolling(w, min_periods=max(2, w // 2))
            for stat in stats:
                try:
                    out[f"{col}_{stat}_{w}"] = getattr(roll, stat)()
                except (AttributeError, TypeError):
                    continue
    return pd.DataFrame(out, index=df.index)


def interaction_features(df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Pairwise products and ratios; defaults to adjacent-column pairs."""
    cols = [c for c in df.columns if np.issubdtype(np.asarray(df[c]).dtype, np.number)]
    if pairs is None:
        pairs = [(cols[i], cols[i + 1]) for i in range(min(len(cols) - 1, 4))]
    out = {}
    for a, b in pairs:
        out[f"{a}_x_{b}"] = df[a] * df[b]
        out[f"{a}_div_{b}"] = df[a] / df[b].replace(0, np.nan)
    return pd.DataFrame(out, index=df.index)


def ewma_features(df: pd.DataFrame, halflives: tuple[int, ...] = (5, 21, 63)) -> pd.DataFrame:
    """EWM means of every column for several half-lives, suffix ``_ewm{h}``."""
    frames = []
    for h in halflives:
        e = df.ewm(halflife=h, min_periods=max(2, h // 3)).mean()
        e.columns = [f"{col}_ewm{h}" for col in df.columns]
        frames.append(e)
    return pd.concat(frames, axis=1) if frames else df.copy()


# ---------------------------------------------------------------------------
# Targets / labels
# ---------------------------------------------------------------------------

def target_forward_returns(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 21)) -> pd.DataFrame:
    """Forward simple returns for each horizon (columns ``fwd_ret_{h}``)."""
    _, _, _, c, _ = _parts(df)
    out = {}
    for h in horizons:
        out[f"fwd_ret_{h}"] = c.shift(-h) / c - 1.0
    return pd.DataFrame(out, index=c.index)


def target_binary_direction(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.0) -> pd.Series:
    """Binary label: 1 if forward return exceeds ``threshold`` else 0."""
    _, _, _, c, _ = _parts(df)
    fwd = c.shift(-horizon) / c - 1.0
    return (fwd > threshold).astype(float).where(fwd.notna()).rename("target")


def target_quantile_labels(df: pd.DataFrame, horizon: int = 5, n_quantiles: int = 3) -> pd.Series:
    """Quantile-bucketed forward-return labels in ``0..n_quantiles-1``."""
    _, _, _, c, _ = _parts(df)
    fwd = c.shift(-horizon) / c - 1.0
    try:
        labels = pd.qcut(fwd, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        labels = pd.Series(np.nan, index=fwd.index)
    return labels.astype(float).where(fwd.notna()).rename("target")


def target_triple_barrier(
    df: pd.DataFrame,
    horizon: int = 21,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Triple-barrier labels (+1/-1/0) with volatility-scaled barriers.

    Barriers at ``pt_mult * vol`` / ``sl_mult * vol`` where ``vol`` is the
    rolling return std; labels record the first barrier touched within
    ``horizon`` bars (0 if neither), plus the bar distance to the touch.
    """
    _, h, l, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    vol = r.rolling(vol_window, min_periods=vol_window).std()
    cv = c.to_numpy(dtype="float64")
    hv = h.to_numpy(dtype="float64")
    lv = l.to_numpy(dtype="float64")
    vv = vol.to_numpy(dtype="float64")
    n = len(cv)
    labels = np.full(n, np.nan)
    bars = np.full(n, np.nan)
    for i in range(n):
        v = vv[i]
        if not np.isfinite(v) or not np.isfinite(cv[i]):
            continue
        up = cv[i] * (1.0 + pt_mult * v)
        dn = cv[i] * (1.0 - sl_mult * v)
        end = min(i + horizon, n - 1)
        label = 0.0
        touch = end - i
        for j in range(i + 1, end + 1):
            if lv[j] <= dn:
                label, touch = -1.0, j - i
                break
            if hv[j] >= up:
                label, touch = 1.0, j - i
                break
        labels[i] = label
        bars[i] = touch
    return pd.DataFrame({"tb_label": labels, "tb_bars": bars}, index=c.index)


def target_trend_labels(df: pd.DataFrame, horizon: int = 21, ma_window: int = 20) -> pd.Series:
    """1 if the simple MA is higher ``horizon`` bars ahead, else 0."""
    _, _, _, c, _ = _parts(df)
    ma = sma(c, ma_window)
    fwd = ma.shift(-horizon)
    return (fwd > ma).astype(float).where(fwd.notna()).rename("target")


def target_volatility_labels(df: pd.DataFrame, horizon: int = 21, n_quantiles: int = 3) -> pd.Series:
    """Quantile label of future realized volatility over ``horizon`` bars."""
    _, _, _, c, _ = _parts(df)
    r = c.pct_change(fill_method=None)
    fwd_vol = r.rolling(horizon, min_periods=max(2, horizon // 2)).std().shift(-horizon)
    try:
        labels = pd.qcut(fwd_vol, n_quantiles, labels=False, duplicates="drop")
    except ValueError:
        labels = pd.Series(np.nan, index=fwd_vol.index)
    return labels.astype(float).where(fwd_vol.notna()).rename("target")


def meta_labels(primary: pd.Series, tb_labels: pd.Series) -> pd.Series:
    """Meta-label: 1 when the primary side agrees with the triple-barrier outcome."""
    p = _as_series(primary)
    t = _as_series(tb_labels).reindex(p.index)
    agree = (np.sign(p) == np.sign(t)) & (p != 0) & (t != 0)
    return agree.astype(float).where(p.notna() & t.notna()).rename("meta_label")


def sample_weights_by_vol(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Inverse-volatility sample weights, normalized to mean 1."""
    _, _, _, c, _ = _parts(df)
    vol = c.pct_change(fill_method=None).rolling(window, min_periods=window).std()
    w = 1.0 / vol.replace(0, np.nan)
    return (w / w.mean()).rename("weight")


# ---------------------------------------------------------------------------
# Transforms / selection
# ---------------------------------------------------------------------------

def winsorize_df(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip every column to its empirical ``[lower, upper]`` quantiles."""
    lo = df.quantile(lower)
    hi = df.quantile(upper)
    return df.clip(lower=lo, upper=hi, axis=1)


def scale_zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardize every column."""
    return (df - df.mean()) / df.std().replace(0, np.nan)


def scale_minmax_df(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale every column into [0, 1]."""
    lo = df.min()
    hi = df.max()
    return (df - lo) / (hi - lo).replace(0, np.nan)


def rank_transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every column to percentile ranks in [0, 1]."""
    return df.rank(pct=True)


def impute_median_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with each column median."""
    return df.fillna(df.median(numeric_only=True))


def drop_low_variance(df: pd.DataFrame, threshold: float = 1e-8) -> pd.DataFrame:
    """Drop columns whose variance is below ``threshold``."""
    keep = [c for c in df.columns if pd.notna(df[c].var()) and df[c].var() > threshold]
    return df[keep]


def drop_correlated(df: pd.DataFrame, threshold: float = 0.95, method: str = "pearson") -> pd.DataFrame:
    """Greedily drop later columns correlated above ``threshold`` with an earlier one."""
    cols = list(df.columns)
    if len(cols) < 2:
        return df.copy()
    corr = df.corr(method=method).abs()
    dropped: set[str] = set()
    for i, ci in enumerate(cols):
        if ci in dropped:
            continue
        for cj in cols[i + 1:]:
            if cj in dropped:
                continue
            val = corr.at[ci, cj]
            if pd.notna(val) and val >= threshold:
                dropped.add(cj)
    return df[[c for c in cols if c not in dropped]]


def select_by_ic(X: pd.DataFrame, y: pd.Series, top_k: int = 20, method: str = "spearman") -> list[str]:
    """Select the ``top_k`` features with the largest |IC| against ``y``."""
    fn = spearman if method == "spearman" else kendall_tau
    if len(X) == 0 or len(X.columns) == 0:
        return []
    y_arr = np.asarray(y.reindex(X.index) if isinstance(y, pd.Series) else np.asarray(y), dtype="float64")
    scores = {}
    for col in X.columns:
        x_arr = np.asarray(X[col], dtype="float64")
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 10:
            continue
        try:
            scores[col] = abs(float(fn(x_arr[mask], y_arr[mask])))
        except Exception:
            continue
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def select_by_mutual_info(X: pd.DataFrame, y: pd.Series, top_k: int = 20, bins: int = 10) -> list[str]:
    """Select the ``top_k`` features by mutual information with ``y``."""
    if len(X) == 0 or len(X.columns) == 0:
        return []
    y_arr = np.asarray(y.reindex(X.index) if isinstance(y, pd.Series) else np.asarray(y), dtype="float64")
    scores = {}
    for col in X.columns:
        x_arr = np.asarray(X[col], dtype="float64")
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        if mask.sum() < 10:
            continue
        try:
            scores[col] = float(mutual_information(x_arr[mask], y_arr[mask], bins=bins))
        except Exception:
            continue
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def pca_features(X: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    """Principal components of a numeric frame (rows with NaNs yield NaN PCs)."""
    arr = np.asarray(X, dtype="float64")
    mask = np.isfinite(arr).all(axis=1)
    out = np.full((arr.shape[0], n_components), np.nan)
    if mask.sum() > n_components + 1:
        model = PCA(n_components=n_components).fit(arr[mask])
        out[mask] = model.transform(arr[mask])
    cols = [f"pc_{i + 1}" for i in range(n_components)]
    return pd.DataFrame(out, index=X.index, columns=cols)


# ---------------------------------------------------------------------------
# Pipeline / assembly / summary
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Leakage-safe feature transform pipeline.

    Order of operations: median impute -> winsorize -> scale -> correlation
    drop -> IC selection -> PCA. Every step fits on the training slice passed
    to :meth:`fit` and is re-applied unchanged in :meth:`transform`.
    """

    def __init__(
        self,
        impute: bool = True,
        winsorize: tuple[float, float] | None = (0.01, 0.99),
        scale: str | None = "zscore",
        corr_threshold: float | None = 0.95,
        top_k: int | None = None,
        ic_method: str = "spearman",
        n_components: int | None = None,
    ):
        self.impute = impute
        self.winsorize = winsorize
        self.scale = scale
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.ic_method = ic_method
        self.n_components = n_components
        self.medians_: pd.Series | None = None
        self.lo_: pd.Series | None = None
        self.hi_: pd.Series | None = None
        self.loc_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.columns_: list[str] = []
        self.pca_: PCA | None = None
        self.pc_columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        """Fit all transform parameters on the training data."""
        X = pd.DataFrame(X)
        if self.impute:
            self.medians_ = X.median(numeric_only=True)
            X = self._impute(X)
        if self.winsorize is not None:
            self.lo_ = X.quantile(self.winsorize[0])
            self.hi_ = X.quantile(self.winsorize[1])
            X = X.clip(lower=self.lo_, upper=self.hi_, axis=1)
        if self.scale == "zscore":
            self.loc_ = X.mean()
            self.scale_ = X.std().replace(0, 1.0)
            X = (X - self.loc_) / self.scale_
        elif self.scale == "minmax":
            self.loc_ = X.min()
            self.scale_ = (X.max() - X.min()).replace(0, 1.0)
            X = (X - self.loc_) / self.scale_
        if self.corr_threshold is not None:
            X = drop_correlated(X, threshold=self.corr_threshold)
        cols = list(X.columns)
        if self.top_k is not None and y is not None:
            selected = select_by_ic(X, y, top_k=self.top_k, method=self.ic_method)
            cols = [c for c in cols if c in set(selected)]
            X = X[cols]
        self.columns_ = cols
        if self.n_components is not None:
            arr = np.asarray(X, dtype="float64")
            mask = np.isfinite(arr).all(axis=1)
            if mask.sum() > self.n_components + 1:
                self.pca_ = PCA(n_components=self.n_components).fit(arr[mask])
            self.pc_columns_ = [f"pc_{i + 1}" for i in range(self.n_components)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transforms to new data."""
        X = pd.DataFrame(X)
        if self.impute and self.medians_ is not None:
            X = self._impute(X)
        if self.winsorize is not None and self.lo_ is not None:
            X = X.clip(lower=self.lo_, upper=self.hi_, axis=1)
        if self.scale in ("zscore", "minmax") and self.loc_ is not None:
            X = (X - self.loc_) / self.scale_
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise KeyError(f"Missing columns required by the pipeline: {missing}")
        X = X[self.columns_]
        if self.pca_ is not None:
            arr = np.asarray(X, dtype="float64")
            mask = np.isfinite(arr).all(axis=1)
            out = np.full((arr.shape[0], len(self.pc_columns_)), np.nan)
            if mask.sum():
                out[mask] = self.pca_.transform(arr[mask])
            return pd.DataFrame(out, index=X.index, columns=self.pc_columns_)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit on ``X`` (with optional labels for IC selection) and transform it."""
        return self.fit(X, y).transform(X)

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            return X
        return X.fillna(self.medians_.reindex(X.columns)).fillna(0.0)


def build_feature_matrix(
    df: pd.DataFrame,
    groups: tuple[str, ...] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Assemble common feature groups into a single design matrix.

    Available groups: ``calendar, returns, overnight, intraday, range, price,
    zscore, ranks, volatility, semivariance, momentum, mom_quality, high52,
    reversal, mean_reversion, bollinger, vwap, volume, flow, trend,
    oscillators, candles, cycles, stats, entropy``.
    """
    builders: dict[str, object] = {
        "calendar": lambda d: calendar_features(d.index) if isinstance(d.index, pd.DatetimeIndex) else pd.DataFrame(index=d.index),
        "returns": return_features,
        "overnight": overnight_feature,
        "intraday": intraday_feature,
        "range": range_features,
        "price": price_level_features,
        "zscore": price_zscore_features,
        "ranks": rolling_rank_features,
        "volatility": volatility_features,
        "semivariance": semivariance_features,
        "momentum": momentum_features,
        "mom_quality": momentum_quality_features,
        "high52": high_52w_features,
        "reversal": reversal_features,
        "mean_reversion": mean_reversion_features,
        "bollinger": bollinger_features,
        "vwap": vwap_features,
        "volume": volume_features,
        "flow": volume_flow_features,
        "trend": trend_features,
        "oscillators": oscillator_features,
        "candles": candlestick_features,
        "cycles": cycle_features,
        "stats": statistical_features,
        "entropy": entropy_features,
    }
    chosen = list(groups) if groups is not None else list(builders)
    frames = []
    for name in chosen:
        fn = builders.get(name)
        if fn is None:
            raise ValueError(f"Unknown feature group: {name}")
        block = fn(df)
        block.columns = [f"{name}__{c}" for c in block.columns]
        frames.append(block)
    matrix = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    return matrix.dropna() if dropna else matrix


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype/nan/mean/std/min/max overview of a feature frame."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_nan": df.isna().sum(),
            "nan_frac": df.isna().mean(),
            "mean": df.mean(numeric_only=True),
            "std": df.std(numeric_only=True),
            "min": df.min(numeric_only=True),
            "max": df.max(numeric_only=True),
        }
    )


# ---------------------------------------------------------------------------
# Internal numeric helpers
# ---------------------------------------------------------------------------

def _linreg_slope(y) -> float:
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype="float64")[mask]
    yy = y[mask]
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    if denom <= 0:
        return np.nan
    return float((xc * (yy - yy.mean())).sum() / denom)


def _linreg_r2(y) -> float:
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(len(y), dtype="float64")[mask]
    yy = y[mask]
    xc = x - x.mean()
    yc = yy - yy.mean()
    denom = float((xc ** 2).sum()) * float((yc ** 2).sum())
    if denom <= 0:
        return np.nan
    return float((xc * yc).sum() ** 2 / denom)


def _roll_spread_series(close: pd.Series, window: int = 21) -> pd.Series:
    c = _as_series(close)
    dp = c.diff()
    cov = dp.rolling(window, min_periods=window).cov(dp.shift(1))
    return 2.0 * np.sqrt((-cov).clip(lower=0.0))


def _perm_entropy(x, order: int = 3) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < order + 1:
        return np.nan
    from itertools import permutations
    perms = list(permutations(range(order)))
    index_map = {p: i for i, p in enumerate(perms)}
    counts = np.zeros(len(perms))
    for i in range(len(x) - order + 1):
        idx = tuple(np.argsort(x[i:i + order], kind="stable"))
        counts[index_map[idx]] += 1.0
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(perms)))


def _approx_entropy(x, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        s = x.std()
        if s <= 0:
            return np.nan
        r = 0.2 * s

    def phi(mm: int) -> float:
        patterns = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        dist = np.max(np.abs(patterns[:, None, :] - patterns[None, :, :]), axis=2)
        c = (dist <= r).mean(axis=1)
        return float(np.log(np.where(c > 0, c, 1e-12)).mean())

    return float(abs(phi(m) - phi(m + 1)))


def _spectral_entropy(x) -> float:
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.nan
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    total = spec.sum()
    if total <= 0:
        return np.nan
    p = spec / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))
