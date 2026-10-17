"""pandas-ta Integration Wrapper.

Provides access to 130+ technical indicators from the pandas-ta library
through a unified interface. Falls back to our own quant.indicators
implementation if pandas-ta is not installed.

All functions accept wide-format pandas DataFrames (dates x assets) and
return DataFrames of indicator values.

Reference: https://github.com/twopirllc/pandas-ta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Check if pandas-ta is available
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from quant.indicators import (
    rsi, macd, bollinger_bands, atr, adx, stoch_rsi,
    williams_r, cci, obv, vwap, supertrend, ichimoku,
    parabolic_sar, stochastic, ema, sma, wma, hma, kama,
    vwma, alma, trix, dpo, cmo, coppock_curve, ppo, pvo,
    rvi, elder_ray, choppiness_index, relative_vigor,
    know_sure_thing, schaff_trend, fisher_transform,
    awesome_oscillator, accelerator_oscillator, squeeze_momentum,
    mass_index, vortex, hurst_exponent, entropy, fractdim,
    skewness, kurtosis, zscore, percentile_rank,
    realized_volatility, ewma_volatility, parkinson_volatility,
    garman_klass_volatility, rogers_satchell_volatility,
    yang_zhang_volatility, volatility_cone,
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    drawdown_series, ulcer_index,
)
from quant.risk import (
    pain_index, pain_ratio, tail_ratio, gain_to_pain_ratio, common_sense_ratio,
    var_historical, var_parametric, cvar_historical, cvar_parametric,
    expected_shortfall,
)
from quant.indicators import (
    omega_ratio, beta, alpha, tracking_error, information_ratio,
    linreg_slope, linreg_channel, linreg_forecast,
    rolling_corr, rolling_cov,
)

__all__ = [
    "PANDAS_TA_AVAILABLE",
    "ta_rsi", "ta_macd", "ta_bbands", "ta_atr", "ta_adx",
    "ta_stochrsi", "ta_willr", "ta_cci", "ta_obv", "ta_vwap",
    "ta_supertrend", "ta_ichimoku", "ta_psar", "ta_stoch",
    "ta_ema", "ta_sma", "ta_wma", "ta_hma", "ta_kama",
    "ta_vwma", "ta_alma", "ta_trix", "ta_dpo", "ta_cmo",
    "ta_coppock", "ta_ppo", "ta_pvo", "ta_rvi", "ta_elder",
    "ta_choppiness", "ta_rvi_cross", "ta_kst", "ta_schaff",
    "ta_fisher", "ta_ao", "ta_ac", "ta_squeeze", "ta_mass",
    "ta_vortex", "ta_hurst", "ta_entropy", "ta_fractdim",
    "ta_skew", "ta_kurt", "ta_zscore", "ta_percentile",
    "ta_rvol", "ta_ewmavol", "ta_parkinson", "ta_gk",
    "ta_rs", "ta_yz", "ta_volcone",
    "ta_sharpe", "ta_sortino", "ta_calmar", "ta_mdd",
    "ta_ulcer", "ta_var", "ta_cvar",
    "ta_beta", "ta_alpha", "ta_te", "ta_ir",
    "ta_linreg", "ta_linreg_channel", "ta_linreg_forecast",
    "ta_rollcorr", "ta_rollcov",
    "get_available_indicators",
    "apply_indicator",
]


# ---------------------------------------------------------------------------
# Helper: apply a pandas-ta indicator to a wide-format DataFrame
# ---------------------------------------------------------------------------

def _apply_pta_indicator(
    df: pd.DataFrame,
    indicator_name: str,
    **kwargs,
) -> pd.DataFrame:
    """Apply a pandas-ta indicator column-wise to a wide-format DataFrame."""
    if not PANDAS_TA_AVAILABLE:
        raise ImportError("pandas-ta is not installed. Install with: pip install pandas-ta")

    result = pd.DataFrame(index=df.index, columns=df.columns, dtype="float64")
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 2:
            result[col] = np.nan
            continue
        try:
            # pandas-ta indicators typically work on Series with open/high/low/close/volume
            # For single-series indicators, we pass close
            out = getattr(ta, indicator_name)(close=series, **kwargs)
            if isinstance(out, pd.DataFrame):
                # Take the first column if multi-output
                result[col] = out.iloc[:, 0]
            else:
                result[col] = out
        except Exception:
            result[col] = np.nan
    return result


def _norm(result, ref):
    """Normalize fallback output: DataFrame in -> DataFrame out."""
    if isinstance(ref, pd.DataFrame):
        if isinstance(result, pd.Series):
            col = ref.columns[0] if ref.shape[1] == 1 else "value"
            return result.to_frame(name=col)
        if isinstance(result, dict):
            first = next(iter(result.values()))
            return _norm(first, ref)
    return result


# ---------------------------------------------------------------------------
# Wrapper functions (pandas-ta preferred, fallback to quant.indicators)
# ---------------------------------------------------------------------------

def ta_rsi(close: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """RSI via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "rsi", length=length)
    return _norm(rsi(close, length), close)


def ta_macd(
    close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD via pandas-ta (returns MACD line)."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "macd", fast=fast, slow=slow, signal=signal)
    return _norm(macd(close, fast, slow, signal)["macd"], close)


def ta_bbands(
    close: pd.DataFrame, length: int = 20, std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands via pandas-ta (returns middle band)."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "bbands", length=length, std=std)
    return _norm(bollinger_bands(close, length, std)["mid"], close)


def ta_atr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 14
) -> pd.DataFrame:
    """ATR via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        # pandas-ta ATR needs high/low/close as separate Series
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.atr(high=high[col], low=low[col], close=close[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return atr(high, low, close, length)


def ta_adx(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 14
) -> pd.DataFrame:
    """ADX via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.adx(high=high[col], low=low[col], close=close[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return adx(high, low, close, length)


def ta_stochrsi(close: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Stochastic RSI via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "stochrsi", length=length)
    return stoch_rsi(close, length)["k"]


def ta_willr(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 14
) -> pd.DataFrame:
    """Williams %R via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.willr(high=high[col], low=low[col], close=close[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return williams_r(high, low, close, length)


def ta_cci(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 20
) -> pd.DataFrame:
    """CCI via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.cci(high=high[col], low=low[col], close=close[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return cci(high, low, close, length)


def ta_obv(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """OBV via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.obv(close=close[col], volume=volume[col])
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return obv(close, volume)


def ta_vwap(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame
) -> pd.DataFrame:
    """VWAP via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.vwap(high=high[col], low=low[col], close=close[col], volume=volume[col])
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return vwap(high, low, close, volume)


def ta_supertrend(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    length: int = 10, multiplier: float = 3.0,
) -> pd.DataFrame:
    """Supertrend via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.supertrend(
                    high=high[col], low=low[col], close=close[col],
                    length=length, multiplier=multiplier,
                )
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return supertrend(high, low, close, length, multiplier)["supertrend"]


def ta_ichimoku(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku via pandas-ta (returns Tenkan-sen)."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.ichimoku(
                    high=high[col], low=low[col], close=close[col],
                    tenkan=tenkan, kijun=kijun, senkou_b=senkou_b,
                )
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return ichimoku(high, low, close, tenkan, kijun, senkou_b)["tenkan"]


def ta_psar(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    step: float = 0.02, max_step: float = 0.2,
) -> pd.DataFrame:
    """Parabolic SAR via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.psar(
                    high=high[col], low=low[col], close=close[col],
                    step=step, max_step=max_step,
                )
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return parabolic_sar(high, low, step, max_step)


def ta_stoch(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    k: int = 14, d: int = 3,
) -> pd.DataFrame:
    """Stochastic via pandas-ta (returns %K)."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.stoch(high=high[col], low=low[col], close=close[col], k=k, d=d)
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return stochastic(high, low, close, k, d)["k"]


def ta_ema(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """EMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "ema", length=length)
    return ema(close, length)


def ta_sma(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """SMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "sma", length=length)
    return sma(close, length)


def ta_wma(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """WMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "wma", length=length)
    return wma(close, length)


def ta_hma(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """HMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "hma", length=length)
    return hma(close, length)


def ta_kama(close: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    """KAMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "kama", length=length)
    return kama(close, length)


def ta_vwma(close: pd.DataFrame, volume: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """VWMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.vwma(close=close[col], volume=volume[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return vwma(close, volume, length)


def ta_alma(close: pd.DataFrame, length: int = 9) -> pd.DataFrame:
    """ALMA via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "alma", length=length)
    return alma(close, length)


def ta_trix(close: pd.DataFrame, length: int = 15) -> pd.DataFrame:
    """TRIX via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "trix", length=length)
    return trix(close, length)


def ta_dpo(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """DPO via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "dpo", length=length)
    return dpo(close, length)


def ta_cmo(close: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """CMO via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "cmo", length=length)
    return cmo(close, length)


def ta_coppock(close: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    """Coppock Curve via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "coppock", length=length)
    return coppock_curve(close)


def ta_ppo(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """PPO via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "ppo", fast=fast, slow=slow, signal=signal)
    return ppo(close, fast, slow, signal)["ppo"]


def ta_pvo(volume: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """PVO via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(volume, "pvo", fast=fast, slow=slow, signal=signal)
    return pvo(volume, fast, slow, signal)["pvo"]


def ta_rvi(
    open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    length: int = 10,
) -> pd.DataFrame:
    """RVI via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.rvi(
                    open=open_[col], high=high[col], low=low[col], close=close[col],
                    length=length,
                )
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return rvi(open_, high, low, close, length)["rvi"]


def ta_elder(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 13
) -> pd.DataFrame:
    """Elder Ray via pandas-ta (returns Bull Power)."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.elder(high=high[col], low=low[col], close=close[col], length=length)
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return elder_ray(high, low, close, length)["bull"]


def ta_choppiness(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 14
) -> pd.DataFrame:
    """Choppiness Index via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.chop(high=high[col], low=low[col], close=close[col], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return choppiness_index(high, low, close, length)


def ta_kst(
    close: pd.DataFrame,
    roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
    sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15,
) -> pd.DataFrame:
    """KST via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(
            close, "kst",
            roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4,
            sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4,
        )
    return know_sure_thing(
        close, roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4
    )["kst"]


def ta_schaff(close: pd.DataFrame, fast: int = 23, slow: int = 50, cycle: int = 10) -> pd.DataFrame:
    """Schaff Trend Cycle via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "schaff", fast=fast, slow=slow, cycle=cycle)
    return schaff_trend(close, fast, slow, cycle)


def ta_fisher(
    high: pd.DataFrame, low: pd.DataFrame, length: int = 10
) -> pd.DataFrame:
    """Fisher Transform via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.fisher(high=high[col], low=low[col], length=length)
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return fisher_transform(high, low, length)["fisher"]


def ta_ao(high: pd.DataFrame, low: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.DataFrame:
    """Awesome Oscillator via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.ao(high=high[col], low=low[col], fast=fast, slow=slow)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return awesome_oscillator(high, low, fast, slow)


def ta_ac(high: pd.DataFrame, low: pd.DataFrame) -> pd.DataFrame:
    """Accelerator Oscillator via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.ac(high=high[col], low=low[col])
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return accelerator_oscillator(high, low)


def ta_squeeze(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    bb_length: int = 20, bb_mult: float = 2.0,
    kc_length: int = 20, kc_mult: float = 1.5,
) -> pd.DataFrame:
    """Squeeze Momentum via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.squeeze(
                    high=high[col], low=low[col], close=close[col],
                    bb_length=bb_length, bb_mult=bb_mult,
                    kc_length=kc_length, kc_mult=kc_mult,
                )
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return squeeze_momentum(high, low, close, bb_length, bb_mult, kc_length, kc_mult)["momentum"]


def ta_mass(high: pd.DataFrame, low: pd.DataFrame, fast: int = 9, slow: int = 25) -> pd.DataFrame:
    """Mass Index via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.massi(high=high[col], low=low[col], fast=fast, slow=slow)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return mass_index(high, low, fast, slow)


def ta_vortex(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, length: int = 14
) -> pd.DataFrame:
    """Vortex Indicator via pandas-ta (returns +VI)."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.vort(high=high[col], low=low[col], close=close[col], length=length)
                if isinstance(out, pd.DataFrame):
                    result[col] = out.iloc[:, 0]
                else:
                    result[col] = out
            except Exception:
                result[col] = np.nan
        return result
    return vortex(high, low, close, length)["vi_plus"]


def ta_hurst(close: pd.DataFrame, max_lag: int = 100) -> pd.DataFrame:
    """Hurst Exponent via pandas-ta (approximation)."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "hurst", max_lag=max_lag)
    # Fallback: scalar per column
    result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for col in close.columns:
        try:
            val = hurst_exponent(close[col], max_lag)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_entropy(close: pd.DataFrame, length: int = 60, bins: int = 10) -> pd.DataFrame:
    """Entropy via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "entropy", length=length, bins=bins)
    return entropy(close, bins, length)


def ta_fractdim(close: pd.DataFrame, length: int = 30) -> pd.DataFrame:
    """Fractal Dimension via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "fractdim", length=length)
    return fractdim(close, length)


def ta_skew(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Skewness via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "skew", length=length)
    return skewness(close, length)


def ta_kurt(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Kurtosis via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "kurt", length=length)
    return kurtosis(close, length)


def ta_zscore(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Z-Score via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "zscore", length=length)
    return zscore(close, length)


def ta_percentile(close: pd.DataFrame, length: int = 252) -> pd.DataFrame:
    """Percentile Rank via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "percentile", length=length)
    return percentile_rank(close, length)


def ta_rvol(close: pd.DataFrame, length: int = 20, annualize: int = 252) -> pd.DataFrame:
    """Realized Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "rvol", length=length, annualize=annualize)
    return realized_volatility(close, length, annualize)


def ta_ewmavol(close: pd.DataFrame, lam: float = 0.94, annualize: int = 252) -> pd.DataFrame:
    """EWMA Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "ewmavol", lam=lam, annualize=annualize)
    return ewma_volatility(close, lam, annualize)


def ta_parkinson(
    high: pd.DataFrame, low: pd.DataFrame, length: int = 20, annualize: int = 252
) -> pd.DataFrame:
    """Parkinson Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.parkinson(high=high[col], low=low[col], length=length, annualize=annualize)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return parkinson_volatility(high, low, length, annualize)


def ta_gk(
    open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    length: int = 20, annualize: int = 252,
) -> pd.DataFrame:
    """Garman-Klass Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.gk(
                    open=open_[col], high=high[col], low=low[col], close=close[col],
                    length=length, annualize=annualize,
                )
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return garman_klass_volatility(open_, high, low, close, length, annualize)


def ta_rs(
    open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    length: int = 20, annualize: int = 252,
) -> pd.DataFrame:
    """Rogers-Satchell Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.rs(
                    open=open_[col], high=high[col], low=low[col], close=close[col],
                    length=length, annualize=annualize,
                )
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return rogers_satchell_volatility(open_, high, low, close, length, annualize)


def ta_yz(
    open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    length: int = 20, annualize: int = 252,
) -> pd.DataFrame:
    """Yang-Zhang Volatility via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
        for col in close.columns:
            try:
                out = ta.yz(
                    open=open_[col], high=high[col], low=low[col], close=close[col],
                    length=length, annualize=annualize,
                )
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return yang_zhang_volatility(open_, high, low, close, length, annualize)


def ta_volcone(
    close: pd.DataFrame, windows=(5, 10, 20, 30, 60, 90, 120, 252),
    quantiles=(0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Volatility Cone via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "volcone", windows=windows, quantiles=quantiles)
    return volatility_cone(close, windows, quantiles)


# ---------------------------------------------------------------------------
# Performance metrics wrappers
# ---------------------------------------------------------------------------

def ta_sharpe(returns: pd.DataFrame, rf: float = 0.0, annualize: int = 252) -> pd.DataFrame:
    """Sharpe Ratio via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(returns, "sharpe", rf=rf, annualize=annualize)
    # Fallback: scalar per column
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype="float64")
    for col in returns.columns:
        try:
            val = sharpe_ratio(returns[col], rf, annualize)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_sortino(returns: pd.DataFrame, rf: float = 0.0, annualize: int = 252) -> pd.DataFrame:
    """Sortino Ratio via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(returns, "sortino", rf=rf, annualize=annualize)
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype="float64")
    for col in returns.columns:
        try:
            val = sortino_ratio(returns[col], rf, annualize)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_calmar(equity: pd.DataFrame, annualize: int = 252) -> pd.DataFrame:
    """Calmar Ratio via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(equity, "calmar", annualize=annualize)
    result = pd.DataFrame(index=equity.index, columns=equity.columns, dtype="float64")
    for col in equity.columns:
        try:
            val = calmar_ratio(equity[col], annualize)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_mdd(equity: pd.DataFrame) -> pd.DataFrame:
    """Max Drawdown via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(equity, "mdd")
    result = pd.DataFrame(index=equity.index, columns=equity.columns, dtype="float64")
    for col in equity.columns:
        try:
            val = max_drawdown(equity[col])
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_ulcer(equity: pd.DataFrame) -> pd.DataFrame:
    """Ulcer Index via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(equity, "ulcer")
    result = pd.DataFrame(index=equity.index, columns=equity.columns, dtype="float64")
    for col in equity.columns:
        try:
            val = ulcer_index(equity[col])
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_var(returns: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Value at Risk via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(returns, "var", confidence=confidence)
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype="float64")
    for col in returns.columns:
        try:
            val = var_historical(returns[col], confidence)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_cvar(returns: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Conditional VaR via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(returns, "cvar", confidence=confidence)
    result = pd.DataFrame(index=returns.index, columns=returns.columns, dtype="float64")
    for col in returns.columns:
        try:
            val = cvar_historical(returns[col], confidence)
            result[col] = val
        except Exception:
            result[col] = np.nan
    return result


def ta_beta(
    asset: pd.DataFrame, benchmark: pd.DataFrame, length: int = 60
) -> pd.DataFrame:
    """Beta via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=asset.index, columns=asset.columns, dtype="float64")
        for col in asset.columns:
            try:
                out = ta.beta(close=asset[col], benchmark=benchmark[col] if col in benchmark.columns else benchmark.iloc[:, 0], length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return beta(asset, benchmark, length)


def ta_alpha(
    asset: pd.DataFrame, benchmark: pd.DataFrame, length: int = 60, annualize: int = 252
) -> pd.DataFrame:
    """Jensen's Alpha via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(asset, "alpha", benchmark=benchmark, length=length, annualize=annualize)
    return alpha(asset, benchmark, length, annualize)


def ta_te(
    asset: pd.DataFrame, benchmark: pd.DataFrame, length: int = 60, annualize: int = 252
) -> pd.DataFrame:
    """Tracking Error via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(asset, "te", benchmark=benchmark, length=length, annualize=annualize)
    return tracking_error(asset, benchmark, length, annualize)


def ta_ir(
    asset: pd.DataFrame, benchmark: pd.DataFrame, length: int = 60
) -> pd.DataFrame:
    """Information Ratio via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(asset, "ir", benchmark=benchmark, length=length)
    return information_ratio(asset, benchmark, length)


def ta_linreg(close: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """Linear Regression Slope via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "linreg", length=length)
    return linreg_slope(close, length)


def ta_linreg_channel(
    close: pd.DataFrame, length: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Linear Regression Channel via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "linreg_channel", length=length, num_std=num_std)
    return linreg_channel(close, length, num_std)


def ta_linreg_forecast(
    close: pd.DataFrame, length: int = 20, forecast: int = 1
) -> pd.DataFrame:
    """Linear Regression Forecast via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        return _apply_pta_indicator(close, "linreg_forecast", length=length, forecast=forecast)
    return linreg_forecast(close, length, forecast)


def ta_rollcorr(
    a: pd.DataFrame, b: pd.DataFrame, length: int = 60
) -> pd.DataFrame:
    """Rolling Correlation via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=a.index, columns=a.columns, dtype="float64")
        for col in a.columns:
            try:
                b_col = b[col] if col in b.columns else b.iloc[:, 0]
                out = ta.rollcorr(close=a[col], benchmark=b_col, length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return rolling_corr(a, b, length)


def ta_rollcov(
    a: pd.DataFrame, b: pd.DataFrame, length: int = 60
) -> pd.DataFrame:
    """Rolling Covariance via pandas-ta."""
    if PANDAS_TA_AVAILABLE:
        result = pd.DataFrame(index=a.index, columns=a.columns, dtype="float64")
        for col in a.columns:
            try:
                b_col = b[col] if col in b.columns else b.iloc[:, 0]
                out = ta.rollcov(close=a[col], benchmark=b_col, length=length)
                result[col] = out if isinstance(out, pd.Series) else out.iloc[:, 0]
            except Exception:
                result[col] = np.nan
        return result
    return rolling_cov(a, b, length)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_available_indicators() -> dict[str, bool]:
    """Return a dictionary of indicator availability."""
    return {
        "pandas_ta": PANDAS_TA_AVAILABLE,
        "quant_indicators": True,
        "alpha101": True,
    }


def apply_indicator(
    name: str,
    data: pd.DataFrame | dict[str, pd.DataFrame],
    **kwargs,
) -> pd.DataFrame:
    """Generic indicator dispatcher.

    Parameters
    ----------
    name : str
        Indicator name (e.g., 'rsi', 'macd', 'bbands', 'alpha001').
    data : DataFrame or dict of DataFrames
        Input data. If dict, keys should match expected parameter names.
    **kwargs : dict
        Additional parameters passed to the indicator function.

    Returns
    -------
    DataFrame
        Indicator values.
    """
    # Alpha101 dispatch
    if name.startswith("alpha") and name[5:].isdigit():
        from quant.alpha101 import compute_all_alphas
        if isinstance(data, dict):
            close = data.get("close")
            volume = data.get("volume")
            vwap = data.get("vwap")
        else:
            close = data
            volume = data
            vwap = None
        if close is None:
            raise ValueError("close data required for alpha indicators")
        all_alphas = compute_all_alphas(close, volume, vwap)
        return all_alphas.xs(name, level="alpha", axis=1)

    # pandas-ta / quant.indicators dispatch
    func_map = {
        "rsi": ta_rsi,
        "macd": ta_macd,
        "bbands": ta_bbands,
        "atr": ta_atr,
        "adx": ta_adx,
        "stochrsi": ta_stochrsi,
        "willr": ta_willr,
        "cci": ta_cci,
        "obv": ta_obv,
        "vwap": ta_vwap,
        "supertrend": ta_supertrend,
        "ichimoku": ta_ichimoku,
        "psar": ta_psar,
        "stoch": ta_stoch,
        "ema": ta_ema,
        "sma": ta_sma,
        "wma": ta_wma,
        "hma": ta_hma,
        "kama": ta_kama,
        "vwma": ta_vwma,
        "alma": ta_alma,
        "trix": ta_trix,
        "dpo": ta_dpo,
        "cmo": ta_cmo,
        "coppock": ta_coppock,
        "ppo": ta_ppo,
        "pvo": ta_pvo,
        "rvi": ta_rvi,
        "elder": ta_elder,
        "choppiness": ta_choppiness,
        "kst": ta_kst,
        "schaff": ta_schaff,
        "fisher": ta_fisher,
        "ao": ta_ao,
        "ac": ta_ac,
        "squeeze": ta_squeeze,
        "mass": ta_mass,
        "vortex": ta_vortex,
        "hurst": ta_hurst,
        "entropy": ta_entropy,
        "fractdim": ta_fractdim,
        "skew": ta_skew,
        "kurt": ta_kurt,
        "zscore": ta_zscore,
        "percentile": ta_percentile,
        "rvol": ta_rvol,
        "ewmavol": ta_ewmavol,
        "parkinson": ta_parkinson,
        "gk": ta_gk,
        "rs": ta_rs,
        "yz": ta_yz,
        "volcone": ta_volcone,
        "sharpe": ta_sharpe,
        "sortino": ta_sortino,
        "calmar": ta_calmar,
        "mdd": ta_mdd,
        "ulcer": ta_ulcer,
        "var": ta_var,
        "cvar": ta_cvar,
        "beta": ta_beta,
        "alpha": ta_alpha,
        "te": ta_te,
        "ir": ta_ir,
        "linreg": ta_linreg,
        "linreg_channel": ta_linreg_channel,
        "linreg_forecast": ta_linreg_forecast,
        "rollcorr": ta_rollcorr,
        "rollcov": ta_rollcov,
    }

    if name not in func_map:
        raise ValueError(f"Unknown indicator: {name}. Available: {list(func_map.keys())}")

    if isinstance(data, dict):
        return func_map[name](**data, **kwargs)
    return func_map[name](data, **kwargs)
