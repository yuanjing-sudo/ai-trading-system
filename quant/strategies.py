"""Composable trading strategies with risk overlays.

Each strategy function accepts market data (wide-format DataFrames) and
returns a signal DataFrame in [-1, 1] (long-short weights before risk
scaling). Strategies can be combined, filtered, and blended.

Conventions
-----------
- Inputs are wide-format DataFrames: rows = dates, columns = assets.
- Output signals are in [-1, 1]; positive = long, negative = short.
- All functions are vectorized across assets and dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import (
    rsi, macd, bollinger_bands, keltner_channels, adx, atr,
    supertrend, parabolic_sar, ichimoku, stochastic, williams_r,
    cci, adx as adx_indicator,
)
from quant.factors import (
    momentum_factor, reversal_factor, volatility_factor,
    composite_value_factor, composite_quality_factor,
    composite_momentum_factor, composite_volatility_factor,
    composite_liquidity_factor, composite_factor,
    rank_cross_sectional,
)

__all__ = [
    "dual_momentum_strategy", "mean_reversion_strategy",
    "trend_following_strategy", "pairs_trading_strategy",
    "breakout_strategy", "rsi_mean_reversion_strategy",
    "bollinger_mean_reversion_strategy", "macd_cross_strategy",
    "supertrend_strategy", "ichimoku_strategy",
    "multi_factor_strategy", "risk_parity_strategy",
    "minimum_variance_strategy", "maximum_sharpe_strategy",
    "black_litterman_strategy", "volatility_targeting_strategy",
    "drawdown_control_strategy", "regime_switching_strategy",
    "seasonality_strategy", "earnings_momentum_strategy",
    "combine_signals", "blend_signals", "apply_position_limits",
    "apply_turnover_penalty", "signal_to_weights",
]


# ---------------------------------------------------------------------------
# Signal combination helpers
# ---------------------------------------------------------------------------

def combine_signals(*signals: pd.DataFrame, weights: list[float] | None = None) -> pd.DataFrame:
    """Blend multiple signal DataFrames with optional weights."""
    if not signals:
        raise ValueError("At least one signal required")
    w = weights or [1.0 / len(signals)] * len(signals)
    if len(w) != len(signals):
        raise ValueError("weights length must match signals length")
    out = pd.DataFrame(0.0, index=signals[0].index, columns=signals[0].columns)
    for sig, weight in zip(signals, w):
        aligned = sig.reindex_like(out).fillna(0.0)
        out = out + weight * aligned
    return out.clip(-1.0, 1.0)


def blend_signals(
    signals: dict[str, pd.DataFrame], regime: pd.Series | None = None,
    regime_weights: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Blend signals with optional regime-dependent weights."""
    names = list(signals.keys())
    base = {n: signals[n].reindex_like(signals[names[0]]).fillna(0.0) for n in names}
    if regime is None or regime_weights is None:
        return combine_signals(*base.values())
    out = pd.DataFrame(0.0, index=base[names[0]].index, columns=base[names[0]].columns)
    for date in out.index:
        reg = regime.loc[date] if date in regime.index else "neutral"
        rw = regime_weights.get(reg, {n: 1.0 / len(names) for n in names})
        total = sum(rw.values()) or 1.0
        for n in names:
            out.loc[date] += (rw.get(n, 0.0) / total) * base[n].loc[date]
    return out.clip(-1.0, 1.0)


def apply_position_limits(
    signals: pd.DataFrame, max_position: float = 0.1, max_gross: float = 1.0
) -> pd.DataFrame:
    """Clip individual positions and rescale to gross exposure limit."""
    clipped = signals.clip(-max_position, max_position)
    gross = clipped.abs().sum(axis=1)
    scale = _safe_div(max_gross, gross.replace(0, np.nan), fill=1.0)
    return clipped.mul(scale, axis=0)


def apply_turnover_penalty(
    current: pd.DataFrame, previous: pd.DataFrame, penalty: float = 0.001
) -> pd.DataFrame:
    """Reduce signal magnitude proportional to turnover cost."""
    turnover = (current - previous).abs().sum(axis=1)
    scale = (1.0 - penalty * turnover).clip(0.0, 1.0)
    return current.mul(scale, axis=0)


def signal_to_weights(
    signals: pd.DataFrame, target_vol: float | None = None,
    returns: pd.DataFrame | None = None, lookback: int = 60,
) -> pd.DataFrame:
    """Convert raw signals to position weights, optionally vol-targeted."""
    weights = signals.copy()
    if target_vol is not None and returns is not None:
        vol = returns.rolling(lookback).std(ddof=1) * np.sqrt(252)
        vol = vol.reindex_like(weights).fillna(vol.median().median())
        scale = _safe_div(target_vol / np.sqrt(252), vol, fill=1.0)
        weights = weights.mul(scale, axis=0)
    return weights.clip(-1.0, 1.0)


def _safe_div(a, b, fill: float = 0.0):
    out = a / b.replace(0, np.nan)
    if isinstance(out, pd.Series):
        return out.fillna(fill)
    return np.where(b == 0, fill, out)


# ---------------------------------------------------------------------------
# Momentum / trend strategies
# ---------------------------------------------------------------------------

def dual_momentum_strategy(
    prices: pd.DataFrame, lookback: int = 126, benchmark: pd.Series | None = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Dual momentum: absolute momentum vs benchmark + relative momentum."""
    mom = momentum_factor(prices, lookback)
    if benchmark is not None:
        abs_mom = benchmark / benchmark.shift(lookback) - 1.0
        investable = mom.where(abs_mom > threshold, 0.0)
    else:
        investable = mom.where(mom > threshold, 0.0)
    # Rank within investable universe
    ranked = investable.rank(axis=1, pct=True, method="average")
    return ranked * np.sign(investable)


def trend_following_strategy(
    prices: pd.DataFrame, fast: int = 50, slow: int = 200,
    atr_period: int = 14, atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Trend following: price above/below slow MA with ADX filter."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    trend = (prices > slow_ma).astype(float) - (prices < slow_ma).astype(float)
    # Require fast > slow for long, fast < slow for short
    trend = np.where(fast_ma > slow_ma, trend, -trend)
    # ADX filter: only trade when trend is strong
    # Approximate ADX using price range
    adx_proxy = prices.rolling(atr_period).apply(
        lambda x: float(x.std() / x.mean()) if x.mean() != 0 else 0.0, raw=True
    )
    trend = np.where(adx_proxy > 0.01, trend, 0.0)
    return pd.DataFrame(trend, index=prices.index, columns=prices.columns)


def breakout_strategy(
    prices: pd.DataFrame, lookback: int = 20, atr_period: int = 14,
    atr_mult: float = 1.5,
) -> pd.DataFrame:
    """Donchian breakout: long on new highs, short on new lows."""
    high = prices.rolling(lookback).max()
    low = prices.rolling(lookback).min()
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[prices > high.shift(1)] = 1.0
    signal[prices < low.shift(1)] = -1.0
    # Volatility filter: only trade when ATR-normalized move is significant
    atr_ = atr(prices, prices, prices, atr_period)
    move = prices.pct_change().abs()
    signal = np.where(move > atr_mult * atr_ / prices, signal, 0.0)
    return pd.DataFrame(signal, index=prices.index, columns=prices.columns)


def supertrend_strategy(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    period: int = 10, mult: float = 3.0,
) -> pd.DataFrame:
    """Supertrend trend-following strategy."""
    st = supertrend(high, low, close, period, mult)
    direction = st["direction"]
    return direction


def ichimoku_strategy(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku cloud strategy: price above/below cloud + TK cross."""
    ich = ichimoku(high, low, close, tenkan, kijun, senkou_b)
    c = close
    above_cloud = (c > ich["senkou_a"]) & (c > ich["senkou_b"])
    below_cloud = (c < ich["senkou_a"]) & (c < ich["senkou_b"])
    tk_bull = ich["tenkan"] > ich["kijun"]
    tk_bear = ich["tenkan"] < ich["kijun"]
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    signal[above_cloud & tk_bull] = 1.0
    signal[below_cloud & tk_bear] = -1.0
    return signal


# ---------------------------------------------------------------------------
# Mean reversion strategies
# ---------------------------------------------------------------------------

def mean_reversion_strategy(
    prices: pd.DataFrame, lookback: int = 20, z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Mean reversion: z-score of price vs rolling mean."""
    sma_ = prices.rolling(lookback).mean()
    std_ = prices.rolling(lookback).std(ddof=1)
    z = _safe_div(prices - sma_, std_, fill=0.0)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[z < -z_threshold] = 1.0   # oversold -> long
    signal[z > z_threshold] = -1.0   # overbought -> short
    return signal


def pairs_trading_strategy(
    prices: pd.DataFrame, lookback: int = 60, entry_z: float = 1.5,
    exit_z: float = 0.5, pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Pairs trading on log-price spreads with entry/exit hysteresis.

    Without explicit ``pairs``, each asset is traded against the equal-weight
    basket of the universe as its hedge. With ``pairs``, the spread signal is
    reported on the first leg only (negate it for the second leg).
    """
    log_prices = np.log(prices.where(prices > 0))
    if pairs is None:
        spread = log_prices.sub(log_prices.mean(axis=1), axis=0)
    else:
        spread = pd.DataFrame(index=prices.index)
        for a, b in pairs:
            if a in log_prices.columns and b in log_prices.columns:
                spread[a] = log_prices[a] - log_prices[b]
    mean_ = spread.rolling(lookback).mean()
    std_ = spread.rolling(lookback).std(ddof=1)
    z = _safe_div(spread - mean_, std_, fill=0.0)
    raw = pd.DataFrame(np.nan, index=prices.index, columns=spread.columns)
    raw[z < -entry_z] = 1.0
    raw[z > entry_z] = -1.0
    raw[z.abs() < exit_z] = 0.0
    signal = raw.ffill().fillna(0.0)
    return signal.clip(-1.0, 1.0)


def rsi_mean_reversion_strategy(
    prices: pd.DataFrame, period: int = 14, oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """RSI mean reversion."""
    rsi_ = rsi(prices, period)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[rsi_ < oversold] = 1.0
    signal[rsi_ > overbought] = -1.0
    return signal


def bollinger_mean_reversion_strategy(
    prices: pd.DataFrame, period: int = 20, num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands mean reversion."""
    bb = bollinger_bands(prices, period, num_std)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[prices < bb["lower"]] = 1.0
    signal[prices > bb["upper"]] = -1.0
    # Exit when price returns to middle band
    signal[(prices > bb["mid"]) & (signal.shift(1) == 1.0)] = 0.0
    signal[(prices < bb["mid"]) & (signal.shift(1) == -1.0)] = 0.0
    return signal


def macd_cross_strategy(
    prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    """MACD cross strategy."""
    macd_df = macd(prices, fast, slow, signal)
    signal_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal_df[macd_df["macd"] > macd_df["signal"]] = 1.0
    signal_df[macd_df["macd"] < macd_df["signal"]] = -1.0
    return signal_df


# ---------------------------------------------------------------------------
# Multi-factor strategies
# ---------------------------------------------------------------------------

def multi_factor_strategy(
    prices: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
    book_value: pd.DataFrame, earnings: pd.DataFrame, sales: pd.DataFrame,
    cash_flow: pd.DataFrame, roe: pd.DataFrame, roa: pd.DataFrame,
    roic: pd.DataFrame, debt_to_equity: pd.DataFrame,
    current_ratio: pd.DataFrame, gross_margin: pd.DataFrame,
    operating_margin: pd.DataFrame, fcf_yield: pd.DataFrame,
    market_cap: pd.DataFrame, weights: dict | None = None,
) -> pd.DataFrame:
    """Multi-factor long-short strategy using composite alpha factor."""
    comp = composite_factor(
        prices=prices, returns=returns, volume=volume,
        book_value=book_value, earnings=earnings, sales=sales,
        cash_flow=cash_flow, roe=roe, roa=roa, roic=roic,
        debt_to_equity=debt_to_equity, current_ratio=current_ratio,
        gross_margin=gross_margin, operating_margin=operating_margin,
        fcf_yield=fcf_yield, market_cap=market_cap,
        weights=weights,
    )
    # Rank and normalize to [-1, 1]
    ranked = comp.rank(axis=1, pct=True, method="average")
    return 2.0 * ranked - 1.0


def momentum_quality_strategy(
    prices: pd.DataFrame, returns: pd.DataFrame,
    roe: pd.DataFrame, roa: pd.DataFrame, roic: pd.DataFrame,
    lookback: int = 126,
) -> pd.DataFrame:
    """Momentum filtered by quality (only long high-quality momentum)."""
    mom = composite_momentum_factor(prices)
    qual = (rank_cross_sectional(roe) + rank_cross_sectional(roa) +
            rank_cross_sectional(roic)) / 3.0
    # Long top momentum among high quality, short bottom momentum among low quality
    mom_rank = mom.rank(axis=1, pct=True)
    qual_rank = qual.rank(axis=1, pct=True)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[(mom_rank > 0.7) & (qual_rank > 0.5)] = 1.0
    signal[(mom_rank < 0.3) & (qual_rank < 0.5)] = -1.0
    return signal


def low_vol_momentum_strategy(
    prices: pd.DataFrame, returns: pd.DataFrame,
    lookback: int = 126, vol_lookback: int = 60,
) -> pd.DataFrame:
    """Low-volatility momentum: momentum among low-volatility assets."""
    mom = momentum_factor(prices, lookback)
    vol = volatility_factor(returns, vol_lookback)
    vol_rank = vol.rank(axis=1, pct=True)
    mom_rank = mom.rank(axis=1, pct=True)
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signal[(vol_rank < 0.4) & (mom_rank > 0.6)] = 1.0
    signal[(vol_rank > 0.6) & (mom_rank < 0.4)] = -1.0
    return signal


# ---------------------------------------------------------------------------
# Risk overlay strategies
# ---------------------------------------------------------------------------

def risk_parity_strategy(
    returns: pd.DataFrame, lookback: int = 60, target_vol: float = 0.10,
) -> pd.DataFrame:
    """Risk parity: equal risk contribution weights."""
    cov = returns.rolling(lookback).cov()
    vol = returns.rolling(lookback).std(ddof=1)
    inv_vol = 1.0 / vol.replace(0, np.nan)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    # Scale to target vol
    port_vol = (weights * returns.rolling(lookback).std(ddof=1)).sum(axis=1) * np.sqrt(252)
    scale = _safe_div(target_vol, port_vol, fill=1.0)
    return weights.mul(scale, axis=0).clip(-1.0, 1.0)


def minimum_variance_strategy(
    returns: pd.DataFrame, lookback: int = 60, target_vol: float = 0.10,
) -> pd.DataFrame:
    """Minimum variance portfolio (analytic approximation)."""
    cov = returns.rolling(lookback).cov()
    n = returns.shape[1]
    # Equal-weight starting point, then tilt by inverse variance
    vol = returns.rolling(lookback).std(ddof=1)
    inv_var = 1.0 / (vol ** 2).replace(0, np.nan)
    weights = inv_var.div(inv_var.sum(axis=1), axis=0)
    port_vol = np.sqrt((weights ** 2 * (vol ** 2)).sum(axis=1)) * np.sqrt(252)
    scale = _safe_div(target_vol, port_vol, fill=1.0)
    return weights.mul(scale, axis=0).clip(-1.0, 1.0)


def maximum_sharpe_strategy(
    returns: pd.DataFrame, lookback: int = 252, rf: float = 0.0,
    target_vol: float = 0.10,
) -> pd.DataFrame:
    """Maximum Sharpe ratio portfolio (sample mean-variance approximation)."""
    mean = returns.rolling(lookback).mean()
    cov = returns.rolling(lookback).cov()
    n = returns.shape[1]
    # Approximate optimal weights: proportional to expected return / variance
    vol = returns.rolling(lookback).std(ddof=1)
    scores = mean / (vol ** 2).replace(0, np.nan)
    scores = scores.clip(lower=0.0)
    weights = scores.div(scores.sum(axis=1), axis=0)
    port_vol = np.sqrt((weights ** 2 * (vol ** 2)).sum(axis=1)) * np.sqrt(252)
    scale = _safe_div(target_vol, port_vol, fill=1.0)
    return weights.mul(scale, axis=0).clip(-1.0, 1.0)


def black_litterman_strategy(
    returns: pd.DataFrame, market_caps: pd.DataFrame,
    views: pd.DataFrame | None = None, view_confidence: float = 0.5,
    tau: float = 0.05, risk_aversion: float = 3.0,
    lookback: int = 252, target_vol: float = 0.10,
) -> pd.DataFrame:
    """Black-Litterman blended expected returns (simplified)."""
    cov = returns.rolling(lookback).cov()
    pi = risk_aversion * cov.dot(market_caps.div(market_caps.sum(axis=1), axis=0))
    if views is not None:
        # Blend views with equilibrium returns
        blended = (1 - view_confidence) * pi + view_confidence * views
    else:
        blended = pi
    vol = returns.rolling(lookback).std(ddof=1)
    scores = blended / (vol ** 2).replace(0, np.nan)
    scores = scores.clip(lower=0.0)
    weights = scores.div(scores.sum(axis=1), axis=0)
    port_vol = np.sqrt((weights ** 2 * (vol ** 2)).sum(axis=1)) * np.sqrt(252)
    scale = _safe_div(target_vol, port_vol, fill=1.0)
    return weights.mul(scale, axis=0).clip(-1.0, 1.0)


def volatility_targeting_strategy(
    signals: pd.DataFrame, returns: pd.DataFrame,
    target_vol: float = 0.10, lookback: int = 60,
) -> pd.DataFrame:
    """Scale signals to target portfolio volatility."""
    port_ret = (signals * returns).sum(axis=1)
    port_vol = port_ret.rolling(lookback).std(ddof=1) * np.sqrt(252)
    scale = _safe_div(target_vol, port_vol, fill=1.0)
    return signals.mul(scale, axis=0).clip(-1.0, 1.0)


def drawdown_control_strategy(
    signals: pd.DataFrame, equity: pd.Series,
    max_dd: float = 0.10, lookback: int = 60,
) -> pd.DataFrame:
    """Reduce exposure when drawdown exceeds threshold."""
    dd = equity.rolling(lookback).apply(
        lambda x: float((x.cummax() - x).max() / x.cummax().iloc[-1]) if len(x) else 0.0,
        raw=True,
    )
    scale = np.where(dd > max_dd, 0.5, 1.0)
    return signals.mul(scale, axis=0)


def regime_switching_strategy(
    signals: pd.DataFrame, regime: pd.Series,
    regime_multipliers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Scale signals by market regime."""
    mult = regime_multipliers or {
        "bull": 1.0, "bear": 0.5, "neutral": 0.75, "crisis": 0.0,
    }
    out = signals.copy()
    for date in out.index:
        reg = regime.loc[date] if date in regime.index else "neutral"
        out.loc[date] *= mult.get(reg, 0.75)
    return out


def seasonality_strategy(
    prices: pd.DataFrame, month: int | None = None,
) -> pd.DataFrame:
    """Simple seasonality: long in strong historical months."""
    if month is None:
        month = pd.Timestamp.now().month
    # Use historical monthly returns to score current month
    monthly_ret = prices.resample("ME").last().pct_change()
    month_scores = monthly_ret.groupby(monthly_ret.index.month).mean()
    score = month_scores.get(month, 0.0)
    signal = pd.DataFrame(score, index=prices.index, columns=prices.columns)
    return signal.clip(-1.0, 1.0)


def earnings_momentum_strategy(
    earnings_surprise: pd.DataFrame, lookback: int = 60,
) -> pd.DataFrame:
    """Earnings momentum: rank by recent earnings surprises."""
    return rank_cross_sectional(earnings_surprise.rolling(lookback).mean()) * 2.0 - 1.0
