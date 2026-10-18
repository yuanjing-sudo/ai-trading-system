"""Universe screening and scanning utilities.

A *universe* is a ``dict[str, pandas.DataFrame]`` mapping a symbol to an OHLCV
frame (columns may include ``open``, ``high``, ``low``, ``close``, ``volume``)
ordered by a DatetimeIndex. Every function takes the universe as its first
argument and returns aligned pandas objects; inputs are never mutated.

Example
-------
>>> from quant.screener_ext import screener_table, screen_top_n
>>> table = screener_table(universe)
>>> screen_top_n(universe, n=5, metric="sharpe")
['AAPL', 'MSFT', ...]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant import strategies_ext
from quant.indicators import atr, rsi, sma

__all__ = [
    "add_asset", "remove_asset", "universe_symbols", "universe_bars",
    "synchronize_universe", "universe_close_matrix", "universe_returns_matrix",
    "universe_volume_matrix", "universe_dollar_volume", "universe_summary",
    "filter_universe", "filter_by_last_return", "filter_trending",
    "screen_by_return", "screen_by_volatility", "screen_by_sharpe",
    "screen_by_trend", "screen_by_rsi", "screen_by_momentum",
    "screen_by_atr_pct", "screen_by_volume_spike", "screen_by_new_high",
    "screen_by_new_low", "screen_by_drawdown", "screen_by_gap",
    "screen_by_relative_strength", "screen_crossovers", "screen_breakouts",
    "screen_reversion_candidates", "screen_volatility_contraction",
    "screen_squeeze", "rank_universe", "composite_score", "screen_top_n",
    "screen_bottom_n", "screener_table", "screen_summary", "format_screen",
    "screen_by_strategy", "strategy_scan", "latest_signals", "active_signals",
    "scan_all", "screen_report",
]

_METRICS = {
    "return": lambda df, lb: _last_return(df, lb),
    "volatility": lambda df, lb: _last_volatility(df, lb),
    "sharpe": lambda df, lb: _last_sharpe(df, lb),
    "momentum": lambda df, lb: _last_momentum(df, lb),
    "drawdown": lambda df, lb: _last_drawdown(df, lb),
    "rsi": lambda df, lb: _last_rsi(df),
    "atr_pct": lambda df, lb: _last_atr_pct(df),
}


def _close(df: pd.DataFrame) -> pd.Series:
    col = "close" if "close" in df.columns else df.columns[0]
    return df[col].astype("float64")


def _log_ret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _last_return(df: pd.DataFrame, lookback: int = 60) -> float:
    close = _close(df).dropna()
    if len(close) <= lookback:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)


def _last_volatility(df: pd.DataFrame, lookback: int = 60, annualize: bool = True) -> float:
    ret = _log_ret(_close(df)).dropna()
    if len(ret) < 2:
        return np.nan
    vol = float(ret.tail(lookback).std())
    return vol * np.sqrt(252.0) if annualize else vol


def _last_sharpe(df: pd.DataFrame, lookback: int = 252, rf: float = 0.0) -> float:
    ret = _log_ret(_close(df)).dropna().tail(lookback)
    if ret.std() == 0 or len(ret) < 2:
        return np.nan
    excess = ret.mean() - rf / 252.0
    return float(excess / ret.std() * np.sqrt(252.0))


def _last_momentum(df: pd.DataFrame, lookback: int = 126, skip: int = 21) -> float:
    close = _close(df).dropna()
    if len(close) <= lookback:
        return np.nan
    start = close.iloc[-lookback - 1]
    end = close.iloc[-skip - 1] if skip < len(close) else close.iloc[-1]
    return float(end / start - 1.0)


def _last_drawdown(df: pd.DataFrame, lookback: int = 252) -> float:
    close = _close(df).dropna().tail(lookback)
    if close.empty:
        return np.nan
    return float(close.iloc[-1] / close.cummax().iloc[-1] - 1.0)


def _last_rsi(df: pd.DataFrame, period: int = 14) -> float:
    vals = rsi(_close(df), period=period).dropna()
    return float(vals.iloc[-1]) if not vals.empty else np.nan


def _last_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    if not {"high", "low", "close"} <= set(df.columns):
        return np.nan
    a = atr(df["high"], df["low"], df["close"], period=period).dropna()
    if a.empty:
        return np.nan
    return float(a.iloc[-1] / _close(df).iloc[-1])


def add_asset(universe: dict, symbol: str, df: pd.DataFrame) -> dict:
    """Return a new universe with ``symbol`` mapped to a copy of ``df``."""
    out = dict(universe)
    out[symbol] = df.copy()
    return out


def remove_asset(universe: dict, symbol: str) -> dict:
    """Return a new universe without ``symbol``."""
    out = dict(universe)
    out.pop(symbol, None)
    return out


def universe_symbols(universe: dict) -> list[str]:
    """Sorted symbol list."""
    return sorted(universe)


def universe_bars(universe: dict) -> pd.Series:
    """Number of rows per symbol."""
    return pd.Series({s: len(df) for s, df in universe.items()}, dtype="int64").sort_index()


def synchronize_universe(universe: dict) -> dict:
    """Restrict every frame to the intersection of all indices."""
    if not universe:
        return {}
    common = None
    for df in universe.values():
        idx = df.index
        common = idx if common is None else common.intersection(idx)
    return {s: df.loc[common].sort_index() for s, df in universe.items()}


def universe_close_matrix(universe: dict) -> pd.DataFrame:
    """Wide close-price matrix, columns = symbols."""
    return pd.DataFrame({s: _close(df) for s, df in universe.items()}).sort_index()


def universe_returns_matrix(universe: dict, log: bool = False) -> pd.DataFrame:
    """Wide return matrix, columns = symbols."""
    close = universe_close_matrix(universe)
    return np.log(close).diff() if log else close.pct_change(fill_method=None)


def universe_volume_matrix(universe: dict) -> pd.DataFrame:
    """Wide volume matrix (NaN when a symbol lacks volume)."""
    out = {}
    for s, df in universe.items():
        out[s] = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(np.nan, index=df.index)
    return pd.DataFrame(out).sort_index()


def universe_dollar_volume(universe: dict) -> pd.DataFrame:
    """Wide dollar-volume matrix (close * volume)."""
    close = universe_close_matrix(universe)
    vol = universe_volume_matrix(universe)
    return (close * vol).sort_index()


def universe_summary(universe: dict) -> pd.DataFrame:
    """Per-symbol first/last date, row count, and last close."""
    rows = []
    for s, df in universe.items():
        close = _close(df)
        rows.append({
            "symbol": s,
            "start": df.index.min(),
            "end": df.index.max(),
            "bars": len(df),
            "last_close": float(close.iloc[-1]) if len(close) else np.nan,
        })
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def filter_universe(universe: dict, min_bars: int = 100, min_dollar_volume: float | None = None,
                    max_nan_frac: float = 0.1, drop_zero_volume: bool = False) -> dict:
    """Keep symbols passing bar-count, liquidity, NaN, and volume checks."""
    out = {}
    for s, df in universe.items():
        close = _close(df)
        if len(df) < min_bars:
            continue
        if close.isna().mean() > max_nan_frac:
            continue
        if min_dollar_volume is not None:
            dv = (close * (df["volume"].astype("float64") if "volume" in df.columns else np.nan)).dropna()
            if dv.empty or float(dv.tail(min(min_bars, len(dv))).mean()) < min_dollar_volume:
                continue
        if drop_zero_volume and "volume" in df.columns and (df["volume"] == 0).all():
            continue
        out[s] = df
    return out


def filter_by_last_return(universe: dict, min_return: float = 0.0, lookback: int = 60) -> dict:
    """Keep symbols whose ``lookback`` return exceeds ``min_return``."""
    return {s: df for s, df in universe.items() if _last_return(df, lookback) >= min_return}


def filter_trending(universe: dict, fast: int = 20, slow: int = 50, up: bool = True) -> dict:
    """Keep symbols trading above (``up``) or below their slow SMA."""
    out = {}
    for s, df in universe.items():
        close = _close(df)
        if len(close) < slow:
            continue
        f = sma(close, fast).iloc[-1]
        sl = sma(close, slow).iloc[-1]
        if np.isnan(f) or np.isnan(sl):
            continue
        if (f > sl) == up:
            out[s] = df
    return out


def _apply(universe: dict, fn) -> pd.Series:
    vals = {s: fn(df) for s, df in universe.items()}
    return pd.Series(vals, dtype="float64").sort_index()


def screen_by_return(universe: dict, lookback: int = 60) -> pd.Series:
    """Latest ``lookback`` simple return per symbol."""
    return _apply(universe, lambda df: _last_return(df, lookback))


def screen_by_volatility(universe: dict, lookback: int = 60, annualize: bool = True) -> pd.Series:
    """Latest realized volatility per symbol."""
    return _apply(universe, lambda df: _last_volatility(df, lookback, annualize))


def screen_by_sharpe(universe: dict, lookback: int = 252, rf: float = 0.0) -> pd.Series:
    """Latest rolling Sharpe ratio per symbol."""
    return _apply(universe, lambda df: _last_sharpe(df, lookback, rf))


def screen_by_trend(universe: dict, fast: int = 20, slow: int = 50) -> pd.Series:
    """Percent distance of price above the slow SMA per symbol."""
    def f(df):
        close = _close(df)
        if len(close) < slow:
            return np.nan
        sl = sma(close, slow).iloc[-1]
        return np.nan if np.isnan(sl) or sl == 0 else float(close.iloc[-1] / sl - 1.0)
    return _apply(universe, f)


def screen_by_rsi(universe: dict, period: int = 14) -> pd.Series:
    """Latest RSI per symbol."""
    return _apply(universe, lambda df: _last_rsi(df, period))


def screen_by_momentum(universe: dict, lookback: int = 126, skip: int = 21) -> pd.Series:
    """Latest momentum (lookback return skipping the most recent ``skip`` bars)."""
    return _apply(universe, lambda df: _last_momentum(df, lookback, skip))


def screen_by_atr_pct(universe: dict, period: int = 14) -> pd.Series:
    """Latest ATR as a fraction of price per symbol."""
    return _apply(universe, lambda df: _last_atr_pct(df, period))


def screen_by_volume_spike(universe: dict, lookback: int = 20) -> pd.Series:
    """Latest volume divided by its trailing average per symbol."""
    def f(df):
        if "volume" not in df.columns:
            return np.nan
        vol = df["volume"].astype("float64")
        avg = vol.rolling(lookback, min_periods=max(2, lookback // 2)).mean().iloc[-1]
        return np.nan if not np.isfinite(avg) or avg == 0 else float(vol.iloc[-1] / avg)
    return _apply(universe, f)


def screen_by_new_high(universe: dict, lookback: int = 252) -> pd.Series:
    """1.0 when the latest close equals the ``lookback`` high, else 0.0."""
    def f(df):
        close = _close(df)
        if len(close) < 2:
            return np.nan
        return float(close.iloc[-1] >= close.tail(lookback).max())
    return _apply(universe, f)


def screen_by_new_low(universe: dict, lookback: int = 252) -> pd.Series:
    """1.0 when the latest close equals the ``lookback`` low, else 0.0."""
    def f(df):
        close = _close(df)
        if len(close) < 2:
            return np.nan
        return float(close.iloc[-1] <= close.tail(lookback).min())
    return _apply(universe, f)


def screen_by_drawdown(universe: dict, lookback: int = 252) -> pd.Series:
    """Current drawdown from the trailing peak per symbol."""
    return _apply(universe, lambda df: _last_drawdown(df, lookback))


def screen_by_gap(universe: dict) -> pd.Series:
    """Latest open-to-previous-close gap per symbol."""
    def f(df):
        if not {"open", "close"} <= set(df.columns) or len(df) < 2:
            return np.nan
        prev = df["close"].iloc[-2]
        return np.nan if prev == 0 else float(df["open"].iloc[-1] / prev - 1.0)
    return _apply(universe, f)


def screen_by_relative_strength(universe: dict, benchmark: pd.DataFrame, lookback: int = 126) -> pd.Series:
    """Excess return of each symbol versus ``benchmark`` over ``lookback``."""
    bench = _last_return(benchmark, lookback)
    return screen_by_return(universe, lookback) - bench


def screen_crossovers(universe: dict, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Fast/slow SMA crossover flags: ``golden`` and ``death`` booleans per symbol."""
    rows = {}
    for s, df in universe.items():
        close = _close(df)
        if len(close) < slow + 1:
            continue
        f = sma(close, fast)
        sl = sma(close, slow)
        above = (f > sl)
        rows[s] = {
            "golden": bool(above.iloc[-1] and not above.iloc[-2]),
            "death": bool((not above.iloc[-1]) and above.iloc[-2]),
        }
    if not rows:
        return pd.DataFrame(columns=["golden", "death"])
    return pd.DataFrame(rows).T.sort_index().astype(bool)


def screen_breakouts(universe: dict, period: int = 20) -> pd.Series:
    """1.0 when the latest close exceeds the prior ``period`` high, else 0.0."""
    def f(df):
        close = _close(df)
        if len(close) < period + 1:
            return np.nan
        prior_high = close.iloc[-period - 1:-1].max()
        return float(close.iloc[-1] > prior_high)
    return _apply(universe, f)


def screen_reversion_candidates(universe: dict, rsi_period: int = 14, rsi_low: float = 30.0) -> pd.Series:
    """Symbols with latest RSI below ``rsi_low`` (sorted ascending)."""
    r = screen_by_rsi(universe, rsi_period)
    return r[r < rsi_low].sort_values()


def screen_volatility_contraction(universe: dict, short: int = 10, long: int = 60) -> pd.Series:
    """Ratio of short- to long-window volatility; below 1 means contracting."""
    def f(df):
        ret = _log_ret(_close(df)).dropna()
        if len(ret) < long:
            return np.nan
        lo = ret.tail(long).std()
        hi = ret.tail(short).std()
        return np.nan if lo == 0 else float(hi / lo)
    return _apply(universe, f)


def screen_squeeze(universe: dict, period: int = 20, mult: float = 2.0) -> pd.Series:
    """1.0 when the latest Bollinger band sits inside the Keltner channel."""
    def f(df):
        close = _close(df)
        if len(close) < period or not {"high", "low"} <= set(df.columns):
            return np.nan
        mid = close.rolling(period).mean()
        sd = close.rolling(period).std()
        a = atr(df["high"], df["low"], close, period=period)
        bb_upper = mid.iloc[-1] + mult * sd.iloc[-1]
        bb_lower = mid.iloc[-1] - mult * sd.iloc[-1]
        kc_upper = mid.iloc[-1] + 1.5 * a.iloc[-1]
        kc_lower = mid.iloc[-1] - 1.5 * a.iloc[-1]
        return float(bb_upper < kc_upper and bb_lower > kc_lower)
    return _apply(universe, f)


def rank_universe(universe: dict, metric: str = "sharpe", lookback: int = 252,
                  ascending: bool = True) -> pd.Series:
    """Percentile rank (0-1) of symbols by ``metric``; by default larger values rank higher."""
    if metric not in _METRICS:
        raise KeyError(f"Unknown metric '{metric}'. Available: {', '.join(sorted(_METRICS))}")
    vals = _apply(universe, lambda df: _METRICS[metric](df, lookback))
    return vals.rank(ascending=ascending, pct=True)


def composite_score(universe: dict, metrics: list[str] | None = None,
                    lookback: int = 252, weights: dict | None = None) -> pd.DataFrame:
    """Z-scored metric table plus a weighted ``composite`` column per symbol."""
    metrics = metrics or ["sharpe", "momentum", "return"]
    for m in metrics:
        if m not in _METRICS:
            raise KeyError(f"Unknown metric '{m}'. Available: {', '.join(sorted(_METRICS))}")
    w = weights or {}
    cols = {}
    for m in metrics:
        raw = _apply(universe, lambda df, m=m: _METRICS[m](df, lookback))
        sd = raw.std()
        cols[m] = (raw - raw.mean()) / sd if sd not in (0, np.nan) and np.isfinite(sd) else raw * 0.0
    table = pd.DataFrame(cols).sort_index()
    total_w = sum(w.get(m, 1.0) for m in metrics)
    table["composite"] = sum(table[m] * w.get(m, 1.0) for m in metrics) / (total_w or 1.0)
    return table.sort_values("composite", ascending=False)


def screen_top_n(universe: dict, n: int = 10, metric: str = "sharpe", lookback: int = 252) -> list[str]:
    """Symbols with the highest value of ``metric``."""
    vals = _apply(universe, lambda df: _METRICS[metric](df, lookback)).dropna()
    return vals.sort_values(ascending=False).head(n).index.tolist()


def screen_bottom_n(universe: dict, n: int = 10, metric: str = "sharpe", lookback: int = 252) -> list[str]:
    """Symbols with the lowest value of ``metric``."""
    vals = _apply(universe, lambda df: _METRICS[metric](df, lookback)).dropna()
    return vals.sort_values(ascending=True).head(n).index.tolist()


def screener_table(universe: dict, metrics: list[str] | None = None,
                   lookback: int = 252) -> pd.DataFrame:
    """Master table: rows = symbols, columns = requested metrics."""
    metrics = metrics or ["return", "volatility", "sharpe", "momentum", "drawdown", "rsi", "atr_pct"]
    cols = {}
    for m in metrics:
        if m not in _METRICS:
            raise KeyError(f"Unknown metric '{m}'. Available: {', '.join(sorted(_METRICS))}")
        cols[m] = _apply(universe, lambda df, m=m: _METRICS[m](df, lookback))
    return pd.DataFrame(cols).sort_index()


def screen_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Count/mean/std/min/max per column of a screener table."""
    return table.describe().T


def format_screen(table: pd.DataFrame, top_n: int = 10, sort_by: str | None = None,
                  ascending: bool = False) -> pd.DataFrame:
    """Sorted, top-``n`` slice of a screener table."""
    if sort_by is None:
        sort_by = table.columns[0]
    if sort_by not in table.columns:
        raise KeyError(f"Column '{sort_by}' not in table")
    return table.sort_values(sort_by, ascending=ascending).head(top_n)


def screen_by_strategy(universe: dict, name: str, min_bars: int = 0,
                       side: str = "long") -> pd.Series:
    """Count of bars each symbol is long (or short) under strategy ``name``."""
    func = strategies_ext.STRATEGIES.get(name)
    if func is None:
        raise KeyError(f"Unknown strategy '{name}'. Try list_strategies().")
    out = {}
    for s, df in universe.items():
        try:
            pos = strategies_ext.strategy_positions(name, df)
        except Exception:
            out[s] = np.nan
            continue
        active = pos > 0 if side == "long" else pos < 0
        count = int(active.sum())
        out[s] = np.nan if count < min_bars else float(count)
    return pd.Series(out, dtype="float64").sort_index()


def strategy_scan(universe: dict, names: list[str] | None = None) -> pd.DataFrame:
    """Latest position per symbol (rows) and strategy (columns)."""
    names = names or strategies_ext.list_strategies()
    table = {}
    for name in names:
        if name not in strategies_ext.STRATEGIES:
            raise KeyError(f"Unknown strategy '{name}'. Try list_strategies().")
        col = {}
        for s, df in universe.items():
            try:
                pos = strategies_ext.strategy_positions(name, df)
                col[s] = float(pos.iloc[-1]) if len(pos) else np.nan
            except Exception:
                col[s] = np.nan
        table[name] = col
    return pd.DataFrame(table).sort_index()


def latest_signals(universe: dict, strategies: list[str] | None = None) -> pd.DataFrame:
    """Symbols (rows) with a tri-state latest signal per strategy: 1 long, -1 short, 0 flat."""
    scan = strategy_scan(universe, strategies)
    return scan.map(lambda v: np.sign(v) if np.isfinite(v) else np.nan)


def active_signals(universe: dict, strategies: list[str] | None = None) -> dict:
    """Map each symbol to the strategies currently signalling long or short."""
    scan = latest_signals(universe, strategies)
    out = {s: [] for s in scan.index}
    for name in scan.columns:
        for s, v in scan[name].items():
            if np.isfinite(v) and v != 0:
                out[s].append((name, "long" if v > 0 else "short"))
    return out


def scan_all(universe: dict, metrics: list[str] | None = None, lookback: int = 252,
             strategies: list[str] | None = None) -> pd.DataFrame:
    """Metrics table joined with latest tri-state strategy signals."""
    table = screener_table(universe, metrics, lookback)
    signals = latest_signals(universe, strategies)
    signals.columns = [f"sig_{c}" for c in signals.columns]
    return table.join(signals, how="left")


def screen_report(universe: dict, metrics: list[str] | None = None, lookback: int = 252) -> str:
    """Markdown report of the screener table."""
    table = screener_table(universe, metrics, lookback)
    lines = ["| symbol | " + " | ".join(table.columns) + " |",
             "|" + "---|" * (len(table.columns) + 1)]
    for s, row in table.iterrows():
        cells = [s] + [f"{v:.4f}" if np.isfinite(v) else "n/a" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)