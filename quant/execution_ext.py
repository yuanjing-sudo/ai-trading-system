"""Execution scheduling, transaction-cost analysis (TCA), and market-microstructure
analytics.

Focus is *scheduling and measurement* - TWAP/VWAP/POV slicing, arrival and
interval benchmarks, implementation shortfall, spread/microprice estimators,
order-flow imbalance, and liquidity/impact approximations. Inputs are pandas
objects and are never mutated.

Example
-------
>>> from quant import execution_ext as ex
>>> ex.twap_schedule(10_000, 5).round(3).tolist()
[0.2, 0.2, 0.2, 0.2, 0.2]
>>> ex.execution_cost_bps(100.0, 100.05, "buy")
5.0...
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "bps_to_return", "return_to_bps", "notional_from_shares", "order_size_from_notional",
    "linear_schedule", "front_loaded_schedule", "back_loaded_schedule",
    "twap_schedule", "vwap_schedule", "pov_schedule", "participation_curve",
    "slice_order", "optimal_execution_horizon", "expected_cost",
    "arrival_price", "twap_from_bars", "bar_vwap", "interval_vwap",
    "execution_cost_bps", "implementation_shortfall", "opportunity_cost",
    "price_improvement", "vwap_error", "twap_error", "participation_rate",
    "fill_rate", "rolling_participation", "weighted_avg_price",
    "slippage_decomposition", "market_impact_estimate", "cost_curve",
    "mid_price", "quoted_spread", "relative_spread", "effective_spread",
    "realized_spread", "adverse_selection_cost", "microprice",
    "order_book_imbalance", "trade_imbalance", "depth_weighted_price",
    "corwin_schultz_spread", "abdi_ranaldo_spread", "tick_classification",
    "lee_ready_classification", "kyle_lambda", "amihud_impact",
    "fill_probability", "limit_fill_probability", "queue_depth",
    "tca_summary", "compare_execution", "execution_report", "volume_profile",
]


def bps_to_return(bps: float) -> float:
    """Convert basis points to a decimal return."""
    return bps / 1e4


def return_to_bps(ret: float) -> float:
    """Convert a decimal return to basis points."""
    return ret * 1e4


def notional_from_shares(shares: float, price: float) -> float:
    """Dollar notional of a share quantity at ``price``."""
    return float(shares) * float(price)


def order_size_from_notional(notional: float, price: float) -> float:
    """Shares required to reach a dollar notional at ``price``."""
    return float("nan") if price == 0 else float(notional) / float(price)


def linear_schedule(order_size: float, n_slices: int) -> pd.Series:
    """Equal-sized slicing of ``order_size`` across ``n_slices``."""
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1")
    return pd.Series(np.full(n_slices, float(order_size) / n_slices), name="size")


def front_loaded_schedule(order_size: float, n_slices: int, decay: float = 2.0) -> pd.Series:
    """Exponentially front-loaded slicing (trade faster early)."""
    w = np.exp(-float(decay) * np.arange(n_slices) / max(n_slices - 1, 1))
    w = w / w.sum()
    return pd.Series(w * float(order_size), name="size")


def back_loaded_schedule(order_size: float, n_slices: int, ramp: float = 2.0) -> pd.Series:
    """Exponentially back-loaded slicing (trade faster late)."""
    w = np.exp(float(ramp) * np.arange(n_slices) / max(n_slices - 1, 1))
    w = w / w.sum()
    return pd.Series(w * float(order_size), name="size")


def twap_schedule(order_size: float, n_slices: int) -> pd.Series:
    """Time-weighted average price schedule (flat slices over time)."""
    return linear_schedule(order_size, n_slices).rename("twap_size")


def vwap_schedule(order_size: float, volume_profile) -> pd.Series:
    """Volume-weighted schedule proportional to a forecast volume profile."""
    v = pd.Series(volume_profile).astype("float64")
    total = float(v.sum())
    if total <= 0:
        return pd.Series(np.full(len(v), float(order_size) / max(len(v), 1)), index=v.index, name="vwap_size")
    return (v / total * float(order_size)).rename("vwap_size")


def pov_schedule(order_size: float, volume_forecast, participation: float = 0.1) -> pd.Series:
    """Percent-of-volume schedule capped by ``participation`` of forecast volume."""
    v = pd.Series(volume_forecast).astype("float64")
    remaining = float(order_size)
    out = []
    for vol in v:
        take = min(remaining, float(participation) * float(vol))
        take = max(take, 0.0)
        out.append(take)
        remaining -= take
        if remaining <= 0:
            out.extend([0.0] * (len(v) - len(out)))
            break
    return pd.Series(out, index=v.index, name="pov_size")


def participation_curve(volume_profile, participation: float = 0.1) -> pd.Series:
    """Tradeable size per bar at a fixed participation rate."""
    v = pd.Series(volume_profile).astype("float64")
    return (v * float(participation)).rename("tradeable")


def slice_order(order_size: float, n_slices: int, ramp: str = "flat", decay: float = 2.0) -> pd.Series:
    """Dispatch to a slicing style: ``flat``, ``front``, or ``back``."""
    if ramp == "flat":
        return linear_schedule(order_size, n_slices)
    if ramp == "front":
        return front_loaded_schedule(order_size, n_slices, decay)
    if ramp == "back":
        return back_loaded_schedule(order_size, n_slices, decay)
    raise ValueError("ramp must be 'flat', 'front', or 'back'")


def optimal_execution_horizon(order_size: float, adv: float, participation: float = 0.1) -> float:
    """Days to complete ``order_size`` at ``participation`` of average daily volume."""
    cap = float(adv) * float(participation)
    return float("nan") if cap <= 0 else float(order_size) / cap


def expected_cost(participation: float, spread_bps: float, sigma: float,
                  horizon: float, impact_coeff: float = 0.1) -> float:
    """Heuristic expected cost in bps: half-spread + temporary impact."""
    half_spread = 0.5 * float(spread_bps)
    impact = float(impact_coeff) * float(sigma) * np.sqrt(max(participation, 0.0)) * 1e4
    return float(half_spread + impact)


def arrival_price(prices, index: int = 0) -> float:
    """Price at order arrival (decision) time."""
    s = pd.Series(prices).dropna()
    if s.empty:
        return float("nan")
    idx = int(np.clip(index, 0, len(s) - 1))
    return float(s.iloc[idx])


def twap_from_bars(bars: pd.DataFrame, price_col: str = "close") -> float:
    """Simple average of the price column (TWAP over the bars)."""
    s = bars[price_col].astype("float64").dropna()
    return float(s.mean()) if not s.empty else float("nan")


def bar_vwap(bars: pd.DataFrame, price_col: str = "close", volume_col: str = "volume") -> float:
    """Volume-weighted average price over bars."""
    p = bars[price_col].astype("float64")
    v = bars[volume_col].astype("float64")
    denom = float(v.sum())
    return float("nan") if denom == 0 else float((p * v).sum() / denom)


def interval_vwap(bars: pd.DataFrame, start: int = 0, end: int | None = None,
                  price_col: str = "close", volume_col: str = "volume") -> float:
    """VWAP over a bar interval ``[start, end)``."""
    sub = bars.iloc[start:end]
    return bar_vwap(sub, price_col, volume_col)


def execution_cost_bps(arrival_pr: float, exec_pr: float, side: str = "buy") -> float:
    """Signed execution cost in bps versus arrival price (positive = slippage)."""
    if arrival_pr == 0:
        return float("nan")
    inv = 1.0 if side == "buy" else -1.0
    return inv * (float(exec_pr) - float(arrival_pr)) / float(arrival_pr) * 1e4


def implementation_shortfall(decision_price: float, avg_exec_price: float, side: str,
                             size: float, unfilled: float = 0.0, close_price: float | None = None) -> float:
    """Implementation shortfall in dollars: executed slippage plus opportunity cost."""
    inv = 1.0 if side == "buy" else -1.0
    executed = inv * (float(avg_exec_price) - float(decision_price)) * float(size)
    held = 0.0
    if close_price is not None and unfilled:
        held = inv * (float(close_price) - float(decision_price)) * float(unfilled)
    return float(executed + held)


def opportunity_cost(decision_price: float, close_price: float, side: str, unfilled: float) -> float:
    """Cost of shares never filled, marked versus the decision price."""
    inv = 1.0 if side == "buy" else -1.0
    return float(inv * (float(close_price) - float(decision_price)) * float(unfilled))


def price_improvement(exec_price: float, quote_price: float, side: str = "buy") -> float:
    """Positive when executed better than the prevailing quote."""
    inv = 1.0 if side == "buy" else -1.0
    return float(inv * (float(quote_price) - float(exec_price)))


def vwap_error(exec_price: float, benchmark_vwap: float, side: str = "buy") -> float:
    """Bps of execution versus the VWAP benchmark."""
    return execution_cost_bps(benchmark_vwap, exec_price, side)


def twap_error(exec_price: float, benchmark_twap: float, side: str = "buy") -> float:
    """Bps of execution versus the TWAP benchmark."""
    return execution_cost_bps(benchmark_twap, exec_price, side)


def participation_rate(filled: float, market_volume: float) -> float:
    """Fraction of market volume traded."""
    return float("nan") if market_volume == 0 else float(filled) / float(market_volume)


def fill_rate(order_size: float, filled: float) -> float:
    """Fraction of the order that was filled."""
    return float("nan") if order_size == 0 else float(filled) / float(order_size)


def rolling_participation(fills, market_volume, window: int = 20) -> pd.Series:
    """Rolling participation rate of fills versus market volume."""
    f = pd.Series(fills).astype("float64")
    v = pd.Series(market_volume).astype("float64")
    num = f.rolling(window, min_periods=1).sum()
    den = v.rolling(window, min_periods=1).sum()
    return (num / den.replace(0.0, np.nan)).rename("participation")


def weighted_avg_price(prices, sizes) -> float:
    """Size-weighted average execution price."""
    p = pd.Series(prices).astype("float64")
    s = pd.Series(sizes).astype("float64")
    denom = float(s.sum())
    return float("nan") if denom == 0 else float((p * s).sum() / denom)


def slippage_decomposition(arrival_pr: float, exec_pr: float, interval_vwap: float,
                           side: str = "buy") -> dict:
    """Split slippage into delay (arrival->interval) and trading (interval->fill) parts."""
    delay = execution_cost_bps(arrival_pr, interval_vwap, side)
    trading = execution_cost_bps(interval_vwap, exec_pr, side)
    total = execution_cost_bps(arrival_pr, exec_pr, side)
    return {"delay_bps": delay, "trading_bps": trading, "total_bps": total}


def market_impact_estimate(order_notional: float, adv_notional: float, sigma: float,
                           eta: float = 0.1) -> float:
    """Square-root market-impact estimate in bps given participation and daily vol."""
    if adv_notional <= 0:
        return float("nan")
    participation = order_notional / adv_notional
    return float(eta * float(sigma) * np.sqrt(max(participation, 0.0)) * 1e4)


def cost_curve(order_sizes, adv: float, sigma: float, eta: float = 0.1) -> pd.DataFrame:
    """Impact cost (bps) and participation across a range of order sizes."""
    sizes = pd.Series(order_sizes).astype("float64")
    parts = sizes / float(adv)
    cost = float(eta) * float(sigma) * np.sqrt(parts.clip(lower=0.0)) * 1e4
    return pd.DataFrame({"size": sizes.to_numpy(), "participation": parts.to_numpy(),
                         "cost_bps": cost.to_numpy()})


def mid_price(bid: float, ask: float) -> float:
    """Midpoint of the best bid and offer."""
    return 0.5 * (float(bid) + float(ask))


def quoted_spread(bid: float, ask: float) -> float:
    """Absolute quoted spread."""
    return float(ask) - float(bid)


def relative_spread(bid: float, ask: float) -> float:
    """Quoted spread divided by mid (a decimal fraction)."""
    mid = mid_price(bid, ask)
    return float("nan") if mid == 0 else quoted_spread(bid, ask) / mid


def effective_spread(trade_price: float, mid: float, side: str = "buy") -> float:
    """Effective spread: ``2 * |trade - mid|`` signed cost."""
    inv = 1.0 if side == "buy" else -1.0
    return float(2.0 * inv * (float(trade_price) - float(mid)))


def realized_spread(trade_price: float, mid_at_exec: float, mid_later: float, side: str = "buy") -> float:
    """Realized spread capturing adverse selection (uses a later mid)."""
    inv = 1.0 if side == "buy" else -1.0
    return float(2.0 * inv * (float(mid_later) - float(mid_at_exec)))


def adverse_selection_cost(trade_price: float, mid_at_exec: float, mid_later: float, side: str = "buy") -> float:
    """Effective minus realized spread (the adverse-selection component)."""
    return effective_spread(trade_price, mid_at_exec, side) - realized_spread(trade_price, mid_at_exec, mid_later, side)


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Size-weighted microprice leaning toward the heavier book side."""
    total = float(bid_size) + float(ask_size)
    if total == 0:
        return mid_price(bid, ask)
    return (float(ask) * float(bid_size) + float(bid) * float(ask_size)) / total


def order_book_imbalance(bid_size: float, ask_size: float) -> float:
    """Book imbalance in [-1, 1]; positive = bid-heavy."""
    total = float(bid_size) + float(ask_size)
    if total == 0:
        return 0.0
    return (float(bid_size) - float(ask_size)) / total


def trade_imbalance(buy_volume: float, sell_volume: float) -> float:
    """Signed trade imbalance in [-1, 1]."""
    total = float(buy_volume) + float(sell_volume)
    if total == 0:
        return 0.0
    return (float(buy_volume) - float(sell_volume)) / total


def depth_weighted_price(bids, asks) -> float:
    """Depth-weighted mid using ``[(price, size), ...]`` ladders for bids and asks."""
    def _w(levels):
        num = sum(p * s for p, s in levels)
        den = sum(s for _, s in levels)
        return num / den if den else np.nan
    b, a = _w(bids), _w(asks)
    if np.isnan(b) or np.isnan(a):
        return float(b if not np.isnan(b) else a)
    return 0.5 * (b + a)


def corwin_schultz_spread(high, low) -> pd.Series:
    """Corwin-Schultz high-low bid-ask spread estimator (decimal)."""
    h = pd.Series(high).astype("float64")
    l = pd.Series(low).astype("float64")
    beta = (np.log(h / l) ** 2 + np.log(h / l).shift(1) ** 2)
    hl2 = np.log((h.rolling(2).max()) / (l.rolling(2).min())) ** 2
    gamma = hl2 - beta
    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma.clip(lower=0.0) / denom)
    spread = 2.0 * (np.exp(alpha.clip(lower=0.0)) - 1.0) / (1.0 + np.exp(alpha.clip(lower=0.0)))
    return spread.clip(lower=0.0).rename("cs_spread")


def abdi_ranaldo_spread(open_, high, low, close) -> pd.Series:
    """Abdi-Ranaldo close-high-low bid-ask spread estimator (decimal)."""
    h = pd.Series(high).astype("float64")
    l = pd.Series(low).astype("float64")
    c = pd.Series(close).astype("float64")
    eta = (np.log(h / c) + np.log(l / c)) / 2.0
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    var_eta = (eta ** 2).rolling(2, min_periods=1).mean()
    spread = 2.0 * np.sqrt((4.0 * var_eta).clip(lower=0.0) / denom)
    return spread.clip(lower=0.0).rename("ar_spread")


def tick_classification(prices) -> pd.Series:
    """Uptick/downtick classification: 1 buy, -1 sell, 0 unchanged."""
    p = pd.Series(prices).astype("float64")
    diff = p.diff()
    return pd.Series(np.sign(diff.to_numpy()), index=p.index).fillna(0.0).rename("side")


def lee_ready_classification(trade_price, bid, ask) -> pd.Series:
    """Lee-Ready trade classification using the quote midpoint rule."""
    tp = pd.Series(trade_price).astype("float64")
    mid = (pd.Series(bid).astype("float64") + pd.Series(ask).astype("float64")) / 2.0
    side = np.where(tp > mid, 1.0, np.where(tp < mid, -1.0, 0.0))
    s = pd.Series(side, index=tp.index)
    zero = s == 0.0
    s[zero] = tick_classification(tp)[zero]
    return s.rename("side")


def kyle_lambda(prices, volume, window: int = 50) -> pd.Series:
    """Rolling Kyle's lambda: price impact per unit signed order flow."""
    p = pd.Series(prices).astype("float64")
    v = pd.Series(volume).astype("float64")
    dp = p.diff()
    signed = v * tick_classification(p)
    cov = dp.rolling(window, min_periods=max(5, window // 3)).cov(signed)
    var = signed.rolling(window, min_periods=max(5, window // 3)).var()
    return (cov / var.replace(0.0, np.nan)).rename("kyle_lambda")


def amihud_impact(returns, dollar_volume) -> pd.Series:
    """Amihud-style illiquidity: absolute return per unit dollar volume."""
    r = pd.Series(returns).astype("float64")
    dv = pd.Series(dollar_volume).astype("float64")
    return (r.abs() / dv.replace(0.0, np.nan)).rename("amihud")


def fill_probability(distance: float, spread: float, volatility: float, horizon: float) -> float:
    """Heuristic probability a passive order at ``distance`` fills within ``horizon``."""
    if volatility <= 0 or horizon <= 0:
        return float("nan")
    reach = volatility * math.sqrt(horizon)
    x = (float(distance) - 0.5 * float(spread)) / reach
    return float(np.clip(1.0 - _norm_cdf(x), 0.0, 1.0))


def limit_fill_probability(market_orders_per_bar: float, ahead_of_queue: float,
                           bars: float) -> float:
    """Approximate fill probability for a limit order given queue depth ahead."""
    if bars <= 0:
        return float("nan")
    effective = max(float(ahead_of_queue), 0.0)
    rate = float(market_orders_per_bar) * float(bars)
    return float(np.clip(1.0 - math.exp(-rate / (effective + 1.0)), 0.0, 1.0))


def queue_depth(bid_size: float, ask_size: float) -> dict:
    """Summary of displayed depth on both sides."""
    return {"bid_size": float(bid_size), "ask_size": float(ask_size),
            "total": float(bid_size) + float(ask_size),
            "imbalance": order_book_imbalance(bid_size, ask_size)}


def tca_summary(trades: pd.DataFrame, arrival_col: str = "arrival_price",
                exec_col: str = "exec_price", size_col: str = "size",
                vwap_col: str | None = "vwap") -> pd.DataFrame:
    """Per-trade VWAP/arrival slippage (bps) and signed dollar cost."""
    required = {arrival_col, exec_col, size_col}
    if not required <= set(trades.columns):
        raise KeyError(f"trades must contain {sorted(required)}")
    out = pd.DataFrame(index=trades.index)
    sign = trades["side"].map({"buy": 1.0, "sell": -1.0}) if "side" in trades.columns else 1.0
    out["slippage_bps"] = sign * (trades[exec_col] - trades[arrival_col]) / trades[arrival_col] * 1e4
    out["cost_dollars"] = sign * (trades[exec_col] - trades[arrival_col]) * trades[size_col]
    if vwap_col and vwap_col in trades.columns:
        out["vwap_slippage_bps"] = sign * (trades[exec_col] - trades[vwap_col]) / trades[vwap_col] * 1e4
    out["notional"] = trades[size_col] * trades[exec_col]
    return out


def compare_execution(results: dict) -> pd.DataFrame:
    """Rank an {algo: trades-DataFrame} mapping by mean slippage and total cost."""
    rows = {}
    for name, trades in results.items():
        summary = tca_summary(trades)
        rows[name] = {
            "mean_slippage_bps": float(summary["slippage_bps"].mean()),
            "total_cost_dollars": float(summary["cost_dollars"].sum()),
            "trades": int(len(summary)),
        }
    return pd.DataFrame(rows).T.sort_values("mean_slippage_bps")


def execution_report(trades: pd.DataFrame) -> str:
    """Markdown TCA report from a trades table."""
    summary = tca_summary(trades)
    lines = ["| metric | value |", "| --- | --- |",
             f"| trades | {len(summary)} |",
             f"| mean slippage (bps) | {summary['slippage_bps'].mean():.3f} |",
             f"| median slippage (bps) | {summary['slippage_bps'].median():.3f} |",
             f"| total cost ($) | {summary['cost_dollars'].sum():.2f} |",
             f"| total notional ($) | {summary['notional'].sum():.2f} |"]
    return "\n".join(lines)


def volume_profile(volume, n_buckets: int = 10) -> pd.Series:
    """Aggregate intraday/bar volume into ``n_buckets`` normalized buckets."""
    v = pd.Series(volume).astype("float64")
    if v.empty:
        return pd.Series(dtype="float64")
    buckets = np.array_split(v.to_numpy(), n_buckets)
    sums = np.array([b.sum() for b in buckets], dtype="float64")
    total = sums.sum()
    weights = sums / total if total > 0 else sums
    return pd.Series(weights, name="volume_weight")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
