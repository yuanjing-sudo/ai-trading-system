"""Performance, risk, and trading metrics for return series and trade lists.

All functions take pandas Series of per-period returns (or per-trade PnL) and
return scalars, Series, or DataFrames aligned to the input index. The number of
periods per year is configurable (``periods_per_year``, default 252) so the same
module works for daily, hourly, or minute bars. Nothing here looks ahead or
mutates inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series
from quant.statistics_ext import chi2_sf, pearson

__all__ = [
    # return metrics
    "total_return", "cagr", "annualized_return", "annualized_volatility",
    "average_return", "median_return", "return_skew", "return_kurtosis",
    # risk-adjusted metrics
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "omega_ratio",
    "gain_to_pain_ratio", "ulcer_index", "martin_ratio", "tail_ratio",
    "common_sense_ratio", "downside_deviation", "downside_ratio",
    "recovery_factor", "risk_return_ratio", "sterling_ratio",
    # drawdown metrics
    "equity_curve", "drawdown_series", "max_drawdown", "average_drawdown",
    "max_drawdown_duration", "drawdown_duration_series", "underwater_series",
    "pain_index", "drawdown_at_risk",
    # value at risk
    "var_historical", "cvar_historical", "var_parametric",
    "cvar_parametric", "modified_var", "kupiec_pof_test",
    # trade metrics
    "win_rate", "profit_factor", "payoff_ratio", "expectancy",
    "average_win", "average_loss", "largest_win", "largest_loss",
    "max_consecutive_wins", "max_consecutive_losses", "kelly_criterion",
    "profit_per_trade", "trade_count",
    # position metrics
    "position_pnl", "position_pnl_curve", "exposure", "turnover",
    "time_in_market", "trades_from_positions", "average_holding_period",
    # benchmark-relative
    "beta", "alpha", "tracking_error", "information_ratio", "treynor_ratio",
    "up_capture", "down_capture", "capture_ratio", "correlation_to_benchmark",
    "benchmark_r_squared", "excess_returns",
    # rolling metrics
    "rolling_volatility", "rolling_sharpe", "rolling_sortino",
    "rolling_max_drawdown", "rolling_calmar", "rolling_beta", "rolling_alpha",
    "rolling_information_ratio",
    # aggregation / reporting
    "monthly_returns_table", "yearly_returns", "day_of_week_returns",
    "strategy_metrics", "compare_strategies",
]


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _clean(returns) -> pd.Series:
    return _as_series(returns).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()


def _max_streak(flags: np.ndarray, value: bool = True) -> int:
    best = cur = 0
    for f in flags:
        if bool(f) == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# ---------------------------------------------------------------------------
# Return metrics
# ---------------------------------------------------------------------------

def total_return(returns) -> float:
    """Compounded total return over the sample."""
    r = _clean(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def cagr(returns, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from per-period returns."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    years = len(r) / float(periods_per_year)
    total = (1.0 + r).prod()
    if total <= 0 or years <= 0:
        return np.nan
    return float(total ** (1.0 / years) - 1.0)


def annualized_return(returns, periods_per_year: int = 252) -> float:
    """Arithmetic mean return scaled to a year."""
    r = _clean(returns)
    return float(r.mean() * periods_per_year) if len(r) else np.nan


def annualized_volatility(returns, periods_per_year: int = 252) -> float:
    """Standard deviation scaled to a year."""
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan


def average_return(returns) -> float:
    """Mean per-period return."""
    r = _clean(returns)
    return float(r.mean()) if len(r) else np.nan


def median_return(returns) -> float:
    """Median per-period return."""
    r = _clean(returns)
    return float(r.median()) if len(r) else np.nan


def return_skew(returns) -> float:
    """Skewness of per-period returns."""
    r = _clean(returns)
    return float(r.skew()) if len(r) > 2 else np.nan


def return_kurtosis(returns) -> float:
    """Excess kurtosis of per-period returns."""
    r = _clean(returns)
    return float(r.kurt()) if len(r) > 3 else np.nan


# ---------------------------------------------------------------------------
# Risk-adjusted metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (per-period risk-free rate subtracted)."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    excess = r - risk_free / float(periods_per_year)
    sd = excess.std(ddof=1)
    if sd <= 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(returns, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized downside deviation below the minimum acceptable return."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    downside = (r - mar / float(periods_per_year)).clip(upper=0.0)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def sortino_ratio(returns, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    dd = downside_deviation(returns, mar, periods_per_year)
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    return float((_clean(returns).mean() * periods_per_year - mar) / dd)


def calmar_ratio(returns, periods_per_year: int = 252) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    return float(cagr(returns, periods_per_year) / abs(mdd))


def sterling_ratio(returns, periods_per_year: int = 252) -> float:
    """CAGR divided by the average drawdown magnitude."""
    avg = average_drawdown(returns)
    if not np.isfinite(avg) or avg >= 0:
        return np.nan
    return float(cagr(returns, periods_per_year) / abs(avg))


def omega_ratio(returns, threshold: float = 0.0) -> float:
    """Omega ratio: gains above threshold over losses below it."""
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    gains = (r - threshold).clip(lower=0.0).sum()
    losses = (threshold - r).clip(lower=0.0).sum()
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def gain_to_pain_ratio(returns) -> float:
    """Sum of returns divided by the absolute sum of negative returns."""
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    pain = float(r.clip(upper=0.0).sum())
    if pain == 0:
        return np.inf if r.sum() > 0 else np.nan
    return float(r.sum() / abs(pain))


def ulcer_index(returns) -> float:
    """Ulcer index: RMS of the drawdown series (as positive percent)."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return np.nan
    return float(np.sqrt((dd ** 2).mean()))


def martin_ratio(returns, periods_per_year: int = 252) -> float:
    """CAGR divided by the ulcer index."""
    ui = ulcer_index(returns)
    if not np.isfinite(ui) or ui <= 0:
        return np.nan
    return float(cagr(returns, periods_per_year) / ui)


def tail_ratio(returns, upper_q: float = 0.95, lower_q: float = 0.05) -> float:
    """Ratio of the upper tail to the absolute lower tail at given quantiles."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    up = float(r.quantile(upper_q))
    dn = abs(float(r.quantile(lower_q)))
    if dn <= 0:
        return np.inf if up > 0 else np.nan
    return float(up / dn)


def common_sense_ratio(returns, periods_per_year: int = 252) -> float:
    """Profit factor times tail ratio - a compact reward/risk composite."""
    pf = profit_factor(_clean(returns).dropna())
    tr = tail_ratio(returns)
    if not np.isfinite(pf) or not np.isfinite(tr):
        return np.nan
    return float(pf * tr)


def downside_ratio(returns, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized excess return over downside deviation."""
    dd = downside_deviation(returns, mar, periods_per_year)
    if not np.isfinite(dd) or dd <= 0:
        return np.nan
    return float((_clean(returns).mean() * periods_per_year - mar) / dd)


def recovery_factor(returns) -> float:
    """Total return divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd >= 0:
        return np.nan
    return float(total_return(returns) / abs(mdd))


def risk_return_ratio(returns) -> float:
    """Mean per-period return divided by per-period standard deviation."""
    r = _clean(returns)
    sd = r.std(ddof=1)
    if len(r) < 2 or sd <= 0:
        return np.nan
    return float(r.mean() / sd)


# ---------------------------------------------------------------------------
# Drawdown metrics
# ---------------------------------------------------------------------------

def equity_curve(returns, initial: float = 1.0) -> pd.Series:
    """Compounded equity curve starting at ``initial``."""
    r = _as_series(returns).fillna(0.0)
    return (initial * (1.0 + r).cumprod()).rename("equity")


def drawdown_series(returns) -> pd.Series:
    """Drawdown series in [-1, 0] relative to the running equity peak."""
    eq = equity_curve(returns)
    return (eq / eq.cummax() - 1.0).rename("drawdown")


def underwater_series(returns) -> pd.Series:
    """Alias of the drawdown series (equity below its high-water mark)."""
    return drawdown_series(returns)


def max_drawdown(returns) -> float:
    """Most negative drawdown (as a negative fraction, 0 if never below peak)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else np.nan


def average_drawdown(returns) -> float:
    """Mean of non-zero drawdown values (negative)."""
    dd = drawdown_series(returns)
    nz = dd[dd < 0]
    return float(nz.mean()) if len(nz) else 0.0


def drawdown_duration_series(returns) -> pd.Series:
    """Bars elapsed since the last equity peak, for each bar."""
    eq = equity_curve(returns)
    peak = eq.cummax()
    at_peak = eq >= peak
    duration = np.zeros(len(eq), dtype="int64")
    run = 0
    for i, flag in enumerate(at_peak.to_numpy()):
        run = 0 if flag else run + 1
        duration[i] = run
    return pd.Series(duration, index=eq.index, name="dd_duration")


def max_drawdown_duration(returns) -> int:
    """Longest number of bars the equity stayed below a prior peak."""
    dd_series = drawdown_series(returns)
    durations = drawdown_duration_series(returns)
    if len(dd_series) == 0:
        return 0
    return int(durations.max())


def pain_index(returns) -> float:
    """Mean drawdown magnitude (positive percent)."""
    dd = drawdown_series(returns)
    return float(-dd.mean()) if len(dd) else np.nan


def drawdown_at_risk(returns, quantile: float = 0.05) -> float:
    """Worst drawdown at the given quantile of the drawdown distribution."""
    dd = drawdown_series(returns)
    if len(dd) == 0:
        return np.nan
    return float(dd.quantile(quantile))


# ---------------------------------------------------------------------------
# Value at risk
# ---------------------------------------------------------------------------

def var_historical(returns, level: float = 0.05) -> float:
    """Historical VaR at ``level`` (negative number for a loss)."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    return float(r.quantile(level))


def cvar_historical(returns, level: float = 0.05) -> float:
    """Historical CVaR/expected shortfall beyond the VaR threshold."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    var = r.quantile(level)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else np.nan


def var_parametric(returns, level: float = 0.05) -> float:
    """Gaussian parametric VaR at ``level``."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    z = _norm_ppf(level)
    return float(r.mean() + z * r.std(ddof=1))


def cvar_parametric(returns, level: float = 0.05) -> float:
    """Gaussian parametric CVaR (expected shortfall)."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    z = _norm_ppf(level)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return float(r.mean() - r.std(ddof=1) * pdf / level)


def modified_var(returns, level: float = 0.05) -> float:
    """Cornish-Fisher modified VaR accounting for skew and excess kurtosis."""
    r = _clean(returns)
    n = len(r)
    if n < 20:
        return np.nan
    z = _norm_ppf(level)
    s = float(r.skew())
    k = float(r.kurt())
    z_cf = z + (z * z - 1.0) * s / 6.0 + (z ** 3 - 3.0 * z) * k / 24.0 - (2.0 * z ** 3 - 5.0 * z) * s * s / 36.0
    return float(r.mean() + z_cf * r.std(ddof=1))


def kupiec_pof_test(returns, level: float = 0.05, window: int = 250) -> dict:
    """Kupiec proportion-of-failures VaR backtest.

    Rolling historical VaR (estimated on the prior ``window`` bars) is compared
    with realized returns; returns exceedance counts, the likelihood-ratio
    statistic, and its chi-square p-value (1 df).
    """
    r = _clean(returns)
    if len(r) <= window + 10:
        return {"exceedances": np.nan, "expected": np.nan, "lr": np.nan, "p_value": np.nan, "reject_5pct": False}
    var = r.rolling(window, min_periods=window).quantile(level).shift(1)
    hits = (r < var).fillna(False)
    hits = hits.iloc[window + 1:]
    t = int(len(hits))
    x = int(hits.sum())
    p = float(level)
    if t == 0:
        return {"exceedances": 0, "expected": 0.0, "lr": np.nan, "p_value": np.nan, "reject_5pct": False}
    xr = x / t
    if x == 0 or xr >= 1.0:
        lr = -2.0 * (t * np.log(1.0 - p))
        p_value = float(chi2_sf(lr, 1))
    else:
        ll_null = t * np.log(1.0 - p) + x * np.log(p)
        ll_alt = (t - x) * np.log(1.0 - xr) + x * np.log(xr)
        lr = max(-2.0 * (ll_null - ll_alt), 0.0)
        p_value = float(chi2_sf(lr, 1))
    return {"exceedances": x, "expected": p * t, "lr": float(lr), "p_value": p_value, "reject_5pct": bool(p_value < 0.05)}


# ---------------------------------------------------------------------------
# Trade metrics
# ---------------------------------------------------------------------------

def trade_count(trades) -> int:
    """Number of trades (finite entries)."""
    return int(len(_clean(trades)))


def win_rate(trades) -> float:
    """Fraction of trades with positive PnL."""
    t = _clean(trades)
    return float((t > 0).mean()) if len(t) else np.nan


def profit_factor(trades) -> float:
    """Gross profits divided by gross losses."""
    t = _clean(trades)
    if len(t) == 0:
        return np.nan
    gains = float(t.clip(lower=0.0).sum())
    losses = float(-t.clip(upper=0.0).sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def average_win(trades) -> float:
    """Mean PnL of winning trades."""
    t = _clean(trades)
    w = t[t > 0]
    return float(w.mean()) if len(w) else np.nan


def average_loss(trades) -> float:
    """Mean PnL of losing trades (negative)."""
    t = _clean(trades)
    l = t[t < 0]
    return float(l.mean()) if len(l) else np.nan


def payoff_ratio(trades) -> float:
    """Average win divided by the absolute average loss."""
    aw = average_win(trades)
    al = average_loss(trades)
    if not np.isfinite(aw) or not np.isfinite(al) or al == 0:
        return np.nan
    return float(aw / abs(al))


def expectancy(trades) -> float:
    """Expected PnL per trade: win rate times payoff minus loss rate."""
    t = _clean(trades)
    if len(t) == 0:
        return np.nan
    return float(t.mean())


def profit_per_trade(trades) -> float:
    """Mean PnL per trade (alias of expectancy)."""
    return expectancy(trades)


def largest_win(trades) -> float:
    """Largest single winning trade."""
    t = _clean(trades)
    return float(t.max()) if len(t) else np.nan


def largest_loss(trades) -> float:
    """Largest single losing trade (negative)."""
    t = _clean(trades)
    return float(t.min()) if len(t) else np.nan


def max_consecutive_wins(trades) -> int:
    """Longest run of consecutive winning trades."""
    t = _clean(trades)
    return _max_streak((t > 0).to_numpy(), True) if len(t) else 0


def max_consecutive_losses(trades) -> int:
    """Longest run of consecutive losing trades."""
    t = _clean(trades)
    return _max_streak((t < 0).to_numpy(), True) if len(t) else 0


def kelly_criterion(trades) -> float:
    """Kelly fraction from win rate and average win/loss sizes."""
    wr = win_rate(trades)
    po = payoff_ratio(trades)
    if not np.isfinite(wr) or not np.isfinite(po) or po <= 0:
        return np.nan
    return float(wr - (1.0 - wr) / po)


# ---------------------------------------------------------------------------
# Position metrics
# ---------------------------------------------------------------------------

def position_pnl(positions, returns) -> pd.Series:
    """Per-bar strategy PnL from positions lagged one bar times returns."""
    pos = _as_series(positions).reindex(_as_series(returns).index).fillna(0.0)
    r = _as_series(returns).fillna(0.0)
    return (pos.shift(1) * r).rename("pnl")


def position_pnl_curve(positions, returns, initial: float = 1.0) -> pd.Series:
    """Compounded equity curve of the position PnL."""
    pnl = position_pnl(positions, returns)
    return (initial * (1.0 + pnl).cumprod()).rename("equity")


def exposure(positions) -> float:
    """Fraction of bars with a non-zero position."""
    pos = _as_series(positions)
    return float((pos.fillna(0.0) != 0.0).mean()) if len(pos) else np.nan


def time_in_market(positions) -> float:
    """Fraction of bars in the market (alias of exposure)."""
    return exposure(positions)


def turnover(positions) -> float:
    """Total absolute position change over the sample."""
    pos = _as_series(positions).fillna(0.0)
    return float(pos.diff().abs().sum()) if len(pos) else np.nan


def trades_from_positions(positions, returns) -> pd.DataFrame:
    """Extract trades from a position series.

    Contiguous non-zero position runs become trades with entry/exit bars,
    duration, direction, compounded return, and a win flag.
    """
    pos = _as_series(positions).fillna(0.0)
    r = _as_series(returns).reindex(pos.index).fillna(0.0)
    rows = []
    open_dir = 0.0
    entry_bar = None
    seg_rets: list[float] = []
    for i, p in enumerate(pos.to_numpy()):
        if p != open_dir:
            if open_dir != 0.0 and entry_bar is not None:
                pnl = float(np.prod([1.0 + x for x in seg_rets]) - 1.0) if seg_rets else 0.0
                rows.append(
                    {
                        "entry_idx": entry_bar,
                        "exit_idx": i,
                        "bars": i - entry_bar,
                        "direction": open_dir,
                        "return": pnl * open_dir,
                        "win": bool(pnl * open_dir > 0),
                    }
                )
            open_dir = p
            entry_bar = i if p != 0.0 else None
            seg_rets = []
        elif open_dir != 0.0:
            seg_rets.append(float(r.iloc[i]))
    if open_dir != 0.0 and entry_bar is not None:
        pnl = float(np.prod([1.0 + x for x in seg_rets]) - 1.0) if seg_rets else 0.0
        rows.append(
            {
                "entry_idx": entry_bar,
                "exit_idx": len(pos),
                "bars": len(pos) - entry_bar,
                "direction": open_dir,
                "return": pnl * open_dir,
                "win": bool(pnl * open_dir > 0),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.index = [
            pos.index[int(e)].isoformat() if hasattr(pos.index[int(e)], "isoformat") else pos.index[int(e)]
            for e in frame["entry_idx"]
        ]
        frame.index.name = "entry"
    return frame


def average_holding_period(positions, returns=None) -> float:
    """Mean bar duration of non-zero position runs."""
    pos = _as_series(positions).fillna(0.0)
    runs = []
    run = 0
    for p in pos.to_numpy():
        if p != 0.0:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return float(np.mean(runs)) if runs else np.nan


# ---------------------------------------------------------------------------
# Benchmark-relative metrics
# ---------------------------------------------------------------------------

def _aligned(r, b) -> tuple[pd.Series, pd.Series]:
    rr = _as_series(r).astype("float64")
    bb = _as_series(b).reindex(rr.index).astype("float64")
    mask = np.isfinite(rr) & np.isfinite(bb)
    return rr[mask], bb[mask]


def beta(returns, benchmark) -> float:
    """OLS beta of returns against a benchmark."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 10 or b.var(ddof=1) <= 0:
        return np.nan
    return float(r.cov(b) / b.var(ddof=1))


def alpha(returns, benchmark, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Jensen's alpha against a benchmark."""
    r, b = _aligned(returns, benchmark)
    bt = beta(r, b)
    if not np.isfinite(bt):
        return np.nan
    rf = risk_free / float(periods_per_year)
    return float((r.mean() - rf - bt * (b.mean() - rf)) * periods_per_year)


def tracking_error(returns, benchmark, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of excess returns."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 10:
        return np.nan
    return float((r - b).std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year: int = 252) -> float:
    """Annualized mean excess return divided by tracking error."""
    te = tracking_error(returns, benchmark, periods_per_year)
    r, b = _aligned(returns, benchmark)
    if not np.isfinite(te) or te <= 0 or len(r) < 10:
        return np.nan
    return float((r - b).mean() / te * np.sqrt(periods_per_year))


def treynor_ratio(returns, benchmark, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized excess return per unit of beta."""
    bt = beta(returns, benchmark)
    r = _clean(returns)
    if not np.isfinite(bt) or bt == 0 or len(r) < 10:
        return np.nan
    return float((r.mean() * periods_per_year - risk_free) / bt)


def up_capture(returns, benchmark) -> float:
    """Average up-bar return divided by the benchmark's over up bars."""
    r, b = _aligned(returns, benchmark)
    up = b > 0
    if up.sum() < 5 or b[up].mean() == 0:
        return np.nan
    return float(r[up].mean() / b[up].mean())


def down_capture(returns, benchmark) -> float:
    """Average down-bar return divided by the benchmark's over down bars."""
    r, b = _aligned(returns, benchmark)
    dn = b < 0
    if dn.sum() < 5 or b[dn].mean() == 0:
        return np.nan
    return float(r[dn].mean() / b[dn].mean())


def capture_ratio(returns, benchmark) -> float:
    """Up capture divided by down capture."""
    uc = up_capture(returns, benchmark)
    dc = down_capture(returns, benchmark)
    if not np.isfinite(uc) or not np.isfinite(dc) or dc == 0:
        return np.nan
    return float(uc / dc)


def correlation_to_benchmark(returns, benchmark) -> float:
    """Pearson correlation of returns with the benchmark."""
    r, b = _aligned(returns, benchmark)
    if len(r) < 10:
        return np.nan
    return float(pearson(r.to_numpy(), b.to_numpy()))


def benchmark_r_squared(returns, benchmark) -> float:
    """R-squared of the returns/benchmark regression."""
    corr = correlation_to_benchmark(returns, benchmark)
    return float(corr ** 2) if np.isfinite(corr) else np.nan


def excess_returns(returns, benchmark) -> pd.Series:
    """Per-bar excess return of the strategy over the benchmark."""
    r, b = _aligned(returns, benchmark)
    return (r - b).rename("excess")


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

def rolling_volatility(returns, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized volatility."""
    r = _as_series(returns).astype("float64")
    return (r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)).rename("rolling_vol")


def rolling_sharpe(returns, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    r = _as_series(returns).astype("float64")
    mean = r.rolling(window, min_periods=window).mean()
    std = r.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std.replace(0, np.nan) * np.sqrt(periods_per_year)).rename("rolling_sharpe")


def rolling_sortino(returns, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized Sortino ratio."""
    r = _as_series(returns).astype("float64")
    mean = r.rolling(window, min_periods=window).mean()
    downside = r.clip(upper=0.0).rolling(window, min_periods=window).apply(
        lambda w: float(np.sqrt(np.nanmean(w ** 2))), raw=True
    )
    return (mean / downside.replace(0, np.nan) * np.sqrt(periods_per_year)).rename("rolling_sortino")


def rolling_max_drawdown(returns, window: int = 126) -> pd.Series:
    """Rolling worst drawdown within the trailing window."""
    r = _as_series(returns).astype("float64")

    def mdd(w) -> float:
        w = np.asarray(w, dtype="float64")
        if not np.isfinite(w).all() or len(w) == 0:
            return np.nan
        eq = np.cumprod(1.0 + w)
        peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1.0).min())

    return r.rolling(window, min_periods=window).apply(mdd, raw=True).rename("rolling_mdd")


def rolling_calmar(returns, window: int = 126, periods_per_year: int = 252) -> pd.Series:
    """Rolling Calmar: window mean return over the window max drawdown."""
    mean_ann = _as_series(returns).astype("float64").rolling(window, min_periods=window).mean() * periods_per_year
    mdd = rolling_max_drawdown(returns, window)
    return (mean_ann / mdd.abs().replace(0, np.nan)).rename("rolling_calmar")


def rolling_beta(returns, benchmark, window: int = 63) -> pd.Series:
    """Rolling OLS beta against the benchmark."""
    r, b = _aligned(returns, benchmark)
    cov = r.rolling(window, min_periods=window).cov(b)
    var = b.rolling(window, min_periods=window).var(ddof=1)
    return (cov / var.replace(0, np.nan)).rename("rolling_beta")


def rolling_alpha(returns, benchmark, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized alpha against the benchmark."""
    r, b = _aligned(returns, benchmark)
    bt = rolling_beta(r, b, window)
    mr = r.rolling(window, min_periods=window).mean()
    mb = b.rolling(window, min_periods=window).mean()
    return ((mr - bt * mb) * periods_per_year).rename("rolling_alpha")


def rolling_information_ratio(returns, benchmark, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized information ratio."""
    r, b = _aligned(returns, benchmark)
    ex = r - b
    mean = ex.rolling(window, min_periods=window).mean()
    std = ex.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std.replace(0, np.nan) * np.sqrt(periods_per_year)).rename("rolling_ir")


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------

def monthly_returns_table(returns) -> pd.DataFrame:
    """Year-by-month compounded return table."""
    r = _as_series(returns).astype("float64")
    if not isinstance(r.index, pd.DatetimeIndex):
        return pd.DataFrame()
    monthly = r.resample("ME").apply(lambda x: float((1.0 + x).prod() - 1.0))
    frame = pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.to_numpy()})
    return frame.pivot_table(index="year", columns="month", values="ret")


def yearly_returns(returns) -> pd.Series:
    """Compounded calendar-year returns."""
    r = _as_series(returns).astype("float64")
    if not isinstance(r.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    out = r.resample("YE").apply(lambda x: float((1.0 + x).prod() - 1.0))
    out.index = out.index.year
    return out.rename("yearly_return")


def day_of_week_returns(returns) -> pd.Series:
    """Mean return by weekday (0 = Monday)."""
    r = _as_series(returns).astype("float64")
    if not isinstance(r.index, pd.DatetimeIndex):
        return pd.Series(dtype="float64")
    return r.groupby(r.index.dayofweek).mean().rename("mean_return")


def strategy_metrics(
    returns,
    benchmark=None,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> dict:
    """Compact metric bundle for a strategy return stream."""
    r = _clean(returns)
    out = {
        "total_return": total_return(r),
        "cagr": cagr(r, periods_per_year),
        "annual_return": annualized_return(r, periods_per_year),
        "annual_volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "sortino": sortino_ratio(r, risk_free, periods_per_year),
        "calmar": calmar_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "average_drawdown": average_drawdown(r),
        "max_dd_duration": max_drawdown_duration(r),
        "ulcer_index": ulcer_index(r),
        "var_95": var_historical(r, 0.05),
        "cvar_95": cvar_historical(r, 0.05),
        "skew": return_skew(r),
        "excess_kurtosis": return_kurtosis(r),
        "bar_win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "bar_profit_factor": profit_factor(r),
        "tail_ratio": tail_ratio(r),
        "best_period": float(r.max()) if len(r) else np.nan,
        "worst_period": float(r.min()) if len(r) else np.nan,
    }
    if benchmark is not None:
        out.update(
            {
                "beta": beta(r, benchmark),
                "alpha": alpha(r, benchmark, risk_free, periods_per_year),
                "tracking_error": tracking_error(r, benchmark, periods_per_year),
                "information_ratio": information_ratio(r, benchmark, periods_per_year),
                "up_capture": up_capture(r, benchmark),
                "down_capture": down_capture(r, benchmark),
                "correlation": correlation_to_benchmark(r, benchmark),
            }
        )
    return out


def compare_strategies(returns, benchmark=None, periods_per_year: int = 252) -> pd.DataFrame:
    """Metrics table with one column per strategy.

    ``returns`` may be a dict of name -> return series or a DataFrame whose
    columns are strategies.
    """
    if isinstance(returns, pd.DataFrame):
        series_map = {str(col): returns[col] for col in returns.columns}
    elif isinstance(returns, dict):
        series_map = {str(k): v for k, v in returns.items()}
    else:
        series_map = {"strategy": returns}
    table = {}
    for name, series in series_map.items():
        table[name] = strategy_metrics(series, benchmark, periods_per_year)
    return pd.DataFrame(table)
