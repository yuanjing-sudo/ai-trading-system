"""Risk analytics: tail risk, hedging, portfolio risk decomposition, and stress tests.

Functions operate on return Series (single asset) or return DataFrames (asset
columns), plus covariance matrices and weight vectors. All outputs are aligned
to the inputs and reproject nothing; inputs are never mutated. The number of
periods per year is configurable where annualization applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series, atr
from quant.statistics_ext import bootstrap_ci

__all__ = [
    # tail risk / VaR extensions
    "var_scale", "var_bootstrap_ci", "expected_tail_loss", "tail_index_hill",
    "worst_window_return", "max_loss_streak", "drawdown_recovery_bars",
    "time_underwater", "rolling_var", "rolling_cvar", "ewma_volatility",
    "volatility_cone", "vol_spike_flag", "return_quantile_profile",
    # beta / hedging
    "hedge_ratio", "dollar_hedge_ratio", "minimum_variance_hedge",
    "hedge_effectiveness", "rolling_hedge_ratio", "downside_beta",
    "upside_beta", "tail_beta", "volatility_beta",
    # portfolio risk
    "portfolio_volatility", "portfolio_beta", "marginal_risk_contribution",
    "component_risk_contribution", "risk_contribution_pct",
    "inverse_vol_weights", "equal_risk_contribution", "min_variance_weights",
    "max_diversification_weights", "diversification_ratio",
    "effective_number_of_bets", "hhi_concentration", "risk_budget_weights",
    "covariance_eigenvalues", "absorption_ratio", "effective_rank",
    "average_pairwise_correlation", "correlation_dispersion",
    "turbulence_index",
    # stress testing
    "stress_scenario_impact", "historical_stress_windows",
    "scenario_shock_analysis", "expected_shortfall_portfolio",
    # position sizing
    "volatility_target_position", "kelly_from_moments", "fractional_kelly",
    "position_size_by_risk", "atr_position_size", "gross_exposure",
    "net_exposure", "leverage_ratio", "position_limit_scale",
    "budget_scale_weights",
    # volatility regime / drawdown
    "volatility_regime", "volatility_regime_z", "drawdown_speed",
    # correlation diagnostics
    "rolling_correlation_to_market", "correlation_breakdown_flag",
    "downside_correlation",
    # stop helpers
    "chandelier_stop", "atr_stop", "trailing_stop", "stop_distance_to_volatility",
]


def _clean(x) -> pd.Series:
    return _as_series(x).astype("float64").replace([np.inf, -np.inf], np.nan).dropna()


def _weights_array(weights) -> np.ndarray:
    w = np.asarray(weights, dtype="float64").ravel()
    return w


def _cov_frame(cov) -> pd.DataFrame:
    if isinstance(cov, pd.Series):
        cov = cov.to_frame()
    if not isinstance(cov, pd.DataFrame):
        arr = np.atleast_2d(np.asarray(cov, dtype="float64"))
        return pd.DataFrame(arr)
    return cov.astype("float64")


# ---------------------------------------------------------------------------
# Tail risk / VaR extensions
# ---------------------------------------------------------------------------

def var_scale(returns, horizon: int = 10, level: float = 0.05) -> float:
    """Historical VaR scaled to ``horizon`` bars by the square-root-of-time rule."""
    r = _clean(returns)
    if len(r) < 10:
        return np.nan
    return float(r.quantile(level) * np.sqrt(horizon))


def var_bootstrap_ci(returns, level: float = 0.05, n_boot: int = 500, alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap confidence interval for the historical VaR estimate."""
    r = _clean(returns)
    if len(r) < 30:
        return {"var": np.nan, "lo": np.nan, "hi": np.nan}
    stat = lambda arr: float(np.quantile(np.asarray(arr), level))
    ci = bootstrap_ci(r.to_numpy(), stat, n_boot=n_boot, alpha=alpha, seed=seed)
    return {"var": stat(r.to_numpy()), "lo": ci.get("lower"), "hi": ci.get("upper")}


def expected_tail_loss(returns, tail_pct: float = 0.05) -> float:
    """Mean loss beyond the worst ``tail_pct`` of observations (positive number)."""
    r = _clean(returns)
    if len(r) < 20:
        return np.nan
    cutoff = r.quantile(tail_pct)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


def tail_index_hill(returns, k: int = 50) -> float:
    """Hill estimator of the tail index from the k largest losses."""
    r = _clean(returns)
    losses = -r[r < 0].sort_values(ascending=False)
    k = min(int(k), max(len(losses) - 1, 1))
    if len(losses) <= k or losses.iloc[k] <= 0:
        return np.nan
    xk = losses.iloc[k]
    xi = losses.iloc[:k]
    return float(k / np.log(xi / xk).sum()) if (xi / xk).gt(0).all() else np.nan


def worst_window_return(returns, window: int = 21) -> float:
    """Worst compounded return over any rolling window of ``window`` bars."""
    r = _clean(returns)
    if len(r) < window:
        return np.nan
    rolled = r.rolling(window, min_periods=window).apply(lambda w: float((1.0 + w).prod() - 1.0), raw=True)
    return float(rolled.min())


def max_loss_streak(returns) -> int:
    """Longest run of consecutive negative bars."""
    r = _clean(returns)
    best = cur = 0
    for x in r.to_numpy():
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def drawdown_recovery_bars(returns) -> float:
    """Mean number of bars from a drawdown trough back to a new equity high."""
    r = _as_series(returns).fillna(0.0)
    if len(r) < 2:
        return np.nan
    eq = (1.0 + r).cumprod()
    peak = eq.iloc[0]
    trough_since = False
    dd_start = None
    durations = []
    for i in range(len(eq)):
        if eq.iloc[i] >= peak:
            if trough_since and dd_start is not None:
                durations.append(i - dd_start)
            peak = eq.iloc[i]
            trough_since = False
            dd_start = None
        elif dd_start is None:
            dd_start = i
        else:
            trough_since = True
    return float(np.mean(durations)) if durations else np.nan


def time_underwater(returns) -> float:
    """Fraction of bars spent below a prior equity peak."""
    r = _as_series(returns).fillna(0.0)
    if len(r) == 0:
        return np.nan
    eq = (1.0 + r).cumprod()
    return float((eq < eq.cummax()).mean())


def rolling_var(returns, window: int = 250, level: float = 0.05) -> pd.Series:
    """Rolling historical VaR."""
    r = _as_series(returns).astype("float64")
    return r.rolling(window, min_periods=window).apply(lambda w: float(np.quantile(w, level)), raw=True).rename("rolling_var")


def rolling_cvar(returns, window: int = 250, level: float = 0.05) -> pd.Series:
    """Rolling historical CVaR (expected shortfall)."""
    r = _as_series(returns).astype("float64")

    def cvar(w) -> float:
        w = np.asarray(w, dtype="float64")
        cut = np.quantile(w, level)
        tail = w[w <= cut]
        return float(tail.mean()) if len(tail) else np.nan

    return r.rolling(window, min_periods=window).apply(cvar, raw=True).rename("rolling_cvar")


def ewma_volatility(returns, halflife: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Annualized EWMA volatility."""
    r = _as_series(returns).astype("float64")
    var = r.pow(2).ewm(halflife=halflife, min_periods=max(2, halflife // 2)).mean()
    return (np.sqrt(var) * np.sqrt(periods_per_year)).rename("ewma_vol")


def volatility_cone(
    returns,
    windows: tuple[int, ...] = (5, 10, 21, 63, 126, 252),
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Volatility cone: quantiles of annualized rolling volatility per window."""
    r = _clean(returns)
    if len(r) < max(windows):
        return pd.DataFrame()
    data = {}
    for w in windows:
        vol = r.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(periods_per_year)
        data[w] = vol.quantile(list(quantiles)).to_numpy()
    return pd.DataFrame(data, index=[f"q{int(q * 100)}" for q in quantiles])


def vol_spike_flag(returns, window: int = 21, z_thresh: float = 2.0) -> pd.Series:
    """Boolean flags where rolling volatility deviates above its history."""
    r = _as_series(returns).astype("float64")
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    hist = vol.rolling(window * 5, min_periods=window * 3)
    z = (vol - hist.mean()) / hist.std().replace(0, np.nan)
    return (z > z_thresh).fillna(False).rename("vol_spike")


def return_quantile_profile(returns, quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> pd.Series:
    """Quantile profile of the return distribution."""
    r = _clean(returns)
    if len(r) < 20:
        return pd.Series(dtype="float64")
    return r.quantile(list(quantiles)).rename("return_quantiles")


# ---------------------------------------------------------------------------
# Beta / hedging
# ---------------------------------------------------------------------------

def hedge_ratio(asset, hedge) -> float:
    """OLS hedge ratio: beta of the asset on the hedging instrument."""
    a = _as_series(asset).astype("float64")
    h = _as_series(hedge).reindex(a.index).astype("float64")
    mask = np.isfinite(a) & np.isfinite(h)
    a, h = a[mask], h[mask]
    if len(a) < 10 or h.var(ddof=1) <= 0:
        return np.nan
    return float(a.cov(h) / h.var(ddof=1))


def dollar_hedge_ratio(asset_exposure: float, asset_price: float, hedge_price: float, beta_value: float) -> float:
    """Units of the hedge instrument per unit of asset exposure."""
    if hedge_price <= 0 or hedge_price == 0:
        return np.nan
    return float(beta_value * asset_exposure * asset_price / hedge_price)


def minimum_variance_hedge(asset, hedge) -> float:
    """Minimum-variance hedge ratio: cov(asset, hedge) / var(hedge)."""
    a = _as_series(asset).astype("float64")
    h = _as_series(hedge).reindex(a.index).astype("float64")
    mask = np.isfinite(a) & np.isfinite(h)
    a, h = a[mask], h[mask]
    if len(a) < 10 or h.var(ddof=1) <= 0:
        return np.nan
    return float(a.cov(h) / h.var(ddof=1))


def hedge_effectiveness(asset, hedge, hedge_ratio_value: float | None = None) -> float:
    """R-squared of the hedged portfolio variance versus the unhedged asset."""
    a = _as_series(asset).astype("float64")
    h = _as_series(hedge).reindex(a.index).astype("float64")
    mask = np.isfinite(a) & np.isfinite(h)
    a, h = a[mask], h[mask]
    if len(a) < 10 or a.var(ddof=1) <= 0:
        return np.nan
    hr = hedge_ratio(a, h) if hedge_ratio_value is None else float(hedge_ratio_value)
    if not np.isfinite(hr):
        return np.nan
    hedged = a - hr * h
    return float(1.0 - hedged.var(ddof=1) / a.var(ddof=1))


def rolling_hedge_ratio(asset, hedge, window: int = 63) -> pd.Series:
    """Rolling minimum-variance hedge ratio."""
    a = _as_series(asset).astype("float64")
    h = _as_series(hedge).reindex(a.index).astype("float64")
    cov = a.rolling(window, min_periods=window).cov(h)
    var = h.rolling(window, min_periods=window).var(ddof=1)
    return (cov / var.replace(0, np.nan)).rename("hedge_ratio")


def downside_beta(returns, benchmark) -> float:
    """Beta computed only over benchmark-down bars."""
    r = _as_series(returns).astype("float64")
    b = _as_series(benchmark).reindex(r.index).astype("float64")
    mask = np.isfinite(r) & np.isfinite(b) & (b < 0)
    if mask.sum() < 10:
        return np.nan
    bb = b[mask]
    if bb.var(ddof=1) <= 0:
        return np.nan
    return float(r[mask].cov(bb) / bb.var(ddof=1))


def upside_beta(returns, benchmark) -> float:
    """Beta computed only over benchmark-up bars."""
    r = _as_series(returns).astype("float64")
    b = _as_series(benchmark).reindex(r.index).astype("float64")
    mask = np.isfinite(r) & np.isfinite(b) & (b > 0)
    if mask.sum() < 10:
        return np.nan
    bb = b[mask]
    if bb.var(ddof=1) <= 0:
        return np.nan
    return float(r[mask].cov(bb) / bb.var(ddof=1))


def tail_beta(returns, benchmark, tail_pct: float = 0.1) -> float:
    """Beta computed over the worst ``tail_pct`` of benchmark bars."""
    r = _as_series(returns).astype("float64")
    b = _as_series(benchmark).reindex(r.index).astype("float64")
    mask = np.isfinite(r) & np.isfinite(b)
    r, b = r[mask], b[mask]
    if len(b) < 30:
        return np.nan
    cutoff = b.quantile(tail_pct)
    sel = b <= cutoff
    if sel.sum() < 5 or b[sel].var(ddof=1) <= 0:
        return np.nan
    return float(r[sel].cov(b[sel]) / b[sel].var(ddof=1))


def volatility_beta(returns, benchmark, window: int = 63) -> float:
    """Ratio of rolling volatility to the benchmark's over ``window`` bars."""
    r = _as_series(returns).astype("float64")
    b = _as_series(benchmark).reindex(r.index).astype("float64")
    rv = r.rolling(window, min_periods=window).std(ddof=1).iloc[-1]
    bv = b.rolling(window, min_periods=window).std(ddof=1).iloc[-1]
    if not np.isfinite(rv) or not np.isfinite(bv) or bv <= 0:
        return np.nan
    return float(rv / bv)


# ---------------------------------------------------------------------------
# Portfolio risk
# ---------------------------------------------------------------------------

def portfolio_volatility(weights, cov) -> float:
    """Portfolio volatility from weights and a covariance matrix."""
    w = _weights_array(weights)
    c = _cov_frame(cov).to_numpy(dtype="float64")
    if c.shape[0] != len(w):
        raise ValueError("weights and covariance dimension mismatch")
    var = float(w @ c @ w)
    return float(np.sqrt(max(var, 0.0)))


def portfolio_beta(weights, betas) -> float:
    """Weighted average of asset betas."""
    w = _weights_array(weights)
    b = _weights_array(betas)
    if len(w) != len(b):
        raise ValueError("weights and betas dimension mismatch")
    return float(w @ b)


def marginal_risk_contribution(weights, cov) -> pd.Series:
    """Marginal contribution of each asset to portfolio volatility."""
    w = _weights_array(weights)
    c = _cov_frame(cov)
    arr = c.to_numpy(dtype="float64")
    pvol = portfolio_volatility(w, c)
    if pvol <= 0:
        return pd.Series(np.nan, index=c.columns, name="mrc")
    mrc = (arr @ w) / pvol
    return pd.Series(mrc, index=c.columns, name="mrc")


def component_risk_contribution(weights, cov) -> pd.Series:
    """Component risk contribution: weight times marginal risk contribution."""
    w = _weights_array(weights)
    mrc = marginal_risk_contribution(w, cov)
    return (w * mrc).rename("component_risk")


def risk_contribution_pct(weights, cov) -> pd.Series:
    """Percentage risk contribution per asset (sums to 1)."""
    comp = component_risk_contribution(weights, cov)
    total = float(comp.sum())
    if not np.isfinite(total) or total == 0:
        return pd.Series(np.nan, index=comp.index, name="risk_pct")
    return (comp / total).rename("risk_pct")


def inverse_vol_weights(returns) -> pd.Series:
    """Weights proportional to inverse volatility."""
    if isinstance(returns, pd.DataFrame):
        vol = returns.std(ddof=1)
    else:
        vol = _as_series(returns).to_frame("asset").std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    w = inv / inv.sum()
    return w.rename("weight")


def equal_risk_contribution(returns, iters: int = 200, tol: float = 1e-8) -> pd.Series:
    """Iterative equal-risk-contribution (risk parity) weights.

    Uses a square-root damped multiplicative update (Spinu-style) initialized
    from inverse-vol weights; falls back to inverse-vol weights if a
    contribution turns non-positive mid-iteration.
    """
    if isinstance(returns, pd.Series):
        return pd.Series([1.0], index=[returns.name or "asset"], name="weight")
    cov = returns.cov()
    n = cov.shape[0]
    arr = cov.to_numpy(dtype="float64")
    sigma = np.sqrt(np.clip(np.diag(arr), 1e-18, None))
    w = (1.0 / sigma) / (1.0 / sigma).sum()
    target = 1.0 / n
    for _ in range(iters):
        mrc = arr @ w
        pvol = float(np.sqrt(max(w @ mrc, 1e-18)))
        contrib = w * mrc / pvol
        total = float(contrib.sum())
        if total <= 0 or not np.isfinite(total):
            break
        ratio = (target * total) / np.where(contrib <= 0, np.nan, contrib)
        if not np.isfinite(ratio).all():
            inv = 1.0 / sigma
            w = inv / inv.sum()
            break
        w_new = np.clip(w * np.sqrt(ratio), 1e-12, None)
        w_new = w_new / w_new.sum()
        gap = float(np.max(np.abs(w_new - w)))
        w = w_new
        if gap < tol:
            break
    return pd.Series(w, index=cov.columns, name="weight")


def min_variance_weights(cov) -> pd.Series:
    """Closed-form global minimum-variance weights."""
    c = _cov_frame(cov)
    arr = c.to_numpy(dtype="float64")
    try:
        inv = np.linalg.pinv(arr)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=c.columns, name="weight")
    ones = np.ones(len(c.columns))
    raw = inv @ ones
    denom = float(ones @ raw)
    if denom == 0:
        return pd.Series(np.nan, index=c.columns, name="weight")
    return pd.Series(raw / denom, index=c.columns, name="weight")


def max_diversification_weights(cov) -> pd.Series:
    """Closed-form maximum-diversification weights w ~ inv(cov) @ sigma."""
    c = _cov_frame(cov)
    arr = c.to_numpy(dtype="float64")
    try:
        inv = np.linalg.pinv(arr)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=c.columns, name="weight")
    sigma = np.sqrt(np.clip(np.diag(arr), 0.0, None))
    raw = inv @ sigma
    total = raw.sum()
    if total == 0:
        return pd.Series(np.nan, index=c.columns, name="weight")
    return pd.Series(raw / total, index=c.columns, name="weight")


def diversification_ratio(weights, cov, vols=None) -> float:
    """Weighted average volatility divided by portfolio volatility."""
    w = _weights_array(weights)
    c = _cov_frame(cov)
    if vols is None:
        sigma = np.sqrt(np.clip(np.diag(c.to_numpy(dtype="float64")), 0.0, None))
    else:
        sigma = _weights_array(vols)
    pvol = portfolio_volatility(w, c)
    if pvol <= 0:
        return np.nan
    return float((np.abs(w) @ sigma) / pvol)


def effective_number_of_bets(weights, cov=None) -> float:
    """Effective number of independent risk bets (1 / HHI of risk contributions)."""
    if cov is not None:
        pct = risk_contribution_pct(weights, cov).to_numpy()
    else:
        w = np.abs(_weights_array(weights))
        total = w.sum()
        pct = w / total if total > 0 else w
    pct = pct[np.isfinite(pct)]
    denom = float((pct ** 2).sum())
    return float(1.0 / denom) if denom > 0 else np.nan


def hhi_concentration(weights) -> float:
    """Herfindahl-Hirschman concentration index of (absolute) weights."""
    w = np.abs(_weights_array(weights))
    total = w.sum()
    if total <= 0:
        return np.nan
    p = w / total
    return float((p ** 2).sum())


def risk_budget_weights(cov, budgets=None, iters: int = 200, tol: float = 1e-8) -> pd.Series:
    """Multiplicative-update weights that match per-asset risk budgets."""
    c = _cov_frame(cov)
    arr = c.to_numpy(dtype="float64")
    n = arr.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype="float64") / np.sum(budgets)
    w = b.copy()
    for _ in range(iters):
        mrc = arr @ w
        pvol = np.sqrt(max(float(w @ mrc), 1e-18))
        contrib = w * mrc / pvol
        total = contrib.sum()
        if total <= 0 or not np.isfinite(total):
            break
        if (contrib <= 0).any():
            w = b / b.sum()
            break
        target = b * total
        adj = target / contrib
        w_new = w * np.sqrt(adj)
        w_new = np.clip(w_new, 1e-12, None)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=c.columns, name="weight")


def covariance_eigenvalues(cov) -> pd.Series:
    """Eigenvalues of a covariance matrix, sorted descending."""
    arr = _cov_frame(cov).to_numpy(dtype="float64")
    if arr.size == 0:
        return pd.Series(dtype="float64")
    vals = np.linalg.eigvalsh((arr + arr.T) / 2.0)
    vals = np.sort(vals)[::-1]
    return pd.Series(vals, name="eigenvalue")


def absorption_ratio(cov, n_factors: int = 1) -> float:
    """Share of total variance explained by the top ``n_factors`` eigenvalues."""
    vals = covariance_eigenvalues(cov).to_numpy()
    total = vals.sum()
    if total <= 0:
        return np.nan
    k = max(int(n_factors), 1)
    return float(vals[:k].sum() / total)


def effective_rank(cov) -> float:
    """Entropy-based effective rank of the covariance matrix."""
    vals = covariance_eigenvalues(cov).to_numpy()
    vals = vals[vals > 0]
    total = vals.sum()
    if total <= 0:
        return np.nan
    p = vals / total
    return float(np.exp(-(p * np.log(p)).sum()))


def average_pairwise_correlation(returns) -> float:
    """Mean off-diagonal pairwise correlation of an asset return matrix."""
    if isinstance(returns, pd.Series):
        return np.nan
    c = returns.corr()
    n = c.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    vals = c.to_numpy()[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def correlation_dispersion(returns) -> float:
    """Standard deviation of pairwise correlations (concentration of co-movement)."""
    if isinstance(returns, pd.Series):
        return np.nan
    c = returns.corr()
    n = c.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    vals = c.to_numpy()[mask]
    vals = vals[np.isfinite(vals)]
    return float(vals.std(ddof=1)) if len(vals) > 1 else np.nan


def turbulence_index(returns) -> pd.Series:
    """Mahalanobis distance of each bar's cross-asset returns from the mean."""
    if isinstance(returns, pd.Series):
        frame = returns.to_frame("asset")
    else:
        frame = returns
    x = frame.dropna()
    if len(x) < 10 or x.shape[1] < 1:
        return pd.Series(np.nan, index=frame.index)
    mu = x.mean().to_numpy()
    cov = np.cov(x.to_numpy(), rowvar=False)
    cov = np.atleast_2d(cov)
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.Series(np.nan, index=frame.index)
    d = x.to_numpy() - mu
    turb = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0))
    return pd.Series(turb, index=x.index, name="turbulence").reindex(frame.index)


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def _portfolio_series(weights, asset_returns) -> pd.Series:
    w = _weights_array(weights)
    frame = asset_returns if isinstance(asset_returns, pd.DataFrame) else pd.DataFrame(asset_returns)
    if frame.shape[1] != len(w):
        raise ValueError("weights and asset_returns dimension mismatch")
    return pd.Series(frame.to_numpy(dtype="float64") @ w, index=frame.index, name="portfolio")


def stress_scenario_impact(weights, asset_returns, scenario=None) -> float:
    """Portfolio return under a per-asset shock vector.

    ``scenario`` may be a Series/dict of asset -> shock; if omitted, the worst
    historical bar of the portfolio is used as the scenario.
    """
    w = _weights_array(weights)
    frame = asset_returns if isinstance(asset_returns, pd.DataFrame) else pd.DataFrame(asset_returns)
    if frame.shape[1] != len(w):
        raise ValueError("weights and asset_returns dimension mismatch")
    if scenario is None:
        p = _portfolio_series(w, frame)
        return float(p.min()) if len(p.dropna()) else np.nan
    if isinstance(scenario, dict):
        shocks = np.array([scenario.get(col, 0.0) for col in frame.columns], dtype="float64")
    else:
        shocks = _weights_array(scenario)
    return float(w @ shocks)


def historical_stress_windows(weights, asset_returns, window: int = 21, top_n: int = 5) -> pd.DataFrame:
    """Worst overlapping historical windows of the portfolio return series."""
    p = _portfolio_series(weights, asset_returns).dropna()
    if len(p) < window:
        return pd.DataFrame(columns=["start", "end", "return"])
    rolled = p.rolling(window, min_periods=window).apply(lambda w: float((1.0 + w).prod() - 1.0), raw=True)
    worst = rolled.nsmallest(int(top_n))
    rows = []
    for end_idx, ret in worst.items():
        pos = p.index.get_loc(end_idx)
        start = p.index[max(pos - window + 1, 0)]
        rows.append({"start": start, "end": end_idx, "return": float(ret)})
    return pd.DataFrame(rows)


def scenario_shock_analysis(weights, shocks) -> dict:
    """Per-asset contribution to a shock scenario plus the total impact."""
    w = _weights_array(weights)
    if isinstance(shocks, dict):
        items = list(shocks.items())
        assets = [k for k, _ in items]
        vals = np.array([v for _, v in items], dtype="float64")
    else:
        s = _as_series(shocks)
        assets = [str(i) for i in s.index]
        vals = s.to_numpy(dtype="float64")
    if len(vals) != len(w):
        raise ValueError("weights and shocks dimension mismatch")
    contrib = pd.Series(w * vals, index=assets, name="contribution")
    return {"total": float(contrib.sum()), "by_asset": contrib}


def expected_shortfall_portfolio(weights, asset_returns, level: float = 0.05) -> float:
    """Historical expected shortfall of the portfolio return stream."""
    p = _portfolio_series(weights, asset_returns).dropna()
    if len(p) < 20:
        return np.nan
    cutoff = p.quantile(level)
    tail = p[p <= cutoff]
    return float(tail.mean()) if len(tail) else np.nan


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def volatility_target_position(target_vol: float, realized_vol: float, max_leverage: float = 1.0) -> float:
    """Position scale that targets ``target_vol`` given realized volatility."""
    if not np.isfinite(realized_vol) or realized_vol <= 0:
        return np.nan
    return float(min(target_vol / realized_vol, max_leverage))


def kelly_from_moments(mean_return: float, std_return: float, risk_free: float = 0.0) -> float:
    """Continuous Kelly fraction from mean/std moments."""
    var = std_return ** 2
    if not np.isfinite(var) or var <= 0:
        return np.nan
    return float((mean_return - risk_free) / var)


def fractional_kelly(kelly_value: float, fraction: float = 0.5) -> float:
    """Fractional Kelly (``fraction`` of the full Kelly bet)."""
    return float(kelly_value * fraction)


def position_size_by_risk(account_size: float, risk_pct: float, stop_distance_price: float) -> float:
    """Units to trade so a stop at ``stop_distance_price`` risks ``risk_pct``."""
    if stop_distance_price <= 0:
        return np.nan
    return float(account_size * risk_pct / stop_distance_price)


def atr_position_size(account_size: float, risk_pct: float, atr_value: float, atr_multiple: float = 2.0) -> float:
    """Units sized so ``atr_multiple`` ATRs of adverse move risk ``risk_pct``."""
    stop = atr_value * atr_multiple
    if stop <= 0:
        return np.nan
    return float(account_size * risk_pct / stop)


def gross_exposure(weights) -> float:
    """Sum of absolute weights."""
    return float(np.abs(_weights_array(weights)).sum())


def net_exposure(weights) -> float:
    """Sum of signed weights."""
    return float(_weights_array(weights).sum())


def leverage_ratio(weights) -> float:
    """Gross exposure divided by net capital (alias of gross exposure)."""
    return gross_exposure(weights)


def position_limit_scale(weights, max_gross: float = 1.0) -> float:
    """Scale factor (<= 1) that caps gross exposure at ``max_gross``."""
    gross = gross_exposure(weights)
    if not np.isfinite(gross) or gross <= 0:
        return np.nan
    return float(min(1.0, max_gross / gross))


def budget_scale_weights(weights, budgets, cov) -> pd.Series:
    """Rescale weights so realized risk contributions match target budgets."""
    w = _weights_array(weights)
    contrib = risk_contribution_pct(w, cov).to_numpy()
    b = _weights_array(budgets)
    if len(b) != len(w):
        raise ValueError("weights and budgets dimension mismatch")
    b = b / b.sum()
    factor = b / np.where(contrib == 0, 1e-12, contrib)
    scaled = w * factor
    total = scaled.sum()
    if total == 0:
        return pd.Series(np.nan, index=_cov_frame(cov).columns, name="weight")
    return pd.Series(scaled / total, index=_cov_frame(cov).columns, name="weight")


# ---------------------------------------------------------------------------
# Volatility regime / drawdown
# ---------------------------------------------------------------------------

def volatility_regime(returns, window: int = 21, low_q: float = 0.33, high_q: float = 0.66) -> pd.Series:
    """Regime labels from rolling volatility vs its expanding quantiles: -1/0/+1."""
    r = _as_series(returns).astype("float64")
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    hist = vol.expanding(min_periods=max(window * 3, 20))
    lo = hist.quantile(low_q)
    hi = hist.quantile(high_q)
    regime = pd.Series(0.0, index=r.index)
    regime[vol > hi] = 1.0
    regime[vol < lo] = -1.0
    regime[vol.isna()] = np.nan
    return regime.rename("vol_regime")


def volatility_regime_z(returns, window: int = 21, lookback: int = 252) -> pd.Series:
    """Z-score of current rolling volatility within its trailing history."""
    r = _as_series(returns).astype("float64")
    vol = r.rolling(window, min_periods=window).std(ddof=1)
    hist = vol.rolling(lookback, min_periods=max(lookback // 4, 20))
    return ((vol - hist.mean()) / hist.std().replace(0, np.nan)).rename("vol_regime_z")


def drawdown_speed(returns) -> float:
    """Max drawdown magnitude divided by the drawdown duration in bars."""
    r = _as_series(returns).fillna(0.0)
    if len(r) < 2:
        return np.nan
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    mdd = abs(float(dd.min()))
    if mdd == 0:
        return 0.0
    underwater = (dd < 0).to_numpy()
    best = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    if best == 0:
        return np.nan
    return float(mdd / best)


# ---------------------------------------------------------------------------
# Correlation diagnostics
# ---------------------------------------------------------------------------

def rolling_correlation_to_market(returns, market_returns, window: int = 63) -> pd.Series:
    """Rolling correlation of the strategy with the market."""
    r = _as_series(returns).astype("float64")
    m = _as_series(market_returns).reindex(r.index).astype("float64")
    return r.rolling(window, min_periods=window).corr(m).rename("corr_to_market")


def correlation_breakdown_flag(returns, market_returns, short_window: int = 21, long_window: int = 126, threshold: float = 0.5) -> pd.Series:
    """Flag bars where short-run correlation falls well below long-run correlation."""
    short = rolling_correlation_to_market(returns, market_returns, short_window)
    long = rolling_correlation_to_market(returns, market_returns, long_window)
    return ((long - short) > threshold).fillna(False).rename("corr_breakdown")


def downside_correlation(returns, market_returns) -> float:
    """Correlation measured only over market-down bars."""
    r = _as_series(returns).astype("float64")
    m = _as_series(market_returns).reindex(r.index).astype("float64")
    mask = np.isfinite(r) & np.isfinite(m) & (m < 0)
    if mask.sum() < 10:
        return np.nan
    rr, mm = r[mask], m[mask]
    if rr.std(ddof=1) <= 0 or mm.std(ddof=1) <= 0:
        return np.nan
    return float(rr.corr(mm))


# ---------------------------------------------------------------------------
# Stop helpers
# ---------------------------------------------------------------------------

def chandelier_stop(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 22, multiple: float = 3.0) -> pd.Series:
    """Chandelier exit for long positions: rolling high minus ``multiple`` ATR."""
    h = _as_series(high)
    a = atr(h, low, close, period)
    return (h.rolling(period, min_periods=period).max() - multiple * a).rename("chandelier_stop")


def atr_stop(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14, multiple: float = 2.0, direction: int = 1) -> pd.Series:
    """ATR stop level below (long) or above (short) the close."""
    c = _as_series(close)
    a = atr(high, low, c, period)
    if direction >= 0:
        return (c - multiple * a).rename("atr_stop")
    return (c + multiple * a).rename("atr_stop")


def trailing_stop(close: pd.Series, pct: float = 0.05, direction: int = 1) -> pd.Series:
    """Percentage trailing stop from the running extreme."""
    c = _as_series(close)
    if direction >= 0:
        return (c.cummax() * (1.0 - pct)).rename("trailing_stop")
    return (c.cummin() * (1.0 + pct)).rename("trailing_stop")


def stop_distance_to_volatility(stop_distance_pct: float, volatility: float, periods_per_year: int = 252) -> float:
    """Stop distance expressed as a multiple of annualized volatility."""
    if volatility <= 0:
        return np.nan
    period_vol = volatility / np.sqrt(periods_per_year)
    if period_vol <= 0:
        return np.nan
    return float(stop_distance_pct / period_vol)
