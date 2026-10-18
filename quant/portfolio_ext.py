"""Portfolio construction, optimization, rebalancing, and attribution.

Works on a returns DataFrame (columns = assets) and weight Series/arrays.
Provides weight estimators (equal, inverse-vol, inverse-variance, mean-variance,
max-Sharpe, risk parity, max diversification, hierarchical-style risk parity,
Black-Litterman, shrinkage, capped/long-only normalization), covariance
estimators (sample, exponentially weighted, shrinkage, Ledoit-Wolf),
rebalancing utilities (calendar, threshold/drift bands, turnover), an efficient
frontier, and attribution/reporting tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.indicators import _as_series
from quant.metrics_ext import drawdown_series, strategy_metrics
from quant.risk_ext import (
    equal_risk_contribution,
    max_diversification_weights,
    min_variance_weights,
    portfolio_volatility,
    risk_contribution_pct,
)
from quant.statistics_ext import KMeans, ewma_cov, ledoit_wolf_shrinkage

__all__ = [
    # basics
    "returns_matrix", "align_returns", "portfolio_returns", "portfolio_equity",
    "normalize_weights", "long_only_weights", "weight_cap", "weights_from_scores",
    # weight estimators
    "equal_weight", "market_cap_weights", "inverse_volatility_weights",
    "inverse_variance_weights", "volatility_scaled_weights", "risk_parity_weights",
    "mean_variance_weights", "max_sharpe_weights", "global_min_variance_weights",
    "max_diversification_portfolio", "hierarchical_risk_parity",
    "black_litterman_weights", "shrinkage_weights", "target_volatility_weights",
    "turnover_penalized_weights",
    # covariance
    "sample_covariance", "exponentially_weighted_covariance", "shrinkage_covariance",
    "ledoit_wolf_covariance", "covariance_to_correlation", "correlation_to_covariance",
    # rebalancing
    "rebalance_dates", "drift_weights", "rebalance_schedule",
    "calendar_rebalance_returns", "threshold_rebalance_returns",
    "rebalancing_turnover",
    # attribution / reporting
    "contribution_series", "contribution_by_asset", "attribution_table",
    "risk_budget_table", "performance_summary", "portfolio_drawdown",
    "downside_risk_portfolio", "marginal_contribution_to_var",
    "active_weights", "active_returns", "diversification_benefit",
    "effective_number_of_assets", "concentration_metrics",
    "rolling_portfolio_volatility", "efficient_frontier",
]


def _frame(returns) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        return returns.astype("float64")
    if isinstance(returns, pd.Series):
        return returns.to_frame(returns.name or "asset").astype("float64")
    return pd.DataFrame(np.asarray(returns, dtype="float64"))


def _w(weights) -> np.ndarray:
    return np.asarray(weights, dtype="float64").ravel()


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------

def returns_matrix(prices) -> pd.DataFrame:
    """Per-period returns of a price matrix."""
    p = _frame(prices)
    return p.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def align_returns(*frames) -> list[pd.DataFrame]:
    """Align several return frames on their common index (inner join)."""
    prepared = [_frame(f) for f in frames]
    if not prepared:
        return []
    common = prepared[0].index
    for f in prepared[1:]:
        common = common.intersection(f.index)
    return [f.reindex(common) for f in prepared]


def portfolio_returns(weights, asset_returns) -> pd.Series:
    """Weighted portfolio return series from static weights."""
    w = _w(weights)
    r = _frame(asset_returns)
    if r.shape[1] != len(w):
        raise ValueError("weights and asset_returns dimension mismatch")
    return pd.Series(r.to_numpy(dtype="float64") @ w, index=r.index, name="portfolio_return")


def portfolio_equity(weights, asset_returns, initial: float = 1.0) -> pd.Series:
    """Compounded equity curve of a static-weight portfolio."""
    r = portfolio_returns(weights, asset_returns).fillna(0.0)
    return (initial * (1.0 + r).cumprod()).rename("equity")


def normalize_weights(weights) -> pd.Series:
    """Scale weights to sum to one (absolute sum when the net is zero)."""
    w = _as_series(weights) if isinstance(weights, pd.Series) else pd.Series(_w(weights))
    total = w.sum()
    if total == 0:
        total = w.abs().sum()
    if total == 0:
        return w.rename("weight")
    return (w / total).rename("weight")


def long_only_weights(weights) -> pd.Series:
    """Clip negatives to zero and renormalize to sum to one."""
    w = _as_series(weights)
    w = w.clip(lower=0.0)
    total = w.sum()
    if total == 0:
        n = len(w)
        return pd.Series(np.full(n, 1.0 / n), index=w.index, name="weight")
    return (w / total).rename("weight")


def weight_cap(weights, max_weight: float = 0.25, iters: int = 50, tol: float = 1e-8) -> pd.Series:
    """Cap individual weights at ``max_weight``, redistributing excess to the rest."""
    w = _as_series(weights).astype("float64")
    w = w / w.sum() if w.sum() != 0 else w
    for _ in range(iters):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        rest = ~over
        if not rest.any():
            break
        room = float((max_weight - w[rest]).clip(lower=0.0).sum())
        if room <= 0:
            break
        w[rest] = w[rest] + excess * (max_weight - w[rest]).clip(lower=0.0) / room
    return (w / w.sum()).rename("weight")


def weights_from_scores(scores, long_short: bool = True) -> pd.Series:
    """Convert cross-sectional scores to weights (demeaned and gross-normalized)."""
    s = _as_series(scores).astype("float64")
    if long_short:
        s = s - s.mean()
    gross = float(s.abs().sum())
    if gross == 0:
        return pd.Series(0.0, index=s.index, name="weight")
    return (s / gross).rename("weight")


# ---------------------------------------------------------------------------
# Weight estimators
# ---------------------------------------------------------------------------

def equal_weight(assets) -> pd.Series:
    """Equal weights across assets (accepts a list, index, or DataFrame)."""
    if isinstance(assets, pd.DataFrame):
        labels = list(assets.columns)
    elif isinstance(assets, pd.Series):
        labels = list(assets.index)
    else:
        labels = list(assets)
    n = max(len(labels), 1)
    return pd.Series(np.full(len(labels), 1.0 / n), index=labels, name="weight")


def market_cap_weights(caps) -> pd.Series:
    """Capitalization weights."""
    c = _as_series(caps).astype("float64").clip(lower=0.0)
    total = c.sum()
    if total == 0:
        return pd.Series(np.full(len(c), 1.0 / max(len(c), 1)), index=c.index, name="weight")
    return (c / total).rename("weight")


def inverse_volatility_weights(returns) -> pd.Series:
    """Weights proportional to inverse volatility."""
    r = _frame(returns)
    vol = r.std(ddof=1)
    inv = 1.0 / vol.replace(0, np.nan)
    total = inv.sum()
    if not np.isfinite(total) or total == 0:
        return equal_weight(r)
    return (inv / total).rename("weight")


def inverse_variance_weights(returns) -> pd.Series:
    """Weights proportional to inverse variance."""
    r = _frame(returns)
    var = r.var(ddof=1)
    inv = 1.0 / var.replace(0, np.nan)
    total = inv.sum()
    if not np.isfinite(total) or total == 0:
        return equal_weight(r)
    return (inv / total).rename("weight")


def volatility_scaled_weights(returns, target_vol: float = 0.10, periods_per_year: int = 252) -> pd.Series:
    """Inverse-vol weights rescaled so the portfolio targets ``target_vol``."""
    base = inverse_volatility_weights(returns)
    r = _frame(returns)
    cov = r.cov()
    pvol = portfolio_volatility(base.to_numpy(), cov) * np.sqrt(periods_per_year)
    if not np.isfinite(pvol) or pvol <= 0:
        return base
    return (base * (target_vol / pvol)).rename("weight")


def risk_parity_weights(returns, iters: int = 200, tol: float = 1e-8) -> pd.Series:
    """Equal-risk-contribution (risk parity) weights."""
    return equal_risk_contribution(returns, iters, tol).rename("weight")


def mean_variance_weights(returns, risk_aversion: float = 1.0) -> pd.Series:
    """Unconstrained mean-variance weights (risk aversion scaled, gross-normalized)."""
    r = _frame(returns)
    mu = r.mean().to_numpy()
    cov = r.cov().to_numpy()
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return equal_weight(r)
    raw = inv @ mu / max(risk_aversion, 1e-9)
    gross = float(np.abs(raw).sum())
    if gross == 0:
        return equal_weight(r)
    return pd.Series(raw / gross, index=r.columns, name="weight")


def max_sharpe_weights(returns, risk_free: float = 0.0) -> pd.Series:
    """Tangency-portfolio weights (closed form, normalized to unit gross)."""
    r = _frame(returns)
    mu = (r.mean() - risk_free / 252.0).to_numpy()
    cov = r.cov().to_numpy()
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return equal_weight(r)
    raw = inv @ mu
    gross = float(np.abs(raw).sum())
    if gross == 0:
        return equal_weight(r)
    return pd.Series(raw / gross, index=r.columns, name="weight")


def global_min_variance_weights(returns) -> pd.Series:
    """Global minimum-variance weights from the sample covariance."""
    r = _frame(returns)
    return min_variance_weights(r.cov()).rename("weight")


def max_diversification_portfolio(returns) -> pd.Series:
    """Maximum-diversification weights."""
    r = _frame(returns)
    return max_diversification_weights(r.cov()).rename("weight")


def hierarchical_risk_parity(returns, n_clusters: int = 2, iters: int = 200) -> pd.Series:
    """Simplified hierarchical risk parity.

    Assets are clustered on correlation structure (KMeans on correlation rows);
    capital is split across clusters by inverse cluster volatility and allocated
    inside each cluster by risk parity.
    """
    r = _frame(returns).dropna()
    if r.shape[1] < 2:
        return equal_weight(r)
    corr = r.corr().fillna(0.0)
    k = int(np.clip(n_clusters, 1, r.shape[1]))
    labels = KMeans(n_clusters=k, seed=0).fit(corr.to_numpy()).labels
    cluster_vol = {}
    for label in np.unique(labels):
        cols = [c for c, lab in zip(r.columns, labels) if lab == label]
        cluster_vol[label] = r[cols].mean(axis=1).std(ddof=1)
    inv = {lab: 1.0 / v if v > 0 else 0.0 for lab, v in cluster_vol.items()}
    total = sum(inv.values())
    if total <= 0:
        return equal_weight(r)
    weights = pd.Series(0.0, index=r.columns, name="weight")
    for label in np.unique(labels):
        cols = [c for c, lab in zip(r.columns, labels) if lab == label]
        share = inv[label] / total
        sub = r[cols]
        sub_w = equal_risk_contribution(sub, iters, 1e-8) if len(cols) > 1 else pd.Series([1.0], index=cols)
        for col, value in sub_w.items():
            weights[col] = share * value
    return weights.rename("weight")


def black_litterman_weights(
    returns,
    market_weights,
    views=None,
    view_confidence: float = 0.5,
    tau: float = 0.05,
    risk_aversion: float = 2.0,
) -> pd.Series:
    """Black-Litterman posterior weights from market-implied equilibrium returns.

    ``views`` is a dict of asset -> expected excess view. Posterior returns blend
    the equilibrium prior with the views, then mean-variance weights are formed.
    """
    r = _frame(returns).dropna()
    cov = r.cov().to_numpy()
    w_mkt = _w(market_weights)
    n = cov.shape[0]
    if len(w_mkt) != n:
        raise ValueError("market_weights and returns dimension mismatch")
    delta = max(risk_aversion, 1e-9)
    pi = delta * cov @ w_mkt
    if views:
        p = np.zeros((len(views), n))
        q = np.zeros(len(views))
        for i, (asset, view) in enumerate(views.items()):
            if asset not in r.columns:
                raise KeyError(f"view asset not in returns: {asset}")
            p[i, list(r.columns).index(asset)] = 1.0
            q[i] = float(view)
        tau_cov = tau * cov
        omega = np.diag(np.diag(p @ tau_cov @ p.T)) / max(view_confidence, 1e-6)
        posterior = np.linalg.pinv(
            np.linalg.pinv(tau_cov) + p.T @ np.linalg.pinv(omega) @ p
        ) @ (np.linalg.pinv(tau_cov) @ pi + p.T @ np.linalg.pinv(omega) @ q)
    else:
        posterior = pi
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return equal_weight(r)
    raw = inv @ posterior / delta
    gross = float(np.abs(raw).sum())
    if gross == 0:
        return equal_weight(r)
    return pd.Series(raw / gross, index=r.columns, name="weight")


def shrinkage_weights(returns, shrink: float = 0.5, risk_aversion: float = 1.0) -> pd.Series:
    """Blend mean-variance weights toward equal weights by ``shrink``."""
    mv = mean_variance_weights(returns, risk_aversion)
    eq = equal_weight(_frame(returns))
    s = float(np.clip(shrink, 0.0, 1.0))
    return ((1.0 - s) * mv + s * eq).rename("weight")


def target_volatility_weights(weights, returns, target_vol: float = 0.10, periods_per_year: int = 252) -> pd.Series:
    """Scale weights (leverage) so realized portfolio volatility hits the target."""
    w = _as_series(weights).astype("float64")
    r = _frame(returns)
    cov = r.cov()
    pvol = portfolio_volatility(w.to_numpy(), cov) * np.sqrt(periods_per_year)
    if not np.isfinite(pvol) or pvol <= 0:
        return w.rename("weight")
    return (w * (target_vol / pvol)).rename("weight")


def turnover_penalized_weights(weights, previous_weights, penalty: float = 0.5) -> pd.Series:
    """Partial-adjust weights toward the previous allocation by ``penalty``."""
    w = _as_series(weights).astype("float64")
    prev = _as_series(previous_weights).reindex(w.index).astype("float64")
    p = float(np.clip(penalty, 0.0, 1.0))
    return ((1.0 - p) * w + p * prev).rename("weight")


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def sample_covariance(returns) -> pd.DataFrame:
    """Sample covariance matrix."""
    return _frame(returns).cov()


def exponentially_weighted_covariance(returns, halflife: int = 63) -> pd.DataFrame:
    """Exponentially weighted covariance matrix."""
    r = _frame(returns)
    lam = float(np.exp(np.log(0.5) / max(halflife, 1)))
    arr = ewma_cov(r.dropna(), lam=lam)
    return pd.DataFrame(arr, index=r.columns, columns=r.columns)


def shrinkage_covariance(returns, shrink: float = 0.2) -> pd.DataFrame:
    """Shrink the sample covariance toward its diagonal."""
    r = _frame(returns)
    cov = r.cov()
    s = float(np.clip(shrink, 0.0, 1.0))
    diag = pd.DataFrame(np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns)
    return ((1.0 - s) * cov + s * diag)


def ledoit_wolf_covariance(returns) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance matrix."""
    r = _frame(returns)
    out = ledoit_wolf_shrinkage(r.dropna())
    return pd.DataFrame(out["cov"], index=r.columns, columns=r.columns)


def covariance_to_correlation(cov) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    c = _frame(cov)
    vals = c.to_numpy(dtype="float64")
    std = np.sqrt(np.clip(np.diag(vals), 1e-18, None))
    corr = vals / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=c.index, columns=c.columns)


def correlation_to_covariance(corr, vols) -> pd.DataFrame:
    """Convert a correlation matrix plus volatilities to a covariance matrix."""
    c = _frame(corr).to_numpy(dtype="float64")
    s = _w(vols)
    if len(s) != c.shape[0]:
        raise ValueError("vols length mismatch")
    return pd.DataFrame(c * np.outer(s, s))


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

def rebalance_dates(index, freq: str = "ME") -> pd.DatetimeIndex:
    """Trading dates on which a periodic rebalance occurs."""
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return idx
    marks = pd.Series(1.0, index=idx).resample(freq).last().dropna().index
    dates = []
    for mark in marks:
        pos = idx.searchsorted(mark, side="right") - 1
        if pos >= 0:
            dates.append(idx[pos])
    return pd.DatetimeIndex(sorted(set(dates)))


def drift_weights(weights, asset_returns) -> pd.Series:
    """Weights after buy-and-hold drift over the sample (no rebalancing)."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    growth = (1.0 + r.fillna(0.0)).cumprod().iloc[-1]
    values = w * growth
    total = values.sum()
    if total == 0:
        return w.rename("weight")
    return (values / total).rename("weight")


def rebalance_schedule(weights, asset_returns, freq: str = "ME", band: float | None = None) -> pd.DataFrame:
    """Target weights on each rebalance date given calendar and/or drift bands."""
    w0 = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    dates = set(rebalance_dates(r.index, freq)) if freq else set()
    current = w0.to_numpy(dtype="float64")
    rows = {}
    for i, dt in enumerate(r.index):
        ret = r.iloc[i].fillna(0.0).to_numpy() if i > 0 else np.zeros(len(current))
        if i > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                values = current * (1.0 + ret)
            total = values.sum()
            current = values / total if total != 0 else current
        trigger = dt in dates
        if band is not None and w0.to_numpy().sum() != 0:
            drift = np.abs(current - w0.to_numpy())
            if (drift > band).any():
                trigger = True
        if trigger:
            rows[dt] = w0.to_numpy().copy()
            current = w0.to_numpy().copy()
    if not rows:
        rows[r.index[0]] = w0.to_numpy()
    return pd.DataFrame(rows, index=w0.index).T


def calendar_rebalance_returns(weights, asset_returns, freq: str = "ME", cost: float = 0.0) -> pd.Series:
    """Portfolio returns with calendar rebalancing at each period end.

    Weights are held into each bar, drift with returns, and reset to target on
    rebalance dates; ``cost`` is charged per unit of turnover (|target - held|).
    """
    r = _frame(asset_returns)
    target = _as_series(weights).astype("float64").to_numpy()
    rebal_set = set(rebalance_dates(r.index, freq)) if freq else set()
    held = target.copy()
    out = np.full(len(r), np.nan)
    for i, dt in enumerate(r.index):
        ret = r.iloc[i].fillna(0.0).to_numpy()
        pnl = float(held @ ret)
        if i > 0 and dt in rebal_set and cost:
            pnl -= cost * float(np.abs(target - held).sum())
        out[i] = pnl
        if dt in rebal_set:
            held = target.copy()
        else:
            values = held * (1.0 + ret)
            total = values.sum()
            held = values / total if total != 0 else held
    return pd.Series(out, index=r.index, name="portfolio_return")


def threshold_rebalance_returns(weights, asset_returns, band: float = 0.05, cost: float = 0.0) -> pd.Series:
    """Portfolio returns with drift-band (threshold) rebalancing."""
    effective = band if band > 0 else 1e-9
    return _band_returns(weights, asset_returns, effective, cost)


def _band_returns(weights, asset_returns, band: float, cost: float) -> pd.Series:
    r = _frame(asset_returns)
    target = _as_series(weights).astype("float64").to_numpy()
    held = target.copy()
    out = np.full(len(r), np.nan)
    for i, dt in enumerate(r.index):
        ret = r.iloc[i].fillna(0.0).to_numpy()
        pnl = float(held @ ret)
        breach = bool((np.abs(held - target) > band).any())
        if i > 0 and breach and cost:
            pnl -= cost * float(np.abs(target - held).sum())
        out[i] = pnl
        if breach:
            held = target.copy()
        else:
            values = held * (1.0 + ret)
            total = values.sum()
            held = values / total if total != 0 else held
    return pd.Series(out, index=r.index, name="portfolio_return")


def rebalancing_turnover(schedule: pd.DataFrame) -> pd.Series:
    """One-way turnover between consecutive target-weight rows of a schedule.

    The first row is treated as a full initial build (turnover = gross weight).
    """
    if len(schedule) == 0:
        return pd.Series(dtype="float64")
    turn = schedule.diff().abs().sum(axis=1)
    turn.iloc[0] = float(schedule.iloc[0].abs().sum())
    return turn.rename("turnover")


# ---------------------------------------------------------------------------
# Attribution / reporting
# ---------------------------------------------------------------------------

def contribution_series(weights, asset_returns) -> pd.DataFrame:
    """Per-period contribution of each asset to the portfolio return."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    return r.mul(w, axis=1)


def contribution_by_asset(weights, asset_returns) -> pd.Series:
    """Total compounded-times-weight contribution per asset (sum of contributions)."""
    return contribution_series(weights, asset_returns).sum().rename("contribution")


def attribution_table(weights, asset_returns) -> pd.DataFrame:
    """Weights, mean return, total contribution, and risk contribution per asset."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    contrib = contribution_series(w, r)
    table = pd.DataFrame(
        {
            "weight": w,
            "mean_return": r.mean(),
            "volatility": r.std(ddof=1),
            "total_contribution": contrib.sum(),
            "risk_contribution_pct": risk_contribution_pct(w.to_numpy(), r.cov()),
        }
    )
    return table


def risk_budget_table(weights, asset_returns) -> pd.DataFrame:
    """Absolute and percentage risk contributions per asset."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    cov = r.cov()
    pct = risk_contribution_pct(w.to_numpy(), cov)
    return pd.DataFrame(
        {
            "weight": w,
            "risk_pct": pct,
            "weight_minus_risk": w - pct,
        }
    )


def performance_summary(weights, asset_returns, benchmark=None, periods_per_year: int = 252) -> dict:
    """Metric bundle of a static-weight portfolio (see ``strategy_metrics``)."""
    pr = portfolio_returns(weights, asset_returns)
    return strategy_metrics(pr, benchmark, periods_per_year)


def portfolio_drawdown(weights, asset_returns) -> pd.Series:
    """Drawdown series of a static-weight portfolio."""
    return drawdown_series(portfolio_returns(weights, asset_returns))


def downside_risk_portfolio(weights, asset_returns, mar: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized downside deviation of the portfolio return stream."""
    pr = portfolio_returns(weights, asset_returns)
    downside = pr.clip(upper=mar / periods_per_year)
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def marginal_contribution_to_var(weights, asset_returns, level: float = 0.05) -> pd.Series:
    """Parametric component VaR per asset (sums to total parametric VaR)."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    cov = r.cov().to_numpy(dtype="float64")
    mu = r.mean().to_numpy()
    pvol = np.sqrt(max(float(w.to_numpy() @ cov @ w.to_numpy()), 0.0))
    if pvol <= 0:
        return pd.Series(np.nan, index=w.index, name="component_var")
    from quant.metrics_ext import _norm_ppf
    z = _norm_ppf(level)
    mrc = (cov @ w.to_numpy()) / pvol
    return pd.Series(w.to_numpy() * (mu + z * mrc), index=w.index, name="component_var")


def active_weights(weights, benchmark_weights) -> pd.Series:
    """Difference between portfolio and benchmark weights."""
    w = _as_series(weights).astype("float64")
    b = _as_series(benchmark_weights).reindex(w.index).astype("float64")
    return (w - b).rename("active_weight")


def active_returns(weights, asset_returns, benchmark_weights) -> pd.Series:
    """Active (portfolio minus benchmark) return stream."""
    return (portfolio_returns(weights, asset_returns) - portfolio_returns(benchmark_weights, asset_returns)).rename("active_return")


def diversification_benefit(weights, asset_returns, periods_per_year: int = 252) -> float:
    """Weighted-average asset volatility minus portfolio volatility."""
    w = _as_series(weights).astype("float64")
    r = _frame(asset_returns)
    port = portfolio_volatility(w.to_numpy(), r.cov()) * np.sqrt(periods_per_year)
    avg = float((w.abs() * r.std(ddof=1) * np.sqrt(periods_per_year)).sum())
    return float(avg - port)


def effective_number_of_assets(weights) -> float:
    """Effective number of holdings (1 / Herfindahl index of weights)."""
    w = _as_series(weights).astype("float64").abs()
    total = w.sum()
    if total == 0:
        return np.nan
    p = w / total
    return float(1.0 / (p ** 2).sum())


def concentration_metrics(weights) -> dict:
    """Concentration diagnostics: max weight, top-5 share, HHI, effective number."""
    w = _as_series(weights).astype("float64").abs()
    total = w.sum()
    p = (w / total) if total > 0 else w
    return {
        "max_weight": float(p.max()) if len(p) else np.nan,
        "top5_share": float(p.nlargest(5).sum()) if len(p) else np.nan,
        "hhi": float((p ** 2).sum()) if len(p) else np.nan,
        "effective_number": effective_number_of_assets(w),
    }


def rolling_portfolio_volatility(weights, asset_returns, window: int = 63, periods_per_year: int = 252) -> pd.Series:
    """Rolling annualized volatility of the static-weight portfolio."""
    pr = portfolio_returns(weights, asset_returns)
    return (pr.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)).rename("portfolio_vol")


def efficient_frontier(returns, n_points: int = 20) -> pd.DataFrame:
    """Efficient frontier points (annualized risk/return) via two-fund separation."""
    r = _frame(returns).dropna()
    mu = r.mean().to_numpy() * 252.0
    cov = r.cov().to_numpy() * 252.0
    if r.shape[1] < 2:
        return pd.DataFrame({"volatility": [np.sqrt(cov[0, 0])], "return": [mu[0]]})
    try:
        inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=["volatility", "return"])
    ones = np.ones(len(mu))
    a = float(ones @ inv @ ones)
    b = float(ones @ inv @ mu)
    c = float(mu @ inv @ mu)
    det = a * c - b * b
    if abs(det) < 1e-18:
        return pd.DataFrame(columns=["volatility", "return"])
    r_mv = b / a if a != 0 else float(mu.min())
    lo, hi = float(max(r_mv, mu.min())), float(mu.max())
    if hi <= lo:
        hi = lo + 1e-9
    rows = []
    for target in np.linspace(lo, hi, int(max(n_points, 2))):
        w = (c - b * target) / det * (inv @ ones) + (a * target - b) / det * (inv @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        rows.append({"volatility": vol, "return": float(target)})
    return pd.DataFrame(rows)