"""Risk analytics and portfolio optimization.

Provides:
- ``risk``: drawdown, tail-risk, regime, and stress-test analytics
- ``portfolio``: mean-variance, risk-parity, Black-Litterman, and
  hierarchical risk parity optimizers
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    # Risk analytics
    "drawdown_series", "max_drawdown", "avg_drawdown", "drawdown_duration",
    "ulcer_index", "pain_index", "pain_ratio", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "omega_ratio", "var_historical", "var_parametric",
    "cvar_historical", "cvar_parametric", "expected_shortfall",
    "tail_ratio", "gain_to_pain_ratio", "common_sense_ratio",
    "cagr", "volatility", "downside_volatility", "upside_volatility",
    "skewness", "kurtosis", "var_cone", "stress_test",
    "regime_detection", "rolling_sharpe", "rolling_sortino",
    "rolling_var", "rolling_cvar",
    # Portfolio optimization
    "mean_variance_optimize", "risk_parity_optimize",
    "minimum_variance_optimize", "maximum_sharpe_optimize",
    "black_litterman_optimize", "hierarchical_risk_parity",
    "equal_weight", "market_cap_weight", "inverse_volatility_weight",
    "diversification_ratio", "effective_number_of_bets",
    "portfolio_volatility", "portfolio_return", "portfolio_sharpe",
    "constraints", "project_weights",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x, dtype="float64")


def _safe_div(a, b, fill: float = 0.0):
    out = a / b.replace(0, np.nan)
    if isinstance(out, pd.Series):
        return out.fillna(fill)
    return np.where(b == 0, fill, out)


# ---------------------------------------------------------------------------
# Risk analytics
# ---------------------------------------------------------------------------

def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown series from an equity curve."""
    e = _as_series(equity)
    peak = e.cummax()
    return _safe_div(e - peak, peak)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (negative number)."""
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd.dropna()) else np.nan


def avg_drawdown(equity: pd.Series) -> float:
    """Average drawdown magnitude."""
    dd = drawdown_series(equity).dropna()
    dd = dd[dd < 0]
    return float(dd.mean()) if len(dd) else np.nan


def drawdown_duration(equity: pd.Series) -> int:
    """Maximum drawdown duration in periods."""
    e = _as_series(equity)
    peak = e.cummax()
    underwater = e < peak
    # Count consecutive underwater periods
    groups = (underwater != underwater.shift(1)).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if len(durations) else 0


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer index (RMS of drawdown in percent)."""
    dd = drawdown_series(equity).dropna() * 100.0
    if dd.empty:
        return np.nan
    return float(np.sqrt((dd ** 2).mean()))


def pain_index(equity: pd.Series) -> float:
    """Pain index (average drawdown in percent)."""
    dd = drawdown_series(equity).dropna() * 100.0
    if dd.empty:
        return np.nan
    return float(dd.mean())


def pain_ratio(equity: pd.Series) -> float:
    """Pain ratio: CAGR / Pain Index."""
    e = _as_series(equity).dropna()
    if len(e) < 2:
        return np.nan
    years = len(e) / 252.0
    if years <= 0:
        return np.nan
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if e.iloc[0] > 0 else np.nan
    pi = pain_index(e)
    if pi == 0 or np.isnan(pi):
        return np.nan
    return float(cagr / pi)


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate."""
    e = _as_series(equity).dropna()
    if len(e) < 2 or e.iloc[0] <= 0:
        return np.nan
    years = len(e) / periods_per_year
    return float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1)


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized volatility."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def downside_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized downside volatility."""
    r = _as_series(returns).dropna()
    downside = r[r < 0]
    if len(downside) < 2:
        return np.nan
    return float(downside.std(ddof=1) * np.sqrt(periods_per_year))


def upside_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized upside volatility."""
    r = _as_series(returns).dropna()
    upside = r[r > 0]
    if len(upside) < 2:
        return np.nan
    return float(upside.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf / periods_per_year
    downside = excess[excess < 0].std(ddof=1)
    if not downside or np.isnan(downside) or downside == 0:
        return np.nan
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio: CAGR / |MaxDD|."""
    e = _as_series(equity).dropna()
    if len(e) < 2:
        return np.nan
    c = cagr(e, periods_per_year)
    mdd = abs(max_drawdown(e))
    if mdd == 0 or np.isnan(mdd) or np.isnan(c):
        return np.nan
    return float(c / mdd)


def omega_ratio(returns: pd.Series, threshold: float = 0.0, periods_per_year: int = 252) -> float:
    """Omega ratio at a return threshold."""
    r = _as_series(returns).dropna() - threshold / periods_per_year
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (negative number)."""
    r = _as_series(returns).dropna()
    if len(r) == 0:
        return np.nan
    return float(np.percentile(r, (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (normal) VaR."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    from scipy import stats
    return float(stats.norm.ppf(1 - confidence, loc=r.mean(), scale=r.std(ddof=1)))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Conditional VaR / Expected Shortfall."""
    r = _as_series(returns).dropna()
    if len(r) == 0:
        return np.nan
    var = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else np.nan


def cvar_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric CVaR."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    from scipy import stats
    z = stats.norm.ppf(1 - confidence)
    pdf = stats.norm.pdf(z)
    return float(r.mean() - r.std(ddof=1) * pdf / (1 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Alias for historical CVaR."""
    return cvar_historical(returns, confidence)


def tail_ratio(returns: pd.Series, confidence: float = 0.95) -> float:
    """Tail ratio: average gain in upper tail / average loss in lower tail."""
    r = _as_series(returns).dropna()
    if len(r) < 2:
        return np.nan
    upper = np.percentile(r, confidence * 100)
    lower = np.percentile(r, (1 - confidence) * 100)
    avg_gain = r[r >= upper].mean()
    avg_loss = r[r <= lower].mean()
    if avg_loss == 0 or np.isnan(avg_loss):
        return np.nan
    return float(avg_gain / abs(avg_loss))


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Gain-to-pain ratio: sum of gains / sum of losses."""
    r = _as_series(returns).dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def common_sense_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Common sense ratio: sum of gains above threshold / sum of losses below."""
    r = _as_series(returns).dropna() - threshold
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def skewness(returns: pd.Series) -> float:
    """Sample skewness."""
    r = _as_series(returns).dropna()
    if len(r) < 3:
        return np.nan
    return float(r.skew())


def kurtosis(returns: pd.Series) -> float:
    """Sample excess kurtosis."""
    r = _as_series(returns).dropna()
    if len(r) < 4:
        return np.nan
    return float(r.kurt())


def var_cone(
    returns: pd.Series, windows=(5, 10, 20, 30, 60, 90, 120, 252),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """VaR cone across lookback windows."""
    r = _as_series(returns)
    rows = {}
    for w in windows:
        rv = r.iloc[-w:] if len(r) >= w else r
        rows[w] = {
            "var": np.percentile(rv, (1 - confidence) * 100),
            "cvar": rv[rv <= np.percentile(rv, (1 - confidence) * 100)].mean(),
            "vol": rv.std(ddof=1) * np.sqrt(252),
        }
    return pd.DataFrame(rows).T


def stress_test(
    returns: pd.Series, shocks: dict[str, float] | None = None
) -> pd.DataFrame:
    """Apply hypothetical shocks to returns and compute portfolio impact."""
    shocks = shocks or {
        "equity_crash": -0.20,
        "vol_spike": -0.10,
        "liquidity_crisis": -0.15,
        "rate_shock": -0.05,
    }
    r = _as_series(returns).dropna()
    results = {}
    for name, shock in shocks.items():
        shocked = r + shock
        results[name] = {
            "mean_return": float(shocked.mean()),
            "var_95": float(np.percentile(shocked, 5)),
            "max_loss": float(shocked.min()),
        }
    return pd.DataFrame(results).T


def regime_detection(returns: pd.Series, window: int = 60) -> pd.Series:
    """Simple regime detection: low/med/high volatility."""
    vol = returns.rolling(window).std(ddof=1)
    vol_median = vol.median()
    regime = pd.Series("medium", index=returns.index)
    regime[vol < vol_median * 0.7] = "low"
    regime[vol > vol_median * 1.3] = "high"
    return regime


def rolling_sharpe(returns: pd.Series, window: int = 60, rf: float = 0.0) -> pd.Series:
    """Rolling Sharpe ratio."""
    r = _as_series(returns)
    excess = r - rf / 252.0
    mean = excess.rolling(window).mean()
    sd = excess.rolling(window).std(ddof=1)
    return _safe_div(mean, sd) * np.sqrt(252)


def rolling_sortino(returns: pd.Series, window: int = 60, rf: float = 0.0) -> pd.Series:
    """Rolling Sortino ratio."""
    r = _as_series(returns)
    excess = r - rf / 252.0
    downside = excess.where(excess < 0).rolling(window).std(ddof=1)
    mean = excess.rolling(window).mean()
    return _safe_div(mean, downside) * np.sqrt(252)


def rolling_var(returns: pd.Series, window: int = 60, confidence: float = 0.95) -> pd.Series:
    """Rolling historical VaR."""
    r = _as_series(returns)
    return r.rolling(window).apply(
        lambda x: float(np.percentile(x, (1 - confidence) * 100)), raw=True
    )


def rolling_cvar(returns: pd.Series, window: int = 60, confidence: float = 0.95) -> pd.Series:
    """Rolling historical CVaR."""
    r = _as_series(returns)

    def _cvar(x):
        var = np.percentile(x, (1 - confidence) * 100)
        tail = x[x <= var]
        return float(tail.mean()) if len(tail) else np.nan

    return r.rolling(window).apply(_cvar, raw=True)


# ---------------------------------------------------------------------------
# Portfolio optimization
# ---------------------------------------------------------------------------

def equal_weight(n_assets: int) -> np.ndarray:
    """Equal-weight portfolio."""
    return np.full(n_assets, 1.0 / n_assets)


def market_cap_weight(market_caps: pd.Series) -> np.ndarray:
    """Market-cap weighted portfolio."""
    mc = market_caps.replace(0, np.nan).fillna(0.0)
    total = mc.sum()
    if total == 0:
        return equal_weight(len(mc))
    return (mc / total).to_numpy(dtype="float64")


def inverse_volatility_weight(returns: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Inverse-volatility weighted portfolio."""
    vol = returns.rolling(lookback).std(ddof=1).iloc[-1]
    inv_vol = 1.0 / vol.replace(0, np.nan)
    total = inv_vol.sum()
    if total == 0 or np.isnan(total):
        return equal_weight(len(vol))
    return (inv_vol / total).to_numpy(dtype="float64")


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    """Single-period portfolio return."""
    w = np.asarray(weights, dtype="float64")
    r = returns.iloc[-1].to_numpy(dtype="float64")
    return float(np.dot(w, r))


def portfolio_volatility(weights: np.ndarray, returns: pd.DataFrame, lookback: int = 60) -> float:
    """Portfolio volatility from rolling covariance."""
    w = np.asarray(weights, dtype="float64")
    cov = returns.rolling(lookback).cov().iloc[-lookback:].values.reshape(lookback, -1, -1)
    # Use last covariance matrix
    cov_last = cov[-1]
    port_var = float(w @ cov_last @ w)
    return float(np.sqrt(max(port_var, 0.0)) * np.sqrt(252))


def portfolio_sharpe(weights: np.ndarray, returns: pd.DataFrame, lookback: int = 60, rf: float = 0.0) -> float:
    """Portfolio Sharpe ratio."""
    w = np.asarray(weights, dtype="float64")
    mean = returns.rolling(lookback).mean().iloc[-1].to_numpy(dtype="float64")
    excess = mean - rf / 252.0
    port_ret = float(np.dot(w, excess)) * 252.0
    port_vol = portfolio_volatility(w, returns, lookback)
    if port_vol == 0 or np.isnan(port_vol):
        return np.nan
    return float(port_ret / port_vol)


def diversification_ratio(weights: np.ndarray, returns: pd.DataFrame, lookback: int = 60) -> float:
    """Diversification ratio: weighted avg vol / portfolio vol."""
    w = np.asarray(weights, dtype="float64")
    vol = returns.rolling(lookback).std(ddof=1).iloc[-1].to_numpy(dtype="float64")
    weighted_avg_vol = float(np.dot(w, vol))
    port_vol = portfolio_volatility(w, returns, lookback)
    if port_vol == 0 or np.isnan(port_vol):
        return np.nan
    return float(weighted_avg_vol / port_vol)


def effective_number_of_bets(weights: np.ndarray) -> float:
    """Effective number of bets (Herfindahl-based)."""
    w = np.asarray(weights, dtype="float64")
    w = np.abs(w)
    total = w.sum()
    if total == 0:
        return np.nan
    w_norm = w / total
    hhi = float(np.sum(w_norm ** 2))
    return float(1.0 / hhi) if hhi > 0 else np.nan


def constraints(
    weights: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    gross_exposure: float = 1.0,
    net_exposure: float = 0.0,
) -> np.ndarray:
    """Project weights onto constraint set (simple clip + rescale)."""
    w = np.asarray(weights, dtype="float64").copy()
    w = np.clip(w, min_weight, max_weight)
    # Rescale to gross exposure
    gross = np.abs(w).sum()
    if gross > 0:
        w = w * (gross_exposure / gross)
    # Adjust net exposure
    current_net = w.sum()
    if abs(current_net - net_exposure) > 1e-6:
        adjustment = (net_exposure - current_net) / len(w)
        w = w + adjustment
        w = np.clip(w, min_weight, max_weight)
    return w


def project_weights(weights: np.ndarray, gross_exposure: float = 1.0) -> np.ndarray:
    """Project weights to satisfy gross exposure constraint."""
    w = np.asarray(weights, dtype="float64")
    gross = np.abs(w).sum()
    if gross == 0:
        return w
    return w * (gross_exposure / gross)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 3.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    gross_exposure: float = 1.0,
) -> np.ndarray:
    """Mean-variance optimization (analytic approximation)."""
    mu = expected_returns.to_numpy(dtype="float64")
    Sigma = cov_matrix.to_numpy(dtype="float64")
    n = len(mu)
    # Regularize covariance
    Sigma = Sigma + 1e-6 * np.eye(n)
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.eye(n)
    # Unconstrained optimal: w = (1/risk_aversion) * Sigma^-1 * mu
    w = (1.0 / risk_aversion) * inv_Sigma @ mu
    # Apply constraints
    w = constraints(w, min_weight, max_weight, gross_exposure, 0.0)
    return w


def risk_parity_optimize(
    cov_matrix: pd.DataFrame,
    target_risk_contrib: pd.Series | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Risk parity optimization (equal risk contribution)."""
    Sigma = cov_matrix.to_numpy(dtype="float64")
    Sigma = Sigma + 1e-8 * np.eye(n := Sigma.shape[0])
    if target_risk_contrib is None:
        target = np.full(n, 1.0 / n)
    else:
        target = target_risk_contrib.to_numpy(dtype="float64")
        target = target / target.sum()
    # Iterative risk budgeting
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        port_vol = np.sqrt(max(w @ Sigma @ w, 1e-12))
        marginal_contrib = Sigma @ w / port_vol
        risk_contrib = w * marginal_contrib / port_vol
        error = risk_contrib - target
        if np.max(np.abs(error)) < tol:
            break
        # Gradient descent step
        grad = 2 * (risk_contrib - target) * marginal_contrib
        w = w - 0.01 * grad
        w = np.clip(w, 0.0, None)
        total = w.sum()
        if total > 0:
            w = w / total
    return w


def minimum_variance_optimize(
    cov_matrix: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    gross_exposure: float = 1.0,
) -> np.ndarray:
    """Minimum variance portfolio (analytic approximation)."""
    Sigma = cov_matrix.to_numpy(dtype="float64")
    Sigma = Sigma + 1e-6 * np.eye(n := Sigma.shape[0])
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.eye(n)
    ones = np.ones(n)
    w = inv_Sigma @ ones
    w = w / w.sum()
    w = constraints(w, min_weight, max_weight, gross_exposure, 0.0)
    return w


def maximum_sharpe_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rf: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    gross_exposure: float = 1.0,
) -> np.ndarray:
    """Maximum Sharpe ratio portfolio (tangency portfolio approximation)."""
    mu = expected_returns.to_numpy(dtype="float64") - rf / 252.0
    Sigma = cov_matrix.to_numpy(dtype="float64")
    Sigma = Sigma + 1e-6 * np.eye(n := Sigma.shape[0])
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.eye(n)
    w = inv_Sigma @ mu
    total = w.sum()
    if total != 0:
        w = w / total
    w = constraints(w, min_weight, max_weight, gross_exposure, 0.0)
    return w


def black_litterman_optimize(
    market_caps: pd.Series,
    cov_matrix: pd.DataFrame,
    views: pd.Series | None = None,
    view_confidence: float = 0.5,
    tau: float = 0.05,
    risk_aversion: float = 3.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    gross_exposure: float = 1.0,
) -> np.ndarray:
    """Black-Litterman optimization (simplified)."""
    mc = market_caps.to_numpy(dtype="float64")
    mc = mc / mc.sum()
    Sigma = cov_matrix.to_numpy(dtype="float64")
    Sigma = Sigma + 1e-6 * np.eye(n := Sigma.shape[0])
    # Equilibrium returns
    pi = risk_aversion * Sigma @ mc
    if views is not None:
        v = views.to_numpy(dtype="float64")
        # Blend: blended = (1-tau)*pi + tau*views (simplified)
        blended = (1 - view_confidence) * pi + view_confidence * v
    else:
        blended = pi
    # Optimal weights
    try:
        inv_Sigma = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        inv_Sigma = np.eye(n)
    w = (1.0 / risk_aversion) * inv_Sigma @ blended
    w = constraints(w, min_weight, max_weight, gross_exposure, 0.0)
    return w


def hierarchical_risk_parity(
    returns: pd.DataFrame, lookback: int = 252,
) -> np.ndarray:
    """Hierarchical Risk Parity (HRP) - simplified implementation."""
    cov = returns.rolling(lookback).cov().iloc[-lookback:].values.reshape(lookback, -1, -1)[-1]
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    n = cov.shape[0]
    # Start with equal weights, then recursively bisect by clustering
    # Simplified: use inverse volatility within clusters
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / np.where(vol == 0, np.nan, vol)
    inv_vol = np.where(np.isnan(inv_vol), 0.0, inv_vol)
    w = inv_vol / inv_vol.sum()
    return w
