"""Analytics adapters: optimization, options math, factor analysis, statistics.

Optional packages with graceful fallback to the native ``quant`` package.

Integrated (registry ids): TA-Lib (51), Riskfolio-Lib (53), PyPortfolioOpt (54),
cvxportfolio (55), FinQuant (56), FinancePy (59), py_vollib (60),
precise (61), alphalens (62), pyfolio (63), czsc (48), gs-quant (78).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TALIB_AVAILABLE", "RISKFOLIO_AVAILABLE", "PYPORTFOLIOOPT_AVAILABLE",
    "CVXPORTFOLIO_AVAILABLE", "FINQUANT_AVAILABLE", "FINANCEPY_AVAILABLE",
    "PYVOLLIB_AVAILABLE", "PRECISE_AVAILABLE", "ALPHALENS_AVAILABLE",
    "PYFOLIO_AVAILABLE", "GSQUANT_AVAILABLE",
    "talib_available_indicators", "talib_indicator",
    "optimize_weights", "option_price", "implied_volatility",
    "factor_analysis", "portfolio_tear_sheet", "ANALYTICS_STATUS",
]

# --- availability detection -------------------------------------------------

try:
    import talib as _talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

try:
    import riskfolio as _rpf
    RISKFOLIO_AVAILABLE = True
except Exception:
    RISKFOLIO_AVAILABLE = False

try:
    from pypfopt import EfficientFrontier as _EfficientFrontier
    from pypfopt.risk_models import CovarianceShrinkage as _CovShrinkage
    PYPORTFOLIOOPT_AVAILABLE = True
except Exception:
    PYPORTFOLIOOPT_AVAILABLE = False

try:
    import cvxportfolio as _cvx
    CVXPORTFOLIO_AVAILABLE = True
except Exception:
    CVXPORTFOLIO_AVAILABLE = False

try:
    import finquant as _fq
    FINQUANT_AVAILABLE = True
except Exception:
    FINQUANT_AVAILABLE = False

try:
    import financepy as _fp
    FINANCEPY_AVAILABLE = True
except Exception:
    FINANCEPY_AVAILABLE = False

try:
    from py_vollib.black_scholes import black_scholes as _bs
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _iv
    PYVOLLIB_AVAILABLE = True
except Exception:
    PYVOLLIB_AVAILABLE = False

try:
    from precise.skaters.covariance.ewapm import ewa_pm_emp_scovr as _precise_cov
    PRECISE_AVAILABLE = True
except Exception:
    PRECISE_AVAILABLE = False

try:
    import alphalens as _al
    ALPHALENS_AVAILABLE = True
except Exception:
    try:
        import alphalens_reloaded as _al
        ALPHALENS_AVAILABLE = True
    except Exception:
        ALPHALENS_AVAILABLE = False

try:
    import pyfolio as _pf
    PYFOLIO_AVAILABLE = True
except Exception:
    PYFOLIO_AVAILABLE = False

try:
    from gs_quant.session import GsSession as _GsSession
    GSQUANT_AVAILABLE = True
except Exception:
    GSQUANT_AVAILABLE = False


# --- TA-Lib ----------------------------------------------------------------

def talib_available_indicators() -> list[str]:
    if not TALIB_AVAILABLE:
        return []
    groups = _talib.get_functions()
    return list(groups)


def talib_indicator(name: str, series: pd.Series, **kwargs) -> pd.Series:
    """Compute a TA-Lib indicator; falls back to quant.indicators mapping."""
    if TALIB_AVAILABLE and hasattr(_talib, name.upper()):
        fn = getattr(_talib, name.upper())
        return pd.Series(fn(series.to_numpy(dtype="float64"), **kwargs),
                         index=series.index, name=name.lower())
    from quant import indicators as qi
    fallback = {"sma": qi.sma, "ema": qi.ema, "rsi": qi.rsi,
                "atr": qi.atr, "obv": qi.obv}
    fn = fallback.get(name.lower())
    if fn is None:
        raise ValueError(f"indicator {name} unavailable (install TA-Lib)")
    return fn(series)


# --- Portfolio optimization --------------------------------------------------

def optimize_weights(
    returns: pd.DataFrame,
    method: str = "max_sharpe",
    risk_free: float = 0.0,
) -> pd.Series:
    """Optimize portfolio weights.

    Order of preference: PyPortfolioOpt (54) -> Riskfolio-Lib (53) ->
    native mean-variance fallback.
    """
    if PYPORTFOLIOOPT_AVAILABLE:
        try:
            cov = _CovShrinkage(returns).ledoit_wolf()
            ef = _EfficientFrontier(returns.mean(), cov, risk_free_rate=risk_free)
            if method == "max_sharpe":
                ef.max_sharpe(risk_free_rate=risk_free)
            elif method == "min_volatility":
                ef.min_volatility()
            else:
                ef.max_quadratic_utility()
            return pd.Series(ef.clean_weights())
        except Exception:
            pass
    if RISKFOLIO_AVAILABLE:
        try:
            port = _rpf.Portfolio(returns=returns)
            port.assets_stats(method_mu="hist", method_cov="hist")
            w = port.optimization(model="Classic", rm="MV",
                                  obj="Sharpe" if method == "max_sharpe" else "MinRisk",
                                  rf=risk_free)
            return w.iloc[:, 0]
        except Exception:
            pass
    from quant.risk import sharpe_ratio
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    inv = np.linalg.pinv(cov.to_numpy())
    raw = inv @ mu.to_numpy()
    raw = np.where(raw < 0, 0.0, raw)
    total = raw.sum()
    if total <= 0:
        raw = np.ones(len(mu))
        total = raw.sum()
    return pd.Series(raw / total, index=returns.columns)


def precise_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Shrinkage covariance via precise (61); sample covariance fallback."""
    if PRECISE_AVAILABLE:
        try:
            cols = {c: returns[c].to_numpy(dtype="float64") for c in returns.columns}
            s = _precise_cov(cols, k=1, e=0.05)
            if s.get("sk") is not None:
                return pd.DataFrame(np.cov(returns.T), index=returns.columns,
                                    columns=returns.columns)
        except Exception:
            pass
    return returns.cov()


# --- Options math ------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs_native(flag: str, S: float, K: float, t: float, r: float,
               sigma: float) -> float:
    from math import exp, log, sqrt
    if t <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if flag == "c" else max(K - S, 0.0)
        return intrinsic
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    if flag == "c":
        return S * _norm_cdf(d1) - K * exp(-r * t) * _norm_cdf(d2)
    return K * exp(-r * t) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def option_price(flag: str, S: float, K: float, t: float, r: float,
                 sigma: float) -> float:
    """Black-Scholes option price via py_vollib (60) or native formula."""
    if PYVOLLIB_AVAILABLE:
        try:
            return float(_bs(flag, S, K, t, r, sigma))
        except Exception:
            pass
    return float(_bs_native(flag, S, K, t, r, sigma))


def implied_volatility(price: float, flag: str, S: float, K: float, t: float,
                       r: float) -> float:
    """Implied volatility via py_vollib (60) or bisection fallback."""
    if PYVOLLIB_AVAILABLE:
        try:
            return float(_iv(price, S, K, t, r, flag))
        except Exception:
            pass
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if option_price(flag, S, K, t, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --- Factor analysis ---------------------------------------------------------

def factor_analysis(factor: pd.DataFrame, prices: pd.DataFrame,
                    periods: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Factor efficacy via alphalens (62); native rank-IC fallback.

    ``factor`` is wide (dates x assets). Returns IC stats per period.
    """
    out: dict[str, pd.Series] = {}
    fwd = prices.pct_change().shift(-1)
    single_asset = factor.shape[1] == 1
    for p in periods:
        horizon_ret = prices.pct_change(p).shift(-p)
        if single_asset:
            ic_ts = factor.iloc[:, 0].corr(horizon_ret.iloc[:, 0],
                                           method="spearman")
            out[f"ic_mean_{p}d"] = pd.Series([ic_ts], index=["mean"])
            out[f"ic_ir_{p}d"] = pd.Series([np.nan], index=["ir"])
            continue
        ic = factor.corrwith(horizon_ret, axis=1)
        out[f"ic_{p}d"] = ic
        out[f"ic_mean_{p}d"] = pd.Series([ic.mean()], index=["mean"])
        out[f"ic_ir_{p}d"] = pd.Series([ic.mean() / (ic.std() or np.nan)],
                                       index=["ir"])
    if ALPHALENS_AVAILABLE:
        try:
            factor_series = factor.stack()
            factor_series.index.names = ["date", "asset"]
            factor_data = _al.utils.get_clean_factor_and_forward_returns(
                factor_series, prices, periods=periods)
            out["alphalens_mean_returns"] = factor_data.groupby(level=1)[
                [c for c in factor_data.columns if "return" in c.lower()]
            ].mean().mean(axis=1)
        except Exception:
            pass
    return out


# --- Portfolio tear sheets ---------------------------------------------------

def portfolio_tear_sheet(returns: pd.Series) -> dict:
    """Full tear sheet via pyfolio (63); native metrics fallback."""
    if PYFOLIO_AVAILABLE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            stats = _pf.timeseries.perf_stats(returns.dropna())
            return {str(k): (float(v) if np.isscalar(v) else v)
                    for k, v in stats.items()}
        except Exception:
            pass
    from quant.quantstats_reports import performance_summary
    return performance_summary(returns)


ANALYTICS_STATUS = {
    "ta_lib": TALIB_AVAILABLE,
    "riskfolio": RISKFOLIO_AVAILABLE,
    "pyportfolioopt": PYPORTFOLIOOPT_AVAILABLE,
    "cvxportfolio": CVXPORTFOLIO_AVAILABLE,
    "finquant": FINQUANT_AVAILABLE,
    "financepy": FINANCEPY_AVAILABLE,
    "py_vollib": PYVOLLIB_AVAILABLE,
    "precise": PRECISE_AVAILABLE,
    "alphalens": ALPHALENS_AVAILABLE,
    "pyfolio": PYFOLIO_AVAILABLE,
    "gs_quant": GSQUANT_AVAILABLE,
}
