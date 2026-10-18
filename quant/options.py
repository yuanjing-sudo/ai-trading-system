"""Options analytics: pricing, Greeks, implied volatility, vol surface helpers,
strategy payoffs, and risk-neutral probabilities.

Pure numpy/pandas (no scipy). Prices are European Black-Scholes unless a
function name says otherwise; ``binomial_tree`` supports American exercise.
Inputs are never mutated.

Example
-------
>>> from quant import options
>>> options.black_scholes_price(100, 100, 0.5, 0.04, 0.2, "call")
10.45...
>>> g = options.black_scholes_greeks(100, 100, 0.5, 0.04, 0.2, "call")
>>> round(g["delta"], 4)
0.56...
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "norm_cdf", "norm_pdf", "inv_norm_cdf", "forward_price", "discount_factor",
    "black_scholes_price", "bs_delta", "bs_gamma", "bs_vega", "bs_theta", "bs_rho",
    "black_scholes_greeks", "implied_volatility", "implied_vol_bisection",
    "binomial_tree", "option_price", "put_call_parity", "parity_gap",
    "intrinsic_value", "time_value", "expected_move", "prob_itm", "prob_otm",
    "prob_touch", "moneyness", "log_moneyness", "delta_to_strike",
    "strike_to_delta", "greeks_series", "price_series", "vol_smile",
    "vol_skew", "vol_term_structure", "risk_neutral_density",
    "breeden_litzenberger", "straddle", "strangle", "bull_call_spread",
    "bear_put_spread", "bull_put_spread", "bear_call_spread", "butterfly",
    "iron_condor", "iron_butterfly", "condor", "collar", "covered_call",
    "protective_put", "risk_reversal", "box_spread", "synthetic_forward",
    "ratio_spread", "strategy_payoff", "strategy_value", "strategy_breakevens",
    "strategy_max_profit", "strategy_max_loss", "profit_table",
    "option_elasticity", "gamma_scalp_pnl", "delta_hedge_shares",
    "variance_swap_strike", "vix_proxy",
]

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x):
    """Standard normal CDF (vectorized)."""
    arr = np.asarray(x, dtype="float64")
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal PDF (vectorized)."""
    arr = np.asarray(x, dtype="float64")
    return np.exp(-0.5 * arr * arr) / SQRT_2PI


def inv_norm_cdf(p):
    """Inverse standard normal CDF (Acklam rational approximation)."""
    q = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if q < p_low:
        r = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    if q > p_high:
        r = math.sqrt(-2.0 * math.log(1.0 - q))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    r = q - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
           (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0)


def discount_factor(r: float, t: float) -> float:
    """Continuous-compounding discount factor ``exp(-r t)``."""
    return math.exp(-r * t)


def forward_price(spot: float, r: float, t: float, q: float = 0.0) -> float:
    """No-arbitrage forward price ``S e^{(r-q)t}``."""
    return spot * math.exp((r - q) * t)


def _d1_d2(spot, strike, t, r, sigma, q):
    if t <= 0 or sigma <= 0:
        raise ValueError("t and sigma must be positive")
    st = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / st
    return d1, d1 - st


def black_scholes_price(spot: float, strike: float, t: float, r: float, sigma: float,
                        option_type: str = "call", q: float = 0.0) -> float:
    """European Black-Scholes price with continuous dividend yield ``q``."""
    if t <= 0:
        return max(0.0, (spot - strike) if option_type == "call" else (strike - spot))
    d1, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    disc_r, disc_q = math.exp(-r * t), math.exp(-q * t)
    if option_type == "call":
        return spot * disc_q * float(norm_cdf(d1)) - strike * disc_r * float(norm_cdf(d2))
    return strike * disc_r * float(norm_cdf(-d2)) - spot * disc_q * float(norm_cdf(-d1))


def bs_delta(spot: float, strike: float, t: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Option delta (spot sensitivity)."""
    if t <= 0:
        itm = spot > strike
        if option_type == "call":
            return 1.0 if itm else 0.0
        return -1.0 if not itm else 0.0
    d1, _ = _d1_d2(spot, strike, t, r, sigma, q)
    disc_q = math.exp(-q * t)
    return disc_q * float(norm_cdf(d1)) if option_type == "call" else -disc_q * float(norm_cdf(-d1))


def bs_gamma(spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Option gamma (delta sensitivity); identical for calls and puts."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, r, sigma, q)
    return math.exp(-q * t) * float(norm_pdf(d1)) / (spot * sigma * math.sqrt(t))


def bs_vega(spot: float, strike: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Option vega for a 1.0 change in volatility (divide by 100 for per-vol-point)."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(spot, strike, t, r, sigma, q)
    return spot * math.exp(-q * t) * float(norm_pdf(d1)) * math.sqrt(t)


def bs_theta(spot: float, strike: float, t: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Option theta per year (divide by 365 for per-day)."""
    if t <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    disc_r, disc_q = math.exp(-r * t), math.exp(-q * t)
    common = -spot * disc_q * float(norm_pdf(d1)) * sigma / (2.0 * math.sqrt(t))
    if option_type == "call":
        return common - r * strike * disc_r * float(norm_cdf(d2)) + q * spot * disc_q * float(norm_cdf(d1))
    return common + r * strike * disc_r * float(norm_cdf(-d2)) - q * spot * disc_q * float(norm_cdf(-d1))


def bs_rho(spot: float, strike: float, t: float, r: float, sigma: float,
           option_type: str = "call", q: float = 0.0) -> float:
    """Option rho (rate sensitivity) per 1.0 change in ``r``."""
    if t <= 0:
        return 0.0
    _, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    disc_r = math.exp(-r * t)
    if option_type == "call":
        return strike * t * disc_r * float(norm_cdf(d2))
    return -strike * t * disc_r * float(norm_cdf(-d2))


def black_scholes_greeks(spot: float, strike: float, t: float, r: float, sigma: float,
                         option_type: str = "call", q: float = 0.0, per_vol_point: bool = True) -> dict:
    """Bundle of primary Greeks; vega/theta optionally scaled per vol-point/day."""
    vega = bs_vega(spot, strike, t, r, sigma, q)
    theta = bs_theta(spot, strike, t, r, sigma, option_type, q)
    return {
        "price": black_scholes_price(spot, strike, t, r, sigma, option_type, q),
        "delta": bs_delta(spot, strike, t, r, sigma, option_type, q),
        "gamma": bs_gamma(spot, strike, t, r, sigma, q),
        "vega": vega / 100.0 if per_vol_point else vega,
        "theta": theta / 365.0 if per_vol_point else theta,
        "rho": bs_rho(spot, strike, t, r, sigma, option_type, q) / 100.0,
    }


def _vega_raw(spot, strike, t, r, sigma, q):
    return bs_vega(spot, strike, t, r, sigma, q)


def implied_volatility(market_price: float, spot: float, strike: float, t: float, r: float,
                       option_type: str = "call", q: float = 0.0, tol: float = 1e-8,
                       max_iter: int = 100) -> float:
    """Implied volatility by Newton-Raphson with a bisection fallback."""
    if t <= 0:
        return float("nan")
    intrinsic = max(0.0, (spot - strike) if option_type == "call" else (strike - spot))
    time_val = market_price - intrinsic * math.exp(-r * t)
    if time_val <= 0:
        return 0.0
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_price(spot, strike, t, r, sigma, option_type, q)
        diff = price - market_price
        if abs(diff) < tol:
            return float(np.clip(sigma, 1e-6, 5.0))
        vega = _vega_raw(spot, strike, t, r, sigma, q)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5.0 or not np.isfinite(sigma):
            break
    return implied_vol_bisection(market_price, spot, strike, t, r, option_type, q, tol)


def implied_vol_bisection(market_price: float, spot: float, strike: float, t: float, r: float,
                          option_type: str = "call", q: float = 0.0, tol: float = 1e-8,
                          max_iter: int = 200) -> float:
    """Implied volatility by bisection on a wide volatility bracket."""
    if t <= 0:
        return float("nan")
    lo, hi = 1e-4, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = black_scholes_price(spot, strike, t, r, mid, option_type, q)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def binomial_tree(spot: float, strike: float, t: float, r: float, sigma: float, steps: int = 200,
                  option_type: str = "call", american: bool = False, q: float = 0.0) -> float:
    """Cox-Ross-Rubinstein binomial price (supports early American exercise)."""
    if t <= 0 or steps < 1:
        return max(0.0, (spot - strike) if option_type == "call" else (strike - spot))
    dt = t / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    sT = spot * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    if option_type == "call":
        values = np.maximum(sT - strike, 0.0)
    else:
        values = np.maximum(strike - sT, 0.0)
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            nj = np.arange(i, -1, -1)
            s = spot * u ** nj * d ** (i - nj)
            intrinsic = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)
            values = np.maximum(values, intrinsic)
    return float(values[0])


def option_price(spot: float, strike: float, t: float, r: float, sigma: float,
                 option_type: str = "call", q: float = 0.0, engine: str = "bs", **kwargs) -> float:
    """Dispatch to ``bs`` or ``binomial`` pricing."""
    if engine == "bs":
        return black_scholes_price(spot, strike, t, r, sigma, option_type, q)
    if engine == "binomial":
        return binomial_tree(spot, strike, t, r, sigma, option_type=option_type, q=q, **kwargs)
    raise ValueError("engine must be 'bs' or 'binomial'")


def put_call_parity(call_price: float, spot: float, strike: float, t: float, r: float, q: float = 0.0) -> float:
    """Theoretical put price implied by call price via put-call parity."""
    return call_price - spot * math.exp(-q * t) + strike * math.exp(-r * t)


def parity_gap(call_price: float, put_price: float, spot: float, strike: float, t: float,
               r: float, q: float = 0.0) -> float:
    """Observed put-call parity violation ``C - P - (S e^{-q t} - K e^{-r t})``."""
    return call_price - put_price - (spot * math.exp(-q * t) - strike * math.exp(-r * t))


def intrinsic_value(spot: float, strike: float, option_type: str = "call") -> float:
    """Exercise value of the option."""
    return max(0.0, (spot - strike) if option_type == "call" else (strike - spot))


def time_value(price: float, spot: float, strike: float, option_type: str = "call") -> float:
    """Option price minus intrinsic value."""
    return price - intrinsic_value(spot, strike, option_type)


def expected_move(spot: float, sigma: float, t: float, z: float = 1.0) -> float:
    """One-standard-deviation (or ``z``-sigma) price move over horizon ``t``."""
    return spot * sigma * math.sqrt(t) * z


def prob_itm(spot: float, strike: float, t: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Risk-neutral probability of finishing in the money."""
    if t <= 0:
        return 1.0 if intrinsic_value(spot, strike, option_type) > 0 else 0.0
    _, d2 = _d1_d2(spot, strike, t, r, sigma, q)
    return float(norm_cdf(d2)) if option_type == "call" else float(norm_cdf(-d2))


def prob_otm(spot: float, strike: float, t: float, r: float, sigma: float,
             option_type: str = "call", q: float = 0.0) -> float:
    """Risk-neutral probability of finishing out of the money."""
    return 1.0 - prob_itm(spot, strike, t, r, sigma, option_type, q)


def prob_touch(spot: float, barrier: float, t: float, sigma: float, drift: float = 0.0) -> float:
    """Approximate probability a driftless lognormal touches ``barrier`` by ``t``."""
    if spot <= 0 or barrier <= 0 or t <= 0 or sigma <= 0:
        return float("nan")
    mu = drift - 0.5 * sigma * sigma
    b = math.log(barrier / spot)
    denom = sigma * math.sqrt(t)
    up = float(norm_cdf((-b + mu * t) / denom))
    down = math.exp(2.0 * mu * b / (sigma * sigma)) * float(norm_cdf((-b - mu * t) / denom)) if mu != 0 else 0.0
    if barrier >= spot:
        return min(1.0, up + down)
    return min(1.0, float(norm_cdf((b - mu * t) / denom)) + down)


def moneyness(spot: float, strike: float) -> float:
    """Simple moneyness ``S / K``."""
    return spot / strike


def log_moneyness(spot: float, strike: float, sigma: float, t: float) -> float:
    """Standardized log-moneyness ``ln(S/K) / (sigma sqrt(t))``."""
    denom = sigma * math.sqrt(t)
    return float("nan") if denom == 0 else math.log(spot / strike) / denom


def delta_to_strike(spot: float, t: float, r: float, sigma: float, delta: float,
                    option_type: str = "call", q: float = 0.0) -> float:
    """Strike that produces a given BS delta."""
    if option_type == "call":
        d1 = float(inv_norm_cdf(delta * math.exp(q * t)))
    else:
        d1 = -float(inv_norm_cdf(-delta * math.exp(q * t)))
    return spot * math.exp((r - q + 0.5 * sigma * sigma) * t - d1 * sigma * math.sqrt(t))


def strike_to_delta(spot: float, strike: float, t: float, r: float, sigma: float,
                    option_type: str = "call", q: float = 0.0) -> float:
    """BS delta for a given strike."""
    return bs_delta(spot, strike, t, r, sigma, option_type, q)


def greeks_series(spot: pd.Series, strike, t, r: float, sigma, option_type: str = "call",
                  q: float = 0.0) -> pd.DataFrame:
    """Vectorized Greeks across a spot series (strike/sigma broadcastable)."""
    s = pd.Series(spot).astype("float64")
    out = {}
    out["price"] = _apply_by_row(s, strike, sigma, lambda a, b, v: black_scholes_price(a, b, t, r, v, option_type, q))
    out["delta"] = _apply_by_row(s, strike, sigma, lambda a, b, v: bs_delta(a, b, t, r, v, option_type, q))
    out["gamma"] = _apply_by_row(s, strike, sigma, lambda a, b, v: bs_gamma(a, b, t, r, v, q))
    out["vega"] = _apply_by_row(s, strike, sigma, lambda a, b, v: bs_vega(a, b, t, r, v, q))
    out["theta"] = _apply_by_row(s, strike, sigma, lambda a, b, v: bs_theta(a, b, t, r, v, option_type, q))
    out["rho"] = _apply_by_row(s, strike, sigma, lambda a, b, v: bs_rho(a, b, t, r, v, option_type, q))
    return pd.DataFrame(out, index=s.index)


def _apply_by_row(s: pd.Series, k, vol, fn) -> pd.Series:
    kk = pd.Series(k, index=s.index) if np.ndim(k) else pd.Series(k, index=s.index)
    vv = pd.Series(vol, index=s.index) if np.ndim(vol) else pd.Series(vol, index=s.index)
    return pd.Series([fn(a, b, v) for a, b, v in zip(s.to_numpy(), kk.to_numpy(), vv.to_numpy())], index=s.index)


def price_series(spot: pd.Series, strike, t, r: float, sigma, option_type: str = "call", q: float = 0.0) -> pd.Series:
    """Vectorized Black-Scholes prices across a spot series."""
    s = pd.Series(spot).astype("float64")
    return _apply_by_row(s, strike, sigma, lambda a, b, v: black_scholes_price(a, b, t, r, v, option_type, q))


def vol_smile(prices, spot: float, strikes, t: float, r: float,
              option_type: str = "call", q: float = 0.0) -> pd.Series:
    """Implied-volatility smile for a range of strikes."""
    ks = pd.Index(strikes, name="strike")
    vols = [implied_volatility(float(p), spot, float(k), t, r, option_type, q) for p, k in zip(prices, strikes)]
    return pd.Series(vols, index=ks)


def vol_skew(vols: pd.Series, strikes: pd.Series | None = None) -> float:
    """25-delta-style skew proxy: low-strike IV minus high-strike IV."""
    v = pd.Series(vols).dropna()
    if strikes is not None:
        k = pd.Series(strikes, index=v.index)
        order = k.sort_values().index
        v = v.loc[order]
    if len(v) < 2:
        return float("nan")
    return float(v.iloc[0] - v.iloc[-1])


def vol_term_structure(vols_by_expiry: dict) -> pd.Series:
    """Sort implied vols by expiry (keys = tenor in years or ordered labels)."""
    return pd.Series(vols_by_expiry).sort_index()


def breeden_litzenberger(call_prices, strikes, t: float, r: float) -> pd.Series:
    """Risk-neutral density from call prices: ``e^{rT} d^2C/dK^2``."""
    c = pd.Series(call_prices, index=pd.Index(strikes, name="strike")).sort_index()
    k = c.index.to_numpy(dtype="float64")
    second = np.gradient(np.gradient(c.to_numpy(dtype="float64"), k), k)
    density = np.exp(r * t) * second
    return pd.Series(np.clip(density, 0.0, None), index=c.index)


def risk_neutral_density(prices, strikes, t: float, r: float) -> pd.Series:
    """Alias of :func:`breeden_litzenberger`."""
    return breeden_litzenberger(prices, strikes, t, r)


def _leg(option_type, strike, qty, premium=0.0):
    return {"type": option_type, "strike": float(strike), "qty": float(qty), "premium": float(premium)}


def straddle(strike, premium_call=0.0, premium_put=0.0, qty=1) -> list:
    """Long straddle: long call + long put at the same strike."""
    return [_leg("call", strike, qty, premium_call), _leg("put", strike, qty, premium_put)]


def strangle(put_strike, call_strike, premium_call=0.0, premium_put=0.0, qty=1) -> list:
    """Long strangle: long OTM put + long OTM call."""
    return [_leg("put", put_strike, qty, premium_put), _leg("call", call_strike, qty, premium_call)]


def bull_call_spread(low_strike, high_strike, qty=1) -> list:
    """Bull call spread: long low-strike call, short high-strike call."""
    return [_leg("call", low_strike, qty), _leg("call", high_strike, -qty)]


def bear_put_spread(high_strike, low_strike, qty=1) -> list:
    """Bear put spread: long high-strike put, short low-strike put."""
    return [_leg("put", high_strike, qty), _leg("put", low_strike, -qty)]


def bull_put_spread(high_strike, low_strike, qty=1) -> list:
    """Bull put spread: short high-strike put, long low-strike put."""
    return [_leg("put", high_strike, -qty), _leg("put", low_strike, qty)]


def bear_call_spread(low_strike, high_strike, qty=1) -> list:
    """Bear call spread: short low-strike call, long high-strike call."""
    return [_leg("call", low_strike, -qty), _leg("call", high_strike, qty)]


def butterfly(low_strike, mid_strike, high_strike, qty=1) -> list:
    """Long call butterfly: +1/-2/+1 at increasing strikes."""
    return [_leg("call", low_strike, qty), _leg("call", mid_strike, -2 * qty), _leg("call", high_strike, qty)]


def iron_condor(put_lo, put_hi, call_lo, call_hi, qty=1) -> list:
    """Short iron condor: short inner strangle, long outer wings."""
    return [_leg("put", put_lo, qty), _leg("put", put_hi, -qty), _leg("call", call_lo, -qty), _leg("call", call_hi, qty)]


def iron_butterfly(put_strike, call_strike, put_wing, call_wing, qty=1) -> list:
    """Short iron butterfly: short straddle with long OTM wings."""
    return [_leg("put", put_strike, -qty), _leg("call", call_strike, -qty),
            _leg("put", put_wing, qty), _leg("call", call_wing, qty)]


def condor(put_lo, put_hi, call_lo, call_hi, qty=1) -> list:
    """Long condor: four calls spaced across strikes."""
    return [_leg("call", put_lo, qty), _leg("call", put_hi, -qty),
            _leg("call", call_lo, qty), _leg("call", call_hi, -qty)]


def collar(put_strike, call_strike, qty=1) -> list:
    """Collar: long protective put + short covered call."""
    return [_leg("put", put_strike, qty), _leg("call", call_strike, -qty)]


def covered_call(strike, qty=1) -> list:
    """Covered call: short call against long stock (stock leg implied)."""
    return [_leg("call", strike, -qty)]


def protective_put(strike, qty=1) -> list:
    """Protective put: long put against long stock."""
    return [_leg("put", strike, qty)]


def risk_reversal(put_strike, call_strike, qty=1) -> list:
    """Risk reversal: long OTM call + short OTM put."""
    return [_leg("call", call_strike, qty), _leg("put", put_strike, -qty)]


def box_spread(low_strike, high_strike, qty=1) -> list:
    """Box spread using a call and a put spread between two strikes."""
    return [_leg("call", low_strike, qty), _leg("call", high_strike, -qty),
            _leg("put", high_strike, qty), _leg("put", low_strike, -qty)]


def synthetic_forward(strike, qty=1) -> list:
    """Synthetic long forward: long call + short put at the same strike."""
    return [_leg("call", strike, qty), _leg("put", strike, -qty)]


def ratio_spread(low_strike, high_strike, ratio=2, option_type="call", qty=1) -> list:
    """Ratio spread: long 1x low strike, short ``ratio`` high strike."""
    return [_leg(option_type, low_strike, qty), _leg(option_type, high_strike, -ratio * qty)]


def strategy_payoff(legs, spot_range) -> pd.Series:
    """Expiry payoff of a leg list across a spot range (premium excluded)."""
    spots = pd.Series(np.asarray(spot_range, dtype="float64"))
    payoff = pd.Series(0.0, index=spots.index)
    for leg in legs:
        intrinsic = np.maximum((spots - leg["strike"]) if leg["type"] == "call" else (leg["strike"] - spots), 0.0)
        payoff = payoff + leg["qty"] * intrinsic
    return payoff


def strategy_value(legs, spot_range, t, r, sigma, q: float = 0.0) -> pd.Series:
    """Strategy value across spot range using Black-Scholes (premiums excluded)."""
    spots = pd.Series(np.asarray(spot_range, dtype="float64"))
    value = pd.Series(0.0, index=spots.index)
    for leg in legs:
        value = value + leg["qty"] * price_series(spots, leg["strike"], t, r, sigma, leg["type"], q)
    return value


def strategy_breakevens(legs, spot_range) -> list:
    """Approximate breakeven spots by locating payoff sign changes."""
    spots = np.asarray(spot_range, dtype="float64")
    net = strategy_payoff(legs, spots)
    premium = sum(leg["qty"] * leg.get("premium", 0.0) for leg in legs)
    pnl = net - premium
    sign = np.sign(pnl.to_numpy())
    breaks = []
    for i in range(1, len(spots)):
        if sign[i - 1] == 0 or sign[i] == 0 or sign[i - 1] != sign[i]:
            x0, x1, y0, y1 = spots[i - 1], spots[i], pnl.iloc[i - 1], pnl.iloc[i]
            if y1 != y0:
                breaks.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return breaks


def strategy_max_profit(legs, spot_range) -> float:
    """Maximum payoff minus net premium over the spot range."""
    spots = np.asarray(spot_range, dtype="float64")
    premium = sum(leg["qty"] * leg.get("premium", 0.0) for leg in legs)
    return float(strategy_payoff(legs, spots).max() - premium)


def strategy_max_loss(legs, spot_range) -> float:
    """Minimum payoff minus net premium over the spot range."""
    spots = np.asarray(spot_range, dtype="float64")
    premium = sum(leg["qty"] * leg.get("premium", 0.0) for leg in legs)
    return float(strategy_payoff(legs, spots).min() - premium)


def profit_table(legs, spot_range) -> pd.DataFrame:
    """Payoff, premium, and net PnL per spot level."""
    spots = np.asarray(spot_range, dtype="float64")
    premium = sum(leg["qty"] * leg.get("premium", 0.0) for leg in legs)
    payoff = strategy_payoff(legs, spots)
    return pd.DataFrame({"spot": spots, "payoff": payoff.to_numpy(),
                         "premium": premium, "pnl": payoff.to_numpy() - premium})


def option_elasticity(spot: float, strike: float, t: float, r: float, sigma: float,
                      option_type: str = "call", q: float = 0.0) -> float:
    """Percent option move per percent underlying move (delta * S / price)."""
    price = black_scholes_price(spot, strike, t, r, sigma, option_type, q)
    if price <= 0:
        return float("nan")
    return bs_delta(spot, strike, t, r, sigma, option_type, q) * spot / price


def delta_hedge_shares(delta: float, contracts: int = 1, contract_size: int = 100) -> float:
    """Shares required to delta-hedge ``contracts`` (negative delta offsets longs)."""
    return -delta * contracts * contract_size


def gamma_scalp_pnl(gamma: float, spot_move: float, theta_daily: float, days: float = 1.0,
                    contract_size: int = 100) -> float:
    """Approximate gamma-scalping PnL: ``0.5 gamma (dS)^2 - theta`` per contract block."""
    return (0.5 * gamma * spot_move ** 2 - theta_daily * days) * contract_size


def variance_swap_strike(implied_vols, weights=None) -> float:
    """Approximate variance-swap fair strike ``sqrt(E[sigma^2])`` from a vol strip."""
    v = pd.Series(implied_vols).dropna().to_numpy(dtype="float64")
    if v.size == 0:
        return float("nan")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype="float64")
    w = w / w.sum()
    return float(np.sqrt(np.sum(w * v ** 2)))


def vix_proxy(implied_vols: pd.Series) -> float:
    """Simple 30-day VIX-style proxy: mean of near-the-money implied vols."""
    v = pd.Series(implied_vols).dropna()
    return float(v.mean()) if not v.empty else float("nan")
