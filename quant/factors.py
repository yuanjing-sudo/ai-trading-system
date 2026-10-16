"""Cross-sectional and time-series alpha factor library.

All functions accept a pandas DataFrame indexed by date with columns for each
asset (wide format) and return a DataFrame of factor values aligned to the
input index. Factor values are typically used as inputs to a portfolio
optimizer or as signals in a long-short strategy.

Conventions
-----------
- Inputs are wide-format DataFrames: rows = dates, columns = assets.
- Returns are simple (not log) unless noted.
- All functions are vectorized across assets and dates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "momentum_factor", "reversal_factor", "volatility_factor", "volume_factor",
    "liquidity_factor", "quality_factor", "value_factor", "size_factor",
    "momentum_quality_factor", "low_vol_factor", "high_div_factor",
    "earnings_yield_factor", "book_to_price_factor", "return_on_equity_factor",
    "debt_to_equity_factor", "current_ratio_factor", "gross_margin_factor",
    "operating_margin_factor", "free_cash_flow_yield_factor",
    "peg_ratio_factor", "ev_to_ebitda_factor", "price_to_sales_factor",
    "dividend_growth_factor", "payout_ratio_factor", "insider_buying_factor",
    "short_interest_factor", "analyst_revision_factor", "earnings_surprise_factor",
    "return_on_assets_factor", "return_on_invested_capital_factor",
    "asset_turnover_factor", "inventory_turnover_factor",
    "receivables_turnover_factor", "capex_to_sales_factor",
    "r_and_d_to_sales_factor", "sales_growth_factor", "eps_growth_factor",
    "book_growth_factor", "cash_conversion_cycle_factor",
    "interest_coverage_factor", "altman_z_score_factor",
    "beneish_m_score_factor", "piotroski_f_score_factor",
    "composite_value_factor", "composite_quality_factor",
    "composite_momentum_factor", "composite_volatility_factor",
    "composite_liquidity_factor", "composite_factor",
    "zscore_cross_sectional", "rank_cross_sectional",
    "winsorize_cross_sectional", "neutralize_sector",
    "demean_cross_sectional", "standardize_cross_sectional",
]


# ---------------------------------------------------------------------------
# Cross-sectional helpers
# ---------------------------------------------------------------------------

def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score across assets (date axis)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)


def rank_cross_sectional(df: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Row-wise rank across assets."""
    return df.rank(axis=1, pct=pct, method="average")


def winsorize_cross_sectional(df: pd.DataFrame, limits: float = 0.05) -> pd.DataFrame:
    """Row-wise winsorization across assets at given quantile limits."""
    lo = df.quantile(limits, axis=1)
    hi = df.quantile(1 - limits, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def demean_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise demeaning across assets."""
    return df.sub(df.mean(axis=1), axis=0)


def standardize_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise standardization (z-score) across assets."""
    return zscore_cross_sectional(df)


def neutralize_sector(df: pd.DataFrame, sectors: pd.DataFrame) -> pd.DataFrame:
    """Neutralize factor values within each sector (row-wise group demean)."""
    out = df.copy()
    for date in df.index:
        row = df.loc[date]
        sec_row = sectors.loc[date] if date in sectors.index else pd.Series(index=df.columns)
        for sector in sec_row.unique():
            members = sec_row[sec_row == sector].index
            members = [m for m in members if m in row.index]
            if len(members) > 1:
                out.loc[date, members] = row[members] - row[members].mean()
    return out


# ---------------------------------------------------------------------------
# Price-based factors
# ---------------------------------------------------------------------------

def momentum_factor(prices: pd.DataFrame, lookback: int = 252, skip: int = 20) -> pd.DataFrame:
    """Price momentum: total return over lookback skipping recent skip days."""
    p = prices.shift(skip)
    return p / p.shift(lookback) - 1.0


def reversal_factor(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Short-term reversal: negative recent return."""
    return -prices.pct_change(lookback)


def volatility_factor(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Rolling realized volatility (annualized)."""
    return returns.rolling(lookback).std(ddof=1) * np.sqrt(252)


def low_vol_factor(returns: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Low-volatility factor: negative volatility."""
    return -volatility_factor(returns, lookback)


def volume_factor(volume: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Volume momentum: recent volume relative to history."""
    v = volume.rolling(lookback).mean()
    return v / v.shift(lookback) - 1.0


def liquidity_factor(prices: pd.DataFrame, volume: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Amihud illiquidity inverted: higher = more liquid."""
    ret = prices.pct_change()
    illiq = ret.abs() / (prices * volume).replace(0, np.nan)
    return -illiq.rolling(lookback).mean()


# ---------------------------------------------------------------------------
# Fundamental factors (expect wide-format DataFrames per metric)
# ---------------------------------------------------------------------------

def value_factor(
    price: pd.DataFrame, book_value: pd.DataFrame, earnings: pd.DataFrame,
    sales: pd.DataFrame, cash_flow: pd.DataFrame,
) -> pd.DataFrame:
    """Composite value: average of book-to-price, earnings-to-price,
    sales-to-price, cash-flow-to-price (each cross-sectionally ranked)."""
    bp = rank_cross_sectional(book_value / price)
    ep = rank_cross_sectional(earnings / price)
    sp = rank_cross_sectional(sales / price)
    cfp = rank_cross_sectional(cash_flow / price)
    return (bp + ep + sp + cfp) / 4.0


def size_factor(market_cap: pd.DataFrame) -> pd.DataFrame:
    """Size factor: negative log market cap (small-cap tilt)."""
    return -np.log(market_cap.replace(0, np.nan))


def quality_factor(
    roe: pd.DataFrame, roa: pd.DataFrame, roic: pd.DataFrame,
    debt_to_equity: pd.DataFrame, current_ratio: pd.DataFrame,
    gross_margin: pd.DataFrame, operating_margin: pd.DataFrame,
    fcf_yield: pd.DataFrame,
) -> pd.DataFrame:
    """Composite quality: profitability + balance-sheet strength."""
    prof = (rank_cross_sectional(roe) + rank_cross_sectional(roa) +
            rank_cross_sectional(roic) + rank_cross_sectional(gross_margin) +
            rank_cross_sectional(operating_margin) + rank_cross_sectional(fcf_yield)) / 6.0
    bal = (rank_cross_sectional(-debt_to_equity) + rank_cross_sectional(current_ratio)) / 2.0
    return (prof + bal) / 2.0


def low_vol_factor_alt(returns: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Low-volatility factor using longer lookback."""
    return -volatility_factor(returns, lookback)


def high_div_factor(dividend_yield: pd.DataFrame) -> pd.DataFrame:
    """High dividend yield factor."""
    return rank_cross_sectional(dividend_yield)


def earnings_yield_factor(earnings: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Earnings-to-price factor."""
    return rank_cross_sectional(earnings / price)


def book_to_price_factor(book_value: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Book-to-price factor."""
    return rank_cross_sectional(book_value / price)


def return_on_equity_factor(roe: pd.DataFrame) -> pd.DataFrame:
    """ROE factor."""
    return rank_cross_sectional(roe)


def debt_to_equity_factor(debt_to_equity: pd.DataFrame) -> pd.DataFrame:
    """Negative debt-to-equity (lower leverage preferred)."""
    return rank_cross_sectional(-debt_to_equity)


def current_ratio_factor(current_ratio: pd.DataFrame) -> pd.DataFrame:
    """Current ratio factor."""
    return rank_cross_sectional(current_ratio)


def gross_margin_factor(gross_margin: pd.DataFrame) -> pd.DataFrame:
    """Gross margin factor."""
    return rank_cross_sectional(gross_margin)


def operating_margin_factor(operating_margin: pd.DataFrame) -> pd.DataFrame:
    """Operating margin factor."""
    return rank_cross_sectional(operating_margin)


def free_cash_flow_yield_factor(fcf: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Free cash flow yield factor."""
    return rank_cross_sectional(fcf / price)


def peg_ratio_factor(peg: pd.DataFrame) -> pd.DataFrame:
    """Negative PEG ratio (lower PEG preferred)."""
    return rank_cross_sectional(-peg)


def ev_to_ebitda_factor(ev: pd.DataFrame, ebitda: pd.DataFrame) -> pd.DataFrame:
    """Negative EV/EBITDA (lower multiple preferred)."""
    return rank_cross_sectional(-ev / ebitda)


def price_to_sales_factor(sales: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Negative price-to-sales (lower multiple preferred)."""
    return rank_cross_sectional(-price / sales)


def dividend_growth_factor(dividends: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Dividend growth factor."""
    return dividends / dividends.shift(lookback) - 1.0


def payout_ratio_factor(dividends: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Negative payout ratio (lower payout preferred for growth)."""
    return rank_cross_sectional(-dividends / earnings)


def short_interest_factor(short_interest: pd.DataFrame, shares_out: pd.DataFrame) -> pd.DataFrame:
    """Negative short interest ratio (lower short interest preferred)."""
    return rank_cross_sectional(-short_interest / shares_out)


def analyst_revision_factor(revisions: pd.DataFrame) -> pd.DataFrame:
    """Analyst earnings revision factor."""
    return rank_cross_sectional(revisions)


def earnings_surprise_factor(surprise: pd.DataFrame) -> pd.DataFrame:
    """Earnings surprise factor (actual vs estimate)."""
    return rank_cross_sectional(surprise)


def return_on_assets_factor(roa: pd.DataFrame) -> pd.DataFrame:
    """ROA factor."""
    return rank_cross_sectional(roa)


def return_on_invested_capital_factor(roic: pd.DataFrame) -> pd.DataFrame:
    """ROIC factor."""
    return rank_cross_sectional(roic)


def asset_turnover_factor(sales: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """Asset turnover factor."""
    return rank_cross_sectional(sales / assets)


def inventory_turnover_factor(cogs: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Inventory turnover factor."""
    return rank_cross_sectional(cogs / inventory)


def receivables_turnover_factor(sales: pd.DataFrame, receivables: pd.DataFrame) -> pd.DataFrame:
    """Receivables turnover factor."""
    return rank_cross_sectional(sales / receivables)


def capex_to_sales_factor(capex: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Negative capex-to-sales (lower capex intensity preferred)."""
    return rank_cross_sectional(-capex / sales)


def r_and_d_to_sales_factor(rd: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """R&D-to-sales factor (innovation intensity)."""
    return rank_cross_sectional(rd / sales)


def sales_growth_factor(sales: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Sales growth factor."""
    return sales / sales.shift(lookback) - 1.0


def eps_growth_factor(eps: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """EPS growth factor."""
    return eps / eps.shift(lookback) - 1.0


def book_growth_factor(book_value: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Book value growth factor."""
    return book_value / book_value.shift(lookback) - 1.0


def cash_conversion_cycle_factor(
    inventory: pd.DataFrame, receivables: pd.DataFrame, payables: pd.DataFrame,
    cogs: pd.DataFrame, sales: pd.DataFrame,
) -> pd.DataFrame:
    """Negative cash conversion cycle (shorter cycle preferred)."""
    dso = receivables / sales * 365
    dio = inventory / cogs * 365
    dpo = payables / cogs * 365
    ccc = dso + dio - dpo
    return rank_cross_sectional(-ccc)


def interest_coverage_factor(ebit: pd.DataFrame, interest: pd.DataFrame) -> pd.DataFrame:
    """Interest coverage factor."""
    return rank_cross_sectional(ebit / interest)


def altman_z_score_factor(
    working_capital: pd.DataFrame, retained_earnings: pd.DataFrame,
    ebit: pd.DataFrame, market_cap: pd.DataFrame, total_liabilities: pd.DataFrame,
    sales: pd.DataFrame, total_assets: pd.DataFrame,
) -> pd.DataFrame:
    """Altman Z-score factor (bankruptcy risk)."""
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    return rank_cross_sectional(z)


def beneish_m_score_factor(
    dsri: pd.DataFrame, gmi: pd.DataFrame, aqi: pd.DataFrame, sgi: pd.DataFrame,
    depi: pd.DataFrame, sgai: pd.DataFrame, lvgi: pd.DataFrame,
    tata: pd.DataFrame,
) -> pd.DataFrame:
    """Beneish M-score factor (earnings manipulation risk)."""
    m = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi +
         0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)
    return rank_cross_sectional(-m)


def piotroski_f_score_factor(
    roa: pd.DataFrame, cfo: pd.DataFrame, delta_roa: pd.DataFrame,
    accrual: pd.DataFrame, delta_leverage: pd.DataFrame,
    delta_liquidity: pd.DataFrame, eq_offered: pd.DataFrame,
    delta_margin: pd.DataFrame, delta_turnover: pd.DataFrame,
) -> pd.DataFrame:
    """Piotroski F-score factor (fundamental strength, 0-9)."""
    score = (
        (roa > 0).astype(int) +
        (cfo > 0).astype(int) +
        (delta_roa > 0).astype(int) +
        (cfo > roa).astype(int) +
        (delta_leverage <= 0).astype(int) +
        (delta_liquidity > 0).astype(int) +
        (eq_offered <= 0).astype(int) +
        (delta_margin > 0).astype(int) +
        (delta_turnover > 0).astype(int)
    )
    return score


def composite_value_factor(
    price: pd.DataFrame, book_value: pd.DataFrame, earnings: pd.DataFrame,
    sales: pd.DataFrame, cash_flow: pd.DataFrame,
) -> pd.DataFrame:
    """Composite value factor (equal-weighted sub-factors)."""
    return value_factor(price, book_value, earnings, sales, cash_flow)


def composite_quality_factor(
    roe: pd.DataFrame, roa: pd.DataFrame, roic: pd.DataFrame,
    debt_to_equity: pd.DataFrame, current_ratio: pd.DataFrame,
    gross_margin: pd.DataFrame, operating_margin: pd.DataFrame,
    fcf_yield: pd.DataFrame,
) -> pd.DataFrame:
    """Composite quality factor."""
    return quality_factor(
        roe, roa, roic, debt_to_equity, current_ratio,
        gross_margin, operating_margin, fcf_yield,
    )


def composite_momentum_factor(prices: pd.DataFrame, lookbacks=(21, 63, 126, 252)) -> pd.DataFrame:
    """Composite momentum across multiple lookbacks."""
    moms = [momentum_factor(prices, lb) for lb in lookbacks]
    return pd.concat(moms, axis=1).mean(axis=1)


def composite_volatility_factor(returns: pd.DataFrame, lookbacks=(20, 60, 126)) -> pd.DataFrame:
    """Composite volatility factor (negative, lower vol preferred)."""
    vols = [volatility_factor(returns, lb) for lb in lookbacks]
    return -pd.concat(vols, axis=1).mean(axis=1)


def composite_liquidity_factor(
    prices: pd.DataFrame, volume: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """Composite liquidity factor."""
    return liquidity_factor(prices, volume, lookback)


def composite_factor(
    prices: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame,
    book_value: pd.DataFrame, earnings: pd.DataFrame, sales: pd.DataFrame,
    cash_flow: pd.DataFrame, roe: pd.DataFrame, roa: pd.DataFrame,
    roic: pd.DataFrame, debt_to_equity: pd.DataFrame, current_ratio: pd.DataFrame,
    gross_margin: pd.DataFrame, operating_margin: pd.DataFrame,
    fcf_yield: pd.DataFrame, market_cap: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Equal-weighted composite alpha factor across value, quality, momentum,
    low-volatility, and liquidity."""
    w = weights or {
        "value": 0.2, "quality": 0.2, "momentum": 0.2,
        "low_vol": 0.2, "liquidity": 0.2,
    }
    val = composite_value_factor(price=prices, book_value=book_value,
                                 earnings=earnings, sales=sales,
                                 cash_flow=cash_flow)
    qual = composite_quality_factor(
        roe=roe, roa=roa, roic=roic, debt_to_equity=debt_to_equity,
        current_ratio=current_ratio, gross_margin=gross_margin,
        operating_margin=operating_margin, fcf_yield=fcf_yield,
    )
    mom = composite_momentum_factor(prices)
    lv = composite_volatility_factor(returns)
    liq = composite_liquidity_factor(prices, volume)
    comp = (
        w["value"] * val +
        w["quality"] * qual +
        w["momentum"] * mom +
        w["low_vol"] * lv +
        w["liquidity"] * liq
    )
    return comp
