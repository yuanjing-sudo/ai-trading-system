from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats


def rolling_sharpe(rets: pd.Series, window: int = 126) -> pd.Series:
    return (rets.rolling(window).mean() * 252 - 0.04) / (rets.rolling(window).std() * np.sqrt(252) + 1e-9)


def rolling_vol(rets: pd.Series, window: int = 63) -> pd.Series:
    return rets.rolling(window).std() * np.sqrt(252)


def value_at_risk(rets: pd.Series, confidence: float = 0.95) -> float:
    return float(np.percentile(rets, (1 - confidence) * 100))


def conditional_var(rets: pd.Series, confidence: float = 0.95) -> float:
    var = value_at_risk(rets, confidence)
    return float(rets[rets <= var].mean())


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def beta_alpha(rets: pd.Series, bench_rets: pd.Series) -> Dict[str, float]:
    df = pd.concat([rets, bench_rets], axis=1).dropna()
    if len(df) < 20:
        return {"beta": np.nan, "alpha": np.nan}
    x, y = df.iloc[:, 1].values, df.iloc[:, 0].values
    beta, alpha, _, _, _ = stats.linregress(x, y)
    return {"beta": float(beta), "alpha": float(alpha * 252)}


def information_ratio(rets: pd.Series, bench_rets: pd.Series) -> float:
    active = rets - bench_rets
    if active.std() == 0:
        return np.nan
    return float(active.mean() / active.std() * np.sqrt(252))


def tail_ratio(rets: pd.Series) -> float:
    q = rets.quantile([0.95, 0.05])
    return float(abs(q.iloc[0]) / (abs(q.iloc[1]) + 1e-9))


def summary_report(equity: pd.Series, rets: pd.Series,
                   bench: pd.Series = None) -> str:
    mdd = max_drawdown(equity)
    var95 = value_at_risk(rets)
    cvar95 = conditional_var(rets)
    lines = [
        "=" * 70,
        "PERFORMANCE SUMMARY",
        "=" * 70,
        f"Periods            : {len(rets)}",
        f"Total Return       : {(equity.iloc[-1]/equity.iloc[0]-1):.2%}",
        f"Annualized Return  : {(rets.mean()*252):.2%}",
        f"Annualized Vol     : {(rets.std()*np.sqrt(252)):.2%}",
        f"Sharpe Ratio       : {((rets.mean()*252-0.04)/(rets.std()*np.sqrt(252)+1e-9)):.3f}",
        f"Max Drawdown       : {mdd:.2%}",
        f"VaR (95%)          : {var95:.2%}",
        f"CVaR (95%)         : {cvar95:.2%}",
        f"Win Rate           : {(rets>0).mean():.2%}",
        f"Tail Ratio         : {tail_ratio(rets):.3f}",
    ]
    if bench is not None:
        ba = beta_alpha(rets, bench.pct_change())
        lines += [
            f"Beta vs Benchmark  : {ba['beta']:.3f}",
            f"Alpha (annualized) : {ba['alpha']:.2%}",
            f"Info Ratio         : {information_ratio(rets, bench.pct_change()):.3f}",
        ]
    lines.append("=" * 70)
    return "\n".join(lines)
