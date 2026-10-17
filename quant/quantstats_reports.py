"""QuantStats Performance Reports Wrapper.

Provides professional performance tear sheets, risk reports, and
tear-sheet HTML/PDF output using the quantstats library. Falls back to
our own quant.risk module if quantstats is not installed.

Reference: https://github.com/ranaroussi/quantstats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Check if quantstats is available
try:
    import quantstats as qs
    QUANTSTATS_AVAILABLE = True
except ImportError:
    QUANTSTATS_AVAILABLE = False

from quant.risk import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    cagr, volatility, var_historical, cvar_historical,
    ulcer_index, pain_ratio, drawdown_series,
)

__all__ = [
    "QUANTSTATS_AVAILABLE",
    "generate_tear_sheet",
    "generate_html_report",
    "generate_pdf_report",
    "performance_summary",
    "risk_summary",
    "monthly_returns",
    "drawdown_report",
    "compare_strategies",
    "rolling_metrics",
    "export_report",
]


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
) -> dict:
    """Compute comprehensive performance summary.

    Parameters
    ----------
    returns : Series
        Strategy returns.
    benchmark : Series, optional
        Benchmark returns for comparison.
    rf : float
        Risk-free rate (annualized).

    Returns
    -------
    dict
        Performance metrics dictionary.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    metrics = {
        "total_return": float((1 + r).prod() - 1),
        "cagr": float(cagr((1 + r).cumprod())),
        "volatility": float(volatility(r)),
        "sharpe": float(sharpe_ratio(r, rf)),
        "sortino": float(sortino_ratio(r, rf)),
        "calmar": float(calmar_ratio((1 + r).cumprod())),
        "max_drawdown": float(max_drawdown((1 + r).cumprod())),
        "ulcer_index": float(ulcer_index((1 + r).cumprod())),
        "pain_ratio": float(pain_ratio((1 + r).cumprod())),
        "var_95": float(var_historical(r, 0.95)),
        "cvar_95": float(cvar_historical(r, 0.95)),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss": float(r[r < 0].mean()) if (r < 0).any() else 0.0,
        "profit_factor": float(
            r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf
        ),
        "skewness": float(r.skew()),
        "kurtosis": float(r.kurt()),
    }

    if benchmark is not None:
        b = benchmark.dropna()
        common = r.index.intersection(b.index)
        if len(common) > 1:
            r_aligned = r.loc[common]
            b_aligned = b.loc[common]
            from quant.risk import beta, alpha, tracking_error, information_ratio
            metrics["beta"] = float(beta(r_aligned, b_aligned).iloc[-1] if hasattr(beta(r_aligned, b_aligned), 'iloc') else beta(r_aligned, b_aligned))
            metrics["alpha"] = float(alpha(r_aligned, b_aligned).iloc[-1] if hasattr(alpha(r_aligned, b_aligned), 'iloc') else alpha(r_aligned, b_aligned))
            metrics["tracking_error"] = float(tracking_error(r_aligned, b_aligned).iloc[-1] if hasattr(tracking_error(r_aligned, b_aligned), 'iloc') else tracking_error(r_aligned, b_aligned))
            metrics["information_ratio"] = float(information_ratio(r_aligned, b_aligned).iloc[-1] if hasattr(information_ratio(r_aligned, b_aligned), 'iloc') else information_ratio(r_aligned, b_aligned))

    return metrics


def risk_summary(
    returns: pd.Series,
    equity: pd.Series | None = None,
) -> dict:
    """Compute risk-focused summary metrics."""
    r = returns.dropna()
    if equity is None:
        equity = (1 + r).cumprod()

    e = equity.dropna()

    return {
        "max_drawdown": float(max_drawdown(e)),
        "avg_drawdown": float(
            drawdown_series(e)[drawdown_series(e) < 0].mean()
            if (drawdown_series(e) < 0).any() else 0.0
        ),
        "drawdown_duration": int(
            (e < e.cummax()).groupby(
                (e < e.cummax()).diff().ne(0).cumsum()
            ).sum().max() if len(e) > 1 else 0
        ),
        "ulcer_index": float(ulcer_index(e)),
        "pain_index": float(
            drawdown_series(e).dropna()[drawdown_series(e).dropna() < 0].mean() * 100
            if (drawdown_series(e).dropna() < 0).any() else 0.0
        ),
        "pain_ratio": float(pain_ratio(e)),
        "var_95": float(var_historical(r, 0.95)),
        "cvar_95": float(cvar_historical(r, 0.95)),
        "expected_shortfall": float(cvar_historical(r, 0.95)),
        "tail_ratio": float(
            (r[r >= r.quantile(0.95)].mean() / abs(r[r <= r.quantile(0.05)].mean()))
            if (r <= r.quantile(0.05)).any() and r[r <= r.quantile(0.05)].mean() != 0
            else np.nan
        ),
        "gain_to_pain": float(
            r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf
        ),
    }


# ---------------------------------------------------------------------------
# QuantStats integration
# ---------------------------------------------------------------------------

def generate_tear_sheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    title: str = "Performance Tear Sheet",
    save_path: str | None = None,
) -> dict:
    """Generate a full quantstats tear sheet.

    Parameters
    ----------
    returns : Series
        Strategy returns.
    benchmark : Series, optional
        Benchmark returns.
    rf : float
        Risk-free rate.
    title : str
        Report title.
    save_path : str, optional
        Path to save HTML report.

    Returns
    -------
    dict
        Performance metrics.
    """
    r = returns.dropna()

    if QUANTSTATS_AVAILABLE:
        try:
            # Generate quantstats tear sheet
            if save_path:
                qs.reports.html(r, benchmark=benchmark, rf=rf, title=title, output=save_path)
            else:
                qs.reports.html(r, benchmark=benchmark, rf=rf, title=title)

            # Also generate key metrics
            metrics = qs.stats.risk(r, rf=rf)
            return {
                "quantstats_available": True,
                "metrics": metrics if isinstance(metrics, dict) else {},
                "report_saved": bool(save_path),
                "save_path": save_path,
            }
        except Exception:
            pass

    # Fallback to our own metrics
    metrics = performance_summary(r, benchmark, rf)
    return {
        "quantstats_available": False,
        "metrics": metrics,
        "report_saved": False,
        "save_path": None,
    }


def generate_html_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    title: str = "AI Trading System - Performance Report",
    output_path: str = "performance_report.html",
) -> str:
    """Generate an HTML performance report.

    Returns the file path of the saved report.
    """
    if QUANTSTATS_AVAILABLE:
        try:
            qs.reports.html(
                returns, benchmark=benchmark, rf=rf,
                title=title, output=output_path,
            )
            return output_path
        except Exception:
            pass

    # Fallback: generate a simple HTML report
    metrics = performance_summary(returns, benchmark, rf)
    risk = risk_summary(returns)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: 'JetBrains Mono', monospace; background: #0a0e17; color: #e0e0e0; padding: 20px; }}
            .card {{ background: #131a2a; border: 1px solid #1e2a3a; border-radius: 12px; padding: 20px; margin: 10px 0; }}
            .metric {{ display: inline-block; margin: 10px 20px; }}
            .metric-value {{ font-size: 1.5rem; font-weight: 700; color: #00d4ff; }}
            .metric-label {{ color: #6b7280; font-size: 0.85rem; text-transform: uppercase; }}
            h1 {{ color: #00d4ff; }}
            h2 {{ color: #7b2ff7; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #1e2a3a; }}
            th {{ color: #00d4ff; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="card">
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {"".join(f'<tr><td>{k.replace("_", " ").title()}</td><td>{v:.4f}</td></tr>' for k, v in metrics.items() if isinstance(v, (int, float)))}
            </table>
        </div>
        <div class="card">
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {"".join(f'<tr><td>{k.replace("_", " ").title()}</td><td>{v:.4f}</td></tr>' for k, v in risk.items() if isinstance(v, (int, float)))}
            </table>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def generate_pdf_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    title: str = "Performance Report",
    output_path: str = "performance_report.pdf",
) -> str:
    """Generate a PDF performance report.

    Requires quantstats with PDF support or weasyprint.
    """
    if QUANTSTATS_AVAILABLE:
        try:
            qs.reports.pdf(returns, benchmark=benchmark, rf=rf, title=title, output=output_path)
            return output_path
        except Exception:
            pass

    # Fallback: generate HTML first, note PDF not available
    html_path = output_path.replace(".pdf", ".html")
    generate_html_report(returns, benchmark, rf, title, html_path)
    return html_path


# ---------------------------------------------------------------------------
# Returns analysis
# ---------------------------------------------------------------------------

def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compute monthly returns matrix."""
    r = returns.dropna()
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.strftime("%Y-%m")
    return monthly.to_frame("return")


def drawdown_report(equity: pd.Series) -> pd.DataFrame:
    """Generate drawdown report with top drawdown periods."""
    e = equity.dropna()
    dd = drawdown_series(e)

    # Find drawdown periods
    in_dd = dd < 0
    groups = (in_dd != in_dd.shift(1)).cumsum()
    dd_periods = []

    for group, data in dd.groupby(groups):
        if data.iloc[0] < 0:  # This is a drawdown period
            start = data.index[0]
            end = data.index[-1]
            max_dd = data.min()
            dd_periods.append({
                "start": start,
                "end": end,
                "max_drawdown": max_dd,
                "duration": len(data),
            })

    return pd.DataFrame(dd_periods)


def compare_strategies(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Compare multiple strategies side by side."""
    rows = []
    for name, returns in strategies.items():
        metrics = performance_summary(returns, benchmark)
        metrics["strategy"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows)
    if "strategy" in df.columns:
        df = df.set_index("strategy")
    return df


def rolling_metrics(
    returns: pd.Series,
    window: int = 60,
    metric: str = "sharpe",
) -> pd.Series:
    """Compute rolling performance metrics."""
    r = returns.dropna()

    if metric == "sharpe":
        return r.rolling(window).apply(
            lambda x: sharpe_ratio(x) if len(x) > 1 else np.nan, raw=True
        )
    elif metric == "sortino":
        return r.rolling(window).apply(
            lambda x: sortino_ratio(x) if len(x) > 1 else np.nan, raw=True
        )
    elif metric == "volatility":
        return r.rolling(window).std(ddof=1) * np.sqrt(252)
    elif metric == "var_95":
        return r.rolling(window).apply(
            lambda x: float(np.percentile(x, 5)) if len(x) > 0 else np.nan, raw=True
        )
    elif metric == "cvar_95":
        return r.rolling(window).apply(
            lambda x: float(x[x <= np.percentile(x, 5)].mean())
            if len(x) > 0 and (x <= np.percentile(x, 5)).any()
            else np.nan,
            raw=True,
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float = 0.0,
    title: str = "Performance Report",
    output_dir: str = "./reports",
) -> dict[str, str]:
    """Export performance report in multiple formats.

    Returns a dictionary of format -> file path.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # HTML report
    html_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.html")
    results["html"] = generate_html_report(returns, benchmark, rf, title, html_path)

    # PDF report (may fall back to HTML)
    pdf_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.pdf")
    results["pdf"] = generate_pdf_report(returns, benchmark, rf, title, pdf_path)

    # Metrics JSON
    import json
    metrics = performance_summary(returns, benchmark, rf)
    json_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    results["json"] = json_path

    return results
