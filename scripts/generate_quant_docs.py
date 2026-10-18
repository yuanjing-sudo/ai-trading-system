"""Generate docs/INDICATORS.md and docs/NEW_FEATURES.md from quant.catalog."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant import catalog  # noqa: E402

DOCS = ROOT / "docs"

MODULE_BLURB = {
    "indicators_ext": "137 additional vectorized technical indicators (oscillators, trend, volume, volatility, statistics).",
    "ehlers": "Ehlers-style cycle and signal-processing filters (sine wave, dominant cycle, roofing filter, adaptive filters).",
    "candlesticks": "90 candlestick pattern detectors plus counts and a normalized pattern score.",
    "statistics_ext": "Statistical toolkit: covariance estimulators, PCA, KMeans, GMM, Kalman, GARCH/OU, entropy, bootstrap.",
    "features_ext": "Feature engineering: returns, volatility, momentum, mean-reversion, calendar, entropy, targets, selectors, FeaturePipeline.",
    "patterns_ext": "Chart structure and pattern detection: pivots, break-of-structure, formations, divergences.",
    "metrics_ext": "Performance metrics: risk-adjusted returns, drawdowns, VaR, trade stats, benchmark-relative, strategy_metrics.",
    "risk_ext": "Risk analytics: tail/VaR, portfolio risk contributions, allocation, stress tests, sizing, regime and correlation diagnostics.",
    "strategies_ext": "66 trading strategies across trend, mean-reversion, momentum, volatility, volume, pattern and composite families.",
    "portfolio_ext": "Portfolio construction: weight estimators, covariance estimators, rebalancing, attribution and reporting.",
    "options": "Options analytics: Black-Scholes pricing and Greeks, implied volatility, binomial American pricing, vol surface helpers, strategy payoffs, risk-neutral density.",
    "execution_ext": "Execution and TCA: TWAP/VWAP/POV scheduling, arrival/interval benchmarks, implementation shortfall, spread/microprice estimators, order-flow imbalance, impact approximations.",
    "screener_ext": "Universe screening and scanning across symbols, metrics and strategies.",
}


def _fmt_default(p) -> str:
    if p.default is inspect.Parameter.empty:
        return ""
    return f"={p.default!r}"


def _signature(fn) -> str:
    try:
        if inspect.isclass(fn):
            sig = inspect.signature(fn.__init__)
            params = [str(p) for p in sig.parameters.values() if p.name != "self"]
        else:
            sig = inspect.signature(fn)
            params = [str(p) for p in sig.parameters.values()]
    except (TypeError, ValueError):
        return "()"
    return "(" + ", ".join(params) + ")"


def write_indicators_md() -> Path:
    lines = [
        "# Indicator Reference",
        "",
        "Auto-generated from `quant.catalog` - do not edit by hand.",
        "Regenerate with `python scripts/generate_quant_docs.py`.",
        "",
        "## Registry summary",
        "",
        "| Kind | Count |",
        "| --- | --- |",
    ]
    counts = catalog.registry_counts()
    for kind, n in sorted(counts.items()):
        lines.append(f"| {kind} | {n} |")
    lines += [f"| **Total** | **{sum(counts.values())}** |", ""]

    groups = [("close", "Close-only indicators"), ("ohlc", "OHLC indicators"),
              ("ohlcv", "OHLCV indicators"), ("other", "Multi-series / other indicators")]
    for key, title in groups:
        names = catalog.indicators_by_input(key)
        lines += [f"## {title} ({len(names)})", "",
                  "| Indicator | Signature | Description |", "| --- | --- | --- |"]
        for name in names:
            fn = catalog.get("indicator", name)
            desc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
            lines.append(f"| `{name}` | `{name}{_signature(fn)}` | {desc[0] if desc else ''} |")
        lines.append("")

    lines += [f"## Candlestick patterns ({counts['candle']})", "",
              "| Pattern | Description |", "| --- | --- |"]
    for name in catalog.list_by_kind("candle"):
        fn = catalog.CATALOG["candle"][name]
        desc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        lines.append(f"| `{name}` | {desc[0] if desc else ''} |")
    lines.append("")

    lines += [f"## Chart-structure detectors ({counts['chart_pattern']})", "",
              "| Detector | Signature |", "| --- | --- |"]
    for name in catalog.list_by_kind("chart_pattern"):
        fn = catalog.get("chart_pattern", name)
        lines.append(f"| `{name}` | `{name}{_signature(fn)}` |")
    lines.append("")

    lines += [f"## Strategies ({counts['strategy']})", "",
              "| Strategy | Signature | Description |", "| --- | --- | --- |"]
    for name in catalog.list_by_kind("strategy"):
        fn = catalog.get("strategy", name)
        desc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        lines.append(f"| `{name}` | `{name}{_signature(fn)}` | {desc[0] if desc else ''} |")
    lines.append("")

    out = DOCS / "INDICATORS.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_new_features_md() -> Path:
    counts = catalog.registry_counts()
    lines = [
        "# New Features",
        "",
        "Auto-generated from `quant.catalog` - do not edit by hand.",
        "Regenerate with `python scripts/generate_quant_docs.py`.",
        "",
        "## Overview",
        "",
        "| Module | Kind | Items |",
        "| --- | --- | --- |",
    ]
    mapping = [
        ("indicators_ext", "indicator"), ("ehlers", "indicator"),
        ("candlesticks", "candle"), ("patterns_ext", "chart_pattern"),
        ("statistics_ext", "statistic"), ("features_ext", "feature"),
        ("metrics_ext", "metric"), ("risk_ext", "risk"),
        ("portfolio_ext", "portfolio"), ("strategies_ext", "strategy"),
        ("options", "option"), ("execution_ext", "execution"),
    ]
    for module, kind in mapping:
        mod = __import__(f"quant.{module}", fromlist=["x"])
        if kind == "strategy":
            count = len(mod.list_strategies())
        elif kind == "candle":
            count = len(mod.ALL_PATTERNS)
        else:
            count = len(getattr(mod, "__all__", []))
        lines.append(f"| `quant.{module}` | {kind} | {count} |")
    screener = __import__("quant.screener_ext", fromlist=["x"])
    lines += [f"| `quant.screener_ext` | scanner | {len(screener.__all__)} |",
              f"| `quant.catalog` | registry | {sum(counts.values())} |", ""]

    lines += ["## Module details", ""]
    for module, kind in mapping:
        lines += [f"### `quant.{module}`", "", MODULE_BLURB.get(module, ""), ""]
        names = catalog.list_by_kind(kind) if kind not in ("indicator",) else None
        if kind == "indicator":
            lines += ["See [INDICATORS.md](INDICATORS.md) for the full indicator tables.", ""]
            continue
        lines += ["| Name | Signature | Description |", "| --- | --- | --- |"]
        for name in names:
            fn = catalog.get(kind, name)
            desc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
            lines.append(f"| `{name}` | `{name}{_signature(fn)}` | {desc[0] if desc else ''} |")
        lines.append("")

    lines += ["## Usage", "", "```python", "from quant import catalog, features_ext, strategies_ext, portfolio_ext",
              "", "catalog.registry_counts()                 # discover everything",
              "catalog.search('divergence')              # search names/docstrings",
              "X = features_ext.build_feature_matrix(df) # engineered feature matrix",
              "pos = strategies_ext.strategy_positions('sma_cross', df)",
              "w = portfolio_ext.risk_parity_weights(returns)",
              "```", ""]

    out = DOCS / "NEW_FEATURES.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    a = write_indicators_md()
    b = write_new_features_md()
    for path in (a, b):
        print(f"wrote {path} ({len(path.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
