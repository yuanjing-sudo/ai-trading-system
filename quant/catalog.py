"""Unified catalog of every indicator, feature, pattern, strategy, metric, and
risk/portfolio tool in the :mod:`quant` package.

The catalog is built once at import time by scanning the public ``__all__`` of
each module, so it always reflects the live code. Use it to discover tools,
describe their signatures, classify indicators by the inputs they require, and
search across the whole library.

Example
-------
>>> from quant import catalog
>>> catalog.registry_counts()
{'indicator': 180, 'statistic': 95, ...}
>>> catalog.search("divergence")
[('pattern', 'bullish_divergence', ...), ('pattern', 'rsi_divergence', ...), ...]
"""

from __future__ import annotations

import inspect

from quant import (
    candlesticks,
    ehlers,
    features_ext,
    indicators,
    indicators_ext,
    metrics_ext,
    patterns_ext,
    portfolio_ext,
    risk_ext,
    statistics_ext,
    strategies_ext,
)

__all__ = [
    "CATALOG", "registry_counts", "list_by_kind", "search", "describe", "get",
    "catalog_summary", "indicators_by_input", "indicator_inputs",
    "features_by_group", "catalog_report",
]

_PRICE_INPUTS = {"series", "close", "high", "low", "open", "open_", "volume", "index"}


def _collect(module, kind: str) -> dict:
    out = {}
    for name in getattr(module, "__all__", []):
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if callable(obj):
            out[name] = obj
    return out


def _detector_names() -> dict:
    out = {}
    for name in getattr(patterns_ext, "__all__", []):
        obj = getattr(patterns_ext, name, None)
        if callable(obj) and not inspect.isclass(obj):
            out[name] = obj
    return out


def _build() -> dict:
    catalog = {}
    catalog["indicator"] = {
        **_collect(indicators, "indicator"),
        **_collect(indicators_ext, "indicator"),
        **_collect(ehlers, "indicator"),
    }
    catalog["statistic"] = _collect(statistics_ext, "statistic")
    catalog["feature"] = _collect(features_ext, "feature")
    catalog["candle"] = {name: fn for name, fn in candlesticks.ALL_PATTERNS.items()}
    catalog["chart_pattern"] = _detector_names()
    catalog["metric"] = _collect(metrics_ext, "metric")
    catalog["risk"] = _collect(risk_ext, "risk")
    catalog["portfolio"] = _collect(portfolio_ext, "portfolio")
    catalog["strategy"] = {name: meta["func"] for name, meta in strategies_ext.STRATEGIES.items()}
    return catalog


def _entry_summary(kind: str, name: str, obj) -> dict:
    params = {}
    if inspect.isclass(obj):
        try:
            params = {
                p.name: p
                for p in inspect.signature(obj.__init__).parameters.values()
                if p.name != "self"
            }
        except (TypeError, ValueError):
            params = {}
        doc = (obj.__doc__ or "").strip().splitlines()
    else:
        try:
            params = dict(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            params = {}
        doc = (obj.__doc__ or "").strip().splitlines()
    required = [n for n, p in params.items() if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return {
        "kind": kind,
        "name": name,
        "class": inspect.isclass(obj),
        "n_params": len(params),
        "required": required,
        "summary": doc[0] if doc else "",
    }


CATALOG: dict[str, dict] = _build()


def registry_counts() -> dict:
    """Number of registered items per kind."""
    return {kind: len(entries) for kind, entries in CATALOG.items()}


def list_by_kind(kind: str) -> list[str]:
    """Sorted names registered under ``kind``."""
    if kind not in CATALOG:
        raise KeyError(f"Unknown kind '{kind}'. Available: {', '.join(sorted(CATALOG))}")
    return sorted(CATALOG[kind])


def get(kind: str, name: str):
    """Fetch the callable registered under ``kind``/``name``."""
    if kind not in CATALOG:
        raise KeyError(f"Unknown kind '{kind}'. Available: {', '.join(sorted(CATALOG))}")
    if name not in CATALOG[kind]:
        raise KeyError(f"Unknown {kind} '{name}'. Try catalog.search('{name}').")
    return CATALOG[kind][name]


def search(term: str, kind: str | None = None) -> list[tuple[str, str, str]]:
    """Substring search over names and docstrings; returns (kind, name, summary)."""
    needle = term.lower()
    hits = []
    kinds = [kind] if kind else list(CATALOG)
    for k in kinds:
        for name, obj in CATALOG[k].items():
            summary = _entry_summary(k, name, obj)["summary"]
            if needle in name.lower() or needle in summary.lower():
                hits.append((k, name, summary))
    return sorted(hits)


def describe(name: str, kind: str | None = None) -> dict:
    """Full description (signature, defaults, docstring) for a named item."""
    kinds = [kind] if kind else list(CATALOG)
    for k in kinds:
        if name in CATALOG.get(k, {}):
            obj = CATALOG[k][name]
            entry = _entry_summary(k, name, obj)
            params = {}
            if inspect.isclass(obj):
                try:
                    sig = inspect.signature(obj.__init__)
                    params = {n: p for n, p in sig.parameters.items() if n != "self"}
                except (TypeError, ValueError):
                    params = {}
            else:
                try:
                    params = dict(inspect.signature(obj).parameters)
                except (TypeError, ValueError):
                    params = {}
            entry["defaults"] = {
                n: (None if p.default is inspect.Parameter.empty else p.default)
                for n, p in params.items()
                if n != "self"
            }
            entry["doc"] = (obj.__doc__ or "").strip()
            return entry
    raise KeyError(f"'{name}' not found in the catalog")


def catalog_summary(kind: str | None = None) -> list[dict]:
    """Table-ready summary rows (kind, name, class, n_params, required, summary)."""
    kinds = [kind] if kind else list(CATALOG)
    rows = []
    for k in kinds:
        for name, obj in CATALOG[k].items():
            rows.append(_entry_summary(k, name, obj))
    return sorted(rows, key=lambda r: (r["kind"], r["name"]))


def indicator_inputs(name: str) -> list[str]:
    """Required price-like inputs an indicator needs (e.g. ['high','low','close'])."""
    obj = get("indicator", name)
    if inspect.isclass(obj):
        return []
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return []
    return [
        n for n, p in params.items()
        if p.default is inspect.Parameter.empty and n in _PRICE_INPUTS
    ]


def indicators_by_input(level: str = "close") -> list[str]:
    """Indicators grouped by input shape: 'close', 'ohlc', 'ohlcv', or 'other'."""
    out = []
    for name in CATALOG["indicator"]:
        inputs = set(indicator_inputs(name))
        if inputs and level == "close" and inputs <= {"series", "close", "open", "open_"}:
            out.append(name)
        elif level == "ohlc" and {"high", "low", "close"} <= inputs and "volume" not in inputs:
            out.append(name)
        elif level == "ohlcv" and "volume" in inputs:
            out.append(name)
        elif level == "other" and not (
            (inputs and inputs <= {"series", "close", "open", "open_"})
            or {"high", "low", "close"} <= inputs
            or "volume" in inputs
        ):
            out.append(name)
    return sorted(out)


def features_by_group() -> dict:
    """Feature builders grouped by their namespace prefix in ``__all__``."""
    text = features_ext.__doc__ or ""
    groups = {}
    for name in getattr(features_ext, "__all__", []):
        obj = getattr(features_ext, name, None)
        if not callable(obj):
            continue
        doc = (getattr(obj, "__doc__", "") or "").strip().splitlines()
        groups.setdefault("features", []).append({"name": name, "doc": doc[0] if doc else ""})
    groups["_source"] = text.strip().splitlines()[0] if text else ""
    return groups


def catalog_report() -> str:
    """Human-readable markdown report of every registry."""
    lines = ["# quant catalog", ""]
    for kind, entries in sorted(CATALOG.items()):
        lines.append(f"## {kind} ({len(entries)})")
        lines.append("")
        for name in sorted(entries):
            summary = _entry_summary(kind, name, entries[name])["summary"]
            lines.append(f"- `{name}` - {summary}" if summary else f"- `{name}`")
        lines.append("")
    return "\n".join(lines)