"""Generate docs/INTEGRATIONS.md from integrations.registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.registry import CATEGORIES, REPOS, integration_summary

OUT = ROOT / "docs" / "INTEGRATIONS.md"

STATUS_LABEL = {
    "core": "Core (wired in)",
    "adapter": "Adapter (optional pip)",
    "wrapped": "Wrapped (native reimplementation)",
    "reference": "Reference (cataloged)",
}


def main() -> int:
    summary = integration_summary()
    lines = [
        "# Third-Party Integration Catalog",
        "",
        "This repository consolidates the 100-repo trading/quant ecosystem watchlist.",
        "Every entry below is tracked in `integrations/registry.py` and exposed via",
        "`GET /api/integrations` and `GET /api/integrations/repos`.",
        "",
        "## Status summary",
        "",
        "| Status | Count | Meaning |",
        "| --- | --- | --- |",
    ]
    for status in ("core", "adapter", "wrapped", "reference"):
        lines.append(
            f"| {STATUS_LABEL[status]} | {summary.get(status, 0)} | "
            f"{_meaning(status)} |"
        )
    lines.append(f"| **Total** | **{summary['total']}** | |")
    lines.append("")

    for cat_key, cat_name in CATEGORIES.items():
        entries = [r for r in REPOS if r.category == cat_key]
        lines.append(f"## {cat_name}")
        lines.append("")
        lines.append("| # | Repository | Status | Where |")
        lines.append("| --- | --- | --- | --- |")
        for r in entries:
            where = f"`{r.where}`" if r.where else "-"
            lines.append(
                f"| {r.id} | [{r.name}]({r.url}) | {STATUS_LABEL[r.status]} | {where} |"
            )
        lines.append("")

    lines += [
        "## Using the integrations",
        "",
        "```python",
        "from integrations import full_status",
        "print(full_status())        # availability of every optional dependency",
        "",
        "from integrations.agents import analyze_symbol",
        "result = analyze_symbol('AAPL', ohlcv_df)   # multi-agent research note",
        "",
        "from integrations.backends import compare_backends",
        "compare_backends(prices, signals)           # native vs installed frameworks",
        "```",
        "",
        "Install optional dependencies: `pip install -r requirements-optional.txt`",
        "or any subset - every adapter falls back to the native implementation.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


def _meaning(status: str) -> str:
    return {
        "core": "Primary dependency or first-class module",
        "adapter": "Optional package with graceful fallback",
        "wrapped": "Techniques reimplemented in quant/ or models/",
        "reference": "Cataloged for discovery (apps, awesome-lists, non-Python)",
    }[status]


if __name__ == "__main__":
    raise SystemExit(main())
