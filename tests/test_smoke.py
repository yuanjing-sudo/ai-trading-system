"""Fast, offline smoke tests for CI (no network, no optional deps required)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_quant_indicators():
    from quant.indicators import rsi, macd, bollinger_bands

    close = _synthetic_ohlcv()["close"]
    r = rsi(close, 14).dropna()
    assert len(r) > 200
    assert r.between(0, 100).all()
    macd_df = macd(close)
    assert len(macd_df) == len(close)
    bb = bollinger_bands(close, 20, 2.0)
    assert len(bb) == len(close)


def test_quant_alpha101():
    from quant.alpha101 import compute_all_alphas

    df = _synthetic_ohlcv()
    close = df[["close"]].rename(columns={"close": "A"})
    volume = df[["volume"]].rename(columns={"volume": "A"})
    alphas = compute_all_alphas(close, volume)
    assert alphas.columns.get_level_values(0).nunique() >= 90
    assert alphas.notna().sum().sum() > 1000


def test_quant_backtest_runs():
    from quant.backtest import BacktestEngine
    from quant.strategies import dual_momentum_strategy

    df = _synthetic_ohlcv()
    prices_wide = pd.DataFrame({"A": df["close"], "B": df["close"] * 1.1})
    signals = dual_momentum_strategy(prices_wide, lookback=60)
    result = BacktestEngine().run(prices_wide, signals)
    assert "total_return" in result.metrics
    assert "sharpe" in result.metrics


def test_integrations_registry_has_100_repos():
    from integrations.registry import REPOS

    assert len(REPOS) == 100
    for repo in REPOS:
        assert repo.name
        assert repo.category


def test_integrations_agents_offline_fallback():
    from integrations.agents import analyze_symbol

    df = _synthetic_ohlcv()
    result = analyze_symbol("TEST", df)
    assert result["ticker"] == "TEST"
    assert result["decision"]["decision"] in {"BUY", "SELL", "HOLD"}
    assert "report_markdown" in result


def test_integrations_storage_json_snapshot(tmp_path):
    from integrations.storage import export_json_snapshot

    out = tmp_path / "snap.json"
    export_json_snapshot({"AAPL": {"close": 123.45}}, out)
    assert out.exists()
    assert "AAPL" in out.read_text(encoding="utf-8")


def test_assistant_context_compression():
    from assistant import AIAssistant

    closes = list(np.linspace(100, 120, 120) + np.sin(np.arange(120)) * 2)
    context = {
        "symbol": "AAPL",
        "price_history": {"dates": [f"d{i}" for i in range(120)], "close": closes},
    }
    compressed = AIAssistant._compress_context(context)
    assert "price_history" not in compressed
    stats = compressed["price_stats"]
    assert stats["last_close"] > 0
    assert 0 <= stats["rsi14"] <= 100
    assert AIAssistant._wants_research("give me a deep dive on NVDA")
