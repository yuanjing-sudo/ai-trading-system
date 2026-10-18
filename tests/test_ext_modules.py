"""Offline tests for the extended quant modules (no network, no optional deps)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


def _ohlcv(n: int = 300, seed: int = 42, mu: float = 0.0005, sigma: float = 0.012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ret = rng.normal(mu, sigma, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _ohlcv()


@pytest.fixture(scope="module")
def universe() -> dict[str, pd.DataFrame]:
    specs = [("AAPL", 42, 0.0006, 0.012), ("MSFT", 7, 0.0004, 0.010),
             ("TSLA", 13, 0.0009, 0.030), ("AMD", 99, -0.0002, 0.020)]
    return {s: _ohlcv(seed=sd, mu=m, sigma=sig) for s, sd, m, sig in specs}


def _routable_args(fn, data: dict) -> dict | None:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return None
    kwargs = {}
    for name, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name in data:
            kwargs[name] = data[name]
        elif p.default is inspect.Parameter.empty:
            return None
    return kwargs


def test_indicators_ext_all_callable_and_partitioned(df):
    from quant import catalog, indicators_ext

    data = {
        "series": df["close"], "close": df["close"], "open_": df["open"],
        "open": df["open"], "high": df["high"], "low": df["low"],
        "volume": df["volume"],
    }
    assert len(indicators_ext.__all__) >= 130
    ok = 0
    for name in indicators_ext.__all__:
        fn = getattr(indicators_ext, name)
        kwargs = _routable_args(fn, data)
        if kwargs is None:
            continue
        fn(**kwargs)
        ok += 1
    assert ok >= 100

    close = set(catalog.indicators_by_input("close"))
    ohlc = set(catalog.indicators_by_input("ohlc"))
    ohlcv = set(catalog.indicators_by_input("ohlcv"))
    other = set(catalog.indicators_by_input("other"))
    total = set(catalog.list_by_kind("indicator"))
    assert close | ohlc | ohlcv | other == total
    assert len(close) + len(ohlc) + len(ohlcv) + len(other) == len(total)


def test_ehlers(df):
    from quant import ehlers

    sw = ehlers.sine_wave(df["close"])
    assert {"sine", "lead_sine", "period"} <= set(sw.columns)
    assert len(sw) == len(df)
    dc = ehlers.dominant_cycle(df["close"], 10, 40)
    assert dc is not None


def test_candlesticks(df):
    from quant import candlesticks

    det = candlesticks.detect_all(df["open"], df["high"], df["low"], df["close"])
    assert isinstance(det, pd.DataFrame)
    assert det.shape[1] == len(candlesticks.ALL_PATTERNS)
    assert det.dtypes.eq(bool).all()
    score = candlesticks.pattern_score(df["open"], df["high"], df["low"], df["close"])
    assert score.between(-1, 1).all()
    assert len(candlesticks.pattern_counts(det)) == det.shape[1]


def test_statistics_ext(df):
    from quant import statistics_ext as st

    ret = df["close"].pct_change(fill_method=None).dropna()
    cov = st.ewma_cov(pd.DataFrame({"a": ret, "b": ret.shift(1)}).dropna())
    assert np.asarray(cov).shape == (2, 2)
    lw = st.ledoit_wolf_shrinkage(ret.to_frame("a"))
    assert {"cov", "shrinkage", "mu"} <= set(lw)
    pca = st.PCA(n_components=2).fit_transform(pd.DataFrame({"a": ret, "b": ret.rolling(2).mean()}).dropna())
    assert np.asarray(pca).shape[1] == 2
    ci = st.bootstrap_ci(ret.to_numpy(), np.mean, n_boot=50, seed=1)
    assert ci["lower"] <= ci["upper"]


def test_features_ext(df):
    from quant import features_ext as fx

    mat = fx.build_feature_matrix(df)
    assert isinstance(mat, pd.DataFrame)
    assert mat.shape[0] == len(df)
    assert mat.notna().sum().sum() > 0
    y = fx.target_forward_returns(df, (1, 5))
    assert isinstance(y, pd.DataFrame)
    assert {"fwd_ret_1", "fwd_ret_5"} <= set(y.columns)
    assert y["fwd_ret_5"].notna().sum() > 0


def test_patterns_ext(df):
    from quant import patterns_ext as px

    out = px.detect_patterns(df)
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == len(df)
    state = px.chart_state(df)
    assert isinstance(state, pd.DataFrame) and state.shape[0] == len(df)


def test_metrics_ext(df):
    from quant import metrics_ext as mx

    ret = df["close"].pct_change(fill_method=None).dropna()
    m = mx.strategy_metrics(ret)
    assert isinstance(m, dict) and len(m) >= 20
    assert "sharpe" in m or "sharpe_ratio" in m
    trades = pd.Series(np.random.default_rng(0).choice([0.0, 1.0], len(df)), index=df.index)
    tm = mx.trades_from_positions(trades, df["close"])
    assert isinstance(tm, (pd.DataFrame, pd.Series))


def test_risk_ext(df):
    from quant import risk_ext as rx

    rng = np.random.default_rng(3)
    rets = pd.DataFrame({s: rng.normal(0.0004, s2, 400) for s, s2 in
                         [("a", 0.01), ("b", 0.02), ("c", 0.015), ("d", 0.012)]})
    w = rx.risk_budget_weights(rets.cov())
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w >= 0).all()
    erc = rx.equal_risk_contribution(rets)
    assert abs(float(erc.sum()) - 1.0) < 1e-6


def test_strategies_ext(df):
    from quant import strategies_ext as sx

    assert len(sx.list_strategies()) >= 60
    pos = sx.strategy_positions("sma_cross", df)
    assert len(pos) == len(df)
    sig = sx.all_strategy_signals(df)
    assert isinstance(sig, pd.DataFrame)
    assert sig.shape[0] == len(df)


def test_portfolio_ext(df):
    from quant import portfolio_ext as ptx

    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-02", periods=400, freq="B")
    A = pd.DataFrame({k: rng.normal(0.0004, s, 400) for k, s in
                      zip("abcd", [0.011, 0.007, 0.014, 0.009])}, index=idx)
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))
    assert abs(float(ptx.equal_weight(list("abcd")).sum()) - 1.0) < 1e-9
    no_cost = ptx.calendar_rebalance_returns(w, A, "ME", 0.0)
    with_cost = ptx.calendar_rebalance_returns(w, A, "ME", 0.001)
    assert np.isfinite(no_cost).all()
    assert float(no_cost.sum()) > float(with_cost.sum())
    assert np.isfinite(ptx.threshold_rebalance_returns(w, A, 0.0)).all()
    lw = ptx.ledoit_wolf_covariance(A)
    assert np.allclose(lw.to_numpy(), lw.to_numpy().T)
    frontier = ptx.efficient_frontier(A, 8)
    assert frontier["volatility"].is_monotonic_increasing


def test_catalog(df):
    from quant import catalog

    counts = catalog.registry_counts()
    assert counts["indicator"] >= 200
    assert counts["strategy"] >= 60
    assert counts["candle"] == 90
    hits = catalog.search("sharpe")
    assert len(hits) > 0
    desc = catalog.describe("rsi")
    assert desc["name"] == "rsi"


def test_screener_ext(universe):
    from quant import screener_ext as sc

    table = sc.screener_table(universe, ["sharpe", "return", "atr_pct"])
    assert table.shape == (len(universe), 3)
    top = sc.screen_top_n(universe, 2, "sharpe")
    assert len(top) == 2
    scan = sc.strategy_scan(universe, ["sma_cross", "rsi_reversal"])
    assert scan.shape == (len(universe), 2)
    assert len(sc.screen_report(universe).splitlines()) == len(universe) + 2
