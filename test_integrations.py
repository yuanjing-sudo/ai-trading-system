import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

np.random.seed(7)
n = 300
dates = pd.date_range("2023-01-01", periods=n, freq="D")
prices = pd.DataFrame(
    {"AAPL": 150 * np.exp(np.cumsum(np.random.randn(n) * 0.015))}, index=dates
)
returns_wide = prices.pct_change().dropna()
signals = pd.DataFrame(1.0, index=dates, columns=["AAPL"])

print("== backends ==")
from integrations.backends import compare_backends, run_native_backtest
print(run_native_backtest(prices, signals))
print(compare_backends(prices, signals))

print("\n== analytics ==")
from integrations.analytics import (
    optimize_weights, option_price, implied_volatility, factor_analysis,
    portfolio_tear_sheet, precise_covariance,
)
w = optimize_weights(returns_wide)
print("optimize_weights:", w.round(3).to_dict())
call = option_price("c", 100, 105, 1.0, 0.05, 0.2)
print("call price:", round(call, 4), "(expected ~8.02)")
iv = implied_volatility(call, "c", 100, 105, 1.0, 0.05)
print("implied vol:", round(iv, 4), "(expected ~0.2)")
fa = factor_analysis(prices, prices)
print("factor_analysis ic_mean_5d:", fa["ic_mean_5d"].to_dict())
ts = portfolio_tear_sheet(returns_wide.iloc[:, 0])
print("tear sheet metrics:", len(ts))
cov = precise_covariance(returns_wide)
print("cov shape:", cov.shape)

print("\n== storage ==")
from integrations.storage import export_json_snapshot, sql_query, to_polars, save_bars, load_bars
import tempfile, os
tmp = tempfile.mkdtemp()
snap = export_json_snapshot({"ticker": "AAPL", "price": 191.3, "date": pd.Timestamp("2024-01-02")}, os.path.join(tmp, "snap.json"))
print("snapshot:", snap.exists())
pq = save_bars(prices, os.path.join(tmp, "bars.parquet"))
print("parquet:", load_bars(pq).shape)
print("duckdb fallback:", sql_query("SELECT 1").iloc[0, 0])

print("\n== agents (offline fallback) ==")
from integrations.agents import analyze_symbol
out = analyze_symbol("AAPL", prices)
print("decision:", out["decision"]["decision"], "| confidence:", round(out["decision"]["confidence"], 2))
print("llm_available:", out["llm_available"])
print(out["report_markdown"][:400])

print("\nALL INTEGRATION CHECKS DONE")
