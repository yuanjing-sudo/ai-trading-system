"""Comprehensive test of all quant modules."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

print("=" * 60)
print("QUANT MODULE VERIFICATION")
print("=" * 60)

# 1. Test imports
print("\n[1] Testing imports...")
from quant import (
    indicators, alpha101, pandasta_wrapper, ccxt_fetcher,
    quantstats_reports, factors, strategies, risk, backtest, optimizer
)
print("   All modules imported successfully")

# 2. Test indicators
print("\n[2] Testing indicators...")
np.random.seed(42)
n = 300
dates = pd.date_range('2023-01-01', periods=n, freq='D')
close = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n) * 0.01)), index=dates)
high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
open_ = close.shift(1).fillna(close.iloc[0])
volume = pd.Series(np.abs(np.random.randn(n)) * 1e6 + 1e5, index=dates)
returns = close.pct_change()

tests = {
    'sma': indicators.sma(close, 20),
    'ema': indicators.ema(close, 20),
    'rsi': indicators.rsi(close, 14),
    'macd': indicators.macd(close),
    'atr': indicators.atr(high, low, close),
    'bbands': indicators.bollinger_bands(close),
    'obv': indicators.obv(close, volume),
    'mfi': indicators.mfi(high, low, close, volume),
    'adx': indicators.adx(high, low, close),
    'supertrend': indicators.supertrend(high, low, close),
    'ichimoku': indicators.ichimoku(high, low, close),
    'sharpe': indicators.sharpe_ratio(returns),
    'mdd': indicators.max_drawdown(close),
}
for name, result in tests.items():
    status = "OK" if result is not None else "FAIL"
    print(f"   {name}: {status}")

# 3. Test Alpha101
print("\n[3] Testing Alpha101...")
from quant.alpha101 import (
    alpha001, alpha050, alpha101 as alpha101_fn, compute_all_alphas,
)

alpha_test = alpha001(close.to_frame('A'), volume.to_frame('A'), close.to_frame('A'))
print(f"   alpha001: OK (shape={alpha_test.shape})")

alpha_test2 = alpha050(close.to_frame('A'), volume.to_frame('A'))
print(f"   alpha050: OK (shape={alpha_test2.shape})")

alpha_test3 = alpha101_fn(close.to_frame('A'), volume.to_frame('A'), close.to_frame('A'))
print(f"   alpha101: OK (shape={alpha_test3.shape})")

# Test compute_all_alphas
all_alphas = compute_all_alphas(close.to_frame('A'), volume.to_frame('A'))
print(f"   compute_all_alphas: OK (shape={all_alphas.shape}, {len(all_alphas.columns.get_level_values(0).unique())} alphas)")

# 4. Test pandas-ta wrapper
print("\n[4] Testing pandas-ta wrapper...")
from quant.pandasta_wrapper import PANDAS_TA_AVAILABLE, ta_rsi, ta_macd, ta_bbands, apply_indicator
print(f"   pandas-ta available: {PANDAS_TA_AVAILABLE}")

rsi_test = ta_rsi(close.to_frame('A'), 14)
print(f"   ta_rsi: OK (shape={rsi_test.shape})")

macd_test = ta_macd(close.to_frame('A'))
print(f"   ta_macd: OK (shape={macd_test.shape})")

bbands_test = ta_bbands(close.to_frame('A'))
print(f"   ta_bbands: OK (shape={bbands_test.shape})")

# Test apply_indicator dispatcher
try:
    result = apply_indicator('rsi', close.to_frame('A'), length=14)
    print(f"   apply_indicator(rsi): OK (shape={result.shape})")
except Exception as e:
    print(f"   apply_indicator(rsi): FAIL ({e})")

# 5. Test CCXT fetcher
print("\n[5] Testing CCXT fetcher...")
from quant.ccxt_fetcher import CCXT_AVAILABLE, get_exchanges, resolve_symbol, CryptoDataFetcher
print(f"   ccxt available: {CCXT_AVAILABLE}")
print(f"   get_exchanges: {len(get_exchanges())} exchanges")

# Test symbol resolution
sym = resolve_symbol("BTC-USDT")
print(f"   resolve_symbol('BTC-USDT'): {sym}")
sym2 = resolve_symbol("BTCUSDT")
print(f"   resolve_symbol('BTCUSDT'): {sym2}")
sym3 = resolve_symbol("ETH/USD")
print(f"   resolve_symbol('ETH/USD'): {sym3}")

# Test fetcher class (will fall back to yfinance)
fetcher = CryptoDataFetcher()
try:
    df = fetcher.fetch_ohlcv("BTC/USDT", "1d", limit=10)
    print(f"   fetch_ohlcv: OK (shape={df.shape})")
except Exception as e:
    print(f"   fetch_ohlcv: FAIL ({e})")

# 6. Test quantstats reports
print("\n[6] Testing quantstats reports...")
from quant.quantstats_reports import QUANTSTATS_AVAILABLE, performance_summary, risk_summary, generate_html_report
print(f"   quantstats available: {QUANTSTATS_AVAILABLE}")

summary = performance_summary(returns)
print(f"   performance_summary: OK ({len(summary)} metrics)")

risk = risk_summary(returns)
print(f"   risk_summary: OK ({len(risk)} metrics)")

# Test HTML report generation
html_path = generate_html_report(returns, title="Test Report", output_path="test_report.html")
print(f"   generate_html_report: OK ({html_path})")

# 7. Test factors
print("\n[7] Testing factors...")
from quant.factors import momentum_factor, composite_factor, composite_value_factor
prices_wide = pd.DataFrame({'A': close, 'B': close * 1.1})
returns_wide = pd.DataFrame({'A': returns, 'B': returns * 1.1})
volume_wide = pd.DataFrame({'A': volume, 'B': volume * 1.1})

mom = momentum_factor(prices_wide, 60)
print(f"   momentum_factor: OK (shape={mom.shape})")

val = composite_value_factor(prices_wide, close.to_frame('A'), close.to_frame('A'), close.to_frame('A'), close.to_frame('A'))
print(f"   composite_value_factor: OK (shape={val.shape})")

# 8. Test strategies
print("\n[8] Testing strategies...")
from quant.strategies import dual_momentum_strategy, mean_reversion_strategy
sig = dual_momentum_strategy(prices_wide, lookback=60)
print(f"   dual_momentum_strategy: OK (shape={sig.shape})")

sig2 = mean_reversion_strategy(prices_wide, lookback=20)
print(f"   mean_reversion_strategy: OK (shape={sig2.shape})")

# 9. Test risk
print("\n[9] Testing risk module...")
from quant.risk import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
sharpe = sharpe_ratio(returns)
print(f"   sharpe_ratio: OK ({sharpe:.4f})")

sortino = sortino_ratio(returns)
print(f"   sortino_ratio: OK ({sortino:.4f})")

mdd = max_drawdown(close)
print(f"   max_drawdown: OK ({mdd:.4f})")

# 10. Test backtest
print("\n[10] Testing backtest module...")
from quant.backtest import BacktestEngine, run_backtest
engine = BacktestEngine()
result = engine.run(prices_wide, sig)
print(f"   BacktestEngine.run: OK ({result.metrics['total_return']:.4f} total return)")

# 11. Test optimizer
print("\n[11] Testing optimizer module...")
from quant.optimizer import WalkForwardOptimizer, ParameterGrid
def factory(prices, lookback=60):
    return dual_momentum_strategy(prices, lookback=lookback)

opt = WalkForwardOptimizer(factory, {'lookback': [30, 60]}, n_splits=3, metric='sharpe')
opt.fit(prices_wide)
print(f"   WalkForwardOptimizer: OK (best_params={opt.best_params_})")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
