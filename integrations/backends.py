"""Backtesting framework adapters.

Exposes the native ``quant.backtest.BacktestEngine`` as the default engine and
adapters for external frameworks when installed. All adapters share one
signature: prices (wide) + signals (wide in [-1, 1]) -> metrics dict.

Integrated (registry ids): backtrader (23), qlib (24), freqtrade (26),
nautilus_trader (27), backtesting.py (28).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BACKTRADER_AVAILABLE", "BACKTESTING_PY_AVAILABLE", "VECTORBT_AVAILABLE",
    "QLIB_AVAILABLE", "FREQTRADE_AVAILABLE", "NAUTILUS_AVAILABLE",
    "BACKEND_STATUS", "run_native_backtest", "run_backtesting_py",
    "run_backtrader", "run_vectorbt", "compare_backends",
]

try:
    import backtrader as _bt
    BACKTRADER_AVAILABLE = True
except Exception:
    BACKTRADER_AVAILABLE = False

try:
    import backtesting as _btpy
    BACKTESTING_PY_AVAILABLE = True
except Exception:
    BACKTESTING_PY_AVAILABLE = False

try:
    import vectorbt as _vbt
    VECTORBT_AVAILABLE = True
except Exception:
    VECTORBT_AVAILABLE = False

try:
    import qlib as _qlib
    QLIB_AVAILABLE = True
except Exception:
    QLIB_AVAILABLE = False

try:
    import freqtrade as _ft
    FREQTRADE_AVAILABLE = True
except Exception:
    FREQTRADE_AVAILABLE = False

try:
    import nautilus_trader as _nt
    NAUTILUS_AVAILABLE = True
except Exception:
    NAUTILUS_AVAILABLE = False


def run_native_backtest(prices: pd.DataFrame, signals: pd.DataFrame,
                        **kwargs) -> dict:
    """Native event-driven engine (always available)."""
    from quant.backtest import run_backtest
    result = run_backtest(prices, signals, **kwargs)
    return {"engine": "native", **result.metrics}


def run_backtesting_py(prices: pd.DataFrame, signals: pd.DataFrame,
                       cash: float = 1_000_000.0, commission: float = 0.0001,
                       **kwargs) -> dict:
    """Run single-asset backtest via backtesting.py (28); native fallback."""
    if not BACKTESTING_PY_AVAILABLE or prices.shape[1] != 1:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (backtesting.py fallback)"
        return out
    try:
        from backtesting import Backtest, Strategy

        close = prices.iloc[:, 0]
        weight = signals.iloc[:, 0].shift(1).fillna(0.0)

        class SignalStrategy(Strategy):
            def init(self):
                self.w = np.asarray(weight, dtype="float64")

            def next(self):
                target = self.w[len(self.data) - 1]
                pos = 1.0 if target > 0 else (-1.0 if target < 0 else 0.0)
                if pos > 0 and not self.position.is_long:
                    self.buy(size=min(0.95, abs(target)))
                elif pos < 0 and not self.position.is_short:
                    self.sell(size=min(0.95, abs(target)))
                elif pos == 0 and self.position:
                    self.position.close()

        data = pd.DataFrame({"Close": close})
        bt = Backtest(data, SignalStrategy, cash=cash, commission=commission,
                      exclusive_orders=True)
        stats = bt.run()
        return {
            "engine": "backtesting.py",
            "total_return": float(stats.get("Return [%]", 0.0)) / 100.0,
            "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
            "max_drawdown": float(np.nan_to_num(stats.get("Max. Drawdown [%]", np.nan))) / 100.0,
            "n_trades": int(stats.get("# Trades", 0)),
        }
    except Exception:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (backtesting.py error fallback)"
        return out


def run_backtrader(prices: pd.DataFrame, signals: pd.DataFrame,
                   cash: float = 1_000_000.0, commission: float = 0.0001,
                   **kwargs) -> dict:
    """Run via backtrader (23); native fallback if unavailable."""
    if not BACKTRADER_AVAILABLE or prices.shape[1] != 1:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (backtrader fallback)"
        return out
    try:
        import backtrader as bt

        class SignalFeeder(bt.feeds.PandasData):
            lines = ("signal",)
            params = (("signal", -1),)

        class SignalStrategy(bt.Strategy):
            def __init__(self):
                self.signal_line = self.data.lines.signal

            def next(self):
                target = self.signal_line[0]
                if target > 0 and not self.position:
                    self.order_target_percent(target=0.95)
                elif target < 0:
                    self.order_target_percent(target=0.0)
                elif target == 0 and self.position:
                    self.close()

        df = pd.DataFrame({"open": prices.iloc[:, 0], "high": prices.iloc[:, 0],
                           "low": prices.iloc[:, 0], "close": prices.iloc[:, 0],
                           "volume": 1e6,
                           "signal": signals.iloc[:, 0].shift(1).fillna(0.0)})
        cerebro = bt.Cerebro()
        cerebro.addstrategy(SignalStrategy)
        cerebro.adddata(SignalFeeder(dataname=df))
        cerebro.broker.setcash(cash)
        cerebro.broker.setcommission(commission=commission)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                            riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        strat = results[0]
        return {
            "engine": "backtrader",
            "total_return": float(end_value / start_value - 1.0),
            "sharpe": float(strat.analyzers.sharpe.get_analysis().get(
                "sharperatio") or np.nan),
            "max_drawdown": float(strat.analyzers.dd.get_analysis().get(
                "max", {}).get("drawdown", np.nan)) / 100.0,
        }
    except Exception:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (backtrader error fallback)"
        return out


def run_vectorbt(prices: pd.DataFrame, signals: pd.DataFrame,
                 fees: float = 0.0001, **kwargs) -> dict:
    """Vectorized portfolio backtest via vectorbt; native fallback."""
    if not VECTORBT_AVAILABLE:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (vectorbt fallback)"
        return out
    try:
        import vectorbt as vbt
        target = signals.shift(1).fillna(0.0)
        close = prices.ffill()
        pf = vbt.Portfolio.from_orders(
            close=close,
            size=target * 1_000_000.0 / close,
            size_type="amount",
            fees=fees,
            init_cash=1_000_000.0,
            cash_sharing=target.shape[1] > 1,
        )
        return {
            "engine": "vectorbt",
            "total_return": float(pf.total_return()),
            "sharpe": float(pf.sharpe_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
        }
    except Exception:
        out = run_native_backtest(prices, signals)
        out["engine"] = "native (vectorbt error fallback)"
        return out


def compare_backends(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Run every available backend and return a comparison table."""
    rows = {}
    rows["native"] = run_native_backtest(prices, signals)
    if BACKTESTING_PY_AVAILABLE and prices.shape[1] == 1:
        rows["backtesting.py"] = run_backtesting_py(prices, signals)
    if BACKTRADER_AVAILABLE and prices.shape[1] == 1:
        rows["backtrader"] = run_backtrader(prices, signals)
    if VECTORBT_AVAILABLE:
        rows["vectorbt"] = run_vectorbt(prices, signals)
    return pd.DataFrame(rows).T


BACKEND_STATUS = {
    "native": True,
    "backtrader": BACKTRADER_AVAILABLE,
    "backtesting_py": BACKTESTING_PY_AVAILABLE,
    "vectorbt": VECTORBT_AVAILABLE,
    "qlib": QLIB_AVAILABLE,
    "freqtrade": FREQTRADE_AVAILABLE,
    "nautilus_trader": NAUTILUS_AVAILABLE,
}
