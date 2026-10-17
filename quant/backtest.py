"""Event-driven backtester with realistic execution costs.

Provides:
- ``BacktestEngine``: event-driven backtest loop
- ``BacktestResult``: performance metrics and equity curve
- ``run_backtest``: convenience function
- ``walk_forward_optimize``: purged walk-forward parameter search
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.risk import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
    cagr, volatility, var_historical, cvar_historical,
    ulcer_index, pain_ratio,
)
from quant.strategies import signal_to_weights, apply_position_limits

__all__ = [
    "BacktestEngine", "BacktestResult", "run_backtest",
    "walk_forward_optimize", "purged_kfold_split",
]


class BacktestResult:
    """Container for backtest results and performance metrics."""

    def __init__(
        self,
        equity: pd.Series,
        returns: pd.Series,
        weights: pd.DataFrame,
        trades: pd.DataFrame,
        costs: pd.Series,
        params: dict | None = None,
    ):
        self.equity = equity
        self.returns = returns
        self.weights = weights
        self.trades = trades
        self.costs = costs
        self.params = params or {}
        self._metrics = None

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            self._metrics = self._compute_metrics()
        return self._metrics

    def _compute_metrics(self) -> dict:
        r = self.returns.dropna()
        e = self.equity.dropna()
        return {
            "total_return": float(e.iloc[-1] / e.iloc[0] - 1.0) if len(e) > 1 else 0.0,
            "cagr": float(cagr(e)),
            "volatility": float(volatility(r)),
            "sharpe": float(sharpe_ratio(r)),
            "sortino": float(sortino_ratio(r)),
            "calmar": float(calmar_ratio(e)),
            "max_drawdown": float(max_drawdown(e)),
            "ulcer_index": float(ulcer_index(e)),
            "pain_ratio": float(pain_ratio(e)),
            "var_95": float(var_historical(r, 0.95)),
            "cvar_95": float(cvar_historical(r, 0.95)),
            "win_rate": float((r > 0).mean()) if len(r) else 0.0,
            "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
            "avg_loss": float(r[r < 0].mean()) if (r < 0).any() else 0.0,
            "profit_factor": float(
                r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.inf
            ),
            "total_costs": float(self.costs.sum()),
            "n_trades": int((self.trades != 0).any(axis=1).sum()) if len(self.trades) else 0,
            "turnover": float(self.weights.diff().abs().sum(axis=1).mean()) if len(self.weights) else 0.0,
        }

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "Backtest Summary",
            "=" * 40,
            f"Total Return : {m['total_return']*100:>8.2f}%",
            f"CAGR         : {m['cagr']*100:>8.2f}%",
            f"Volatility   : {m['volatility']*100:>8.2f}%",
            f"Sharpe       : {m['sharpe']:>8.2f}",
            f"Sortino      : {m['sortino']:>8.2f}",
            f"Calmar       : {m['calmar']:>8.2f}",
            f"Max Drawdown : {m['max_drawdown']*100:>8.2f}%",
            f"Ulcer Index  : {m['ulcer_index']:>8.2f}",
            f"VaR 95%      : {m['var_95']*100:>8.2f}%",
            f"CVaR 95%     : {m['cvar_95']*100:>8.2f}%",
            f"Win Rate     : {m['win_rate']*100:>8.2f}%",
            f"Profit Factor: {m['profit_factor']:>8.2f}",
            f"Total Costs  : {m['total_costs']*100:>8.2f}%",
            f"N Trades     : {m['n_trades']:>8d}",
            f"Turnover     : {m['turnover']:>8.4f}",
            "=" * 40,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Event-driven backtest engine with slippage, commission, and market impact."""

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 1.0,      # 1 bps per side
        slippage_bps: float = 0.5,        # 0.5 bps per side
        market_impact_coeff: float = 0.0, # temporary impact coefficient
        borrow_rate_bps: float = 0.0,     # annualized borrow rate for shorts
        min_trade_size: float = 0.0,      # minimum notional to trade
    ):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.market_impact_coeff = market_impact_coeff
        self.borrow_rate_bps = borrow_rate_bps
        self.min_trade_size = min_trade_size

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
        rebalance_freq: str = "D",
    ) -> BacktestResult:
        """Run the backtest.

        Parameters
        ----------
        prices : DataFrame
            Wide-format close prices (dates x assets).
        signals : DataFrame
            Target weights in [-1, 1] (dates x assets).
        volumes : DataFrame, optional
            Volume data for market impact estimation.
        rebalance_freq : str
            Resampling frequency for rebalance dates.
        """
        prices = prices.sort_index()
        signals = signals.reindex(prices.index).ffill().fillna(0.0)
        if volumes is not None:
            volumes = volumes.reindex(prices.index).ffill()

        # Determine rebalance dates
        if rebalance_freq in ("D", None):
            rebalance_dates = prices.index
        else:
            freq = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}.get(
                rebalance_freq, rebalance_freq
            )
            rebalance_dates = (
                prices.index.to_series()
                .groupby(pd.Grouper(freq=freq))
                .last()
                .dropna()
                .index
            )

        n_assets = prices.shape[1]
        capital = self.initial_capital
        positions = np.zeros(n_assets)  # shares held
        cash = capital
        equity_history = []
        returns_history = []
        weights_history = []
        trades_history = []
        costs_history = []

        prev_equity = capital
        prev_signal = None

        for i, date in enumerate(prices.index):
            price = prices.loc[date].to_numpy(dtype="float64")
            signal = signals.loc[date].to_numpy(dtype="float64")

            # Mark-to-market
            portfolio_value = cash + float(positions @ price)
            if i > 0 and prev_equity > 0:
                period_return = (portfolio_value - prev_equity) / prev_equity
            else:
                period_return = 0.0
            returns_history.append(period_return)
            equity_history.append(portfolio_value)
            prev_equity = portfolio_value

            # Current weights
            if portfolio_value > 0:
                current_weights = (positions * price) / portfolio_value
            else:
                current_weights = np.zeros(n_assets)

            weights_history.append(current_weights)

            # Rebalance decision
            is_rebalance = date in rebalance_dates
            if is_rebalance and (
                prev_signal is None or not np.allclose(signal, prev_signal, atol=1e-6)
            ):
                target_weights = signal.copy()
                # Apply position limits
                target_weights = apply_position_limits(
                    pd.DataFrame(target_weights.reshape(1, -1),
                                 index=[date], columns=prices.columns)
                ).iloc[0].to_numpy(dtype="float64")

                target_value = portfolio_value * target_weights
                target_shares = np.where(price > 0, target_value / price, 0.0)
                trade_shares = target_shares - positions

                # Estimate costs
                trade_notional = np.abs(trade_shares * price)
                commission_cost = trade_notional * (self.commission_bps / 1e4)
                slippage_cost = trade_notional * (self.slippage_bps / 1e4)
                impact_cost = np.zeros(n_assets)
                if volumes is not None and self.market_impact_coeff > 0:
                    vol = volumes.loc[date].to_numpy(dtype="float64")
                    vol = np.where(vol > 0, vol, 1.0)
                    participation = np.abs(trade_shares) / vol
                    impact_cost = self.market_impact_coeff * participation ** 2 * trade_notional
                total_cost = commission_cost + slippage_cost + impact_cost

                # Borrow cost for short positions (pro-rated daily)
                short_notional = np.abs(np.minimum(target_value, 0.0))
                borrow_cost = short_notional * (self.borrow_rate_bps / 1e4 / 252.0)

                # Apply trade notional and costs from cash
                trade_notional_signed = float((trade_shares * price).sum())
                cash -= trade_notional_signed + float(total_cost.sum()) + float(borrow_cost.sum())

                # Execute trades
                positions = target_shares
                trades_history.append(
                    pd.Series(trade_shares, index=prices.columns, name=date)
                )
                costs_history.append(
                    pd.Series(total_cost, index=prices.columns, name=date)
                )
                prev_signal = signal.copy()

        # Build result objects
        equity = pd.Series(equity_history, index=prices.index)
        returns = pd.Series(returns_history, index=prices.index)
        weights = pd.DataFrame(weights_history, index=prices.index, columns=prices.columns)
        trades = pd.DataFrame(trades_history) if trades_history else pd.DataFrame()
        costs = pd.Series([c.sum() for c in costs_history],
                          index=[c.name for c in costs_history]) if costs_history else pd.Series(dtype=float)

        return BacktestResult(
            equity=equity,
            returns=returns,
            weights=weights,
            trades=trades,
            costs=costs,
            params={
                "initial_capital": self.initial_capital,
                "commission_bps": self.commission_bps,
                "slippage_bps": self.slippage_bps,
                "market_impact_coeff": self.market_impact_coeff,
                "borrow_rate_bps": self.borrow_rate_bps,
                "rebalance_freq": rebalance_freq,
            },
        )


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    **kwargs,
) -> BacktestResult:
    """Convenience function to run a backtest."""
    run_keys = {"volumes", "rebalance_freq"}
    run_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in run_keys}
    engine = BacktestEngine(**kwargs)
    return engine.run(prices, signals, **run_kwargs)


# ---------------------------------------------------------------------------
# Walk-forward optimization
# ---------------------------------------------------------------------------

def purged_kfold_split(
    n_periods: int, n_splits: int = 5, embargo_days: int = 5, purge_days: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged k-fold train/test splits."""
    indices = np.arange(n_periods)
    fold_size = n_periods // n_splits
    splits = []
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_periods
        test_idx = indices[test_start:test_end]
        # Train = everything before test_start minus embargo, plus after test_end plus purge
        train_before = indices[: max(0, test_start - embargo_days)]
        train_after = indices[min(n_periods, test_end + purge_days):]
        train_idx = np.concatenate([train_before, train_after]) if len(train_before) or len(train_after) else indices[:test_start]
        splits.append((train_idx, test_idx))
    return splits


def walk_forward_optimize(
    prices: pd.DataFrame,
    signal_factory,
    param_grid: list[dict],
    n_splits: int = 5,
    embargo_days: int = 5,
    purge_days: int = 5,
    metric: str = "sharpe",
) -> tuple[dict, list[dict]]:
    """Walk-forward parameter optimization with purged splits.

    Parameters
    ----------
    prices : DataFrame
        Price data.
    signal_factory : callable
        Function(prices, **params) -> signal DataFrame.
    param_grid : list of dict
        Parameter combinations to try.
    n_splits : int
        Number of walk-forward folds.
    embargo_days : int
        Embargo days between train and test.
    purge_days : int
        Purge days after test before next train.
    metric : str
        Metric to optimize ("sharpe", "sortino", "calmar").

    Returns
    -------
    best_params : dict
        Best parameter set.
    fold_results : list of dict
        Per-fold results.
    """
    n_periods = len(prices)
    splits = purged_kfold_split(n_periods, n_splits, embargo_days, purge_days)

    fold_results = []
    param_scores = {i: [] for i in range(len(param_grid))}

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_prices = prices.iloc[train_idx]
        test_prices = prices.iloc[test_idx]

        best_score = -np.inf
        best_param_idx = 0

        for param_idx, params in enumerate(param_grid):
            try:
                train_signal = signal_factory(train_prices, **params)
                engine = BacktestEngine()
                result = engine.run(train_prices, train_signal)
                score = result.metrics.get(metric, -np.inf)
                if np.isnan(score):
                    score = -np.inf
            except Exception:
                score = -np.inf
            param_scores[param_idx].append(score)
            if score > best_score:
                best_score = score
                best_param_idx = param_idx

        # Evaluate best params on test fold
        best_params = param_grid[best_param_idx]
        test_signal = signal_factory(test_prices, **best_params)
        engine = BacktestEngine()
        test_result = engine.run(test_prices, test_signal)

        fold_results.append({
            "fold": fold_idx,
            "train_start": str(train_prices.index[0]),
            "train_end": str(train_prices.index[-1]),
            "test_start": str(test_prices.index[0]),
            "test_end": str(test_prices.index[-1]),
            "best_params": best_params,
            "train_score": best_score,
            "test_metrics": test_result.metrics,
        })

    # Aggregate: choose params with best average train score
    avg_scores = {i: float(np.mean(s)) for i, s in param_scores.items()}
    best_param_idx = max(avg_scores, key=avg_scores.get)
    best_params = param_grid[best_param_idx]

    return best_params, fold_results
