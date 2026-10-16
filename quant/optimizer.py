"""Walk-forward optimization and parameter search.

Provides:
- ``ParameterGrid``: cartesian product parameter iterator
- ``PurgedKFold``: purged cross-validation splitter
- ``WalkForwardOptimizer``: walk-forward parameter search with out-of-sample
  evaluation
- ``combine_fold_results``: aggregate walk-forward results
"""

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from quant.backtest import BacktestEngine, BacktestResult

__all__ = [
    "ParameterGrid", "PurgedKFold", "WalkForwardOptimizer",
    "combine_fold_results", "summary_table",
]


class ParameterGrid:
    """Iterator over a cartesian product of parameter values.

    Example
    -------
    >>> grid = ParameterGrid({"lookback": [20, 50], "threshold": [0.0, 0.05]})
    >>> for params in grid:
    ...     print(params)
    """

    def __init__(self, param_grid: dict[str, list]):
        self.param_grid = param_grid
        self.keys = list(param_grid.keys())
        self.values = list(param_grid.values())
        self._combinations = list(itertools.product(*self.values))
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._combinations):
            raise StopIteration
        combo = self._combinations[self._index]
        self._index += 1
        return dict(zip(self.keys, combo))

    def __len__(self):
        return len(self._combinations)

    def __getitem__(self, idx):
        return dict(zip(self.keys, self._combinations[idx]))


class PurgedKFold:
    """Purged k-fold cross-validation for time series.

    Removes embargo days before each test fold and purge days after each
    test fold to prevent information leakage from overlapping samples.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 5, purge_days: int = 5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days

    def split(self, n_periods: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, test_idx) splits."""
        indices = np.arange(n_periods)
        fold_size = n_periods // self.n_splits
        splits = []
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_periods
            test_idx = indices[test_start:test_end]
            train_before = indices[: max(0, test_start - self.embargo_days)]
            train_after = indices[min(n_periods, test_end + self.purge_days):]
            if len(train_before) or len(train_after):
                train_idx = np.concatenate([train_before, train_after])
            else:
                train_idx = indices[:test_start]
            splits.append((train_idx, test_idx))
        return splits


class WalkForwardOptimizer:
    """Walk-forward parameter optimization.

    For each fold:
    1. Grid-search parameters on the training period
    2. Evaluate best parameters on the test period
    3. Aggregate out-of-sample results

    Parameters
    ----------
    signal_factory : callable
        Function(prices: DataFrame, **params) -> signal DataFrame.
    param_grid : dict or ParameterGrid
        Parameter grid to search.
    n_splits : int
        Number of walk-forward folds.
    embargo_days : int
        Embargo days before test fold.
    purge_days : int
        Purge days after test fold.
    metric : str
        Metric to optimize ("sharpe", "sortino", "calmar", "total_return").
    initial_capital : float
        Starting capital for backtests.
    commission_bps : float
        Commission in basis points per side.
    slippage_bps : float
        Slippage in basis points per side.
    """

    def __init__(
        self,
        signal_factory,
        param_grid: dict[str, list] | ParameterGrid,
        n_splits: int = 5,
        embargo_days: int = 5,
        purge_days: int = 5,
        metric: str = "sharpe",
        initial_capital: float = 1_000_000.0,
        commission_bps: float = 1.0,
        slippage_bps: float = 0.5,
    ):
        self.signal_factory = signal_factory
        self.param_grid = (
            param_grid if isinstance(param_grid, ParameterGrid)
            else ParameterGrid(param_grid)
        )
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.metric = metric
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.fold_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf

    def fit(self, prices: pd.DataFrame) -> "WalkForwardOptimizer":
        """Run walk-forward optimization on price data."""
        n_periods = len(prices)
        splitter = PurgedKFold(
            n_splits=self.n_splits,
            embargo_days=self.embargo_days,
            purge_days=self.purge_days,
        )
        splits = splitter.split(n_periods)

        param_scores = {i: [] for i in range(len(self.param_grid))}
        self.fold_results_ = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]

            best_score = -np.inf
            best_param_idx = 0

            for param_idx, params in enumerate(self.param_grid):
                try:
                    train_signal = self.signal_factory(train_prices, **params)
                    engine = BacktestEngine(
                        initial_capital=self.initial_capital,
                        commission_bps=self.commission_bps,
                        slippage_bps=self.slippage_bps,
                    )
                    result = engine.run(train_prices, train_signal)
                    score = result.metrics.get(self.metric, -np.inf)
                    if np.isnan(score):
                        score = -np.inf
                except Exception:
                    score = -np.inf
                param_scores[param_idx].append(score)
                if score > best_score:
                    best_score = score
                    best_param_idx = param_idx

            # Evaluate best params on test fold
            best_params = self.param_grid[best_param_idx]
            try:
                test_signal = self.signal_factory(test_prices, **best_params)
                engine = BacktestEngine(
                    initial_capital=self.initial_capital,
                    commission_bps=self.commission_bps,
                    slippage_bps=self.slippage_bps,
                )
                test_result = engine.run(test_prices, test_signal)
                test_metrics = test_result.metrics
            except Exception:
                test_metrics = {}

            self.fold_results_.append({
                "fold": fold_idx,
                "train_start": str(train_prices.index[0]),
                "train_end": str(train_prices.index[-1]),
                "test_start": str(test_prices.index[0]),
                "test_end": str(test_prices.index[-1]),
                "best_params": best_params,
                "train_score": best_score,
                "test_metrics": test_metrics,
            })

        # Aggregate: choose params with best average train score
        avg_scores = {i: float(np.mean(s)) for i, s in param_scores.items()}
        best_param_idx = max(avg_scores, key=avg_scores.get)
        self.best_params_ = self.param_grid[best_param_idx]
        self.best_score_ = avg_scores[best_param_idx]
        return self

    def evaluate_best(self, prices: pd.DataFrame) -> BacktestResult:
        """Run a backtest with best_params_ on the full price series."""
        if self.best_params_ is None:
            raise ValueError("Optimizer has not been fitted yet")
        signal = self.signal_factory(prices, **self.best_params_)
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_bps=self.commission_bps,
            slippage_bps=self.slippage_bps,
        )
        return engine.run(prices, signal)


def combine_fold_results(fold_results: list[dict]) -> dict:
    """Aggregate walk-forward fold results into summary statistics."""
    if not fold_results:
        return {}
    test_sharpes = [f["test_metrics"].get("sharpe", np.nan) for f in fold_results]
    test_sortinos = [f["test_metrics"].get("sortino", np.nan) for f in fold_results]
    test_calmars = [f["test_metrics"].get("calmar", np.nan) for f in fold_results]
    test_mdds = [f["test_metrics"].get("max_drawdown", np.nan) for f in fold_results]
    test_returns = [f["test_metrics"].get("total_return", np.nan) for f in fold_results]

    return {
        "n_folds": len(fold_results),
        "avg_test_sharpe": float(np.nanmean(test_sharpes)),
        "avg_test_sortino": float(np.nanmean(test_sortinos)),
        "avg_test_calmar": float(np.nanmean(test_calmars)),
        "avg_test_max_dd": float(np.nanmean(test_mdds)),
        "avg_test_return": float(np.nanmean(test_returns)),
        "sharpe_std": float(np.nanstd(test_sharpes)),
        "positive_folds": int(sum(1 for s in test_sharpes if s > 0)),
        "best_fold": int(np.nanargmax(test_sharpes)),
        "worst_fold": int(np.nanargmin(test_sharpes)),
    }


def summary_table(fold_results: list[dict]) -> pd.DataFrame:
    """Create a summary DataFrame from fold results."""
    rows = []
    for f in fold_results:
        m = f["test_metrics"]
        rows.append({
            "fold": f["fold"],
            "train_start": f["train_start"],
            "train_end": f["train_end"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "best_params": str(f["best_params"]),
            "train_score": f["train_score"],
            "test_sharpe": m.get("sharpe", np.nan),
            "test_sortino": m.get("sortino", np.nan),
            "test_calmar": m.get("calmar", np.nan),
            "test_max_dd": m.get("max_drawdown", np.nan),
            "test_return": m.get("total_return", np.nan),
        })
    return pd.DataFrame(rows)
