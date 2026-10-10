from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import SystemConfig
from models.ensemble import EnsemblePredictor
from models.ai_analyst import AIAnalyst
from models.xgboost_model import WalkForwardXGB
from scanner.screener import MultiFactorScanner
from scanner.patterns import pattern_score
from strategy.signals import SignalGenerator
from strategy.portfolio import PortfolioConstructor
from strategy.risk import RiskManager
from utils.logger import setup_logger

logger = setup_logger("Backtester")


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg
        self.rm = RiskManager(cfg.risk)

    def run(self, panel: pd.DataFrame, price_data: Dict[str, pd.DataFrame],
            benchmark: pd.DataFrame) -> BacktestResult:
        dates = sorted(panel.index.get_level_values("date").unique())
        dser = pd.Series(dates)
        weekly = dser.groupby(dser.dt.isocalendar().week).last().values

        equity = self.cfg.backtest.initial_capital
        positions: Dict[str, float] = {}
        entry_prices: Dict[str, float] = {}
        equity_curve = {}
        position_history: Dict[pd.Timestamp, Dict[str, float]] = {}
        trades: List[Dict] = []

        rets_wide = self._build_returns_matrix(price_data)

        prev_equity = equity
        for i, dt in enumerate(weekly):
            if i < 20:
                continue
            try:
                snapshot = self._slice(panel, dt)
                if snapshot.empty:
                    continue

                wf = WalkForwardXGB(
                    self.cfg.model,
                    train_window=self.cfg.backtest.train_window,
                    embargo=self.cfg.backtest.embargo_days,
                    horizon=5,
                )
                wf.fit(panel, as_of=pd.Timestamp(dt))
                probs = wf.predict(snapshot).reindex(snapshot.index).fillna(0.5)

                scored = snapshot.copy()
                scored["p_ensemble"] = probs.values
                scored["final_score"] = (scored["p_ensemble"] - 0.5) * 2

                bench_hist = benchmark.loc[:dt, "close"]
                regime = self._regime(bench_hist)
                market_state = {"regime": regime,
                                "risk_multiplier": {"trending_up": 1.1, "trending_down": 0.9,
                                                    "high_vol": 0.7, "crisis": 0.4,
                                                    "low_vol": 1.05, "ranging": 0.9}
                                .get(regime, 1.0)}

                sg = SignalGenerator(self.cfg.strategy)
                signals = sg.generate(scored, market_state)
                pc = PortfolioConstructor(self.cfg.strategy, self.cfg.risk)
                new_weights = pc.construct(signals, rets_wide.loc[:dt].tail(252), market_state)

                trades.extend(self._rebalance(positions, entry_prices, new_weights,
                                              dt, price_data, equity))
                positions = dict(new_weights)

            except Exception as e:
                logger.error(f"Rebalance {dt} failed: {e}")
                continue

            next_idx = i + 1
            next_dt = weekly[next_idx] if next_idx < len(weekly) else dates[-1]
            period_rets = self._period_return(positions, rets_wide, dt, next_dt,
                                              self.cfg.backtest.commission_bps,
                                              self.cfg.backtest.slippage_bps,
                                              first=i == 20)
            equity *= (1 + period_rets)
            equity_curve[pd.Timestamp(dt)] = equity
            position_history[pd.Timestamp(dt)] = dict(positions)

            dd, breach = self.rm.update_drawdown(equity)
            if breach:
                logger.warning(f"Drawdown limit breached ({dd:.2%}); de-risking")
                positions = {k: v * 0.25 for k, v in positions.items()}

            prev_equity = equity

        ec = pd.Series(equity_curve).sort_index()
        rets = ec.pct_change().fillna(0.0)
        pos_df = pd.DataFrame(position_history).T.fillna(0.0)

        metrics = self._compute_metrics(ec, rets)
        return BacktestResult(ec, rets, pos_df, trades, metrics)

    @staticmethod
    def _slice(panel: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        try:
            return panel.xs(dt, level="date")
        except KeyError:
            dates = panel.index.get_level_values("date").unique()
            prior = dates[dates <= dt]
            if len(prior) == 0:
                return pd.DataFrame()
            return panel.xs(prior[-1], level="date")

    @staticmethod
    def _build_returns_matrix(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = {s: d["close"].pct_change() for s, d in price_data.items()}
        return pd.DataFrame(frames).fillna(0.0)

    @staticmethod
    def _regime(bench: pd.Series) -> str:
        if len(bench) < 60:
            return "unknown"
        rets = bench.pct_change().dropna()
        vol = rets.tail(20).std() * np.sqrt(252)
        trend = bench.iloc[-1] / bench.iloc[-60] - 1
        if vol > 0.45:
            return "crisis"
        if vol > 0.28:
            return "high_vol"
        if vol < 0.10:
            return "low_vol"
        if trend > 0.05:
            return "trending_up"
        if trend < -0.05:
            return "trending_down"
        return "ranging"

    def _rebalance(self, old: Dict[str, float], entries: Dict[str, float],
                   new: Dict[str, float], dt, price_data, equity) -> List[Dict]:
        trades = []
        all_syms = set(old) | set(new)
        for s in all_syms:
            o = old.get(s, 0.0)
            n = new.get(s, 0.0)
            if abs(n - o) < 1e-4:
                continue
            trades.append({
                "date": dt, "symbol": s, "old_w": o, "new_w": n,
                "delta_w": n - o, "notional": (n - o) * equity,
            })
            if s in price_data and len(price_data[s]) > 0:
                entries[s] = price_data[s]["close"].asof(dt) if hasattr(price_data[s]["close"], "asof") else np.nan
        return trades

    def _period_return(self, positions: Dict[str, float], rets: pd.DataFrame,
                       start, end, commission_bps: float, slippage_bps: float,
                       first: bool = False) -> float:
        if rets.empty or not positions:
            return 0.0
        window = rets.loc[(rets.index > start) & (rets.index <= end)]
        if window.empty:
            return 0.0
        common = [s for s in positions if s in window.columns]
        if not common:
            return 0.0
        w = np.array([positions[s] for s in common])
        R = window[common].values
        daily = R @ w
        gross = float(np.prod(1 + daily) - 1)
        if first:
            cost = sum(abs(w)) * (commission_bps + slippage_bps) / 1e4
            gross -= cost
        return gross

    @staticmethod
    def _compute_metrics(equity: pd.Series, rets: pd.Series) -> Dict:
        if len(rets) == 0 or equity.empty:
            return {}
        ann = 252
        cum = (1 + rets).cumprod()
        total_ret = cum.iloc[-1] - 1
        years = max(len(rets) / ann, 1e-6)
        cagr = (cum.iloc[-1]) ** (1 / years) - 1
        vol = rets.std() * np.sqrt(ann)
        sharpe = (rets.mean() * ann - 0.04) / (vol + 1e-9)
        downside = rets[rets < 0].std() * np.sqrt(ann)
        sortino = (rets.mean() * ann - 0.04) / (downside + 1e-9)
        dd = (equity / equity.cummax() - 1)
        max_dd = dd.min()
        calmar = cagr / (abs(max_dd) + 1e-9)
        win_rate = (rets > 0).mean()
        return {
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "vol": float(vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar": float(calmar),
            "win_rate": float(win_rate),
            "n_periods": int(len(rets)),
        }
