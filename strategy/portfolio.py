from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import StrategyConfig, RiskConfig
from strategy.risk import RiskManager
from utils.logger import setup_logger

logger = setup_logger("Portfolio")


class PortfolioConstructor:
    def __init__(self, strat_cfg: StrategyConfig, risk_cfg: RiskConfig):
        self.s = strat_cfg
        self.r = risk_cfg
        self.rm = RiskManager(risk_cfg)

    def construct(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
        market_state: Optional[Dict] = None,
    ) -> Dict[str, float]:
        if signals.empty:
            return {}

        longs = signals[signals["adjusted_score"] > 0].head(self.s.top_n_positions)
        shorts = signals[signals["adjusted_score"] < 0].tail(self.s.top_n_positions)

        weights: Dict[str, float] = {}
        if not longs.empty:
            longs = longs.copy()
            longs["inv_atr"] = 1.0 / (longs.get("atr", pd.Series(0.02, index=longs.index))
                                     .fillna(0.02) + 1e-6)
            longs["w"] = longs["inv_atr"] / longs["inv_atr"].sum() * 0.5
            for idx, row in longs.iterrows():
                weights[self._sym(idx)] = float(row["w"])

        if not shorts.empty and self.s.long_short:
            shorts = shorts.copy()
            shorts["inv_atr"] = 1.0 / (shorts.get("atr", pd.Series(0.02, index=shorts.index))
                                      .fillna(0.02) + 1e-6)
            shorts["w"] = shorts["inv_atr"] / shorts["inv_atr"].sum() * 0.5
            for idx, row in shorts.iterrows():
                weights[self._sym(idx)] = weights.get(self._sym(idx), 0.0) - float(row["w"])

        weights = self.rm.apply_position_caps(weights)

        if not returns.empty:
            common = [s for s in weights if s in returns.columns]
            if common:
                w_vec = np.array([weights[s] for s in common])
                cov = returns[common].cov().values * 252
                pv = float(np.sqrt(w_vec @ cov @ w_vec))
                scalar = self.rm.volatility_target_scalar(pv)
                weights = {k: v * scalar for k, v in weights.items()}

        if not returns.empty:
            symbols = [s for s in weights if s in returns.columns]
            if symbols:
                cov = returns[symbols].cov().values * 252
                weights = self.rm.scale_for_var(weights, cov, symbols)

        gross = sum(abs(v) for v in weights.values())
        if gross > self.r.max_portfolio_leverage:
            scale = self.r.max_portfolio_leverage / gross
            weights = {k: v * scale for k, v in weights.items()}

        if market_state and self.s.regime_adaptive:
            mult = market_state.get("risk_multiplier", 1.0)
            weights = {k: v * mult for k, v in weights.items()}

        return weights

    @staticmethod
    def _sym(idx) -> str:
        return idx[1] if isinstance(idx, tuple) else str(idx)
