from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import RiskConfig
from utils.logger import setup_logger

logger = setup_logger("RiskManager")


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.peak_equity: float = 0.0

    def volatility_target_scalar(self, portfolio_vol: float) -> float:
        if portfolio_vol <= 1e-6:
            return 1.0
        return float(np.clip(self.cfg.target_volatility / portfolio_vol, 0.1, 2.0))

    def kelly_size(self, win_prob: float, win_loss_ratio: float) -> float:
        if win_loss_ratio <= 0:
            return 0.0
        k = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        k = max(0.0, k) * self.cfg.kelly_fraction
        return float(np.clip(k, 0.0, self.cfg.max_single_position))

    def apply_position_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        capped = {}
        for k, v in weights.items():
            capped[k] = float(np.clip(v, -self.cfg.max_single_position,
                                      self.cfg.max_single_position))
        return capped

    def sector_neutralize(self, weights: Dict[str, float],
                          sectors: Dict[str, str],
                          max_sector: Optional[float] = None) -> Dict[str, float]:
        max_sector = max_sector or self.cfg.max_sector_exposure
        by_sector: Dict[str, List[str]] = {}
        for s in weights:
            by_sector.setdefault(sectors.get(s, "unknown"), []).append(s)
        out = dict(weights)
        for sec, syms in by_sector.items():
            total = sum(abs(out[s]) for s in syms)
            if total > max_sector:
                scale = max_sector / total
                for s in syms:
                    out[s] *= scale
        return out

    def correlation_check(self, candidate: str, weights: Dict[str, float],
                          returns: pd.DataFrame) -> bool:
        if candidate not in returns.columns:
            return True
        existing = [s for s in weights if s in returns.columns and weights[s] != 0]
        if not existing:
            return True
        corr = returns[existing].corrwith(returns[candidate]).abs()
        return bool(corr.max() < self.cfg.correlation_limit or corr.max() != corr.max())

    def compute_stops(self, price: float, atr: float, side: int) -> Tuple[float, float]:
        stop = price - side * self.cfg.stop_loss_atr_mult * atr
        target = price + side * self.cfg.take_profit_atr_mult * atr
        return stop, target

    def update_drawdown(self, equity: float) -> Tuple[float, bool]:
        self.peak_equity = max(self.peak_equity, equity)
        dd = (equity / self.peak_equity) - 1.0 if self.peak_equity > 0 else 0.0
        breach = dd < -self.cfg.max_drawdown_stop
        return dd, breach

    def portfolio_var(self, weights: np.ndarray, cov: np.ndarray,
                      confidence: float = 0.99) -> float:
        port_vol = float(np.sqrt(weights @ cov @ weights))
        from scipy.stats import norm
        z = norm.ppf(confidence)
        return z * port_vol

    def scale_for_var(self, weights: Dict[str, float], cov: np.ndarray,
                      symbols: List[str]) -> Dict[str, float]:
        w_vec = np.array([weights.get(s, 0.0) for s in symbols])
        var = self.portfolio_var(w_vec, cov)
        if var <= self.cfg.daily_var_limit or var <= 0:
            return weights
        scale = self.cfg.daily_var_limit / var
        logger.warning(f"VaR={var:.4f} > limit; scaling portfolio by {scale:.3f}")
        return {k: v * scale for k, v in weights.items()}
