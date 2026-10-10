from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import StrategyConfig
from utils.logger import setup_logger

logger = setup_logger("Scanner")


class MultiFactorScanner:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def rank(self, scored: pd.DataFrame) -> pd.DataFrame:
        if scored.empty:
            return scored
        latest = scored.index.get_level_values("date").max()
        today = scored.loc[scored.index.get_level_values("date") == latest].copy()
        today["rank"] = today["final_score"].rank(ascending=False)
        today = today.sort_values("final_score", ascending=False)
        logger.info(f"Scanned {len(today)} symbols on {latest}")
        return today

    def select_positions(self, ranked: pd.DataFrame) -> Dict[str, float]:
        if ranked.empty:
            return {}
        n = self.cfg.top_n_positions
        longs = ranked.head(n)
        shorts = ranked.tail(n) if self.cfg.long_short else ranked.iloc[0:0]

        longs = longs[longs["final_score"] > (self.cfg.signal_threshold_buy - 0.5)]
        shorts = shorts[shorts["final_score"] < (self.cfg.signal_threshold_sell - 0.5)]

        weights: Dict[str, float] = {}

        if not longs.empty:
            w = 1.0 / (longs.get("atr", pd.Series(1.0, index=longs.index)).fillna(0.02) + 1e-6)
            w = w / w.sum() * 0.5
            for sym, ww in w.items():
                weights[sym[1] if isinstance(sym, tuple) else sym] = float(ww)

        if not shorts.empty:
            w = 1.0 / (shorts.get("atr", pd.Series(1.0, index=shorts.index)).fillna(0.02) + 1e-6)
            w = w / w.sum() * 0.5
            for sym, ww in w.items():
                s = sym[1] if isinstance(sym, tuple) else sym
                weights[s] = weights.get(s, 0.0) - float(ww)

        weights = {k: float(np.clip(v, -self.cfg.max_position_size, self.cfg.max_position_size))
                   for k, v in weights.items()}
        return weights

    def filter_liquid(self, panel: pd.DataFrame, min_adv: float = 1e6) -> pd.DataFrame:
        if "volume" not in panel.columns or "close" not in panel.columns:
            return panel
        dvol = (panel["volume"] * panel["close"]).groupby(level="symbol").mean()
        good = dvol[dvol > min_adv].index
        mask = panel.index.get_level_values("symbol").isin(good)
        return panel[mask]
