from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import StrategyConfig
from utils.logger import setup_logger

logger = setup_logger("Signals")


class SignalGenerator:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def generate(self, scored: pd.DataFrame,
                 market_state: Optional[Dict] = None) -> pd.DataFrame:
        df = scored.copy()
        risk_mult = 1.0
        if market_state:
            risk_mult = market_state.get("risk_multiplier", 1.0)

        df["adjusted_score"] = df["final_score"] * risk_mult

        buy_hi = self.cfg.signal_threshold_buy * 2 - 1
        buy_lo = self.cfg.signal_threshold_sell * 2 - 1
        df["signal"] = np.select(
            [
                df["adjusted_score"] >= buy_hi,
                df["adjusted_score"] >= buy_lo,
                df["adjusted_score"] <= -buy_hi,
                df["adjusted_score"] <= -buy_lo,
            ],
            ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
            default="NEUTRAL",
        )
        return df.sort_values("adjusted_score", ascending=False)

    @staticmethod
    def position_from_signal(signal: str) -> int:
        return {
            "STRONG_BUY": 2, "BUY": 1, "NEUTRAL": 0,
            "SELL": -1, "STRONG_SELL": -2,
        }.get(signal, 0)
