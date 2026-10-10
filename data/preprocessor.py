from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("Preprocessor")


class DataPreprocessor:
    def __init__(self, fill_method: str = "ffill", max_gap: int = 5):
        self.fill_method = fill_method
        self.max_gap = max_gap

    def process(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        cleaned = {}
        for sym, df in price_data.items():
            if df.empty or len(df) < 250:
                logger.warning(f"Skipping {sym}: insufficient history ({len(df)})")
                continue
            d = df.copy()
            if self.fill_method == "ffill":
                d = d.fillna(method="ffill", limit=self.max_gap)
            elif self.fill_method == "interpolate":
                d = d.interpolate(method="linear", limit=self.max_gap)
            d = d.dropna(subset=["close"])
            if not d.empty:
                cleaned[sym] = d
        return cleaned

    def align_dates(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
        aligned = {}
        for sym, df in price_data.items():
            aligned[sym] = df.reindex(all_dates)
        return aligned
