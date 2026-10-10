from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from crypto.engine import (
    CryptoPredictor, CryptoPredictorConfig, CryptoPredictionTarget,
    CryptoSignalFusion, REGIME_NAMES_CRYPTO, CryptoRegimeDetector,
    SyntheticCryptoMarket, CryptoDiagnostics, CryptoBacktest,
)
from utils.logger import setup_logger

logger = setup_logger("CryptoBridge")


class CryptoPredictorBridge:
    def __init__(self, predictor: Optional[CryptoPredictor] = None,
                 cfg: Optional[CryptoPredictorConfig] = None):
        self.cfg = cfg or CryptoPredictorConfig(
            horizons_hours=(1, 4, 12, 24, 72, 168),
            primary_horizon_hours=24,
            target_type=CryptoPredictionTarget.DIRECTION,
            min_history=500,
            n_bootstrap=15,
            xgb_n_estimators=300,
            xgb_max_depth=5,
            feature_selection_enabled=True,
            max_features=100,
            online_enabled=False,
            log_level="WARNING",
        )
        self.predictor = predictor or CryptoPredictor(self.cfg)
        self.fusion = CryptoSignalFusion()
        self._predictions: Optional[pd.DataFrame] = None
        self._fused: Optional[pd.DataFrame] = None

    def fit(self, price_data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None) -> "CryptoPredictorBridge":
        logger.info("CryptoPredictorBridge: fitting crypto predictor engine...")
        self.predictor.fit(price_data, benchmark)
        logger.info("CryptoPredictorBridge: fit complete")
        return self

    def predict(self, price_data: Dict[str, pd.DataFrame],
                benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        preds = self.predictor.predict(price_data, benchmark)
        self._predictions = preds
        if not preds.empty:
            self._fused = self.fusion.fuse(preds)
        return preds

    def get_final_scores(self) -> pd.DataFrame:
        if self._fused is None:
            raise RuntimeError("Call predict() first")
        out = self._fused.copy()
        out["final_score"] = (out["fused_prob"] - 0.5) * 2
        return out.sort_values("final_score", ascending=False)

    def get_signals_df(self) -> pd.DataFrame:
        scores = self.get_final_scores().reset_index(drop=True)
        scores["adjusted_score"] = scores["final_score"]
        buy_hi = 0.6 * 2 - 1
        buy_lo = 0.4 * 2 - 1
        scores["signal"] = np.select(
            [
                scores["adjusted_score"] >= buy_hi,
                scores["adjusted_score"] >= buy_lo,
                scores["adjusted_score"] <= -buy_hi,
                scores["adjusted_score"] <= -buy_lo,
            ],
            ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL"],
            default="NEUTRAL",
        )
        return scores

    def get_market_state(self, benchmark: Optional[pd.DataFrame] = None) -> Dict:
        regime = "unknown"
        risk_mult = 0.75
        breadth = 0.5
        if self._fused is not None and not self._fused.empty:
            breadth = float((self._fused["fused_prob"] > 0.5).mean())
        if benchmark is not None and not benchmark.empty:
            detector = CryptoRegimeDetector()
            reg = detector.classify(benchmark["close"])
            regime = REGIME_NAMES_CRYPTO.get(reg, "unknown")
            risk_mult = CryptoRegimeDetector.risk_multiplier(reg)
        return {
            "regime": regime,
            "risk_multiplier": risk_mult,
            "breadth": breadth,
            "source": "crypto_predictor_engine",
        }

    def explain(self, symbol: str,
                price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return self.predictor.explain(symbol, price_data)

    def metrics_report(self) -> pd.DataFrame:
        return self.predictor.metrics_report()

    def feature_importance(self, top_k: int = 30) -> pd.Series:
        return self.predictor.feature_importance_report(top_k=top_k)

    def save(self, path: str) -> None:
        self.predictor.save(path)

    @classmethod
    def load(cls, path: str) -> "CryptoPredictorBridge":
        predictor = CryptoPredictor.load(path)
        bridge = cls(predictor=predictor)
        return bridge

    def run_self_test(self) -> None:
        gen = SyntheticCryptoMarket(n_symbols=10, n_hours=5500, random_state=11)
        prices, bench = gen.generate()
        self.fit(prices, bench)
        preds = self.predict(prices, bench)
        print(f"\nSelf-test complete: {len(preds)} predictions")
        print(self.metrics_report().to_string(index=False))
