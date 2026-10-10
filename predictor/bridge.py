from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from predictor.engine import (
    Predictor, PredictorConfig, PredictionTarget, SignalFusion,
    SyntheticMarketGenerator, Diagnostics, SimpleBacktest,
    WalkForwardTrainer, REGIME_NAMES,
)
from utils.logger import setup_logger

logger = setup_logger("PredictorBridge")


class PredictorBridge:
    def __init__(self, predictor: Optional[Predictor] = None,
                 cfg: Optional[PredictorConfig] = None):
        self.cfg = cfg or PredictorConfig(
            horizons=(1, 5, 10, 20),
            primary_horizon=5,
            target_type=PredictionTarget.DIRECTION,
            min_history=260,
            n_bootstrap=20,
            xgb_n_estimators=500,
            feature_selection_enabled=True,
            max_features=100,
            log_level="INFO",
        )
        self.predictor = predictor or Predictor(self.cfg)
        self.fusion = SignalFusion()
        self._predictions: Optional[pd.DataFrame] = None
        self._fused: Optional[pd.DataFrame] = None

    def fit(self, price_data: Dict[str, pd.DataFrame],
            benchmark: Optional[pd.DataFrame] = None) -> "PredictorBridge":
        logger.info("PredictorBridge: fitting predictor engine...")
        self.predictor.fit(price_data, benchmark)
        logger.info("PredictorBridge: fit complete")
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
        risk_mult = 0.85
        breadth = 0.5
        if self._fused is not None and not self._fused.empty:
            breadth = float((self._fused["fused_prob"] > 0.5).mean())
        if benchmark is not None and not benchmark.empty:
            from predictor.engine import RegimeDetector
            detector = RegimeDetector()
            reg = detector.classify_single(benchmark["close"])
            regime = REGIME_NAMES.get(reg, "unknown")
            risk_mult = RegimeDetector.regime_risk_multiplier(reg)
        return {
            "regime": regime,
            "risk_multiplier": risk_mult,
            "breadth": breadth,
            "source": "predictor_engine",
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
    def load(cls, path: str) -> "PredictorBridge":
        predictor = Predictor.load(path)
        bridge = cls(predictor=predictor)
        return bridge

    def run_walk_forward(self, price_data: Dict[str, pd.DataFrame],
                         benchmark: Optional[pd.DataFrame] = None,
                         start_after: int = 500) -> pd.DataFrame:
        wf = WalkForwardTrainer(self.cfg, refit_frequency_days=20)
        return wf.run(price_data, benchmark, start_after=start_after)

    def run_self_test(self) -> None:
        gen = SyntheticMarketGenerator(n_symbols=15, n_days=900, random_state=7)
        prices, bench = gen.generate()
        self.fit(prices, bench)
        preds = self.predict(prices, bench)
        print(f"\nSelf-test complete: {len(preds)} predictions")
        print(self.metrics_report().to_string(index=False))
