from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from config import ModelConfig
from models.xgboost_model import XGBoostPredictor
from models.ai_analyst import AIAnalyst
from utils.logger import setup_logger

logger = setup_logger("Ensemble")


class EnsemblePredictor:
    def __init__(self, cfg: ModelConfig, ai_analyst: Optional[AIAnalyst] = None):
        self.cfg = cfg
        self.xgb = XGBoostPredictor(cfg)
        self.rf: Optional[RandomForestClassifier] = None
        self.gb: Optional[GradientBoostingClassifier] = None
        self.meta: Optional[LogisticRegression] = None
        self.ai = ai_analyst
        self.feature_names: List[str] = []

    def train(self, panel: pd.DataFrame, horizon: int = 5):
        df = panel.dropna(subset=[f"target_{horizon}"]).copy()
        self.feature_names = self.xgb._auto_features(df)
        X = df[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = df[f"target_{horizon}"].astype(int)

        logger.info("Training XGBoost...")
        self.xgb.train(panel, horizon=horizon)

        logger.info("Training Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=20,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )
        self.rf.fit(X, y)

        logger.info("Training Gradient Boosting...")
        self.gb = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, random_state=42,
        )
        self.gb.fit(X, y)

        logger.info("Training meta-model (stacking)...")
        p_xgb = self.xgb.predict(panel, horizon=horizon).reindex(df.index).fillna(0.5).values
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_gb = self.gb.predict_proba(X)[:, 1]
        X_meta = np.column_stack([p_xgb, p_rf, p_gb])
        self.meta = LogisticRegression(C=1.0, max_iter=1000)
        self.meta.fit(X_meta, y)
        logger.info("Ensemble trained")

    def predict_proba(self, panel: pd.DataFrame, horizon: int = 5) -> pd.Series:
        X = panel[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        p_xgb = self.xgb.predict(panel, horizon=horizon).fillna(0.5).values
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_gb = self.gb.predict_proba(X)[:, 1]
        X_meta = np.column_stack([p_xgb, p_rf, p_gb])
        if self.meta is not None:
            probs = self.meta.predict_proba(X_meta)[:, 1]
        else:
            w = self.cfg.ensemble_weights
            probs = (w["xgboost"] * p_xgb + w["random_forest"] * p_rf
                     + w["gradient_boosting"] * p_gb)
        return pd.Series(probs, index=panel.index, name="ensemble_prob")

    def predict_full(self, panel: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        X = panel[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = pd.DataFrame(index=panel.index)
        out["p_xgb"] = self.xgb.predict(panel, horizon=horizon).fillna(0.5)
        out["p_rf"] = self.rf.predict_proba(X)[:, 1]
        out["p_gb"] = self.gb.predict_proba(X)[:, 1]
        out["p_ensemble"] = self.predict_proba(panel, horizon=horizon)

        if self.ai is not None:
            ai_scores, ai_conf = [], []
            for idx, row in panel.iterrows():
                try:
                    r = self.ai.analyze_symbol(
                        symbol=idx[1] if isinstance(idx, tuple) else "unknown",
                        features=row,
                        ml_prob=float(out.loc[idx, "p_ensemble"]),
                    )
                    ai_scores.append(r["composite_score"])
                    ai_conf.append(r["confidence"])
                except Exception:
                    ai_scores.append(0.0)
                    ai_conf.append(0.0)
            out["ai_composite"] = ai_scores
            out["ai_confidence"] = ai_conf
            w = self.cfg.ensemble_weights
            out["final_score"] = (
                w["xgboost"] * (out["p_ensemble"] - 0.5) * 2
                + w["ai_analysis"] * out["ai_composite"]
            )
        else:
            out["final_score"] = (out["p_ensemble"] - 0.5) * 2
        return out
