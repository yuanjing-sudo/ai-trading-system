from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, log_loss

from config import ModelConfig
from utils.logger import setup_logger

logger = setup_logger("XGBoostModel")


@dataclass
class ModelMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    logloss: float
    n_train: int
    n_valid: int


class XGBoostPredictor:
    def __init__(self, cfg: ModelConfig, feature_names: Optional[List[str]] = None):
        self.cfg = cfg
        self.feature_names = feature_names
        self.models: Dict[int, xgb.XGBClassifier] = {}
        self.metrics: Dict[int, ModelMetrics] = {}

    def _build_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.cfg.xgb_n_estimators,
            max_depth=self.cfg.xgb_max_depth,
            learning_rate=self.cfg.xgb_learning_rate,
            subsample=self.cfg.xgb_subsample,
            colsample_bytree=self.cfg.xgb_colsample_bytree,
            min_child_weight=self.cfg.xgb_min_child_weight,
            gamma=self.cfg.xgb_gamma,
            reg_alpha=self.cfg.xgb_reg_alpha,
            reg_lambda=self.cfg.xgb_reg_lambda,
            objective=self.cfg.xgb_objective,
            eval_metric=self.cfg.xgb_eval_metric,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=self.cfg.xgb_early_stopping,
        )

    def train(self, panel: pd.DataFrame, horizon: int = 5,
              feature_cols: Optional[List[str]] = None) -> ModelMetrics:
        target_col = f"target_{horizon}"
        df = panel.dropna(subset=[target_col]).copy()
        feature_cols = feature_cols or self._auto_features(df)
        self.feature_names = feature_cols

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df[target_col].astype(int)
        keep = X.notna().sum(axis=1) >= len(feature_cols) * 0.8
        X, y = X[keep].fillna(0.0), y[keep]

        if len(X) < 500:
            raise RuntimeError(f"Insufficient samples: {len(X)}")

        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, valid_idx = list(tscv.split(X))[-1]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

        model = self._build_model()
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        preds = model.predict_proba(X_va)[:, 1]
        labels = (preds > 0.5).astype(int)
        m = ModelMetrics(
            auc=roc_auc_score(y_va, preds),
            accuracy=accuracy_score(y_va, labels),
            precision=precision_score(y_va, labels, zero_division=0),
            recall=recall_score(y_va, labels, zero_division=0),
            logloss=log_loss(y_va, preds),
            n_train=len(X_tr), n_valid=len(X_va),
        )
        self.models[horizon] = model
        self.metrics[horizon] = m
        logger.info(
            f"[h={horizon}] AUC={m.auc:.4f} | Acc={m.accuracy:.4f} | "
            f"P={m.precision:.4f} | R={m.recall:.4f} | n_train={m.n_train}"
        )
        return m

    def train_all_horizons(self, panel: pd.DataFrame) -> Dict[int, ModelMetrics]:
        out = {}
        for h in self.cfg.prediction_horizons:
            try:
                out[h] = self.train(panel, horizon=h)
            except Exception as e:
                logger.error(f"Training failed h={h}: {e}")
        return out

    def predict(self, panel: pd.DataFrame, horizon: int = 5) -> pd.Series:
        if horizon not in self.models:
            raise KeyError(f"No model for horizon {horizon}")
        model = self.models[horizon]
        X = panel[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        probs = model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=panel.index, name=f"prob_up_{horizon}")

    def predict_all(self, panel: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=panel.index)
        for h in self.models:
            out[f"prob_up_{h}"] = self.predict(panel, h)
        if len(self.models) > 0:
            out["prob_composite"] = out[[c for c in out.columns if c.startswith("prob_up_")]].mean(axis=1)
        return out

    def feature_importance(self, horizon: int = 5) -> pd.Series:
        if horizon not in self.models:
            return pd.Series(dtype=float)
        m = self.models[horizon]
        imp = pd.Series(m.feature_importances_, index=self.feature_names)
        return imp.sort_values(ascending=False)

    @staticmethod
    def _auto_features(df: pd.DataFrame) -> List[str]:
        drop = {"symbol", "close", "volume", "fwd_ret_1", "fwd_ret_5",
                "fwd_ret_10", "fwd_ret_20", "target_1", "target_5",
                "target_10", "target_20"}
        return [c for c in df.columns
                if c not in drop and df[c].dtype.kind in "fi"]


class WalkForwardXGB:
    def __init__(self, cfg: ModelConfig, train_window: int = 504,
                 embargo: int = 5, horizon: int = 5):
        self.cfg = cfg
        self.train_window = train_window
        self.embargo = embargo
        self.horizon = horizon
        self.model: Optional[XGBoostPredictor] = None
        self.last_train_date: Optional[pd.Timestamp] = None

    def fit(self, panel: pd.DataFrame, as_of: pd.Timestamp):
        cutoff = as_of - pd.Timedelta(days=self.embargo)
        hist = panel.loc[panel.index.get_level_values("date") < cutoff]
        if len(hist) < self.train_window:
            raise RuntimeError(f"Insufficient training history: {len(hist)}")
        hist = (hist.reset_index()
                    .sort_values("date")
                    .groupby("symbol").tail(self.train_window)
                    .set_index(["date", "symbol"]))
        m = XGBoostPredictor(self.cfg)
        m.train(hist, horizon=self.horizon)
        self.model = m
        self.last_train_date = as_of
        return m

    def predict(self, panel_slice: pd.DataFrame) -> pd.Series:
        assert self.model is not None
        return self.model.predict(panel_slice, horizon=self.horizon)
