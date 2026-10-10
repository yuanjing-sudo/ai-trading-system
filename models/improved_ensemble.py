from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class ImprovedEnsemble:
    def __init__(self, horizon: int = 5, n_splits: int = 5):
        self.horizon = horizon
        self.n_splits = n_splits
        self.base_models: Dict[str, any] = {}
        self.meta_model = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.regime_models: Dict[str, Dict] = {}
        self.fitted = False

    def _build_base_models(self) -> Dict:
        return {
            "xgb": xgb.XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.015,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
                objective="binary:logistic", eval_metric="auc",
                tree_method="hist", n_jobs=-1, random_state=42,
            ),
            "extratrees": ExtraTreesClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=30,
                min_samples_split=50, max_features="sqrt", n_jobs=-1,
                random_state=42, class_weight="balanced_subsample",
            ),
            "histgbm": HistGradientBoostingClassifier(
                max_iter=400, max_depth=5, learning_rate=0.02,
                max_bins=128, min_samples_leaf=40, l2_regularization=2.0,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=30, random_state=42,
            ),
        }

    def _decay_weights(self, n: int, halflife: int = 126) -> np.ndarray:
        return np.exp(-np.arange(n)[::-1] * np.log(2) / halflife)

    def _triple_barrier_labels(self, close: pd.Series, horizon: int,
                               upper_mult: float = 2.0, lower_mult: float = 2.0) -> pd.Series:
        labels = pd.Series(0, index=close.index)
        for i in range(len(close) - horizon):
            entry = close.iloc[i]
            upper = entry * (1 + upper_mult * close.iloc[i:i + horizon].std())
            lower = entry * (1 - lower_mult * close.iloc[i:i + horizon].std())
            future = close.iloc[i + 1:i + horizon + 1]
            hit_upper = (future >= upper).any()
            hit_lower = (future <= lower).any()
            if hit_upper and not hit_lower:
                labels.iloc[i] = 1
            elif hit_lower and not hit_upper:
                labels.iloc[i] = -1
            elif hit_upper and hit_lower:
                first_up = future[future >= upper].index[0] if (future >= upper).any() else None
                first_low = future[future <= lower].index[0] if (future <= lower).any() else None
                if first_up and first_low:
                    labels.iloc[i] = 1 if first_up < first_low else -1
        return labels

    def _create_labels(self, panel: pd.DataFrame, horizon: int,
                       label_type: str = "direction") -> pd.Series:
        close_df = panel["close"].unstack("symbol")
        if label_type == "direction":
            fwd = close_df.shift(-horizon) / close_df - 1.0
            labels = (fwd > 0).astype(int).stack()
        elif label_type == "triple_barrier":
            labels = pd.Series(index=panel.index, dtype=float)
            for sym in close_df.columns:
                tb = self._triple_barrier_labels(close_df[sym].dropna(), horizon)
                for idx, val in tb.items():
                    if (idx, sym) in labels.index:
                        labels.loc[(idx, sym)] = val
            labels = labels.dropna().astype(int)
        elif label_type == "terciles":
            fwd = close_df.shift(-horizon) / close_df - 1.0
            labels = fwd.stack()
            labels = pd.qcut(labels, q=3, labels=[0, 1, 2]).astype(int)
        else:
            fwd = close_df.shift(-horizon) / close_df - 1.0
            labels = (fwd > 0).astype(int).stack()
        return labels.reindex(panel.index)

    def fit(self, panel: pd.DataFrame, feature_cols: List[str],
            horizon: int = 5, label_type: str = "direction") -> "ImprovedEnsemble":
        self.feature_names = feature_cols
        df = panel.dropna().copy()
        df = df[df[feature_cols].notna().sum(axis=1) >= int(0.6 * len(feature_cols))]

        labels = self._create_labels(df, horizon, label_type)
        df = df.loc[labels.index]
        labels = labels.dropna()
        df = df.loc[labels.index]

        if len(df) < 200:
            raise RuntimeError(f"Insufficient samples: {len(df)}")

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        y = labels.values.astype(int)
        if label_type == "triple_barrier":
            y = (y == 1).astype(int)

        weights = self._decay_weights(len(X))

        self.base_models = self._build_base_models()
        for name, model in self.base_models.items():
            try:
                model.fit(X, y, sample_weight=weights)
            except Exception:
                model.fit(X, y)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        meta_features = []
        meta_labels = []
        for tr_idx, va_idx in tscv.split(X):
            if len(np.unique(y[va_idx])) < 2:
                continue
            fold_preds = {}
            for name, model in self.base_models.items():
                fold_preds[name] = model.predict_proba(X[va_idx])[:, 1]
            meta_X = np.column_stack(list(fold_preds.values()))
            meta_features.append(meta_X)
            meta_labels.append(y[va_idx])

        if meta_features:
            X_meta = np.vstack(meta_features)
            y_meta = np.concatenate(meta_labels)
            self.meta_model = LogisticRegression(
                C=0.5, max_iter=2000, solver="lbfgs",
                random_state=42,
            )
            self.meta_model.fit(X_meta, y_meta)

            self.calibrator = CalibratedClassifierCV(
                self.meta_model, method="isotonic", cv=3
            )
            self.calibrator.fit(X_meta, y_meta)

        self.fitted = True
        return self

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        X = panel[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        preds = []
        for name, model in self.base_models.items():
            try:
                preds.append(model.predict_proba(X)[:, 1])
            except Exception:
                preds.append(np.full(X.shape[0], 0.5))
        X_meta = np.column_stack(preds)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X_meta)[:, 1]
        elif self.meta_model is not None:
            return self.meta_model.predict_proba(X_meta)[:, 1]
        return np.mean(preds, axis=0)

    def feature_importance(self) -> pd.Series:
        if not self.base_models:
            return pd.Series(dtype=float)
        importances = {}
        if "xgb" in self.base_models:
            importances["xgb"] = self.base_models["xgb"].feature_importances_
        if "extratrees" in self.base_models:
            importances["extratrees"] = self.base_models["extratrees"].feature_importances_
        if not importances:
            return pd.Series(dtype=float)
        avg = np.mean(list(importances.values()), axis=0)
        return pd.Series(avg, index=self.feature_names).sort_values(ascending=False)


class CrossSectionalEnsemble:
    def __init__(self, horizons: List[int] = [1, 5, 10, 20]):
        self.horizons = horizons
        self.ensembles: Dict[int, ImprovedEnsemble] = {}
        self.fitted = False

    def fit(self, panel: pd.DataFrame, feature_cols: List[str],
            label_type: str = "direction") -> "CrossSectionalEnsemble":
        for h in self.horizons:
            try:
                ens = ImprovedEnsemble(horizon=h)
                ens.fit(panel, feature_cols, horizon=h, label_type=label_type)
                self.ensembles[h] = ens
            except Exception:
                continue
        self.fitted = len(self.ensembles) > 0
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for h, ens in self.ensembles.items():
            results[h] = ens.predict_proba(panel)
        return pd.DataFrame(results, index=panel.index)


class RegimeSpecificEnsemble:
    def __init__(self, regime_col: str = "regime"):
        self.regime_col = regime_col
        self.regime_models: Dict[str, ImprovedEnsemble] = {}
        self.fallback_model: Optional[ImprovedEnsemble] = None
        self.fitted = False

    def fit(self, panel: pd.DataFrame, feature_cols: List[str],
            horizon: int = 5) -> "RegimeSpecificEnsemble":
        if self.regime_col not in panel.columns:
            self.fallback_model = ImprovedEnsemble(horizon=horizon)
            self.fallback_model.fit(panel, feature_cols, horizon=horizon)
            self.fitted = True
            return self

        for regime_val in panel[self.regime_col].dropna().unique():
            mask = panel[self.regime_col] == regime_val
            regime_panel = panel[mask]
            if len(regime_panel) < 100:
                continue
            try:
                ens = ImprovedEnsemble(horizon=horizon)
                ens.fit(regime_panel, feature_cols, horizon=horizon)
                self.regime_models[str(regime_val)] = ens
            except Exception:
                continue

        if not self.regime_models:
            self.fallback_model = ImprovedEnsemble(horizon=horizon)
            self.fallback_model.fit(panel, feature_cols, horizon=horizon)

        self.fitted = True
        return self

    def predict_proba(self, panel: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        if self.fallback_model is not None:
            return self.fallback_model.predict_proba(panel)
        if self.regime_col not in panel.columns or not self.regime_models:
            return np.full(len(panel), 0.5)
        predictions = np.full(len(panel), 0.5)
        for regime_val, model in self.regime_models.items():
            mask = panel[self.regime_col] == regime_val
            if mask.any():
                preds = model.predict_proba(panel[mask])
                predictions[mask.values] = preds
        return predictions
