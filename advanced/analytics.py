from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings("ignore")


class AdvancedAnalytics:
    def __init__(self):
        pass

    @staticmethod
    def compute_shap_values(model, X: np.ndarray, feature_names: List[str],
                            n_samples: int = 100) -> pd.DataFrame:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sample_idx = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
            X_sample = X[sample_idx]
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": feature_names[:len(mean_abs_shap)],
                "shap_importance": mean_abs_shap
            }).sort_values("shap_importance", ascending=False)
            return shap_df
        except Exception:
            return pd.DataFrame({"feature": feature_names, "shap_importance": [0.0] * len(feature_names)})

    @staticmethod
    def cross_asset_correlation(price_data: Dict[str, pd.DataFrame],
                                lookback: int = 30) -> pd.DataFrame:
        returns = {}
        for sym, df in price_data.items():
            if not df.empty and "close" in df.columns:
                returns[sym] = df["close"].pct_change().tail(lookback)
        if not returns:
            return pd.DataFrame()
        rets_df = pd.DataFrame(returns)
        return rets_df.corr()

    @staticmethod
    def compute_risk_metrics(returns: pd.Series) -> Dict[str, float]:
        if returns.empty or len(returns) < 10:
            return {}
        rets = returns.dropna()
        var_95 = float(np.percentile(rets, 5))
        var_99 = float(np.percentile(rets, 1))
        cvar_95 = float(rets[rets <= var_95].mean())
        cvar_99 = float(rets[rets <= var_99].mean())
        downside = rets[rets < 0]
        downside_dev = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else 0.0
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
        sortino = float(rets.mean() / (downside_dev + 1e-9) * np.sqrt(252))
        equity = (1 + rets).cumprod()
        dd = equity / equity.cummax() - 1
        max_dd = float(dd.min())
        calmar = float(rets.mean() * 252 / (abs(max_dd) + 1e-9))
        skew = float(stats.skew(rets))
        kurt = float(stats.kurtosis(rets))
        return {
            "var_95": var_95, "var_99": var_99,
            "cvar_95": cvar_95, "cvar_99": cvar_99,
            "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_dd, "calmar": calmar,
            "skewness": skew, "kurtosis": kurt,
            "downside_deviation": downside_dev,
        }

    @staticmethod
    def monte_carlo_simulation(returns: pd.Series, n_simulations: int = 1000,
                               horizon: int = 252, confidence: float = 0.95) -> Dict:
        if returns.empty or len(returns) < 30:
            return {}
        rets = returns.dropna()
        mu = rets.mean()
        sigma = rets.std()
        simulations = np.random.normal(mu, sigma, (n_simulations, horizon))
        cumulative = np.cumprod(1 + simulations, axis=1) - 1
        final_returns = cumulative[:, -1]
        var = float(np.percentile(final_returns, (1 - confidence) * 100))
        cvar = float(final_returns[final_returns <= var].mean())
        prob_profit = float((final_returns > 0).mean())
        return {
            "mean_final_return": float(final_returns.mean()),
            "median_final_return": float(np.median(final_returns)),
            "std_final_return": float(final_returns.std()),
            "var": var, "cvar": cvar,
            "prob_profit": prob_profit,
            "percentile_5": float(np.percentile(final_returns, 5)),
            "percentile_95": float(np.percentile(final_returns, 95)),
            "simulations": cumulative,
        }

    @staticmethod
    def walk_forward_analysis(predictions: pd.DataFrame, actual_returns: pd.DataFrame,
                              train_size: int = 252, step: int = 21) -> pd.DataFrame:
        results = []
        n = len(predictions)
        for start in range(0, n - train_size, step):
            end = start + train_size
            if end >= n:
                break
            train_preds = predictions.iloc[start:end]
            test_preds = predictions.iloc[end:end + step]
            if test_preds.empty:
                continue
            train_auc = roc_auc_score(
                (actual_returns.iloc[start:end] > 0).astype(int),
                train_preds
            ) if len(actual_returns) > end else 0.5
            test_auc = roc_auc_score(
                (actual_returns.iloc[end:end + step] > 0).astype(int),
                test_preds
            ) if len(actual_returns) > end + step else 0.5
            results.append({
                "start_idx": start, "end_idx": end,
                "train_auc": train_auc, "test_auc": test_auc,
            })
        return pd.DataFrame(results)

    @staticmethod
    def regime_performance(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
        df = pd.concat([returns, regimes], axis=1).dropna()
        if df.empty:
            return pd.DataFrame()
        df.columns = ["return", "regime"]
        grouped = df.groupby("regime")["return"].agg(["mean", "std", "count"])
        grouped["sharpe"] = grouped["mean"] / (grouped["std"] + 1e-9) * np.sqrt(252)
        return grouped

    @staticmethod
    def feature_interaction_analysis(X: np.ndarray, feature_names: List[str],
                                     top_k: int = 20) -> pd.DataFrame:
        n_features = min(X.shape[1], len(feature_names))
        correlations = np.corrcoef(X[:, :n_features].T)
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append({
                    "feature_1": feature_names[i],
                    "feature_2": feature_names[j],
                    "correlation": abs(correlations[i, j]),
                })
        df = pd.DataFrame(interactions)
        if df.empty:
            return df
        return df.sort_values("correlation", ascending=False).head(top_k)

    @staticmethod
    def prediction_confidence_analysis(predictions: pd.DataFrame) -> Dict:
        if predictions.empty:
            return {}
        probs = predictions["probability_up"] if "probability_up" in predictions.columns else predictions.iloc[:, 0]
        conf = predictions.get("confidence", pd.Series(0.5, index=predictions.index))
        return {
            "mean_probability": float(probs.mean()),
            "std_probability": float(probs.std()),
            "mean_confidence": float(conf.mean()),
            "high_confidence_ratio": float((conf > 0.7).mean()),
            "prediction_entropy": float(-np.sum(probs * np.log(probs + 1e-9) + (1 - probs) * np.log(1 - probs + 1e-9)) / len(probs)),
            "probability_skew": float(stats.skew(probs)),
            "probability_kurtosis": float(stats.kurtosis(probs)),
        }

    @staticmethod
    def multi_timeframe_analysis(price_data: Dict[str, pd.DataFrame],
                                 timeframes: List[str] = ["1d", "1w", "1mo"]) -> Dict[str, pd.DataFrame]:
        results = {}
        for tf in timeframes:
            tf_data = {}
            for sym, df in price_data.items():
                if df.empty:
                    continue
                try:
                    resampled = df.resample(tf).agg({
                        "open": "first", "high": "max", "low": "min",
                        "close": "last", "volume": "sum"
                    }).dropna()
                    if not resampled.empty:
                        tf_data[sym] = resampled
                except Exception:
                    continue
            if tf_data:
                results[tf] = tf_data
        return results

    @staticmethod
    def ensemble_diversity_analysis(predictions: pd.DataFrame) -> Dict:
        if predictions.empty:
            return {}
        prob_cols = [c for c in predictions.columns if c.startswith("p_")]
        if len(prob_cols) < 2:
            return {"diversity": 0.0, "agreement": 1.0}
        probs_df = predictions[prob_cols]
        mean_prob = probs_df.mean(axis=1)
        diversity = float(probs_df.std(axis=1).mean())
        agreement = float((probs_df.std(axis=1) < 0.1).mean())
        return {
            "diversity": diversity,
            "agreement": agreement,
            "mean_pairwise_corr": float(probs_df.corr().mean().mean()),
            "n_models": len(prob_cols),
        }
