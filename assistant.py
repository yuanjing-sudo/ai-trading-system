from __future__ import annotations
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("assistant")


class AIAssistant:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5-coder:7b"
        self.conversation_history: List[Dict[str, str]] = []

    async def ask(self, question: str, context: Optional[Dict] = None) -> str:
        self.conversation_history.append({"role": "user", "content": question})
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "system", "content": self._build_system_prompt(context)},
                                     *self.conversation_history[-6:]],
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result.get("message", {}).get("content", "")
                    if answer:
                        self.conversation_history.append({"role": "assistant", "content": answer})
                        return answer
        except Exception as e:
            logger.warning(f"Ollama assistant failed: {e}")
        fallback = self._fallback_response(question, context)
        self.conversation_history.append({"role": "assistant", "content": fallback})
        return fallback

    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        ctx = ""
        if context:
            ctx = f"\n\nCurrent Analysis Context:\n{json.dumps(context, indent=2, default=str)[:3000]}"

        return f"""You are an expert quantitative trading assistant embedded in an institutional-grade AI trading system.
You help traders understand predictions, risk, and market dynamics.
Always provide clear, actionable insights. Use data from the analysis context when available.
If asked about specific numbers, reference the actual prediction data provided.
Keep responses concise but informative. Format with markdown when helpful.{ctx}"""

    def _fallback_response(self, question: str, context: Optional[Dict]) -> str:
        q_lower = question.lower()
        if any(w in q_lower for w in ["buy", "sell", "trade", "position"]):
            if context and "signals" in context:
                signals = context.get("signals", [])
                if signals:
                    top = signals[0]
                    return f"Based on the current analysis, the top signal is **{top.get('signal', 'NEUTRAL')}** for {top.get('symbol', 'N/A')} with a probability of {(top.get('fused_prob', 0.5) * 100):.1f}%. The model confidence is {(top.get('fused_confidence', 0) * 100):.0f}%."
            return "Based on the model predictions, please review the signal dashboard for current recommendations. Consider position sizing based on the confidence level."
        if any(w in q_lower for w in ["risk", "drawdown", "var", "safe"]):
            if context and "risk_metrics" in context:
                rm = context["risk_metrics"]
                return f"Current risk metrics: Sharpe {rm.get('sharpe', 0):.2f}, Max Drawdown {(rm.get('max_drawdown', 0) * 100):.1f}%, VaR 95% {(rm.get('var_95', 0) * 100):.2f}%. Position size should be adjusted accordingly."
            return "Risk metrics are shown in the dashboard. Consider the Sharpe ratio, max drawdown, and VaR when sizing positions."
        if any(w in q_lower for w in ["feature", "why", "explain", "important"]):
            if context and "feature_importance" in context:
                top = context["feature_importance"][0] if context["feature_importance"] else None
                if top:
                    return f"The most important feature is **{top['feature']}** with importance {top['importance']:.4f}. This is the primary driver of the current prediction."
            return "Check the Feature Importance section for the top drivers of the current prediction."
        if any(w in q_lower for w in ["regime", "market", "volatil"]):
            if context and "market_state" in context:
                ms = context["market_state"]
                return f"Current market regime: **{ms.get('regime', 'unknown')}** with risk multiplier {ms.get('risk_multiplier', 1.0):.2f}. Adjust exposure based on this regime."
            return "Market regime information is shown in the top bar. It affects the risk multiplier applied to predictions."
        if any(w in q_lower for w in ["null", "none", "nothing", "unknown"]):
            return "I couldn't find specific data for that query. Try asking about predictions, risk metrics, feature importance, or the current market regime."
        return "I can help you understand the analysis. Ask me about predictions, risk metrics, feature importance, market regime, or trading recommendations."

    def reset_conversation(self):
        self.conversation_history.clear()


class AlgorithmAnalyzer:
    def __init__(self):
        pass

    def generate_full_analysis(self, data: Dict) -> Dict:
        predictions = data.get("predictions", [])
        signals = data.get("signals", [])
        metrics = data.get("metrics", [])
        feature_importance = data.get("feature_importance", [])
        market_state = data.get("market_state", {})
        asset_specific = data.get("asset_specific", {})

        primary = self._find_primary_prediction(predictions)
        signal = signals[0] if signals else {}
        primary_metrics = self._find_primary_metrics(metrics)

        return {
            "summary": self._generate_summary(primary, signal, market_state),
            "signal_breakdown": self._analyze_signal(signal, primary),
            "model_performance": self._analyze_model_performance(primary_metrics),
            "feature_drivers": self._analyze_feature_drivers(feature_importance),
            "risk_assessment": self._assess_risk(primary, signal, market_state),
            "actionable_recommendations": self._generate_recommendations(primary, signal, market_state, primary_metrics),
            "market_context": self._analyze_market_context(market_state),
            "asset_specific_insights": self._analyze_asset_specific(asset_specific),
            "confidence_explanation": self._explain_confidence(primary, primary_metrics),
        }

    def _find_primary_prediction(self, predictions: List[Dict]) -> Dict:
        if not predictions:
            return {}
        for p in predictions:
            if p.get("horizon") == 5 or p.get("horizon_hours") == 24:
                return p
        return predictions[0]

    def _find_primary_metrics(self, metrics: List[Dict]) -> Dict:
        if not metrics:
            return {}
        for m in metrics:
            if m.get("horizon") == 5 or m.get("horizon_hours") == 24:
                return m
        return metrics[0]

    def _generate_summary(self, primary: Dict, signal: Dict, market_state: Dict) -> str:
        sym = primary.get("symbol", "N/A")
        prob = primary.get("probability_up", 0.5)
        direction = "bullish" if prob > 0.5 else "bearish"
        confidence = primary.get("confidence", 0.5)
        regime = market_state.get("regime", "unknown")
        exp_ret = primary.get("expected_return", 0)
        return (
            f"The AI model is **{direction}** on {sym} with a "
            f"**{(prob * 100):.1f}%** probability of upward movement. "
            f"Confidence level is **{(confidence * 100):.0f}%**. "
            f"The current market regime is **{regime}**. "
            f"Expected return over the primary horizon is "
            f"**{(exp_ret * 100):.2f}%**."
        )

    def _analyze_signal(self, signal: Dict, primary: Dict) -> Dict:
        sig = signal.get("signal", "NEUTRAL")
        score = signal.get("final_score", 0)
        return {
            "signal": sig,
            "score": score,
            "interpretation": (
                "Strong conviction" if abs(score) > 0.3
                else "Moderate conviction" if abs(score) > 0.15
                else "Weak signal / neutral"
            ),
            "threshold_analysis": {
                "buy_threshold": 0.2,
                "sell_threshold": -0.2,
                "current_score": score,
                "distance_to_threshold": abs(abs(score) - 0.2),
            },
        }

    def _analyze_model_performance(self, metrics: Dict) -> Dict:
        auc = metrics.get("auc", 0)
        return {
            "auc": auc,
            "auc_quality": (
                "Excellent" if auc > 0.7
                else "Good" if auc > 0.6
                else "Fair" if auc > 0.55
                else "Poor — model has limited predictive power"
            ),
            "accuracy": metrics.get("accuracy", 0),
            "brier_score": metrics.get("brier", 0),
            "brier_quality": (
                "Well calibrated" if metrics.get("brier", 1) < 0.2
                else "Moderately calibrated" if metrics.get("brier", 1) < 0.25
                else "Poorly calibrated — use probabilities with caution"
            ),
            "hit_rate": metrics.get("hit_rate", 0),
        }

    def _analyze_feature_drivers(self, feature_importance: List[Dict]) -> Dict:
        if not feature_importance:
            return {"top_driver": "N/A", "concentration": 0}
        top = feature_importance[0]
        top5_sum = sum(abs(f.get("importance", 0)) for f in feature_importance[:5])
        total_sum = sum(abs(f.get("importance", 0)) for f in feature_importance) or 1
        return {
            "top_driver": top.get("feature", "N/A"),
            "top_driver_importance": top.get("importance", 0),
            "top_5_features": [f.get("feature") for f in feature_importance[:5]],
            "feature_concentration": top5_sum / total_sum,
            "diversification": (
                "High — many features contribute" if top5_sum / total_sum < 0.5
                else "Concentrated — few features dominate"
            ),
        }

    def _assess_risk(self, primary: Dict, signal: Dict, market_state: Dict) -> Dict:
        conf = primary.get("confidence", 0.5)
        regime_mult = market_state.get("risk_multiplier", 1.0)
        return {
            "confidence_level": conf,
            "regime_multiplier": regime_mult,
            "risk_adjusted_confidence": conf * regime_mult,
            "position_size_recommendation": (
                "Full size" if conf * regime_mult > 0.8
                else "Half size" if conf * regime_mult > 0.5
                else "Quarter size / avoid" if conf * regime_mult > 0.3
                else "No trade — risk too high"
            ),
            "regime_warning": (
                "CRASH regime — extreme caution" if regime_mult < 0.3
                else "High volatility — reduce size" if regime_mult < 0.7
                else "Favorable regime — normal sizing" if regime_mult > 1.1
                else "Normal risk environment"
            ),
        }

    def _generate_recommendations(self, primary: Dict, signal: Dict,
                                  market_state: Dict, metrics: Dict) -> List[str]:
        recs = []
        prob = primary.get("probability_up", 0.5)
        conf = primary.get("confidence", 0.5)
        regime = market_state.get("regime", "unknown")
        auc = metrics.get("auc", 0.5)

        if prob > 0.6 and conf > 0.6:
            recs.append("Consider a long position with standard risk management.")
        elif prob < 0.4 and conf > 0.6:
            recs.append("Consider a short position or hedge existing longs.")
        elif 0.4 <= prob <= 0.6:
            recs.append("Model is uncertain — wait for a clearer signal before acting.")

        if auc < 0.55:
            recs.append("Model AUC is marginal — reduce position size until performance improves.")

        if regime in ["crisis", "crash"]:
            recs.append("CRASH regime detected — minimize exposure and use tight stops.")
        elif regime in ["high_vol", "high_volatility"]:
            recs.append("High volatility regime — widen stops and reduce size.")

        if conf < 0.4:
            recs.append("Low confidence — consider this a watchlist item, not an active trade.")

        recs.append("Always use stop-losses based on ATR (2.5x ATR recommended).")
        recs.append("Re-evaluate the position at each model refresh cycle.")

        return recs

    def _analyze_market_context(self, market_state: Dict) -> Dict:
        regime = market_state.get("regime", "unknown")
        mult = market_state.get("risk_multiplier", 1.0)
        breadth = market_state.get("breadth", 0.5)
        return {
            "regime": regime,
            "breadth": breadth,
            "risk_multiplier": mult,
            "market_condition": (
                "Strong uptrend" if regime in ["trending_up", "bull", "extreme_bull", "melt_up"]
                else "Downtrend" if regime in ["trending_down", "bear", "extreme_bear"]
                else "Crisis / crash" if regime in ["crisis", "crash"]
                else "Ranging / choppy" if regime in ["ranging", "chop"]
                else "Normal"
            ),
        }

    def _analyze_asset_specific(self, asset_specific: Dict) -> Dict:
        if not asset_specific:
            return {}
        return asset_specific

    def _explain_confidence(self, primary: Dict, metrics: Dict) -> Dict:
        conf = primary.get("confidence", 0.5)
        unc = primary.get("uncertainty", 0.3)
        auc = metrics.get("auc", 0.5)
        return {
            "raw_confidence": conf,
            "uncertainty": unc,
            "model_auc": auc,
            "explanation": (
                f"The model's confidence of {(conf * 100):.0f}% is derived from "
                f"the prediction probability distance from 0.5, adjusted for "
                f"bootstrap uncertainty ({(unc * 100):.1f}%). "
                f"The ensemble AUC of {auc:.3f} indicates "
                f"{'strong' if auc > 0.65 else 'moderate' if auc > 0.55 else 'weak'} "
                f"discriminative power."
            ),
        }
