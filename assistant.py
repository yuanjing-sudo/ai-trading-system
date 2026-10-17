from __future__ import annotations
import json
import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger("assistant")

_RESEARCH_WORDS = (
    "deep dive", "deep research", "deep analysis", "full analysis", "research",
    "multi-agent", "multi agent", "agent team", "analyst team", "research note",
    "committee", "second opinion", "full report",
)
_SYMBOL_STOPWORDS = {
    "I", "A", "AI", "OK", "PM", "CEO", "FED", "GDP", "CPI", "USD", "ETF",
    "RSI", "MACD", "ATR", "API", "UI", "LLM", "VAR", "EOD",
}


class AIAssistant:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen2.5-coder:7b"
        self.conversation_history: List[Dict[str, str]] = []
        self._ollama_up: Optional[bool] = None
        self._ollama_checked = 0.0

    async def ask(self, question: str, context: Optional[Dict] = None) -> str:
        context = context or {}
        self.conversation_history.append({"role": "user", "content": question})

        answer = None
        if self._wants_research(question):
            symbol = self._resolve_symbol(question, context)
            if symbol:
                answer = self._run_research(symbol, context)
        if answer is None:
            answer = await self._ask_llm(question, context)
        if answer is None:
            answer = self._fallback_response(question, context)

        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    async def _ask_llm(self, question: str, context: Dict) -> Optional[str]:
        if not self._ollama_available():
            return None
        try:
            import httpx
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": self._build_system_prompt(context)},
                             *self.conversation_history[-6:]],
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code == 200:
                answer = resp.json().get("message", {}).get("content", "").strip()
                if answer:
                    return answer
            logger.warning(f"Ollama assistant returned HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Ollama assistant failed: {e}")
            self._ollama_up = None
        return None

    def _ollama_available(self) -> bool:
        now = time.time()
        if self._ollama_up is not None and now - self._ollama_checked < 30:
            return self._ollama_up
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            self._ollama_up = resp.status_code == 200
        except Exception:
            self._ollama_up = False
        self._ollama_checked = now
        return self._ollama_up

    def _build_system_prompt(self, context: Dict) -> str:
        compressed = self._compress_context(context)
        ctx = ""
        if compressed:
            ctx = (f"\n\nCurrent Analysis Context (price_stats were computed locally "
                   f"from the latest bars - trust these numbers):\n"
                   f"{json.dumps(compressed, indent=2, default=str)[:3500]}")

        return f"""You are an expert quantitative trading assistant embedded in an institutional-grade AI trading system.
You help traders understand predictions, risk, and market dynamics.
Always provide clear, actionable insights. Use data from the analysis context when available.
If asked about specific numbers, reference the actual prediction data provided.
Keep responses concise but informative. Format with markdown when helpful.{ctx}"""

    @staticmethod
    def _compress_context(context: Dict) -> Dict:
        compressed = {k: v for k, v in context.items() if k != "price_history"}
        closes = [c for c in ((context.get("price_history") or {}).get("close") or [])
                  if isinstance(c, (int, float))]
        if len(closes) >= 30:
            try:
                import numpy as np
                c = np.asarray(closes, dtype=float)
                ma20 = float(c[-20:].mean())
                ret = c[1:] / c[:-1] - 1.0
                compressed["price_stats"] = {
                    "last_close": round(float(c[-1]), 4),
                    "change_20d_pct": round(float(c[-1] / c[-21] - 1) * 100, 2) if len(c) >= 21 else None,
                    "ma20": round(ma20, 4),
                    "above_ma20": bool(c[-1] > ma20),
                    "ann_vol_20d_pct": round(float(ret[-20:].std() * (252 ** 0.5)) * 100, 2),
                    "period_high": round(float(c.max()), 4),
                    "period_low": round(float(c.min()), 4),
                    "rsi14": AIAssistant._rsi(c, 14),
                }
            except Exception as e:
                logger.debug(f"Price stats failed: {e}")
        return compressed

    @staticmethod
    def _rsi(closes, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0
        try:
            import numpy as np
            diff = np.diff(np.asarray(closes[-(period + 1):], dtype=float))
            avg_gain = float(np.clip(diff, 0, None).sum()) / period
            avg_loss = float(np.clip(-diff, 0, None).sum()) / period
        except Exception:
            return 50.0
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return round(100.0 - 100.0 / (1.0 + rs), 1)

    @staticmethod
    def _wants_research(question: str) -> bool:
        q = question.lower()
        return any(w in q for w in _RESEARCH_WORDS)

    @staticmethod
    def _resolve_symbol(question: str, context: Dict) -> Optional[str]:
        symbol = context.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip().upper()
        for sig in (context.get("signals") or []):
            if isinstance(sig, dict) and sig.get("symbol"):
                return str(sig["symbol"]).upper()
        for token in re.findall(r"\b[A-Z][A-Z0-9]{0,6}(?:[-/][A-Z]{2,5})?\b", question):
            if token not in _SYMBOL_STOPWORDS:
                return token
        return None

    def _run_research(self, symbol: str, context: Dict) -> Optional[str]:
        try:
            from integrations.agents import analyze_symbol
            prices = self._prices_from_context(context)
            if prices is None or len(prices) < 60:
                prices = self._fetch_prices(symbol)
            if prices is None or len(prices) < 60:
                return None
            result = analyze_symbol(symbol, prices)
            report = (result.get("report_markdown") or "").strip()
            if report:
                mode = ("local AI agents (Ollama)" if result.get("llm_available")
                        else "rule-based agents (Ollama offline - start it for richer copy)")
                return f"{report}\n\n*Generated by the multi-agent research team: {mode}.*"
        except Exception as e:
            logger.warning(f"Deep research failed for {symbol}: {e}")
        return None

    @staticmethod
    def _prices_from_context(context: Dict):
        history = context.get("price_history") or {}
        closes = [c for c in (history.get("close") or []) if isinstance(c, (int, float))]
        if len(closes) < 60:
            return None
        try:
            import pandas as pd
            index = history.get("dates") or list(range(len(closes)))
            frame = pd.DataFrame({
                "open": history.get("open") or closes,
                "high": history.get("high") or closes,
                "low": history.get("low") or closes,
                "close": closes,
                "volume": history.get("volume") or [0] * len(closes),
            }, index=index)
            return frame.dropna(subset=["close"])
        except Exception:
            return None

    @staticmethod
    def _fetch_prices(symbol: str, period: str = "400d"):
        try:
            import yfinance as yf
            frame = yf.download(symbol, period=period, interval="1d",
                                progress=False, auto_adjust=True)
            if frame is None or frame.empty:
                return None
            if getattr(frame.columns, "nlevels", 1) > 1:
                frame.columns = [c[0] if isinstance(c, tuple) else c for c in frame.columns]
            frame.columns = [str(c).lower() for c in frame.columns]
            return frame
        except Exception as e:
            logger.warning(f"Price fetch failed for {symbol}: {e}")
            return None

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
        hint = " For a full multi-agent research note, ask e.g. \"deep dive on AAPL\"."
        return "I can help you understand the analysis. Ask me about predictions, risk metrics, feature importance, market regime, or trading recommendations." + hint

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
