from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("AIAnalyst")


class SentimentEngine:
    POS_WORDS = {"beat", "surge", "record", "upgrade", "growth", "strong",
                 "outperform", "bullish", "upside", "rally", "expand", "profit"}
    NEG_WORDS = {"miss", "plunge", "downgrade", "weak", "loss", "bearish",
                 "lawsuit", "investigation", "cut", "warning", "decline",
                 "default", "bankruptcy", "fraud", "downturn"}

    def __init__(self, model_name: str = "finbert"):
        self.model_name = model_name
        self._pipe = None
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            if "finbert" in self.model_name.lower():
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                )
                logger.info("FinBERT sentiment model loaded")
        except Exception as e:
            logger.info(f"Transformer unavailable, using lexicon fallback ({e})")
            self._pipe = None

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        if self._pipe is not None:
            try:
                out = self._pipe(text[:512])[0]
                label, conf = out["label"].lower(), float(out["score"])
                if "pos" in label:
                    return conf
                if "neg" in label:
                    return -conf
                return 0.0
            except Exception:
                pass
        tokens = re.findall(r"[a-z]+", text.lower())
        pos = sum(t in self.POS_WORDS for t in tokens)
        neg = sum(t in self.NEG_WORDS for t in tokens)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)


class RegimeDetector:
    def __init__(self, vol_lookback: int = 20, trend_lookback: int = 60):
        self.vol_lb = vol_lookback
        self.trend_lb = trend_lookback

    def classify(self, prices: pd.Series) -> str:
        if len(prices) < self.trend_lb + 1:
            return "unknown"
        rets = prices.pct_change().dropna()
        vol = rets.rolling(self.vol_lb).std().iloc[-1] * np.sqrt(252)
        trend = prices.iloc[-1] / prices.iloc[-self.trend_lb] - 1

        if vol > 0.45:
            return "crisis"
        if vol > 0.28:
            return "high_vol"
        if vol < 0.10:
            return "low_vol"
        if trend > 0.05:
            return "trending_up"
        if trend < -0.05:
            return "trending_down"
        return "ranging"


class AIAnalyst:
    def __init__(self, cfg, sentiment_engine: Optional[SentimentEngine] = None):
        self.cfg = cfg
        self.sentiment = sentiment_engine or SentimentEngine(cfg.ai_model_name)
        self.regime = RegimeDetector()
        self.last_commentary: Dict[str, str] = {}

    def analyze_symbol(
        self,
        symbol: str,
        features: pd.Series,
        ml_prob: float,
        news_headlines: Optional[List[str]] = None,
    ) -> Dict:
        tech_score = self._technical_score(features)
        news_score = 0.0
        if news_headlines:
            news_score = np.mean([self.sentiment.score(h) for h in news_headlines])

        composite = (
            0.65 * (ml_prob - 0.5) * 2
            + 0.25 * tech_score
            + 0.10 * news_score
        )
        composite = float(np.clip(composite, -1, 1))

        rationale = self._rationale(features, ml_prob, tech_score, news_score)
        return {
            "symbol": symbol,
            "ml_prob": float(ml_prob),
            "tech_score": float(tech_score),
            "news_score": float(news_score),
            "composite_score": composite,
            "confidence": float(min(1.0, abs(composite) * 1.25)),
            "rationale": rationale,
        }

    def _technical_score(self, f: pd.Series) -> float:
        score = 0.0
        if "dist_ma_20" in f and not np.isnan(f["dist_ma_20"]):
            score += 0.15 * np.tanh(f["dist_ma_20"] * 20)
        if "dist_ma_50" in f and not np.isnan(f["dist_ma_50"]):
            score += 0.10 * np.tanh(f["dist_ma_50"] * 15)
        if "ma_20_50_x" in f and not np.isnan(f["ma_20_50_x"]):
            score += 0.15 * np.tanh(f["ma_20_50_x"] * 40)
        if "rsi" in f and not np.isnan(f["rsi"]):
            score += 0.15 * np.tanh((f["rsi"] - 50) / 20)
        if "macd_hist" in f and not np.isnan(f["macd_hist"]):
            score += 0.10 * np.tanh(f["macd_hist"] * 100)
        if "bb_pos" in f and not np.isnan(f["bb_pos"]):
            score += 0.10 * np.tanh((0.5 - f["bb_pos"]) * 2)
        if "vol_ratio_20" in f and not np.isnan(f["vol_ratio_20"]):
            score += 0.05 * np.tanh((f["vol_ratio_20"] - 1) * 2)
        if "adx" in f and not np.isnan(f["adx"]):
            score += 0.05 * np.tanh((f["adx"] - 25) / 25)
        return float(np.clip(score, -1, 1))

    def _rationale(self, f: pd.Series, ml_prob: float,
                   tech: float, news: float) -> str:
        parts = []
        if ml_prob > 0.6:
            parts.append(f"ML model expects upside (P={ml_prob:.2f})")
        elif ml_prob < 0.4:
            parts.append(f"ML model expects downside (P={ml_prob:.2f})")
        else:
            parts.append(f"ML model is neutral (P={ml_prob:.2f})")

        if "rsi" in f and not np.isnan(f["rsi"]):
            if f["rsi"] > 70:
                parts.append("RSI overbought")
            elif f["rsi"] < 30:
                parts.append("RSI oversold")
        if "adx" in f and not np.isnan(f["adx"]) and f["adx"] > 30:
            parts.append("strong trend (ADX>30)")
        if "vol_ratio_20" in f and not np.isnan(f["vol_ratio_20"]) and f["vol_ratio_20"] > 1.5:
            parts.append("volume expansion")
        if abs(news) > 0.3:
            parts.append(f"news sentiment {'bullish' if news > 0 else 'bearish'}")
        parts.append(f"tech score {tech:+.2f}")
        return "; ".join(parts)

    def market_overview(self, benchmark_prices: pd.Series,
                        breadth: float) -> Dict:
        reg = self.regime.classify(benchmark_prices)
        overview = {
            "regime": reg,
            "breadth": breadth,
            "timestamp": datetime.utcnow().isoformat(),
        }
        risk_multiplier = {
            "trending_up": 1.15,
            "trending_down": 0.85,
            "ranging": 0.90,
            "low_vol": 1.10,
            "high_vol": 0.70,
            "crisis": 0.40,
            "unknown": 0.85,
        }.get(reg, 0.85)
        overview["risk_multiplier"] = risk_multiplier
        return overview


class LLMBridge:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._client = None
        if api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.info(f"OpenAI unavailable: {e}")

    def comment(self, market_state: Dict, top_signals: List[Dict]) -> str:
        if self._client is None:
            return self._local_comment(market_state, top_signals)
        prompt = (
            "You are a sell-side quantitative strategist. In 4 sentences, "
            f"summarize today's setup. Regime: {market_state.get('regime')}; "
            f"breadth: {market_state.get('breadth'):.2f}. "
            f"Top longs: {[s['symbol'] for s in top_signals if s['composite_score'] > 0][:5]}. "
            f"Top shorts: {[s['symbol'] for s in top_signals if s['composite_score'] < 0][:5]}."
        )
        try:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=180,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return self._local_comment(market_state, top_signals)

    @staticmethod
    def _local_comment(state: Dict, signals: List[Dict]) -> str:
        longs = [s["symbol"] for s in signals if s["composite_score"] > 0.2][:5]
        shorts = [s["symbol"] for s in signals if s["composite_score"] < -0.2][:5]
        return (
            f"Regime detected as {state.get('regime')}. "
            f"Breadth at {state.get('breadth', 0):.0%}. "
            f"Long candidates: {longs}. Short candidates: {shorts}. "
            f"Risk multiplier set to {state.get('risk_multiplier', 1.0):.2f}."
        )
