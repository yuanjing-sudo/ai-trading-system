"""Multi-agent AI analysis layer (TradingAgents / AutoHedge inspired).

A team of specialized agents - technical, fundamental, sentiment, risk and
portfolio-manager - analyze a symbol and produce a structured verdict plus a
markdown research note.

The team runs with three LLM tiers, in priority order:

- **Groq (cloud, free tier)**: used automatically when ``LLM_API_KEY`` /
  ``GROQ_API_KEY`` is set - the GitHub Actions snapshot workflow passes the
  ``GROQ_API_KEY`` repo secret here. Default model ``openai/gpt-oss-120b``
  (override with ``LLM_MODEL``). Any OpenAI-compatible endpoint works via
  ``LLM_API_URL``.
- **Local Ollama** (``http://localhost:11434``) for local runs.
- Deterministic, indicator-driven fallback so the pipeline never fails.

Integrated (registry ids): TradingAgents (10), AutoHedge (13),
WallStreetAgents (14), FinGPT-style sentiment (11).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

__all__ = [
    "OllamaClient", "OpenAICompatClient", "build_llm_client", "LLM_STATUS",
    "TechnicalAnalystAgent", "FundamentalsAgent", "SentimentAgent",
    "RiskManagerAgent", "PortfolioManagerAgent", "TradingAgentsOrchestrator",
    "analyze_symbol",
]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL_DEFAULT = "openai/gpt-oss-120b"
LLM_DEBUG = bool(os.environ.get("LLM_DEBUG"))


class OllamaClient:
    """Minimal Ollama HTTP client (stdlib only) with reachability caching."""

    def __init__(self, host: str = OLLAMA_HOST, model: str = OLLAMA_MODEL,
                 timeout: float = 30.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                req = urllib.request.Request(f"{self.host}/api/tags")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    self._available = resp.status == 200
            except Exception:
                self._available = False
        return self._available

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        if not self.available:
            return None
        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"temperature": temperature},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip() or None
        except Exception:
            return None


class OpenAICompatClient:
    """OpenAI-compatible chat completions client (Groq by default).

    Uses any provider exposing ``/chat/completions``; defaults to Groq's free
    tier (``https://api.groq.com/openai/v1``) with
    ``openai/gpt-oss-120b``. Configure via ``LLM_API_KEY`` (or
    ``GROQ_API_KEY``), ``LLM_API_URL`` and ``LLM_MODEL``.
    """

    def __init__(self, api_key: str, model: str = LLM_MODEL_DEFAULT,
                 url: str = GROQ_API_URL, timeout: float = 45.0):
        self.token = api_key or ""
        self.model = model
        self.url = url
        self.backend = "groq" if "groq" in url else "openai-compat"
        try:
            self.host = urllib.parse.urlsplit(url).netloc
        except Exception:
            self.host = url
        self.timeout = timeout
        self.debug = LLM_DEBUG

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _log(self, message: str) -> None:
        if self.debug:
            sys.stderr.write(f"[{self.backend}] {message}\n")

    def _attempt(self, url: str, payload: bytes) -> tuple[str | None, bool]:
        """Returns (content, hit_rate_limit)."""
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ai-trading-system/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.status
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._log(f"non-JSON HTTP {status} from {url}: {raw[:200]!r}")
                return None, False
            choices = data.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content", "")
                if content and content.strip():
                    return content.strip(), False
            self._log(f"no choices in response: {str(data)[:200]}")
            return None, False
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                body = ""
            self._log(f"HTTP {exc.code} from {url}: {body}")
            if exc.code == 429:
                return None, True
            return None, False
        except Exception as exc:
            self._log(f"{type(exc).__name__} from {url}: {exc}")
            return None, False

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> str | None:
        if not self.available:
            return None
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": system or "You are a concise financial markets analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 400,
        }).encode("utf-8")
        for attempt in (1, 2):
            content, rate_limited = self._attempt(self.url, payload)
            if content:
                return content
            if rate_limited and attempt == 1:
                time.sleep(20)
            else:
                break
        return None


def build_llm_client():
    """Pick the best available LLM backend.

    A hosted OpenAI-compatible API (Groq) wins when a key is present - this is
    what the cloud snapshot workflow uses via the ``GROQ_API_KEY`` secret.
    Otherwise fall back to local Ollama.
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
    if api_key:
        return OpenAICompatClient(
            api_key=api_key,
            model=os.environ.get("LLM_MODEL", LLM_MODEL_DEFAULT),
            url=os.environ.get("LLM_API_URL", GROQ_API_URL),
        )
    return OllamaClient()


_LLM = build_llm_client()
LLM_STATUS = {
    "backend": getattr(_LLM, "backend", "ollama"),
    "configured_model": _LLM.model,
    "host": _LLM.host,
}


def _rsi(close: pd.Series, length: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / length, min_periods=length).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1]) if len(rsi.dropna()) else 50.0


class TechnicalAnalystAgent:
    """Momentum / trend / volatility verdict from OHLCV."""

    name = "technical_analyst"

    def analyze(self, prices: pd.DataFrame) -> dict:
        close = prices["close"] if "close" in prices else prices.iloc[:, 0]
        close = close.dropna()
        if len(close) < 60:
            return {"agent": self.name, "signal": "HOLD", "confidence": 0.3,
                    "reasoning": "insufficient history"}
        rsi = _rsi(close)
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        ret20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 21 else 0.0
        vol = close.pct_change().tail(60).std() * np.sqrt(252)

        score = 0.0
        score += 0.35 if close.iloc[-1] > ma60 else -0.35
        score += 0.25 if ma20 > ma60 else -0.25
        score += 0.2 * np.tanh(ret20 / max(vol / np.sqrt(252) * np.sqrt(20), 1e-9))
        if rsi < 30:
            score += 0.1
        elif rsi > 70:
            score -= 0.1

        signal = "BUY" if score > 0.15 else ("SELL" if score < -0.15 else "HOLD")
        facts = (f"RSI={rsi:.1f}, price {close.iloc[-1]:.2f} vs MA20 {ma20:.2f} / "
                 f"MA60 {ma60:.2f}, 20d return {ret20:+.1%}, ann vol {vol:.1%}")
        reasoning = facts
        llm = _LLM.generate(
            f"You are a technical analyst. Data: {facts}. "
            f"Give a 2-sentence technical read and a BUY/HOLD/SELL verdict.",
            system="You are a concise markets technical analyst.",
        )
        if llm:
            reasoning = f"{facts}\n\nLLM: {llm}"
        return {"agent": self.name, "signal": signal,
                "confidence": float(min(0.9, abs(score) + 0.3)),
                "reasoning": reasoning,
                "facts": {"rsi": rsi, "ret20": ret20, "vol_annual": float(vol)}}


class FundamentalsAgent:
    """Valuation / quality verdict from yfinance fundamentals."""

    name = "fundamentals_analyst"

    def analyze(self, ticker: str) -> dict:
        info: dict = {}
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        pe = info.get("trailingPE")
        pm = info.get("profitMargins")
        rg = info.get("revenueGrowth")
        de = info.get("debtToEquity")
        score = 0.0
        if pe and pe > 0:
            score += 0.3 if pe < 25 else (-0.2 if pe > 45 else 0.05)
        if pm is not None:
            score += 0.3 if pm > 0.15 else (-0.2 if pm < 0 else 0.0)
        if rg is not None:
            score += 0.2 if rg > 0.1 else (-0.15 if rg < 0 else 0.0)
        if de is not None:
            score += 0.2 if de < 100 else (-0.15 if de > 250 else 0.0)
        signal = "BUY" if score > 0.2 else ("SELL" if score < -0.2 else "HOLD")
        facts = (f"PE={pe}, margins={pm}, rev growth={rg}, debt/equity={de}, "
                 f"sector={info.get('sector', 'n/a')}")
        reasoning = facts
        llm = _LLM.generate(
            f"You are a fundamentals analyst. Data: {facts}. "
            f"Give a 2-sentence fundamental read and BUY/HOLD/SELL verdict.",
            system="You are a concise fundamental equity analyst.",
        )
        if llm:
            reasoning = f"{facts}\n\nLLM: {llm}"
        return {"agent": self.name, "signal": signal,
                "confidence": 0.55 if info else 0.25,
                "reasoning": reasoning}


class SentimentAgent:
    """Headline sentiment via yfinance news with keyword scoring."""

    name = "sentiment_analyst"

    POSITIVE = ("beat", "surge", "record", "upgrade", "growth", "strong",
                "rally", "profit", "outperform", "raise")
    NEGATIVE = ("miss", "plunge", "downgrade", "weak", "loss", "lawsuit",
                "cut", "decline", "warn", "investigation")

    def analyze(self, ticker: str) -> dict:
        headlines: list[str] = []
        try:
            import yfinance as yf
            news = yf.Ticker(ticker).news or []
            for item in news[:15]:
                title = (item.get("title") or item.get("content", {}).get("title")
                         or "") if isinstance(item, dict) else ""
                if title:
                    headlines.append(title)
        except Exception:
            headlines = []
        text = " ".join(headlines).lower()
        pos = sum(text.count(w) for w in self.POSITIVE)
        neg = sum(text.count(w) for w in self.NEGATIVE)
        score = (pos - neg) / max(pos + neg, 1)
        signal = "BUY" if score > 0.2 else ("SELL" if score < -0.2 else "HOLD")
        reasoning = (f"{len(headlines)} headlines; pos={pos}, neg={neg}. "
                     + (" | ".join(headlines[:3]) if headlines else "no headlines"))
        llm = _LLM.generate(
            f"You are a news sentiment analyst. Headlines: {headlines[:8]}. "
            f"Give a 1-sentence sentiment read and BUY/HOLD/SELL verdict.",
            system="You are a concise market sentiment analyst.")
        if llm:
            reasoning = f"{reasoning}\n\nLLM: {llm}"
        return {"agent": self.name, "signal": signal,
                "confidence": 0.4 if headlines else 0.2,
                "reasoning": reasoning}


class RiskManagerAgent:
    """Portfolio / instrument risk gate."""

    name = "risk_manager"

    def analyze(self, prices: pd.DataFrame, proposed_weight: float = 0.1,
                 max_weight: float = 0.25, max_vol: float = 0.40) -> dict:
        close = prices["close"] if "close" in prices else prices.iloc[:, 0]
        returns = close.pct_change().dropna()
        vol = float(returns.tail(120).std() * np.sqrt(252)) if len(returns) > 5 else 0.5
        dd = float((close / close.cummax() - 1).min()) if len(close) else -0.5
        var95 = float(np.nanpercentile(returns.tail(250), 5)) if len(returns) else -0.05
        scale = 1.0
        notes = []
        if vol > max_vol:
            scale = max_vol / vol
            notes.append(f"vol {vol:.0%} exceeds cap {max_vol:.0%} -> scale {scale:.2f}")
        if proposed_weight * scale > max_weight:
            scale = max_weight / proposed_weight
            notes.append(f"weight capped at {max_weight:.0%}")
        if dd < -0.30:
            scale *= 0.5
            notes.append(f"deep drawdown {dd:.0%} -> halve exposure")
        verdict = "APPROVED" if scale > 0.6 else "REDUCED"
        return {"agent": self.name, "signal": verdict,
                "confidence": 0.8,
                "position_scale": float(scale),
                "reasoning": f"vol={vol:.1%}, maxDD={dd:.1%}, VaR95={var95:.2%}; "
                             + ("; ".join(notes) if notes else "within all limits"),
                "facts": {"vol": vol, "max_drawdown": dd, "var95": var95}}


class PortfolioManagerAgent:
    """Aggregates the team into a final decision."""

    name = "portfolio_manager"

    WEIGHTS = {"technical_analyst": 0.4, "fundamentals_analyst": 0.3,
               "sentiment_analyst": 0.15}

    def decide(self, analyst_views: list[dict], risk_view: dict) -> dict:
        score = 0.0
        votes = []
        for view in analyst_views:
            w = self.WEIGHTS.get(view["agent"], 0.1)
            s = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}[view["signal"]]
            conf = 0.5 + 0.5 * float(view.get("confidence", 0.5))
            score += w * s * conf
            votes.append(f"{view['agent']}: {view['signal']}")
        total_w = sum(self.WEIGHTS.get(v["agent"], 0.1) for v in analyst_views)
        score = score / total_w if total_w else 0.0
        scale = risk_view.get("position_scale", 1.0)
        if score > 0.2:
            decision, direction = "BUY", "LONG"
        elif score < -0.2:
            decision, direction = "SELL", "SHORT"
        else:
            decision, direction = "HOLD", "FLAT"
        suggested_weight = float(np.clip(abs(score) * scale, 0.0, 0.25))
        base = f"Team vote ({'; '.join(votes)}), aggregated score {score:+.2f}"
        llm = _LLM.generate(
            f"You are the portfolio manager. Analyst votes: {votes}. "
            f"Risk manager: {risk_view['reasoning']}. "
            f"Decision {decision}. Write a 3-sentence portfolio committee summary.",
            system="You are a decisive portfolio manager. Be concise.")
        summary = llm or (base + f". Risk: {risk_view['reasoning']}")
        return {"agent": self.name, "decision": decision, "direction": direction,
                "score": float(score),
                "suggested_weight": suggested_weight,
                "confidence": float(min(0.95, abs(score) + 0.3)),
                "summary": summary}


class TradingAgentsOrchestrator:
    """Runs the analyst team -> risk gate -> portfolio committee."""

    def __init__(self, llm_client: OllamaClient | None = None):
        global _LLM
        if llm_client is not None:
            _LLM = llm_client
        self.technical = TechnicalAnalystAgent()
        self.fundamentals = FundamentalsAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskManagerAgent()
        self.pm = PortfolioManagerAgent()

    def run(self, ticker: str, prices: pd.DataFrame,
            proposed_weight: float = 0.1) -> dict:
        views = [
            self.technical.analyze(prices),
            self.fundamentals.analyze(ticker),
            self.sentiment.analyze(ticker),
        ]
        risk_view = self.risk.analyze(prices, proposed_weight)
        decision = self.pm.decide(views, risk_view)
        report = self._render_report(ticker, views, risk_view, decision)
        return {"ticker": ticker, "views": views, "risk": risk_view,
                "decision": decision, "report_markdown": report,
                "llm_available": _LLM.available}

    def _render_report(self, ticker: str, views: list[dict], risk: dict,
                       decision: dict) -> str:
        lines = [f"# Research Note - {ticker}", "",
                 f"**Decision: {decision['decision']} ({decision['direction']})** "
                 f"| confidence {decision['confidence']:.0%} | suggested weight "
                 f"{decision['suggested_weight']:.1%}", "",
                 decision["summary"], "", "## Analyst views", ""]
        for v in views:
            lines.append(f"- **{v['agent']}**: {v['signal']} - {v['reasoning'].splitlines()[0]}")
        lines += ["", "## Risk gate", "",
                  f"- {risk['signal']} (scale {risk['position_scale']:.2f}): {risk['reasoning']}"]
        return "\n".join(lines)


def analyze_symbol(ticker: str, prices: pd.DataFrame,
                   proposed_weight: float = 0.1) -> dict:
    """Convenience wrapper for the full agent pipeline."""
    return TradingAgentsOrchestrator().run(ticker, prices, proposed_weight)
