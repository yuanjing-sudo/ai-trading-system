from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import FeatureConfig
from utils.logger import setup_logger

logger = setup_logger("FeatureEngineer")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_k = df["low"].rolling(k).min()
    high_k = df["high"].rolling(k).max()
    k_pct = 100 * (df["close"] - low_k) / (high_k - low_k).replace(0, np.nan)
    return k_pct, k_pct.rolling(d).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff()).fillna(0)
    return (sign * df["volume"]).cumsum()


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mf = typical * df["volume"]
    pos = mf.where(typical.diff() > 0, 0)
    neg = mf.where(typical.diff() < 0, 0)
    mr = pos.rolling(period).sum() / neg.rolling(period).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mr))


def _bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    return ma, ma + n_std * sd, ma - n_std * sd


class FeatureEngineer:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build(self, price_data: Dict[str, pd.DataFrame],
              benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        frames = []
        for sym, df in price_data.items():
            if df.empty or len(df) < 250:
                logger.warning(f"Skipping {sym}: insufficient history ({len(df)})")
                continue
            f = self._per_symbol_features(df, sym)
            frames.append(f)

        if not frames:
            raise RuntimeError("No valid symbols to build features for")

        panel = pd.concat(frames, axis=0)

        if self.cfg.cross_sectional_rank:
            panel = self._add_cross_sectional(panel)

        if benchmark is not None and not benchmark.empty:
            panel = self._add_benchmark_relative(panel, benchmark)

        return panel

    def _per_symbol_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d = df.copy()
        close = d["close"]
        feat = pd.DataFrame(index=d.index)
        feat["symbol"] = symbol
        feat["close"] = close
        feat["volume"] = d["volume"]

        for w in self.cfg.returns_windows:
            feat[f"ret_{w}"] = close.pct_change(w)
        feat["log_ret_1"] = np.log(close / close.shift(1))

        for w in self.cfg.volatility_windows:
            feat[f"vol_{w}"] = feat["log_ret_1"].rolling(w).std() * np.sqrt(252)
        feat["vol_ratio_5_20"] = feat["vol_5"] / feat["vol_20"].replace(0, np.nan)
        feat["vol_ratio_20_60"] = feat["vol_20"] / feat["vol_60"].replace(0, np.nan)

        for w in self.cfg.ma_windows:
            ma = close.rolling(w).mean()
            feat[f"ma_{w}"] = ma
            feat[f"dist_ma_{w}"] = (close - ma) / ma.replace(0, np.nan)
        feat["ma_20_50_x"] = (feat["ma_20"] - feat["ma_50"]) / close
        feat["ma_50_200_x"] = (feat["ma_50"] - feat["ma_200"]) / close

        rsi = _rsi(close, self.cfg.rsi_period)
        feat["rsi"] = rsi
        feat["rsi_slope_5"] = rsi.diff(5)

        macd, sig, hist = _macd(close, self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal)
        feat["macd"] = macd / close
        feat["macd_signal"] = sig / close
        feat["macd_hist"] = hist / close

        mid, up, lo = _bollinger(close, self.cfg.bb_period, self.cfg.bb_std)
        feat["bb_pos"] = (close - lo) / (up - lo).replace(0, np.nan)
        feat["bb_width"] = (up - lo) / mid.replace(0, np.nan)

        feat["atr"] = _atr(d, self.cfg.atr_period) / close
        feat["adx"] = _adx(d, self.cfg.adx_period)
        k, dd = _stochastic(d, self.cfg.stoch_k, self.cfg.stoch_d)
        feat["stoch_k"] = k
        feat["stoch_d"] = dd

        for w in self.cfg.volume_windows:
            vma = d["volume"].rolling(w).mean()
            feat[f"vol_ratio_{w}"] = d["volume"] / vma.replace(0, np.nan)
        if self.cfg.obv_enabled:
            obv = _obv(d)
            feat["obv_z_20"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, np.nan)
        feat["mfi"] = _mfi(d, self.cfg.mfi_period)

        rng = (d["high"] - d["low"]).replace(0, np.nan)
        feat["upper_shadow"] = (d["high"] - np.maximum(d["open"], close)) / rng
        feat["lower_shadow"] = (np.minimum(d["open"], close) - d["low"]) / rng
        feat["body_pct"] = (close - d["open"]) / rng
        feat["gap"] = (d["open"] - close.shift(1)) / close.shift(1)

        for w in (20, 60):
            feat[f"skew_{w}"] = feat["log_ret_1"].rolling(w).skew()
            feat[f"kurt_{w}"] = feat["log_ret_1"].rolling(w).kurt()

        for h in (1, 5, 10, 20):
            fwd_ret = close.shift(-h) / close - 1.0
            feat[f"fwd_ret_{h}"] = fwd_ret
            feat[f"target_{h}"] = (fwd_ret > 0).astype(int)

        return feat.dropna(subset=["close"])

    def _add_cross_sectional(self, panel: pd.DataFrame) -> pd.DataFrame:
        panel = panel.reset_index().rename(columns={"index": "date"})
        if "date" not in panel.columns:
            panel = panel.rename(columns={panel.columns[0]: "date"})

        if not np.issubdtype(panel["date"].dtype, np.datetime64):
            panel["date"] = pd.to_datetime(panel["date"])

        cols_to_rank = [
            "ret_5", "ret_20", "ret_60", "vol_20", "rsi",
            "macd_hist", "atr", "adx", "vol_ratio_20", "mfi",
        ]
        for c in cols_to_rank:
            if c in panel.columns:
                panel[f"cs_rank_{c}"] = panel.groupby("date")[c].rank(pct=True) - 0.5

        panel = panel.set_index(["date", "symbol"])
        return panel

    def _add_benchmark_relative(self, panel: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
        b = bench["close"].pct_change().rename("bench_ret_1")
        panel = panel.reset_index()
        panel = panel.merge(b, left_on="date", right_index=True, how="left")
        panel["excess_ret_1"] = panel["log_ret_1"] - np.log1p(panel["bench_ret_1"])
        panel = panel.sort_values(["symbol", "date"])

        def beta(group):
            cov = group["log_ret_1"].rolling(60).cov(group["bench_ret_1"])
            var = group["bench_ret_1"].rolling(60).var()
            return cov / var.replace(0, np.nan)

        panel["beta_60"] = panel.groupby("symbol", group_keys=False).apply(beta)
        return panel.set_index(["date", "symbol"])
