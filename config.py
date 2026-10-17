from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import os
from datetime import date, timedelta
from pathlib import Path


def _live_end_date() -> str:
    """Tomorrow (yfinance end is exclusive) so today's bar is included."""
    return (date.today() + timedelta(days=1)).isoformat()


class AssetClass(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FX = "fx"
    FUTURES = "futures"
    OPTIONS = "options"


class SignalStrength(Enum):
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


class RegimeType(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"
    CRISIS = "crisis"


@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA",
    ])
    benchmark: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: str = field(default_factory=_live_end_date)
    interval: str = "1d"
    lookback_days: int = 504
    cache_dir: str = "./cache"
    data_source: str = "yfinance"


@dataclass
class FeatureConfig:
    returns_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 60])
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    obv_enabled: bool = True
    mfi_period: int = 14
    cross_sectional_rank: bool = True
    sector_neutral: bool = True


@dataclass
class ModelConfig:
    xgb_n_estimators: int = 1500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.02
    xgb_subsample: float = 0.85
    xgb_colsample_bytree: float = 0.85
    xgb_min_child_weight: int = 5
    xgb_gamma: float = 0.1
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.5
    xgb_objective: str = "binary:logistic"
    xgb_eval_metric: str = "auc"
    xgb_early_stopping: int = 75
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    target_return_threshold: float = 0.005
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "xgboost": 0.55,
        "random_forest": 0.20,
        "gradient_boosting": 0.15,
        "ai_analysis": 0.10,
    })
    ai_analysis_enabled: bool = True
    ai_model_name: str = "finbert"
    ai_confidence_threshold: float = 0.65


@dataclass
class StrategyConfig:
    signal_threshold_buy: float = 0.60
    signal_threshold_strong_buy: float = 0.75
    signal_threshold_sell: float = 0.40
    signal_threshold_strong_sell: float = 0.25
    top_n_positions: int = 20
    max_position_size: float = 0.10
    min_position_size: float = 0.01
    rebalance_frequency: str = "weekly"
    long_short: bool = True
    market_neutral: bool = False
    regime_adaptive: bool = True


@dataclass
class RiskConfig:
    max_portfolio_leverage: float = 1.5
    max_sector_exposure: float = 0.30
    max_single_position: float = 0.10
    max_drawdown_stop: float = 0.20
    daily_var_limit: float = 0.03
    target_volatility: float = 0.12
    stop_loss_atr_mult: float = 2.5
    take_profit_atr_mult: float = 5.0
    trailing_stop: bool = True
    correlation_limit: float = 0.75
    kelly_fraction: float = 0.25


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 3.0
    borrow_cost_annual: float = 0.02
    risk_free_rate: float = 0.04
    walk_forward: bool = True
    train_window: int = 504
    test_window: int = 63
    embargo_days: int = 5


@dataclass
class ExecutionConfig:
    broker: str = "paper"
    api_key: str = os.getenv("BROKER_API_KEY", "")
    api_secret: str = os.getenv("BROKER_API_SECRET", "")
    base_url: str = ""
    paper_trading: bool = True
    order_type: str = "market"
    max_slippage_bps: float = 20.0


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = "INFO"
    random_seed: int = 42
    output_dir: str = "./output"
    n_jobs: int = -1


CONFIG = SystemConfig()
