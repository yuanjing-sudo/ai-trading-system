from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from config import CONFIG
from data.fetcher import MarketDataFetcher
from data.features import FeatureEngineer
from models.xgboost_model import XGBoostPredictor
from models.ensemble import EnsemblePredictor
from models.ai_analyst import AIAnalyst, SentimentEngine, LLMBridge
from scanner.screener import MultiFactorScanner
from strategy.signals import SignalGenerator
from strategy.portfolio import PortfolioConstructor
from backtest.engine import Backtester
from backtest.metrics import summary_report
from predictor.bridge import PredictorBridge
from predictor.engine import PredictorConfig, PredictionTarget
from utils.logger import setup_logger

logger = setup_logger("Main", CONFIG.log_level, CONFIG.output_dir)


def prepare_data(cfg):
    logger.info("Fetching market data...")
    fetcher = MarketDataFetcher(cfg.data.cache_dir, cfg.data.data_source)
    price_data = fetcher.fetch(cfg.data.symbols, cfg.data.start_date, cfg.data.end_date,
                               cfg.data.interval)
    benchmark = fetcher.fetch_benchmark(cfg.data.benchmark, cfg.data.start_date,
                                        cfg.data.end_date)
    if not price_data:
        raise RuntimeError("No data fetched")
    logger.info(f"Data ready: {len(price_data)} symbols, "
                f"benchmark rows={len(benchmark)}")
    return price_data, benchmark


def build_features(cfg, price_data, benchmark):
    logger.info("Building features...")
    fe = FeatureEngineer(cfg.features)
    panel = fe.build(price_data, benchmark)
    logger.info(f"Panel shape: {panel.shape}, symbols={panel.index.get_level_values('symbol').nunique()}")
    return panel


def train_models(cfg, panel):
    logger.info("Training models...")
    ai = AIAnalyst(cfg.model, SentimentEngine(cfg.model.ai_model_name))
    ens = EnsemblePredictor(cfg.model, ai_analyst=ai)
    ens.train(panel, horizon=5)
    imp = ens.xgb.feature_importance(5)
    if not imp.empty:
        logger.info("Top-15 features:\n" + str(imp.head(15)))
    return ens


def run_scan(cfg, ens, panel, benchmark):
    logger.info("Scanning universe...")
    scored = ens.predict_full(panel, horizon=5)

    ai = ens.ai or AIAnalyst(cfg.model, SentimentEngine(cfg.model.ai_model_name))
    bench_hist = benchmark["close"] if not benchmark.empty else pd.Series(dtype=float)
    breadth = float((scored["final_score"] > 0).mean())
    mstate = ai.market_overview(bench_hist, breadth)
    logger.info(f"Market regime: {mstate}")

    scanner = MultiFactorScanner(cfg.strategy)
    ranked = scanner.rank(scored)

    sg = SignalGenerator(cfg.strategy)
    signals = sg.generate(ranked, mstate)
    logger.info(f"Top 10 longs:\n{signals.head(10)[['p_ensemble','final_score','signal']]}")

    rets = panel.groupby(level="symbol")["log_ret_1"].apply(lambda s: s).unstack(level="symbol") \
                if "log_ret_1" in panel.columns else pd.DataFrame()
    return signals, mstate, scored


def run_backtest(cfg, panel, price_data, benchmark):
    logger.info("Running walk-forward backtest...")
    bt = Backtester(cfg)
    result = bt.run(panel, price_data, benchmark)
    logger.info("\n" + summary_report(result.equity_curve, result.returns,
                                       benchmark["close"] if not benchmark.empty else None))
    return result


def run_predictor_mode(price_data, benchmark):
    logger.info("Running Predictor Engine subsystem...")
    pred_cfg = PredictorConfig(
        horizons=(1, 5, 10, 20),
        primary_horizon=5,
        target_type=PredictionTarget.DIRECTION,
        min_history=260,
        n_bootstrap=20,
        xgb_n_estimators=500,
        feature_selection_enabled=True,
        max_features=100,
        log_level=CONFIG.log_level,
    )
    bridge = PredictorBridge(cfg=pred_cfg)
    bridge.fit(price_data, benchmark)

    preds = bridge.predict(price_data, benchmark)
    logger.info(f"Predictor produced {len(preds)} predictions")

    signals = bridge.get_signals_df()
    logger.info(f"Top 10 signals:\n{signals.head(10)[['symbol','fused_prob','final_score','signal']]}")

    mstate = bridge.get_market_state(benchmark)
    logger.info(f"Market state: {mstate}")

    metrics = bridge.metrics_report()
    logger.info(f"\nPredictor metrics:\n{metrics.to_string(index=False)}")

    imp = bridge.feature_importance(top_k=15)
    logger.info(f"\nTop-15 features:\n{imp.to_string()}")

    out = Path(CONFIG.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out / "predictor_signals.csv", index=False)
    metrics.to_csv(out / "predictor_metrics.csv", index=False)
    with open(out / "predictor_market_state.json", "w") as f:
        json.dump(mstate, f, default=str)
    logger.info(f"Predictor outputs written to {out}")
    return signals, mstate


def save_outputs(cfg, result, signals, mstate):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    result.equity_curve.to_csv(out / f"equity_{stamp}.csv")
    result.returns.to_csv(out / f"returns_{stamp}.csv")
    result.positions.to_csv(out / f"positions_{stamp}.csv")
    signals.to_csv(out / f"signals_{stamp}.csv")
    with open(out / f"market_state_{stamp}.json", "w") as f:
        json.dump(mstate, f, default=str)
    with open(out / f"metrics_{stamp}.json", "w") as f:
        json.dump(result.metrics, f, default=str)
    logger.info(f"Outputs written to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["backtest", "scan", "live", "predictor"], default="backtest")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg = CONFIG
    if args.symbols:
        cfg.data.symbols = args.symbols
    if args.start:
        cfg.data.start_date = args.start
    if args.end:
        cfg.data.end_date = args.end

    price_data, benchmark = prepare_data(cfg)

    if args.mode == "predictor":
        signals, mstate = run_predictor_mode(price_data, benchmark)
        return

    panel = build_features(cfg, price_data, benchmark)

    if args.mode == "backtest":
        ens = train_models(cfg, panel)
        signals, mstate, scored = run_scan(cfg, ens, panel, benchmark)
        result = run_backtest(cfg, panel, price_data, benchmark)
        save_outputs(cfg, result, signals, mstate)

    elif args.mode == "scan":
        ens = train_models(cfg, panel)
        signals, mstate, scored = run_scan(cfg, ens, panel, benchmark)
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        signals.to_csv(out / "latest_signals.csv")
        logger.info(f"Scan written to {out}/latest_signals.csv")

    elif args.mode == "live":
        raise SystemExit("Live mode requires broker credentials; use scan/backtest first.")


if __name__ == "__main__":
    main()
