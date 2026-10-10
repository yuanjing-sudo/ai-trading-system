from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from config import ExecutionConfig
from utils.logger import setup_logger

logger = setup_logger("Broker")


@dataclass
class Order:
    symbol: str
    qty: float
    side: str
    order_type: str = "market"
    limit_price: Optional[float] = None
    order_id: Optional[str] = None
    status: str = "NEW"
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    timestamp: datetime = None


class PaperBroker:
    def __init__(self, cfg: ExecutionConfig, initial_cash: float = 1e6):
        self.cfg = cfg
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.orders: List[Order] = []
        self._oid = 0

    def _next_id(self) -> str:
        self._oid += 1
        return f"PAPER-{self._oid:06d}"

    def submit_order(self, symbol: str, qty: float, side: str,
                     last_price: float, slippage_bps: float = 3.0) -> Order:
        o = Order(symbol=symbol, qty=qty, side=side, order_type="market",
                  order_id=self._next_id(), timestamp=datetime.utcnow())
        direction = 1 if side == "BUY" else -1
        fill = last_price * (1 + direction * slippage_bps / 1e4)
        o.filled_price = fill
        o.filled_qty = qty
        o.status = "FILLED"

        notional = qty * fill
        if side == "BUY":
            self.cash -= notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        else:
            self.cash += notional
            self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
        self.orders.append(o)
        logger.info(f"{o.status} {side} {qty} {symbol} @ {fill:.4f}")
        return o

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        mv = sum(self.positions.get(s, 0.0) * prices.get(s, 0.0) for s in self.positions)
        return self.cash + mv

    def rebalance(self, target_weights: Dict[str, float],
                  prices: Dict[str, float], total_equity: float) -> List[Order]:
        orders = []
        target_notional = {s: w * total_equity for s, w in target_weights.items()}
        for s in list(self.positions.keys()):
            if s not in target_weights:
                qty = abs(self.positions[s])
                side = "SELL" if self.positions[s] > 0 else "BUY"
                if qty > 0 and s in prices:
                    orders.append(self.submit_order(s, qty, side, prices[s]))
        for s, tgt_n in target_notional.items():
            if s not in prices:
                continue
            cur_n = self.positions.get(s, 0.0) * prices[s]
            delta_n = tgt_n - cur_n
            if abs(delta_n) < 100:
                continue
            qty = abs(delta_n) / prices[s]
            side = "BUY" if delta_n > 0 else "SELL"
            orders.append(self.submit_order(s, qty, side, prices[s]))
        return orders


class IBKRBroker:
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg
        self._ib = None

    def connect(self):
        try:
            from ib_insync import IB
            self._ib = IB()
            self._ib.connect("127.0.0.1", 7497, clientId=1)
            logger.info("Connected to IBKR")
        except Exception as e:
            raise RuntimeError(f"IBKR connect failed: {e}")

    def submit_order(self, symbol, qty, side, order_type="MKT"):
        if self._ib is None:
            raise RuntimeError("Not connected")
        from ib_insync import Stock, MarketOrder
        contract = Stock(symbol, "SMART", "USD")
        order = MarketOrder(side.upper(), qty)
        trade = self._ib.placeOrder(contract, order)
        logger.info(f"IBKR order submitted: {trade}")
        return trade


class BrokerFactory:
    @staticmethod
    def create(cfg: ExecutionConfig, initial_cash: float = 1e6):
        if cfg.broker == "paper":
            return PaperBroker(cfg, initial_cash)
        if cfg.broker == "ibkr":
            return IBKRBroker(cfg)
        raise ValueError(f"Unknown broker: {cfg.broker}")
