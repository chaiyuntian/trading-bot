"""Paper trading adapter — simulates trading without real money.

Uses a real exchange for market data but simulates order execution locally.
Essential for testing strategies before going live.
"""

import time
import uuid
from typing import Optional

import pandas as pd

from src.exchange.base import (
    ExchangeAdapter, Ticker, OrderResult, OrderSide, OrderType,
    OrderStatus, MarketInfo
)
from src.exchange.ccxt_adapter import CCXTAdapter
from src.utils.logger import setup_logger

logger = setup_logger("exchange.paper")


class PaperAdapter(ExchangeAdapter):
    """Simulated trading using real market data."""

    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = float(config["trading"]["initial_capital"])
        # Real exchange for market data only
        self._data_source = CCXTAdapter(config)

        self._balances: dict[str, float] = {}
        self._orders: dict[str, OrderResult] = {}
        self._fills: list[OrderResult] = []
        self._fee_rate = 0.001  # 0.1% simulated fee

    def connect(self) -> None:
        self._data_source.connect()
        quote = self.config["trading"]["symbol"].split("/")[1]
        self._balances = {quote: self.initial_capital}
        logger.info(f"Paper trading initialized with {self.initial_capital} {quote}")

    def disconnect(self) -> None:
        self._data_source.disconnect()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                    limit: int = 500) -> pd.DataFrame:
        return self._data_source.fetch_ohlcv(symbol, timeframe, limit)

    def get_ticker(self, symbol: str) -> Ticker:
        return self._data_source.get_ticker(symbol)

    def get_balance(self, currency: str = "USDT") -> float:
        return self._balances.get(currency, 0.0)

    def get_market_info(self, symbol: str) -> MarketInfo:
        return self._data_source.get_market_info(symbol)

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    amount: float, price: Optional[float] = None) -> OrderResult:
        info = self.get_market_info(symbol)
        amount = round(amount, info.amount_precision)

        if order_type == OrderType.MARKET:
            ticker = self.get_ticker(symbol)
            exec_price = ticker.ask if side == OrderSide.BUY else ticker.bid
            if exec_price <= 0:
                exec_price = ticker.last
        else:
            exec_price = price or self.get_price(symbol)

        cost = amount * exec_price
        fee = cost * self._fee_rate
        base, quote = info.base, info.quote

        # Check sufficient balance
        if side == OrderSide.BUY:
            available = self._balances.get(quote, 0)
            needed = cost + fee
            if available < needed:
                logger.warning(f"Insufficient {quote}: need {needed:.2f}, have {available:.2f}")
                return OrderResult(
                    id=str(uuid.uuid4())[:8], symbol=symbol, side=side,
                    order_type=order_type, status=OrderStatus.REJECTED,
                    amount=amount, price=exec_price,
                )
            self._balances[quote] = available - needed
            self._balances[base] = self._balances.get(base, 0) + amount
        else:
            available = self._balances.get(base, 0)
            if available < amount:
                logger.warning(f"Insufficient {base}: need {amount:.8f}, have {available:.8f}")
                return OrderResult(
                    id=str(uuid.uuid4())[:8], symbol=symbol, side=side,
                    order_type=order_type, status=OrderStatus.REJECTED,
                    amount=amount, price=exec_price,
                )
            self._balances[base] = available - amount
            self._balances[quote] = self._balances.get(quote, 0) + cost - fee

        result = OrderResult(
            id=str(uuid.uuid4())[:8], symbol=symbol, side=side,
            order_type=order_type, status=OrderStatus.FILLED,
            amount=amount, price=exec_price, filled=amount,
            cost=cost, fee=fee,
        )
        self._orders[result.id] = result
        self._fills.append(result)

        logger.info(
            f"PAPER {side.value.upper()} {amount:.6f} {symbol} @ {exec_price:.2f} "
            f"cost={cost:.2f} fee={fee:.4f}"
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_open_orders(self, symbol: str) -> list[OrderResult]:
        return [
            o for o in self._orders.values()
            if o.symbol == symbol and o.status == OrderStatus.OPEN
        ]

    def get_order(self, order_id: str, symbol: str) -> OrderResult:
        return self._orders[order_id]

    def get_total_value(self, symbol: str) -> float:
        """Total portfolio value in quote currency."""
        info = self.get_market_info(symbol)
        quote_bal = self._balances.get(info.quote, 0)
        base_bal = self._balances.get(info.base, 0)
        if base_bal > 0:
            price = self.get_price(symbol)
            return quote_bal + base_bal * price
        return quote_bal
