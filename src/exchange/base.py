"""Platform-agnostic trading abstraction layer.

All exchange implementations must inherit from ExchangeAdapter.
This decouples strategies and risk management from any specific exchange.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd
import time


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(slots=True)
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: float = field(default_factory=time.time)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        if self.ask == 0:
            return 0
        return self.spread / self.ask


@dataclass(slots=True)
class OrderResult:
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    amount: float
    price: float
    filled: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    timestamp: float = field(default_factory=time.time)
    raw: Optional[dict] = field(default=None, repr=False)


@dataclass(slots=True)
class MarketInfo:
    symbol: str
    base: str
    quote: str
    min_amount: float
    min_cost: float
    price_precision: int
    amount_precision: int
    maker_fee: float
    taker_fee: float


class ExchangeAdapter(ABC):
    """Abstract base for all exchange implementations.

    Implement this interface to add support for any trading platform:
    - Crypto exchanges (Binance, Bybit, KuCoin, etc.)
    - Stock brokers (Alpaca, Interactive Brokers, etc.)
    - DEXs (Uniswap, dYdX, etc.)
    - Paper/simulated trading
    """

    @abstractmethod
    def connect(self) -> None:
        """Initialize connection and load market data."""

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up connections."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                    limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV candles. Returns DataFrame with columns:
        [open, high, low, close, volume] indexed by timestamp."""

    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker data."""

    @abstractmethod
    def get_balance(self, currency: str = "USDT") -> float:
        """Get available balance for a currency."""

    @abstractmethod
    def get_market_info(self, symbol: str) -> MarketInfo:
        """Get market constraints (min order, precision, fees)."""

    @abstractmethod
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    amount: float, price: Optional[float] = None) -> OrderResult:
        """Place an order. The unified entry point for all order types."""

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order. Returns True if successful."""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[OrderResult]:
        """Get all open orders for a symbol."""

    @abstractmethod
    def get_order(self, order_id: str, symbol: str) -> OrderResult:
        """Get order status by ID."""

    # ── Convenience methods (non-abstract, built on primitives) ──

    def market_buy(self, symbol: str, amount: float) -> OrderResult:
        return self.place_order(symbol, OrderSide.BUY, OrderType.MARKET, amount)

    def market_sell(self, symbol: str, amount: float) -> OrderResult:
        return self.place_order(symbol, OrderSide.SELL, OrderType.MARKET, amount)

    def limit_buy(self, symbol: str, amount: float, price: float) -> OrderResult:
        return self.place_order(symbol, OrderSide.BUY, OrderType.LIMIT, amount, price)

    def limit_sell(self, symbol: str, amount: float, price: float) -> OrderResult:
        return self.place_order(symbol, OrderSide.SELL, OrderType.LIMIT, amount, price)

    def get_price(self, symbol: str) -> float:
        return self.get_ticker(symbol).last

    def round_amount(self, symbol: str, amount: float) -> float:
        info = self.get_market_info(symbol)
        return round(amount, info.amount_precision)

    def round_price(self, symbol: str, price: float) -> float:
        info = self.get_market_info(symbol)
        return round(price, info.price_precision)
