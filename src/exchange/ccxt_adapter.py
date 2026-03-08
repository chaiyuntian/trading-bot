"""CCXT-based exchange adapter. Supports 100+ exchanges via unified API."""

import ccxt
import pandas as pd
from typing import Optional

from src.exchange.base import (
    ExchangeAdapter, Ticker, OrderResult, OrderSide, OrderType,
    OrderStatus, MarketInfo
)
from src.utils.logger import setup_logger

logger = setup_logger("exchange.ccxt")

_STATUS_MAP = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


class CCXTAdapter(ExchangeAdapter):
    """Production adapter for any CCXT-supported exchange."""

    def __init__(self, config: dict):
        exc_cfg = config["exchange"]
        self.exchange_name = exc_cfg["name"]
        self.sandbox = exc_cfg.get("sandbox", True)

        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange: ccxt.Exchange = exchange_class({
            "apiKey": exc_cfg.get("api_key", ""),
            "secret": exc_cfg.get("api_secret", ""),
            "enableRateLimit": exc_cfg.get("rate_limit", True),
            "options": {"defaultType": "spot"},
        })

        # Cache market info to avoid repeated API calls
        self._market_cache: dict[str, MarketInfo] = {}

    def connect(self) -> None:
        if self.sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"SANDBOX mode on {self.exchange_name}")
        else:
            logger.warning(f"LIVE mode on {self.exchange_name}!")

        self.exchange.load_markets()
        logger.info(f"Connected: {self.exchange_name} ({len(self.exchange.markets)} markets)")

    def disconnect(self) -> None:
        self._market_cache.clear()
        logger.info("Disconnected")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                    limit: int = 500) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_ticker(self, symbol: str) -> Ticker:
        t = self.exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            last=float(t.get("last", 0)),
            bid=float(t.get("bid", 0) or 0),
            ask=float(t.get("ask", 0) or 0),
            volume_24h=float(t.get("quoteVolume", 0) or 0),
            timestamp=float(t.get("timestamp", 0) or 0) / 1000,
        )

    def get_balance(self, currency: str = "USDT") -> float:
        bal = self.exchange.fetch_balance()
        return float(bal.get("free", {}).get(currency, 0))

    def get_market_info(self, symbol: str) -> MarketInfo:
        if symbol in self._market_cache:
            return self._market_cache[symbol]

        m = self.exchange.market(symbol)
        limits = m.get("limits", {})
        precision = m.get("precision", {})
        fees = m.get("fees", m.get("fee", {}))

        info = MarketInfo(
            symbol=symbol,
            base=m.get("base", ""),
            quote=m.get("quote", ""),
            min_amount=float(limits.get("amount", {}).get("min", 0) or 0),
            min_cost=float(limits.get("cost", {}).get("min", 0) or 0),
            price_precision=int(precision.get("price", 8) or 8),
            amount_precision=int(precision.get("amount", 8) or 8),
            maker_fee=float(fees.get("maker", 0.001) if isinstance(fees, dict) else 0.001),
            taker_fee=float(fees.get("taker", 0.001) if isinstance(fees, dict) else 0.001),
        )
        self._market_cache[symbol] = info
        return info

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    amount: float, price: Optional[float] = None) -> OrderResult:
        amount = self.round_amount(symbol, amount)
        if price is not None:
            price = self.round_price(symbol, price)

        side_str = side.value
        if order_type == OrderType.MARKET:
            order = self.exchange.create_order(symbol, "market", side_str, amount)
        elif order_type == OrderType.LIMIT:
            order = self.exchange.create_order(symbol, "limit", side_str, amount, price)
        else:
            order = self.exchange.create_order(symbol, "limit", side_str, amount, price)

        result = self._parse_order(order, symbol, side, order_type)
        logger.info(
            f"ORDER {side_str.upper()} {order_type.value} {amount} {symbol}"
            f" @ {price or 'market'} -> {result.status.value} id={result.id}"
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"CANCEL order {order_id}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"Order {order_id} not found for cancel")
            return False

    def get_open_orders(self, symbol: str) -> list[OrderResult]:
        orders = self.exchange.fetch_open_orders(symbol)
        return [
            self._parse_order(o, symbol, OrderSide(o["side"]), OrderType.LIMIT)
            for o in orders
        ]

    def get_order(self, order_id: str, symbol: str) -> OrderResult:
        o = self.exchange.fetch_order(order_id, symbol)
        return self._parse_order(o, symbol, OrderSide(o["side"]), OrderType.LIMIT)

    @staticmethod
    def _parse_order(raw: dict, symbol: str, side: OrderSide,
                     order_type: OrderType) -> OrderResult:
        status = _STATUS_MAP.get(raw.get("status", ""), OrderStatus.PENDING)
        fee_info = raw.get("fee") or {}
        return OrderResult(
            id=str(raw.get("id", "")),
            symbol=symbol,
            side=side,
            order_type=order_type,
            status=status,
            amount=float(raw.get("amount", 0) or 0),
            price=float(raw.get("price", 0) or raw.get("average", 0) or 0),
            filled=float(raw.get("filled", 0) or 0),
            cost=float(raw.get("cost", 0) or 0),
            fee=float(fee_info.get("cost", 0) or 0),
            timestamp=float(raw.get("timestamp", 0) or 0) / 1000,
            raw=raw,
        )
