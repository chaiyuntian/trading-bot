"""Futures/Perpetual contract adapter via CCXT.

Key differences from spot trading:
- Can go SHORT (profit from price drops)
- Leverage: use 2-3x to amplify small capital
- Funding rate: paid/received every 8 hours
- Lower fees: 0.02% maker vs 0.10% taker on spot
- Liquidation risk: must manage margin carefully

For $100 accounts, futures is the ONLY viable path because:
1. Fees are 5x lower (0.02% maker vs 0.10% spot)
2. Can profit in both directions
3. 2-3x leverage turns $100 into $200-300 buying power
"""

import ccxt
import pandas as pd
from typing import Optional

from src.exchange.base import (
    ExchangeAdapter, Ticker, OrderResult, OrderSide, OrderType,
    OrderStatus, MarketInfo
)
from src.utils.logger import setup_logger

logger = setup_logger("exchange.futures")

_STATUS_MAP = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


class FuturesAdapter(ExchangeAdapter):
    """CCXT adapter for perpetual futures trading.

    Supports: Binance USDM, Bybit Linear, OKX Swaps, etc.
    """

    def __init__(self, config: dict):
        exc_cfg = config["exchange"]
        self.exchange_name = exc_cfg["name"]
        self.sandbox = exc_cfg.get("sandbox", True)
        self.leverage = config["trading"].get("leverage", 2)

        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange: ccxt.Exchange = exchange_class({
            "apiKey": exc_cfg.get("api_key", ""),
            "secret": exc_cfg.get("api_secret", ""),
            "enableRateLimit": exc_cfg.get("rate_limit", True),
            "options": {"defaultType": "swap"},  # Perpetual futures
        })

        self._market_cache: dict[str, MarketInfo] = {}

    def connect(self) -> None:
        if self.sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"SANDBOX FUTURES on {self.exchange_name}")
        else:
            logger.warning(f"LIVE FUTURES on {self.exchange_name}!")

        self.exchange.load_markets()
        logger.info(
            f"Futures connected: {self.exchange_name} "
            f"| Leverage={self.leverage}x"
        )

    def set_leverage(self, symbol: str, leverage: int = None):
        """Set leverage for a symbol. Call before placing orders."""
        lev = leverage or self.leverage
        try:
            self.exchange.set_leverage(lev, symbol)
            logger.info(f"Leverage set: {symbol} = {lev}x")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")

    def disconnect(self) -> None:
        self._market_cache.clear()
        logger.info("Futures disconnected")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "4h",
                    limit: int = 500) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
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
            maker_fee=float(fees.get("maker", 0.0002) if isinstance(fees, dict) else 0.0002),
            taker_fee=float(fees.get("taker", 0.0005) if isinstance(fees, dict) else 0.0005),
        )
        self._market_cache[symbol] = info
        return info

    def get_funding_rate(self, symbol: str) -> dict:
        """Get current funding rate. Positive = longs pay shorts."""
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return {
                "rate": float(funding.get("fundingRate", 0) or 0),
                "timestamp": funding.get("fundingTimestamp"),
                "next_timestamp": funding.get("nextFundingTimestamp"),
            }
        except Exception as e:
            logger.debug(f"Funding rate fetch failed: {e}")
            return {"rate": 0.0, "timestamp": None, "next_timestamp": None}

    def get_positions(self, symbol: str = None) -> list[dict]:
        """Get open futures positions."""
        try:
            positions = self.exchange.fetch_positions([symbol] if symbol else None)
            return [
                {
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "size": float(p.get("contracts", 0) or 0),
                    "notional": float(p.get("notional", 0) or 0),
                    "entry_price": float(p.get("entryPrice", 0) or 0),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0) or 0),
                    "liquidation_price": float(p.get("liquidationPrice", 0) or 0),
                    "leverage": float(p.get("leverage", 1) or 1),
                    "margin": float(p.get("initialMargin", 0) or 0),
                }
                for p in positions
                if float(p.get("contracts", 0) or 0) > 0
            ]
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                    amount: float, price: Optional[float] = None) -> OrderResult:
        amount = self.round_amount(symbol, amount)
        if price is not None:
            price = self.round_price(symbol, price)

        side_str = side.value
        if order_type == OrderType.MARKET:
            order = self.exchange.create_order(symbol, "market", side_str, amount)
        elif order_type == OrderType.LIMIT:
            order = self.exchange.create_order(
                symbol, "limit", side_str, amount, price,
                {"postOnly": True}  # Ensure maker fee (0.02%)
            )
        else:
            order = self.exchange.create_order(symbol, "limit", side_str, amount, price)

        result = self._parse_order(order, symbol, side, order_type)
        logger.info(
            f"FUTURES {side_str.upper()} {order_type.value} {amount} {symbol}"
            f" @ {price or 'market'} -> {result.status.value}"
        )
        return result

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except ccxt.OrderNotFound:
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
            symbol=symbol, side=side, order_type=order_type,
            status=status,
            amount=float(raw.get("amount", 0) or 0),
            price=float(raw.get("price", 0) or raw.get("average", 0) or 0),
            filled=float(raw.get("filled", 0) or 0),
            cost=float(raw.get("cost", 0) or 0),
            fee=float(fee_info.get("cost", 0) or 0),
            timestamp=float(raw.get("timestamp", 0) or 0) / 1000,
            raw=raw,
        )
