"""Async WebSocket exchange adapter using ccxt.pro.

Provides real-time data feeds via WebSocket with REST fallback for orders.
ccxt.pro is included in ccxt>=4.0.0 — no extra install needed.

Data flow:
  WebSocket -> kline events -> CandleBuffer -> IncrementalIndicators -> AlphaSignals
  Orders still go via REST (fastest reliable path for execution).
"""

import asyncio
from typing import Callable, Optional

import ccxt.pro as ccxtpro
import pandas as pd

from src.exchange.base import (
    ExchangeAdapter, OrderSide, OrderType, OrderStatus, OrderResult,
    Ticker, MarketInfo,
)
from src.utils.logger import setup_logger

logger = setup_logger("exchange.ws")

_STATUS_MAP = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
}


class CCXTWebSocketAdapter:
    """Async exchange adapter: WebSocket for data, REST for orders."""

    def __init__(self, config: dict):
        ex_cfg = config.get("exchange", {})
        self.exchange_name = ex_cfg.get("name", "binance")
        self.api_key = ex_cfg.get("api_key", "")
        self.api_secret = ex_cfg.get("api_secret", "")
        self.sandbox = ex_cfg.get("sandbox", True)

        self._exchange: Optional[ccxtpro.Exchange] = None
        self._subscriptions: dict[str, asyncio.Task] = {}
        self._market_info_cache: dict[str, MarketInfo] = {}

    async def connect(self):
        exchange_class = getattr(ccxtpro, self.exchange_name, None)
        if exchange_class is None:
            raise ValueError(f"Exchange not supported: {self.exchange_name}")

        self._exchange = exchange_class({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        if self.sandbox:
            self._exchange.set_sandbox_mode(True)

        await self._exchange.load_markets()
        logger.info(f"Connected to {self.exchange_name} (WebSocket)")

    async def disconnect(self):
        for task in self._subscriptions.values():
            task.cancel()
        self._subscriptions.clear()
        if self._exchange:
            await self._exchange.close()
        logger.info("Disconnected")

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                          limit: int = 500) -> pd.DataFrame:
        """REST fetch for initial warmup data."""
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    async def subscribe_klines(self, symbol: str, timeframe: str,
                               callback: Callable):
        """Subscribe to real-time kline/candlestick updates."""
        async def _loop():
            while True:
                try:
                    ohlcv = await self._exchange.watch_ohlcv(symbol, timeframe)
                    if ohlcv:
                        for candle in ohlcv:
                            ts, o, h, l, c, v = candle
                            await callback({
                                "timestamp": pd.Timestamp(ts, unit="ms"),
                                "open": o, "high": h, "low": l,
                                "close": c, "volume": v,
                                "closed": False,  # ccxt.pro doesn't reliably flag this
                            })
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Kline WS error: {e}, reconnecting...")
                    await asyncio.sleep(1)

        task = asyncio.create_task(_loop())
        self._subscriptions[f"klines_{symbol}_{timeframe}"] = task
        logger.info(f"Subscribed to klines: {symbol} {timeframe}")

    async def subscribe_trades(self, symbol: str, callback: Callable):
        """Subscribe to real-time trade updates (tick data)."""
        async def _loop():
            while True:
                try:
                    trades = await self._exchange.watch_trades(symbol)
                    for trade in trades:
                        await callback({
                            "price": trade["price"],
                            "amount": trade["amount"],
                            "side": trade["side"],
                            "timestamp": trade["timestamp"],
                        })
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Trades WS error: {e}, reconnecting...")
                    await asyncio.sleep(1)

        task = asyncio.create_task(_loop())
        self._subscriptions[f"trades_{symbol}"] = task

    async def get_ticker(self, symbol: str) -> Ticker:
        ticker = await self._exchange.fetch_ticker(symbol)
        return Ticker(
            symbol=symbol,
            last=ticker.get("last", 0),
            bid=ticker.get("bid", 0),
            ask=ticker.get("ask", 0),
            volume_24h=ticker.get("quoteVolume", 0),
            timestamp=ticker.get("timestamp", 0),
        )

    async def place_order(self, symbol: str, side: OrderSide,
                          order_type: OrderType, amount: float,
                          price: float = None) -> OrderResult:
        """Place order via REST (most reliable for execution)."""
        try:
            side_str = "buy" if side == OrderSide.BUY else "sell"
            type_str = "market" if order_type == OrderType.MARKET else "limit"

            result = await self._exchange.create_order(
                symbol, type_str, side_str, amount, price
            )

            return OrderResult(
                id=result["id"],
                symbol=symbol,
                side=side,
                type=order_type,
                status=_STATUS_MAP.get(result.get("status", ""), OrderStatus.PENDING),
                amount=amount,
                price=result.get("average", result.get("price", 0)) or 0,
                filled=result.get("filled", 0),
                cost=result.get("cost", 0),
                fee=result.get("fee", {}).get("cost", 0),
                timestamp=result.get("timestamp", 0),
                raw=result,
            )
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return OrderResult(
                id="", symbol=symbol, side=side, type=order_type,
                status=OrderStatus.REJECTED, amount=amount, price=0,
                filled=0, cost=0, fee=0, timestamp=0, raw={"error": str(e)},
            )

    async def market_buy(self, symbol: str, amount: float) -> OrderResult:
        return await self.place_order(symbol, OrderSide.BUY, OrderType.MARKET, amount)

    async def market_sell(self, symbol: str, amount: float) -> OrderResult:
        return await self.place_order(symbol, OrderSide.SELL, OrderType.MARKET, amount)

    async def get_balance(self, currency: str = "USDT") -> float:
        balance = await self._exchange.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0))
