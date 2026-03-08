import ccxt
import pandas as pd
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger("exchange")


class ExchangeClient:
    """Unified exchange client via ccxt."""

    def __init__(self, config: dict):
        self.config = config
        exc_cfg = config["exchange"]
        self.exchange_name = exc_cfg["name"]
        self.sandbox = exc_cfg.get("sandbox", True)

        exchange_class = getattr(ccxt, self.exchange_name)
        self.exchange = exchange_class({
            "apiKey": exc_cfg.get("api_key", ""),
            "secret": exc_cfg.get("api_secret", ""),
            "enableRateLimit": exc_cfg.get("rate_limit", True),
            "options": {"defaultType": "spot"},
        })

        if self.sandbox:
            self.exchange.set_sandbox_mode(True)
            logger.info(f"Running in SANDBOX mode on {self.exchange_name}")
        else:
            logger.warning(f"Running in LIVE mode on {self.exchange_name}!")

        self.exchange.load_markets()
        logger.info(f"Connected to {self.exchange_name}, {len(self.exchange.markets)} markets loaded")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m",
                    limit: int = 500) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def get_ticker(self, symbol: str) -> dict:
        return self.exchange.fetch_ticker(symbol)

    def get_balance(self, currency: str = "USDT") -> float:
        balance = self.exchange.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0))

    def get_price(self, symbol: str) -> float:
        ticker = self.get_ticker(symbol)
        return float(ticker["last"])

    def create_market_buy(self, symbol: str, amount: float,
                          params: Optional[dict] = None) -> dict:
        params = params or {}
        order = self.exchange.create_market_buy_order(symbol, amount, params)
        logger.info(f"BUY {amount} {symbol} @ market | order_id={order['id']}")
        return order

    def create_market_sell(self, symbol: str, amount: float,
                           params: Optional[dict] = None) -> dict:
        params = params or {}
        order = self.exchange.create_market_sell_order(symbol, amount, params)
        logger.info(f"SELL {amount} {symbol} @ market | order_id={order['id']}")
        return order

    def create_limit_buy(self, symbol: str, amount: float, price: float,
                         params: Optional[dict] = None) -> dict:
        params = params or {}
        order = self.exchange.create_limit_buy_order(symbol, amount, price, params)
        logger.info(f"LIMIT BUY {amount} {symbol} @ {price} | order_id={order['id']}")
        return order

    def create_limit_sell(self, symbol: str, amount: float, price: float,
                          params: Optional[dict] = None) -> dict:
        params = params or {}
        order = self.exchange.create_limit_sell_order(symbol, amount, price, params)
        logger.info(f"LIMIT SELL {amount} {symbol} @ {price} | order_id={order['id']}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        result = self.exchange.cancel_order(order_id, symbol)
        logger.info(f"CANCELLED order {order_id}")
        return result

    def get_open_orders(self, symbol: str) -> list:
        return self.exchange.fetch_open_orders(symbol)

    def get_min_order_amount(self, symbol: str) -> float:
        market = self.exchange.market(symbol)
        return float(market.get("limits", {}).get("amount", {}).get("min", 0))

    def get_min_order_cost(self, symbol: str) -> float:
        market = self.exchange.market(symbol)
        return float(market.get("limits", {}).get("cost", {}).get("min", 0))
