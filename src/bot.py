import time
import uuid
import schedule
from datetime import datetime
from src.exchange.client import ExchangeClient
from src.risk.manager import RiskManager
from src.strategies.base import BaseStrategy
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.utils.logger import setup_logger

logger = setup_logger("bot")

STRATEGY_MAP = {
    "rsi_macd": RsiMacdStrategy,
    "mean_reversion": MeanReversionStrategy,
    "grid": GridTradingStrategy,
    "dca_momentum": DCAMomentumStrategy,
}


class TradingBot:
    """Main trading bot that orchestrates exchange, strategy, and risk."""

    def __init__(self, config: dict):
        self.config = config
        self.mode = config["trading"].get("mode", "paper")
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]

        # Paper trading state
        self.paper_balance = float(config["trading"]["initial_capital"])
        self.paper_holdings = 0.0

        # Initialize components
        if self.mode == "live":
            self.exchange = ExchangeClient(config)
        else:
            self.exchange = None
            logger.info("Running in PAPER TRADING mode (no real exchange)")

        strategy_name = config["strategy"]["name"]
        if strategy_name not in STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(STRATEGY_MAP.keys())}"
            )
        self.strategy: BaseStrategy = STRATEGY_MAP[strategy_name](config)
        self.risk_manager = RiskManager(config)

        self.running = False
        self.cycle_count = 0

        logger.info(
            f"Bot initialized | Strategy={self.strategy.name()} | "
            f"Symbol={self.symbol} | TF={self.timeframe} | Mode={self.mode}"
        )

    def fetch_data(self):
        if self.exchange:
            return self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)

        # Paper mode: fetch from public API without auth
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
        raw = exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
        import pandas as pd
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def execute_buy(self, price: float, amount: float):
        cost = amount * price
        fee = cost * 0.001

        if self.mode == "live" and self.exchange:
            try:
                order = self.exchange.create_market_buy(self.symbol, amount)
                actual_price = float(order.get("average", price))
                self.risk_manager.open_trade(
                    order["id"], self.symbol, "buy", actual_price, amount
                )
                return order
            except Exception as e:
                logger.error(f"Buy order failed: {e}")
                return None
        else:
            # Paper trading
            if cost + fee > self.paper_balance:
                amount = (self.paper_balance * 0.99) / price
                cost = amount * price
                fee = cost * 0.001

            self.paper_balance -= (cost + fee)
            self.paper_holdings += amount

            trade_id = f"paper_{uuid.uuid4().hex[:8]}"
            self.risk_manager.open_trade(
                trade_id, self.symbol, "buy", price, amount
            )
            logger.info(
                f"[PAPER] BUY {amount:.6f} @ {price:.2f} "
                f"| Cost=${cost:.2f} Fee=${fee:.4f} "
                f"| Balance=${self.paper_balance:.2f}"
            )
            return {"id": trade_id, "status": "paper"}

    def execute_sell(self, price: float):
        if not self.risk_manager.open_trades:
            return None

        if self.mode == "live" and self.exchange:
            for trade in list(self.risk_manager.open_trades):
                try:
                    order = self.exchange.create_market_sell(
                        self.symbol, trade.amount
                    )
                    actual_price = float(order.get("average", price))
                    self.risk_manager.close_trade(trade, actual_price, "signal")
                    return order
                except Exception as e:
                    logger.error(f"Sell order failed: {e}")
                    return None
        else:
            # Paper trading
            for trade in list(self.risk_manager.open_trades):
                revenue = trade.amount * price
                fee = revenue * 0.001
                self.paper_balance += (revenue - fee)
                self.paper_holdings -= trade.amount
                self.risk_manager.close_trade(trade, price, "signal")
                logger.info(
                    f"[PAPER] SELL {trade.amount:.6f} @ {price:.2f} "
                    f"| Revenue=${revenue:.2f} Fee=${fee:.4f} "
                    f"| Balance=${self.paper_balance:.2f}"
                )
            return {"status": "paper_sold"}

    def run_cycle(self):
        """Execute one trading cycle."""
        self.cycle_count += 1

        try:
            # Fetch market data
            df = self.fetch_data()
            if df is None or len(df) < 60:
                logger.warning("Insufficient data, skipping cycle")
                return

            current_price = float(df.iloc[-1]["close"])

            # Check stops first
            stopped = self.risk_manager.check_stops(self.symbol, current_price)
            if stopped:
                for trade in stopped:
                    if self.mode != "live":
                        revenue = trade.amount * trade.exit_price
                        self.paper_balance += revenue
                        self.paper_holdings -= trade.amount

            # Check if we can trade
            can_trade, reason = self.risk_manager.can_open_trade()

            # Generate signal
            signal = self.strategy.generate_signal(df)

            # Execute
            if signal == "buy" and can_trade:
                stop_price = self.risk_manager.get_stop_loss(current_price, "buy")
                amount = self.risk_manager.calculate_position_size(
                    current_price, stop_price
                )

                min_cost = 5.0
                if amount * current_price >= min_cost:
                    self.execute_buy(current_price, amount)
                else:
                    logger.debug(
                        f"Order too small: ${amount * current_price:.2f} < ${min_cost}"
                    )

            elif signal == "sell" and self.risk_manager.open_trades:
                self.execute_sell(current_price)

            # Log status periodically
            if self.cycle_count % 10 == 0:
                self._log_status(current_price)

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

    def _log_status(self, price: float):
        stats = self.risk_manager.get_stats()
        balance = self.paper_balance if self.mode != "live" else "N/A"
        open_count = len(self.risk_manager.open_trades)
        logger.info(
            f"[Cycle #{self.cycle_count}] Price={price:.2f} | "
            f"Balance=${balance} | Open={open_count} | "
            f"Trades={stats['total_trades']} | "
            f"WR={stats['win_rate']:.0%} | PnL=${stats['total_pnl']:+.2f}"
        )

    def start(self):
        """Start the bot with scheduled execution."""
        self.running = True
        logger.info(f"Starting bot... Timeframe={self.timeframe}")

        # Map timeframe to seconds
        tf_seconds = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
        }
        interval = tf_seconds.get(self.timeframe, 900)

        # Run first cycle immediately
        self.run_cycle()

        # Schedule subsequent cycles
        logger.info(f"Scheduling cycles every {interval}s ({self.timeframe})")

        while self.running:
            try:
                time.sleep(interval)
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Bot stopped by user (Ctrl+C)")
                self.stop()
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(10)

    def stop(self):
        self.running = False
        stats = self.risk_manager.get_stats()
        logger.info("=" * 50)
        logger.info("BOT STOPPED - Final Stats:")
        for key, val in stats.items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 50)
