"""Main trading bot — orchestrates exchange, strategy, and risk management.

Uses the ExchangeAdapter abstraction layer so it works with any platform:
- CCXTAdapter for live trading on 100+ exchanges
- PaperAdapter for simulated trading with real market data
- Any future adapter (Alpaca, IBKR, DEX, etc.)
"""

import time
import uuid
from datetime import datetime, timezone

from src.exchange.base import ExchangeAdapter, OrderSide, OrderType, OrderStatus
from src.exchange.ccxt_adapter import CCXTAdapter
from src.exchange.paper_adapter import PaperAdapter
from src.risk.manager import RiskManager
from src.strategies.base import BaseStrategy, Signal
from src.strategies.rsi_macd import RsiMacdStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.grid_trading import GridTradingStrategy
from src.strategies.dca_momentum import DCAMomentumStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.kama_trend import KamaTrendStrategy
from src.utils.logger import setup_logger

logger = setup_logger("bot")

STRATEGY_MAP = {
    "rsi_macd": RsiMacdStrategy,
    "mean_reversion": MeanReversionStrategy,
    "grid": GridTradingStrategy,
    "dca_momentum": DCAMomentumStrategy,
    "ensemble": EnsembleStrategy,
    "kama_trend": KamaTrendStrategy,
}


class AlphaCombinerStrategy:
    """Adapter: wraps SignalCombiner as a BaseStrategy for backtest compatibility."""
    strategy_name = "Alpha Combiner"

    def __init__(self, config: dict):
        from src.alpha.combiner import SignalCombiner
        self.combiner = SignalCombiner(config)
        self.config = config
        self.params = config.get("strategy", {}).get("params", {})

    def name(self):
        return self.strategy_name

    def generate_signal(self, df):
        sig = self.combiner.combine(df)
        return sig.signal.value

    def generate_rich_signal(self, df):
        return self.combiner.combine(df)


STRATEGY_MAP["alpha"] = AlphaCombinerStrategy


class ThinkingStrategy:
    """Adapter: wraps ThinkingEngine as a strategy for backtest/dashboard."""
    strategy_name = "Thinking Engine"

    def __init__(self, config: dict):
        from src.alpha.thinking import ThinkingEngine
        self.engine = ThinkingEngine(config)
        self.config = config
        self.params = config.get("strategy", {}).get("params", {})

    def name(self):
        return self.strategy_name

    def generate_signal(self, df):
        sig = self.engine.think(df)
        return sig.signal.value

    def generate_rich_signal(self, df):
        return self.engine.think(df)


STRATEGY_MAP["thinking"] = ThinkingStrategy

# Timeframe to seconds mapping
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def create_exchange(config: dict) -> ExchangeAdapter:
    """Factory: create the right exchange adapter based on config."""
    mode = config["trading"].get("mode", "paper")
    if mode == "live":
        adapter = CCXTAdapter(config)
    else:
        adapter = PaperAdapter(config)
    adapter.connect()
    return adapter


class TradingBot:
    """Main trading bot that orchestrates exchange, strategy, and risk.

    Platform-agnostic: operates through ExchangeAdapter interface.
    Can be connected to any exchange or broker by swapping the adapter.
    """

    def __init__(self, config: dict, exchange: ExchangeAdapter = None):
        self.config = config
        self.mode = config["trading"].get("mode", "paper")
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]

        # Exchange adapter (platform-agnostic)
        self.exchange: ExchangeAdapter = exchange or create_exchange(config)

        # Strategy
        strategy_name = config["strategy"]["name"]
        if strategy_name not in STRATEGY_MAP:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {list(STRATEGY_MAP.keys())}"
            )
        self.strategy: BaseStrategy = STRATEGY_MAP[strategy_name](config)
        self.risk_manager = RiskManager(config)

        # Performance: cache last OHLCV fetch to avoid redundant API calls
        self._last_df = None
        self._last_fetch_ts = 0
        self._min_fetch_interval = max(30, _TF_SECONDS.get(self.timeframe, 900) * 0.5)

        self.running = False
        self.cycle_count = 0
        self._last_daily_reset = datetime.now(timezone.utc).date()

        logger.info(
            f"Bot initialized | Strategy={self.strategy.name()} | "
            f"Symbol={self.symbol} | TF={self.timeframe} | Mode={self.mode}"
        )

    def fetch_data(self):
        """Fetch OHLCV data via the exchange adapter, with caching."""
        now = time.monotonic()
        if self._last_df is not None and (now - self._last_fetch_ts) < self._min_fetch_interval:
            return self._last_df

        df = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
        self._last_df = df
        self._last_fetch_ts = now
        return df

    def _execute_buy(self, price: float, amount: float):
        """Execute a buy order through the exchange adapter."""
        try:
            result = self.exchange.market_buy(self.symbol, amount)
            if result.status == OrderStatus.REJECTED:
                logger.warning(f"Buy rejected: insufficient balance")
                return None

            exec_price = result.price if result.price > 0 else price
            self.risk_manager.open_trade(
                result.id, self.symbol, "buy", exec_price, result.filled or amount
            )
            return result
        except Exception as e:
            logger.error(f"Buy order failed: {e}")
            return None

    def _execute_sell(self, price: float):
        """Close all open positions via the exchange adapter."""
        if not self.risk_manager.open_trades:
            return None

        results = []
        for trade in list(self.risk_manager.open_trades):
            try:
                result = self.exchange.market_sell(self.symbol, trade.amount)
                if result.status == OrderStatus.REJECTED:
                    logger.warning(f"Sell rejected for trade {trade.id}")
                    continue
                exec_price = result.price if result.price > 0 else price
                self.risk_manager.close_trade(trade, exec_price, "signal")
                results.append(result)
            except Exception as e:
                logger.error(f"Sell order failed for trade {trade.id}: {e}")

        return results if results else None

    def _check_daily_reset(self):
        """Reset daily loss counters at UTC midnight."""
        today = datetime.now(timezone.utc).date()
        if today != self._last_daily_reset:
            self.risk_manager.reset_daily()
            self._last_daily_reset = today
            logger.info("Daily risk counters reset")

    def run_cycle(self):
        """Execute one trading cycle."""
        self.cycle_count += 1
        self._check_daily_reset()

        try:
            df = self.fetch_data()
            if df is None or len(df) < 60:
                logger.warning("Insufficient data, skipping cycle")
                return

            current_price = float(df.iloc[-1]["close"])

            # Check stop-loss / take-profit on open trades
            stopped = self.risk_manager.check_stops(self.symbol, current_price)
            for trade in stopped:
                # Execute the stop on exchange
                try:
                    self.exchange.market_sell(self.symbol, trade.amount)
                except Exception:
                    pass  # Already closed in risk manager

            # Can we open new trades?
            can_trade, reason = self.risk_manager.can_open_trade()

            # Generate signal (uses rich signal for confidence)
            trade_signal = self.strategy.generate_rich_signal(df)

            if trade_signal.signal == Signal.BUY and can_trade:
                stop_price = self.risk_manager.get_stop_loss(current_price, "buy")
                amount = self.risk_manager.calculate_position_size(
                    current_price, stop_price
                )

                # Check minimum order size
                try:
                    info = self.exchange.get_market_info(self.symbol)
                    min_cost = max(info.min_cost, 5.0)
                except Exception:
                    min_cost = 5.0

                if amount * current_price >= min_cost:
                    # Scale position by confidence
                    if trade_signal.confidence < 0.5:
                        amount *= 0.5  # Half position on low confidence
                    self._execute_buy(current_price, amount)
                else:
                    logger.debug(
                        f"Order too small: ${amount * current_price:.2f} < ${min_cost}"
                    )

            elif trade_signal.signal == Signal.SELL and self.risk_manager.open_trades:
                self._execute_sell(current_price)

            # Log status periodically
            if self.cycle_count % 10 == 0:
                self._log_status(current_price)

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

    def _log_status(self, price: float):
        stats = self.risk_manager.get_stats()
        open_count = len(self.risk_manager.open_trades)

        # Get portfolio value for paper trading
        portfolio = ""
        if isinstance(self.exchange, PaperAdapter):
            total = self.exchange.get_total_value(self.symbol)
            portfolio = f"Portfolio=${total:.2f} | "

        logger.info(
            f"[Cycle #{self.cycle_count}] Price={price:.2f} | "
            f"{portfolio}"
            f"Open={open_count} | "
            f"Trades={stats['total_trades']} | "
            f"WR={stats['win_rate']:.0%} | PnL=${stats['total_pnl']:+.2f} | "
            f"DD={stats['max_drawdown']:.1%}"
        )

    def start(self):
        """Start the bot with scheduled execution."""
        self.running = True
        interval = _TF_SECONDS.get(self.timeframe, 900)

        logger.info(f"Starting bot... Interval={interval}s ({self.timeframe})")

        # Run first cycle immediately
        self.run_cycle()

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

        logger.info("=" * 60)
        logger.info("BOT STOPPED - Final Stats:")
        for key, val in stats.items():
            if isinstance(val, float):
                logger.info(f"  {key}: {val:.4f}")
            else:
                logger.info(f"  {key}: {val}")
        logger.info("=" * 60)

        try:
            self.exchange.disconnect()
        except Exception:
            pass
