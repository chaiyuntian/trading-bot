"""Async Trading Bot — event-driven, WebSocket-fed, multi-alpha architecture.

Key differences from sync bot (bot.py):
  - WebSocket data feed (no polling, lowest latency)
  - Event-driven: fires on candle close, not on timer
  - Alpha signal combiner (21 atomic signals) instead of monolithic strategies
  - Async order execution (non-blocking)
  - Candle buffer (no repeated REST fetches)

Usage:
  python -m src.main --async
"""

import asyncio
import time
from datetime import datetime, timezone

from src.exchange.ccxt_ws_adapter import CCXTWebSocketAdapter
from src.exchange.base import OrderStatus
from src.alpha.combiner import SignalCombiner
from src.indicators.candle_buffer import CandleBuffer
from src.risk.manager import RiskManager
from src.strategies.base import Signal
from src.utils.logger import setup_logger

logger = setup_logger("bot.async")

_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


class AsyncTradingBot:
    """Event-driven trading bot with WebSocket feeds and alpha signal combiner."""

    def __init__(self, config: dict):
        self.config = config
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]
        self.mode = config["trading"].get("mode", "paper")

        self.exchange = CCXTWebSocketAdapter(config)
        self.combiner = SignalCombiner(config)
        self.risk_manager = RiskManager(config)
        self.candle_buffer = CandleBuffer(max_size=500)

        self._running = False
        self._cycle_count = 0
        self._last_candle_ts = 0
        self._last_daily_reset = datetime.now(timezone.utc).date()
        self._tf_seconds = _TF_SECONDS.get(self.timeframe, 900)

    async def start(self):
        """Start the async bot."""
        logger.info(
            f"AsyncBot starting | Symbol={self.symbol} | TF={self.timeframe} | "
            f"Mode={self.mode} | Signals={len(self.combiner.signals)}"
        )

        await self.exchange.connect()

        # Warmup: fetch historical data via REST
        logger.info("Fetching warmup data...")
        df = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=500)
        self.candle_buffer.load(df)
        logger.info(f"Warmup complete: {len(df)} candles loaded")

        # Subscribe to WebSocket kline feed
        await self.exchange.subscribe_klines(
            self.symbol, self.timeframe, self._on_kline
        )

        self._running = True
        logger.info("Listening for candle events...")

        try:
            while self._running:
                await asyncio.sleep(1)
                self._check_daily_reset()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _on_kline(self, candle: dict):
        """Called on each WebSocket kline update."""
        ts = candle.get("timestamp")
        if ts is None:
            return

        # Detect candle close by timestamp change
        ts_epoch = int(ts.timestamp()) if hasattr(ts, "timestamp") else int(ts / 1000)
        candle_start = (ts_epoch // self._tf_seconds) * self._tf_seconds

        if candle_start != self._last_candle_ts and self._last_candle_ts != 0:
            # New candle started — previous candle just closed
            candle["closed"] = True
            self.candle_buffer.update(candle)
            await self._run_cycle()
        else:
            candle["closed"] = False
            self.candle_buffer.update(candle)

        self._last_candle_ts = candle_start

    async def _run_cycle(self):
        """Execute one trading cycle on candle close."""
        self._cycle_count += 1

        try:
            df = self.candle_buffer.to_dataframe()
            if df is None or len(df) < 60:
                return

            current_price = float(df.iloc[-1]["close"])

            # Check stops on open trades
            stopped = self.risk_manager.check_stops(self.symbol, current_price)
            for trade in stopped:
                try:
                    await self.exchange.market_sell(self.symbol, trade.amount)
                except Exception:
                    pass

            # Can we open new trades?
            can_trade, reason = self.risk_manager.can_open_trade()

            # Generate combined alpha signal
            t0 = time.monotonic()
            signal = self.combiner.combine(df)
            latency_ms = (time.monotonic() - t0) * 1000

            if signal.signal == Signal.BUY and can_trade:
                stop_price = self.risk_manager.get_stop_loss(current_price, "buy")
                amount = self.risk_manager.calculate_position_size(current_price, stop_price)

                if amount * current_price >= 5:
                    if signal.confidence < 0.5:
                        amount *= 0.5

                    result = await self.exchange.market_buy(self.symbol, amount)
                    if result.status != OrderStatus.REJECTED:
                        exec_price = result.price if result.price > 0 else current_price
                        self.risk_manager.open_trade(
                            result.id or f"async_{self._cycle_count}",
                            self.symbol, "buy", exec_price,
                            result.filled or amount,
                        )

            elif signal.signal == Signal.SELL and self.risk_manager.open_trades:
                for trade in list(self.risk_manager.open_trades):
                    result = await self.exchange.market_sell(self.symbol, trade.amount)
                    if result.status != OrderStatus.REJECTED:
                        exec_price = result.price if result.price > 0 else current_price
                        self.risk_manager.close_trade(trade, exec_price, "alpha_signal")

            # Periodic status log
            if self._cycle_count % 10 == 0:
                stats = self.risk_manager.get_stats()
                n_signals = signal.metadata.get("active_signals", 0) if signal.metadata else 0
                logger.info(
                    f"[Cycle #{self._cycle_count}] Price={current_price:.2f} | "
                    f"PnL=${stats['total_pnl']:+.2f} | "
                    f"Trades={stats['total_trades']} | "
                    f"WR={stats['win_rate']:.0%} | "
                    f"Signals={n_signals} | "
                    f"Latency={latency_ms:.1f}ms"
                )

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

    def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today != self._last_daily_reset:
            self.risk_manager.reset_daily()
            self._last_daily_reset = today
            logger.info("Daily risk counters reset")

    async def stop(self):
        self._running = False
        stats = self.risk_manager.get_stats()
        logger.info("=" * 60)
        logger.info("ASYNC BOT STOPPED — Final Stats:")
        for key, val in stats.items():
            logger.info(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
        logger.info("=" * 60)
        await self.exchange.disconnect()
